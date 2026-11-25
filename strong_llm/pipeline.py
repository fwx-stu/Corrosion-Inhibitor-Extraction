"""
强 LLM 抽取策略实现

策略清单:
  - E0: 长上下文直接抽取（零检索强 LLM 基线）
  - E1: Wiki-RAG（Milvus 全局检索 → 领域知识蒸馏 → 全文抽取，2次LLM调用）
  - E3: 本地 Local-RAG（ChromaDB 字段级检索 + 重排序）
  - E4: E3 + RAG 验证（基于关键值检索证据块验证/修正）
  - E5: 全局 RAG（Milvus 全局检索，1次LLM调用）
  - E6: 混合 RAG（Local + Global，1次LLM调用）

设计要点:
  - E0 不 import retrieval.py —— 零基础设施依赖
  - E1/E5/E6 lazy import retrieval（Milvus 全局检索）
  - E3/E4 lazy import retrieval（需要 embedding + reranker）
  - 所有策略共享工具函数和统一输出格式
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
# 复用 idea/llm_client.py（由 config.py 已将 idea/ 加入 sys.path）
from llm_client import LLMClient
from utils import load_prompt, render_prompt


# 缓存 metadata_index.json 以避免重复 I/O
_META_INDEX_CACHE: Optional[Dict[str, Any]] = None


def _lookup_doi(paper_id: str) -> Optional[str]:
    """从 metadata_index.json 查找 paper_id 对应的 DOI。"""
    global _META_INDEX_CACHE
    if _META_INDEX_CACHE is None:
        meta_path = config.METADATA_INDEX_PATH
        if meta_path.exists():
            try:
                _META_INDEX_CACHE = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                _META_INDEX_CACHE = {}
        else:
            _META_INDEX_CACHE = {}
    entry = _META_INDEX_CACHE.get(paper_id) or _META_INDEX_CACHE.get(f"{paper_id}.md", {})
    if isinstance(entry, dict):
        return entry.get("doi")
    return None


def extract_json_str(text: str) -> Optional[str]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def safe_parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    js = extract_json_str(text)
    if not js:
        return None, "No JSON object detected in model output."
    try:
        return json.loads(js), None
    except json.JSONDecodeError as e:
        # 尝试容错恢复：截断到最后一个完整的 record，救回已解析数据
        recovered = _try_recover_json(js)
        if recovered is not None:
            return recovered, f"JSON parse error (recovered partial): {e}"
        return None, f"JSON parse error: {e}"


def _build_json_closing(candidate: str) -> str:
    """通过栈跟踪嵌套，生成正确的闭合字符序列（解决 ]]}} vs ]}]} 顺序问题）。"""
    stack = []
    in_string = False
    escape_next = False
    for ch in candidate:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
    return ''.join(']' if c == '[' else '}' for c in reversed(stack))


def _try_recover_json(broken_json: str) -> Optional[Dict[str, Any]]:
    """
    当 JSON 在 records/inhibitors 数组中间断裂时，逐步向前回退到最后一个完整
    record 的 '},' 截断后补全 ]} 来恢复。
    支持扁平格式 {"records": [...]} 和分组格式 {"inhibitors": [...]}.
    """
    # 找到数组的起始位置：优先 "records"，其次 "inhibitors"
    idx = broken_json.find('"records"')
    if idx == -1:
        idx = broken_json.find('"records"', broken_json.find('"corrosion_inhibition_analysis"'))
    if idx == -1:
        idx = broken_json.find('"inhibitors"')
    if idx == -1:
        return None

    # 从末尾向前查找最后一个完整 record 的结束标记 "},"
    last_pos = len(broken_json) - 1
    while last_pos > idx:
        last_pos = broken_json.rfind("},", idx, last_pos)
        if last_pos == -1:
            break
        # 截断：保留到 "}," 的 "}"，然后用栈计算正确的闭合序列
        candidate = broken_json[:last_pos + 1]  # 到 "}" 为止
        closing = _build_json_closing(candidate)
        try:
            obj = json.loads(candidate + closing)
            return obj
        except json.JSONDecodeError:
            last_pos -= 1
            continue
    return None


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """标准化单条抽取记录（13 字段 + evidence）。
    
    支持短字段名（inh/sub/...）和长字段名（inhibitor_name/substrate/...）。
    """
    def ensure_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i) for i in x if i is not None]
        return [str(x)]

    row = {}
    for field in config.FIELD_NAMES:
        # 先尝试长名，再尝试短名
        short = config.FIELD_LONG_TO_SHORT.get(field)
        val = rec.get(field) or (rec.get(short) if short else None)
        row[field] = val
    row["evidence_text"] = ensure_list(rec.get("evidence_text"))
    row["evidence_source"] = ensure_list(rec.get("evidence_source"))
    return row


def normalize_result(parsed: Optional[Dict[str, Any]], paper_id: str) -> Dict[str, Any]:
    """将 LLM 返回的 JSON 标准化为 {paper_id, records} 结构。
    
    自动兼容两种格式:
      - 扁平格式: {"records": [...]}
      - 分组格式: {"inhibitors": [...]}
    """
    if not isinstance(parsed, dict):
        return {"paper_id": paper_id, "records": []}

    # 优先检测分组格式 {"inhibitors": [...]}
    if isinstance(parsed.get("inhibitors"), list):
        return flatten_grouped_result(parsed, paper_id)

    records = None
    if isinstance(parsed.get("records"), list):
        records = parsed["records"]
    elif isinstance(parsed.get("corrosion_inhibition_analysis"), dict):
        cia = parsed["corrosion_inhibition_analysis"]
        if isinstance(cia.get("records"), list):
            records = cia["records"]

    if not isinstance(records, list):
        records = []

    return {
        "paper_id": paper_id,
        "records": [normalize_record(r) for r in records if isinstance(r, dict)],
    }


def remove_duplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """基于核心字段去重。"""
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in records:
        key = json.dumps(
            {k: r.get(k) for k in config.FIELD_NAMES},
            ensure_ascii=False,
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _build_output(
    paper_id: str,
    source_file: str,
    strategy: str,
    model: str,
    result: Dict,
    raw_output: str,
    parse_error: Optional[str],
    resp=None,
    global_chunks: list = None,
    local_chunks: list = None,
) -> Dict[str, Any]:
    """统一输出格式。自动对 records 做后处理去重，防止 LLM 复读膨胀。"""
    # 后处理去重：移除 LLM 输出中完全相同的重复记录
    if isinstance(result, dict) and isinstance(result.get("records"), list):
        before = len(result["records"])
        result["records"] = remove_duplicate_records(result["records"])
        after = len(result["records"])
        if before > after:
            dedup_msg = f"Post-processing removed {before - after} duplicate records ({before} → {after})"
            parse_error = f"{parse_error}; {dedup_msg}" if parse_error else dedup_msg

    token_stats = {}
    if resp is not None:
        token_stats = {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_ms": resp.latency_ms,
        }
    return {
        "paper_id": paper_id,
        "source_file": source_file,
        "strategy": strategy,
        "model": model,
        "result": result,
        "global_retrieved_chunks": global_chunks or [],
        "local_retrieved_chunks": local_chunks or [],
        "raw_output": raw_output,
        "parse_error": parse_error,
        "token_stats": token_stats,
    }


# ============================================================
# 分组结构 → 扁平记录转换
# ============================================================
# 分组字段映射（短名 → 长名）
_GROUP_SHORT_MAP = {
    "inh": "inhibitor_name", "sub": "substrate", "med": "corrosive_medium",
    "homo": "homo", "lumo": "lumo", "dg": "delta_g_ads", "eads": "adsorption_energy",
}
# 数据点字段映射
_DATA_SHORT_MAP = {
    "med_c": "medium_concentration", "inh_c": "inhibitor_concentration",
    "temp": "temperature", "time": "immersion_time",
    "method": "test_method", "ie": "inhibition_efficiency",
}


def flatten_grouped_result(parsed: Optional[Dict[str, Any]], paper_id: str) -> Dict[str, Any]:
    """将分组结构 {inhibitors: [...]} 展开为扁平 {paper_id, records: [...]}。"""
    if not isinstance(parsed, dict):
        return {"paper_id": paper_id, "records": []}

    inhibitors = parsed.get("inhibitors")
    if not isinstance(inhibitors, list):
        # 不是分组结构，回退到标准解析（避免递归回 flatten_grouped_result）
        records = None
        if isinstance(parsed.get("records"), list):
            records = parsed["records"]
        elif isinstance(parsed.get("corrosion_inhibition_analysis"), dict):
            cia = parsed["corrosion_inhibition_analysis"]
            if isinstance(cia.get("records"), list):
                records = cia["records"]
        if not isinstance(records, list):
            records = []
        return {
            "paper_id": paper_id,
            "records": [normalize_record(r) for r in records if isinstance(r, dict)],
        }

    records = []
    for group in inhibitors:
        if not isinstance(group, dict):
            continue
        # 提取组级字段
        shared = {}
        for short, long in _GROUP_SHORT_MAP.items():
            val = group.get(short) or group.get(long)
            if val is not None:
                shared[long] = val

        data_points = group.get("data", [])
        if not isinstance(data_points, list) or not data_points:
            # 没有 data 数组时，仅用组级字段创建一条记录
            records.append(normalize_record({**shared}))
            continue

        for dp in data_points:
            if not isinstance(dp, dict):
                continue
            row = dict(shared)
            for short, long in _DATA_SHORT_MAP.items():
                val = dp.get(short) or dp.get(long)
                if val is not None:
                    row[long] = val
            records.append(normalize_record(row))

    return {"paper_id": paper_id, "records": records}


# ============================================================
# E0 — 长上下文直接抽取 (阶段 A)
# ============================================================
def run_e0(
    paper_id: str,
    md_path: Path,
    llm_client: LLMClient,
    params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    读取 MD 全文 → 拼接抽取提示词 → LLM 单次调用 → 解析结果。
    无检索，1 次 LLM 调用。
    """
    params = params or {}
    max_chars = params.get("max_doc_chars", config.DEFAULT_PARAMS["max_doc_chars"])

    text = Path(md_path).read_text(encoding="utf-8")
    if len(text) > max_chars:
        text = text[:max_chars]

    prompt_template = load_prompt(config.PROMPTS_DIR / "extract.txt")
    prompt = render_prompt(prompt_template, content=text)
    resp = llm_client.call(prompt)
    parsed, parse_err = safe_parse_json(resp.content)
    result = normalize_result(parsed, paper_id)

    return _build_output(
        paper_id=paper_id,
        source_file=Path(md_path).name,
        strategy="E0",
        model=llm_client.model,
        result=result,
        raw_output=resp.content,
        parse_error=parse_err,
        resp=resp,
    )


# ============================================================
# E1 — Wiki-RAG（全局知识蒸馏 + 全文抽取，2 次 LLM 调用）
# ============================================================
def run_e1(
    paper_id: str,
    md_path: Path,
    llm_client: LLMClient,
    params: Dict[str, Any] = None,
    chunks: List[Dict[str, Any]] = None,  # 保留参数兼容性，此策略不实际使用 chunks
) -> Dict[str, Any]:
    """
    Wiki-RAG：从 Milvus 全局知识库检索领域相关 chunks，
    用 LLM 蒸馏为结构化 Domain Wiki，再结合目标论文全文抽取。

    流程：
      1. Milvus 全局检索（排除自身，去重）→ global_chunks
      2. wiki_summarize.txt 调用 → 领域知识摘要（wiki_text）
      3. [Domain Wiki] + [Target Paper] 拼接 → extract_wiki.txt 抽取
    总计 2 次 LLM 调用（对比旧 E1 的 N+1 次）。
    """
    from retrieval import MilvusRetriever  # lazy import

    params = params or {}
    global_top_k = params.get("wiki_global_top_k", params.get("global_top_k", config.WIKI_GLOBAL_TOP_K))
    max_chars = params.get("max_doc_chars", config.DEFAULT_PARAMS.get("max_doc_chars", 80000))

    # 1. 读取目标论文全文
    target_text = Path(md_path).read_text(encoding="utf-8")[:max_chars]

    # 2. 构建字段级查询
    field_queries = list(config.FIELD_QUERIES.values())

    # 3. Milvus 全局检索（排除同 paper 的 chunks）
    milvus = MilvusRetriever()
    exclude_doi = _lookup_doi(paper_id)
    global_hits = milvus.search(
        queries=field_queries,
        top_k=global_top_k,
        exclude_paper_id=paper_id,
        exclude_doi=exclude_doi,
        dedup_threshold=config.MILVUS_DEDUP_THRESHOLD,
    )

    # 4. 格式化全局 chunks 作为 wiki 蒸馏素材
    context_parts = []
    for i, ch in enumerate(global_hits, start=1):
        context_parts.append(f"[Fragment-{i} | {ch['chunk_id']}]\n{ch['text']}\n")
    global_context = "\n".join(context_parts)

    # 5. 第一次 LLM 调用：将全局 chunks 蒸馏为 Domain Wiki
    wiki_template = load_prompt(config.PROMPTS_DIR / "wiki_summarize.txt")
    wiki_prompt = render_prompt(wiki_template, global_context=global_context)
    wiki_resp = llm_client.call(wiki_prompt)
    wiki_text = wiki_resp.content.strip()

    # 6. 拼接：Domain Wiki + 目标论文全文
    combined = (
        f"[Domain Wiki]\n{wiki_text}\n\n"
        f"[Target Paper]\n{target_text}"
    )

    # 7. 第二次 LLM 调用：基于 wiki + 全文抽取
    extract_template = load_prompt(config.PROMPTS_DIR / "extract_wiki.txt")
    extract_prompt = render_prompt(extract_template, content=combined)
    resp = llm_client.call(extract_prompt)
    parsed, parse_err = safe_parse_json(resp.content)
    result = normalize_result(parsed, paper_id)

    return _build_output(
        paper_id=paper_id,
        source_file=Path(md_path).name,
        strategy="E1",
        model=llm_client.model,
        result=result,
        raw_output=resp.content,
        parse_error=parse_err,
        resp=resp,
        global_chunks=global_hits,
    )


# ============================================================
# 共享本地检索管道（E3 / E6 共用）
# ============================================================
def _local_retrieval_pipeline(
    paper_id: str,
    params: Dict[str, Any],
    chunks: List[Dict[str, Any]] = None,
    md_path: Path = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    字段级语义检索 + 逐字段重排 + 窗口扩展 + 占位符过滤。

    Returns: (chunks, local_hits)
    """
    from retrieval import LocalRetriever  # lazy import

    params = params or {}
    top_k = params.get("retrieval_top_k", config.DEFAULT_PARAMS["retrieval_top_k"])
    rerank_top_n = params.get("rerank_top_n", config.DEFAULT_PARAMS["rerank_top_n"])
    window_size = params.get("window_size", config.DEFAULT_PARAMS["window_size"])
    enable_reranker = params.get("enable_reranker", config.DEFAULT_PARAMS["enable_reranker"])
    enable_window = params.get("enable_window_expand", config.DEFAULT_PARAMS["enable_window_expand"])

    # 1. 获取 chunks
    if chunks is None:
        from prepare import split_single_md
        _, chunks = split_single_md(
            md_path,
            chunk_size=params.get("chunk_size", config.DEFAULT_PARAMS["chunk_size"]),
            chunk_overlap=params.get("chunk_overlap", config.DEFAULT_PARAMS["chunk_overlap"]),
        )

    # 2. 字段级语义查询
    field_top_k = params.get("field_retrieval_top_k", config.FIELD_RETRIEVAL_TOP_K)
    max_chunks = params.get("max_final_chunks", config.MAX_FINAL_CHUNKS)
    cs = params.get("chunk_size", config.DEFAULT_PARAMS["chunk_size"])
    co = params.get("chunk_overlap", config.DEFAULT_PARAMS["chunk_overlap"])
    bm25_weight = params.get("bm25_weight", config.DEFAULT_PARAMS.get("bm25_weight", 0.0))
    cache_key = f"{paper_id}_cs{cs}_co{co}"
    local = LocalRetriever()

    seen_ids: dict = {}
    for field_name, field_query in config.FIELD_QUERIES.items():
        hits = local.retrieve(
            chunks, [field_query], top_k=field_top_k,
            cache_key=cache_key, bm25_weight=bm25_weight,
        )
        for h in hits:
            cid = h.get("chunk_id", "")
            if cid not in seen_ids or h.get("retrieval_score", 0) > seen_ids[cid].get("retrieval_score", 0):
                seen_ids[cid] = h

    all_hits = sorted(seen_ids.values(), key=lambda x: x.get("retrieval_score", 0), reverse=True)
    all_hits = all_hits[:top_k]

    # 3. 逐字段重排
    if enable_reranker and len(all_hits) > min(rerank_top_n, max_chunks):
        reranked_ids: dict = {}
        per_field_top = max(1, min(rerank_top_n, max_chunks) // len(config.FIELD_QUERIES))
        for _field_name, field_query in config.FIELD_QUERIES.items():
            field_reranked = local.rerank(field_query, all_hits, top_n=per_field_top)
            for ch in field_reranked:
                cid = ch.get("chunk_id", "")
                score = ch.get("rerank_score", 0.0)
                if cid not in reranked_ids or score > reranked_ids[cid][1]:
                    reranked_ids[cid] = (ch, score)
        all_hits = [v[0] for v in sorted(reranked_ids.values(), key=lambda x: x[1], reverse=True)]
        all_hits = all_hits[:min(rerank_top_n, max_chunks)]
    else:
        all_hits = all_hits[:min(rerank_top_n, max_chunks)]

    # 4. 窗口扩展
    local_hits = all_hits
    if enable_window:
        local_hits = LocalRetriever.expand_window(local_hits, chunks, window_size)
        local_hits = local_hits[:max_chunks]

    # 5. 过滤纯占位符垃圾 chunks
    local_hits = [c for c in local_hits if not re.fullmatch(r'\s*((\[参见表格\s*\d+\]|\[.*?\])\s*)+', c['text'].strip())]

    return chunks, local_hits


# ============================================================
# E3 — 本地 Local-RAG
# ============================================================
def run_e3(
    paper_id: str,
    md_path: Path,
    llm_client: LLMClient,
    params: Dict[str, Any] = None,
    chunks: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    切分文档 → 字段查询 → 本地密集检索 → 重排 → 窗口扩展 → LLM。
    本地检索，1 次 LLM 调用。
    """
    chunks, local_hits = _local_retrieval_pipeline(paper_id, params, chunks, md_path)

    # 格式化内容（简化标签格式）
    parts = []
    for i, ch in enumerate(local_hits, start=1):
        parts.append(f"--- chunk {i} ---\n{ch['text']}\n")
    content = "\n".join(parts)

    # 6. 抽取（使用 extract_local.txt，含碎片去重指令）
    prompt_template = load_prompt(config.PROMPTS_DIR / "extract_local.txt")
    prompt = render_prompt(prompt_template, content=content)
    resp = llm_client.call(prompt)
    parsed, parse_err = safe_parse_json(resp.content)
    result = normalize_result(parsed, paper_id)

    return _build_output(
        paper_id=paper_id,
        source_file=Path(md_path).name,
        strategy="E3",
        model=llm_client.model,
        result=result,
        raw_output=resp.content,
        parse_error=parse_err,
        resp=resp,
        local_chunks=local_hits,
    )


# ============================================================
# E4 — E3 + RAG 验证 (方案B: 基于关键值检索证据块)
# ============================================================
def run_e4(
    paper_id: str,
    md_path: Path,
    llm_client: LLMClient,
    params: Dict[str, Any] = None,
    chunks: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    两阶段策略:
      Stage 1 — 调用 E3 获取初始抽取结果
      Stage 2 — 从 E3 结果中提取关键值作为查询，
                在本地 ChromaDB 中检索证据块，
                将初始结果 + 证据块送入 LLM 验证/修正

    本地检索 + 验证，2 次 LLM 调用（E3 抽取 + 验证）。
    """
    from retrieval import LocalRetriever  # lazy import

    params = params or {}
    rerank_top_n = params.get("rerank_top_n", config.DEFAULT_PARAMS["rerank_top_n"])
    top_k = params.get("retrieval_top_k", config.DEFAULT_PARAMS["retrieval_top_k"])
    enable_reranker = params.get("enable_reranker", config.DEFAULT_PARAMS["enable_reranker"])

    # 1. 获取 chunks（供 E3 和验证检索共用）
    if chunks is None:
        from prepare import split_single_md
        _, chunks = split_single_md(
            md_path,
            chunk_size=params.get("chunk_size", config.DEFAULT_PARAMS["chunk_size"]),
            chunk_overlap=params.get("chunk_overlap", config.DEFAULT_PARAMS["chunk_overlap"]),
        )

    # 2. Stage 1: 调用 E3 获取初始抽取结果
    e3_output = run_e3(paper_id, md_path, llm_client, params, chunks)
    e3_result = e3_output["result"]
    e3_records = e3_result.get("records", [])

    if not e3_records:
        # E3 未抽取到任何记录，直接返回空结果
        return _build_output(
            paper_id=paper_id,
            source_file=Path(md_path).name,
            strategy="E4",
            model=llm_client.model,
            result=e3_result,
            raw_output=e3_output.get("raw_output", ""),
            parse_error=None,
            local_chunks=e3_output.get("local_retrieved_chunks", []),
        )

    # 3. Stage 2: 从 E3 结果中提取关键值作为验证查询
    verify_queries = _extract_verify_queries(e3_records)

    # 4. 使用关键值在本地 ChromaDB 中检索证据块
    local = LocalRetriever()
    cs = params.get("chunk_size", config.DEFAULT_PARAMS["chunk_size"])
    co = params.get("chunk_overlap", config.DEFAULT_PARAMS["chunk_overlap"])
    cache_key = f"{paper_id}_cs{cs}_co{co}"
    evidence_hits = local.retrieve(chunks, verify_queries, top_k=top_k, cache_key=cache_key)

    if enable_reranker:
        joined_query = " ; ".join(verify_queries)
        evidence_hits = local.rerank(joined_query, evidence_hits, top_n=rerank_top_n)
    else:
        evidence_hits = evidence_hits[:rerank_top_n]

    # 5. 格式化证据块
    evidence_parts = []
    for i, ch in enumerate(evidence_hits, start=1):
        evidence_parts.append(f"[Evidence-{i} | {ch['chunk_id']}]\n{ch['text']}\n")
    evidence_text = "\n".join(evidence_parts)

    # 6. 格式化初始抽取结果
    extraction_json = json.dumps(
        {"records": e3_records},
        ensure_ascii=False,
        indent=2,
    )

    # 7. 验证 LLM 调用
    verify_template = load_prompt(config.PROMPTS_DIR / "verify.txt")
    verify_prompt = render_prompt(
        verify_template,
        extraction_json=extraction_json,
        evidence_chunks=evidence_text,
    )
    verify_resp = llm_client.call(verify_prompt)
    parsed, parse_err = safe_parse_json(verify_resp.content)
    final_result = normalize_result(parsed, paper_id)

    # 合并所有检索块信息
    all_local_chunks = e3_output.get("local_retrieved_chunks", []) + evidence_hits

    return _build_output(
        paper_id=paper_id,
        source_file=Path(md_path).name,
        strategy="E4",
        model=llm_client.model,
        result=final_result,
        raw_output=verify_resp.content,
        parse_error=parse_err,
        resp=verify_resp,
        local_chunks=all_local_chunks,
    )


def _extract_verify_queries(records: List[Dict[str, Any]]) -> List[str]:
    """
    从 E3 的抽取记录中提取关键值，构建验证检索查询。
    每条记录提取: inhibitor_name, substrate, inhibition_efficiency, test_method 等关键字段。
    """
    queries = []
    seen = set()

    for rec in records:
        # 提取有意义的字段值作为查询
        key_fields = [
            "inhibitor_name", "substrate", "corrosive_medium",
            "test_method", "inhibition_efficiency",
            "homo", "lumo", "delta_g_ads", "adsorption_energy",
        ]
        values = []
        for f in key_fields:
            v = rec.get(f)
            if v and v != "null":
                values.append(str(v))

        if values:
            q = " ".join(values)
            if q not in seen:
                seen.add(q)
                queries.append(q)

    # 如果无有效查询，回退到字段名
    if not queries:
        queries = list(config.FIELD_NAMES)

    return queries


# ============================================================
# E5 — 全局 RAG（Milvus 跨文档检索，领域知识补充）
# ============================================================
def run_e5_global(
    paper_id: str,
    md_path: Path,
    llm_client: LLMClient,
    params: Dict[str, Any] = None,
    chunks: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Global-RAG：从 Milvus 全局知识库检索领域相关 chunks，
    作为领域背景 + 目标论文全文送入 LLM 抽取。

    特点：
    - 排除与目标论文 DOI 相同的 chunks（防自漏）
    - 相似度去重（防重复）
    - 使用 extract_global.txt prompt（强制来源隔离）
    """
    from retrieval import MilvusRetriever  # lazy import

    params = params or {}
    global_top_k = params.get("global_top_k", config.MILVUS_EXTERNAL_TOP_K)

    # 1. 读取目标论文全文
    target_text = Path(md_path).read_text(encoding="utf-8")
    max_chars = params.get("max_doc_chars", config.DEFAULT_PARAMS.get("max_doc_chars", 80000))
    target_text = target_text[:max_chars]

    # 2. 构建字段级查询
    field_queries = list(config.FIELD_QUERIES.values())

    # 3. Milvus 全局检索（排除同 paper 的 chunks）
    milvus = MilvusRetriever()
    exclude_doi = _lookup_doi(paper_id)
    global_hits = milvus.search(
        queries=field_queries,
        top_k=global_top_k,
        exclude_paper_id=paper_id,
        exclude_doi=exclude_doi,
        dedup_threshold=config.MILVUS_DEDUP_THRESHOLD,
    )

    # 4. 格式化 — 全局 chunks 作为 Context 标签
    context_parts = []
    for i, ch in enumerate(global_hits, start=1):
        context_parts.append(f"[Global-{i} | {ch['chunk_id']}]\n{ch['text']}\n")
    context_text = "\n".join(context_parts)

    # 5. 拼接：目标论文全文 + 全局 context
    combined = (
        f"[Target Paper]\n{target_text}\n\n"
        f"[Domain Context — from other papers, do NOT extract data from these]\n{context_text}"
    )

    # 6. 抽取（使用 extract_global.txt，含来源隔离规则）
    prompt_template = load_prompt(config.PROMPTS_DIR / "extract_global.txt")
    prompt = render_prompt(prompt_template, content=combined)
    resp = llm_client.call(prompt)
    parsed, parse_err = safe_parse_json(resp.content)
    result = normalize_result(parsed, paper_id)

    return _build_output(
        paper_id=paper_id,
        source_file=Path(md_path).name,
        strategy="E5",
        model=llm_client.model,
        result=result,
        raw_output=resp.content,
        parse_error=parse_err,
        resp=resp,
        global_chunks=global_hits,
    )


# ============================================================
# E6 — 本地 RAG + 全局领域知识混合
# ============================================================
def run_e6_hybrid(
    paper_id: str,
    md_path: Path,
    llm_client: LLMClient,
    params: Dict[str, Any] = None,
    chunks: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Hybrid-RAG：先用 E3 字段级本地检索获取高精度 local chunks，
    再从 Milvus 检索少量领域知识块作为背景补充，
    两者拼接后送入 LLM（使用 extract_global.txt 的来源隔离规则）。
    """
    from retrieval import MilvusRetriever  # lazy import

    params = params or {}

    chunks, local_hits = _local_retrieval_pipeline(paper_id, params, chunks, md_path)

    global_top_k = params.get("global_hybrid_top_k", params.get("global_top_k", config.MILVUS_EXTERNAL_TOP_K))

    # 3. Milvus 全局检索（少量领域背景）
    try:
        milvus = MilvusRetriever()
        field_queries = list(config.FIELD_QUERIES.values())
        exclude_doi = _lookup_doi(paper_id)
        global_hits = milvus.search(
            queries=field_queries,
            top_k=global_top_k,
            exclude_paper_id=paper_id,
            exclude_doi=exclude_doi,
            dedup_threshold=config.MILVUS_DEDUP_THRESHOLD,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "E6 Milvus global retrieval failed for %s, proceeding with local-only: %s",
            paper_id, e
        )
        global_hits = []

    # 4. 格式化
    parts = []
    for i, ch in enumerate(local_hits, start=1):
        parts.append(f"[Local-{i}]\n{ch['text']}\n")
    for i, ch in enumerate(global_hits, start=1):
        parts.append(f"[Global-{i}]\n{ch['text']}\n")
    content = "\n".join(parts)

    # 5. 抽取（使用 extract_hybrid.txt：含来源隔离规则 + 本地碎片去重指令）
    prompt_template = load_prompt(config.PROMPTS_DIR / "extract_hybrid.txt")
    prompt = render_prompt(prompt_template, content=content)
    resp = llm_client.call(prompt)
    parsed, parse_err = safe_parse_json(resp.content)
    result = normalize_result(parsed, paper_id)

    return _build_output(
        paper_id=paper_id,
        source_file=Path(md_path).name,
        strategy="E6",
        model=llm_client.model,
        result=result,
        raw_output=resp.content,
        parse_error=parse_err,
        resp=resp,
        local_chunks=local_hits,
        global_chunks=global_hits,
    )


# ============================================================
# 策略调度器
# ============================================================
_STRATEGY_FN = {
    "E0": run_e0,
    "E1": run_e1,
    "E3": run_e3,
    "E4": run_e4,
    "E5": run_e5_global,
    "E6": run_e6_hybrid,
}


def run_strategy(
    strategy: str,
    paper_id: str,
    md_path: Path,
    llm_client: LLMClient,
    params: Dict[str, Any] = None,
    chunks: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    统一调度入口。

    Args:
        strategy: "E0", "E1", "E3", "E4", "E5", "E6"
        paper_id: 文档标识
        md_path: Markdown 文件路径
        llm_client: LLMClient 实例
        params: 覆盖默认超参数
        chunks: 预切分的 chunks（E1/E3/E4/E6 可用，可选）
    """
    fn = _STRATEGY_FN.get(strategy)
    if fn is None:
        raise ValueError(f"未知策略: {strategy}，可选: {list(_STRATEGY_FN.keys())}")

    # 检查部署依赖
    missing = config.check_deps(strategy)
    if missing:
        raise RuntimeError(
            f"策略 {strategy} 缺少依赖: {missing}\n"
            "请确保模型/服务已就绪。"
        )

    kwargs = {"paper_id": paper_id, "md_path": md_path, "llm_client": llm_client, "params": params}
    # E1/E3/E4/E6 支持传入预切分 chunks
    if strategy in ("E1", "E3", "E4", "E6") and chunks is not None:
        kwargs["chunks"] = chunks

    return fn(**kwargs)
