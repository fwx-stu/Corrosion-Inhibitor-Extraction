"""
智谱 Batch API 客户端

Batch API 价格为实时 API 的 50%，适合大批量抽取任务。
支持 E0（单轮 batch）和 E1 Wiki-RAG（两轮 batch: wiki蒸馏 → 全文抽取）。

使用方式:
  # 1. 生成 batch 请求文件
  python batch_client.py prepare --strategy E0

  # 2. 提交 batch（自动等待完成）
  python batch_client.py submit --strategy E0

  # 3. 查询状态
  python batch_client.py status --batch-id batch_xxx

  # 4. 下载结果并解析
  python batch_client.py download --strategy E4 --batch-id batch_2045055490023096320

  # 5. 一键全流程
  python batch_client.py run --strategy E0
"""

from __future__ import annotations

import json
import re
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _batch_dir(strategy: str) -> Path:
    return _ensure_dir(Path(config.RESULTS_DIR) / strategy.lower() / "batch")


# ============================================================
# 1. 准备 Batch 请求文件
# ============================================================
def prepare_e0_batch(
    md_files: List[Path],
    existing_ids: set = None,
) -> Path:
    """生成 E0 的 batch JSONL 请求文件。"""
    from pipeline import load_prompt, render_prompt

    existing_ids = existing_ids or set()
    prompt_template = load_prompt(config.PROMPTS_DIR / "extract.txt")
    max_chars = config.DEFAULT_PARAMS["max_doc_chars"]

    out_dir = _batch_dir("E0")
    out_path = out_dir / "e0_batch_input.jsonl"

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for md_path in md_files:
            paper_id = md_path.stem
            if paper_id in existing_ids:
                continue

            text = md_path.read_text(encoding="utf-8")
            if len(text) > max_chars:
                text = text[:max_chars]

            prompt = render_prompt(prompt_template, content=text)
            request = {
                "custom_id": f"e0_{paper_id}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": config.LLM_PROVIDERS["zhipu"]["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": int(config.BATCH_MAX_TOKENS),
                    "temperature": config.LLM_TEMPERATURE,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1

    print(f"[batch] E0 请求文件: {out_path} ({count} 条)")
    return out_path


# ============================================================
# E1 Wiki-RAG Batch 支持
# (旧的 prepare_e1_extraction_batch / prepare_e1_aggregation_batch 已由
#  prepare_e1_wiki_batch / prepare_e1_extract_batch 替代，见下方)
# ============================================================


# ============================================================
# 2. 提交 Batch
# ============================================================
def submit_batch(input_path: Path, description: str = "") -> str:
    """上传文件并创建 Batch 任务，返回 batch_id。"""
    from zhipuai import ZhipuAI

    api_key = config.LLM_PROVIDERS["zhipu"]["api_key"]
    client = ZhipuAI(api_key=api_key)

    # 上传文件
    with open(input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"[batch] 文件已上传: {file_obj.id}")

    # 创建 batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v4/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"[batch] Batch 已创建: {batch.id}")

    # 保存 batch 信息
    info_path = input_path.parent / f"batch_info_{batch.id}.json"
    info = {
        "batch_id": batch.id,
        "input_file_id": file_obj.id,
        "input_path": str(input_path),
        "status": batch.status,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    return batch.id


# ============================================================
# 3. 等待 Batch 完成
# ============================================================
def wait_batch(batch_id: str, poll_interval: int = 30) -> Dict[str, Any]:
    """轮询等待 Batch 完成，返回 batch 对象。"""
    from zhipuai import ZhipuAI

    api_key = config.LLM_PROVIDERS["zhipu"]["api_key"]
    client = ZhipuAI(api_key=api_key)

    terminal_states = {"completed", "failed", "expired", "cancelled"}
    start = time.time()

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        elapsed = time.time() - start
        completed = getattr(batch, "request_counts", None)

        progress = ""
        if completed:
            progress = f" (完成={getattr(completed, 'completed', '?')}, 失败={getattr(completed, 'failed', '?')}, 总计={getattr(completed, 'total', '?')})"

        print(f"[batch] 状态: {status}{progress}  耗时: {elapsed:.0f}s")

        if status in terminal_states:
            if status != "completed":
                print(f"[batch] 警告: Batch 终态为 {status}")
            return {
                "batch_id": batch_id,
                "status": status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
            }

        time.sleep(poll_interval)


# ============================================================
# 4. 下载结果
# ============================================================
def download_batch_results(batch_id: str, output_path: Path) -> List[Dict]:
    """下载 Batch 结果并解析。"""
    from zhipuai import ZhipuAI

    api_key = config.LLM_PROVIDERS["zhipu"]["api_key"]
    client = ZhipuAI(api_key=api_key)

    def _with_retry(func, what: str, max_attempts: int = 6, base_sleep: float = 2.0):
        """对网络抖动做指数退避重试，避免 WinError 10054 导致整轮失败。"""
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                return func()
            except Exception as e:
                last_err = e
                if attempt >= max_attempts:
                    break
                wait_s = base_sleep * (2 ** (attempt - 1))
                print(f"[batch] {what} 失败 (attempt {attempt}/{max_attempts}): {e}; {wait_s:.1f}s 后重试...")
                time.sleep(wait_s)
        raise last_err

    batch = _with_retry(lambda: client.batches.retrieve(batch_id), f"retrieve batch {batch_id}")

    # 优先下载错误文件以便诊断
    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        try:
            err_content = _with_retry(lambda: client.files.content(error_file_id), f"download error file {error_file_id}")
            err_raw = err_content.content if hasattr(err_content, "content") else err_content
            if isinstance(err_raw, bytes):
                err_raw = err_raw.decode("utf-8")
            else:
                err_raw = str(err_raw)
            err_path = output_path.parent / f"batch_errors_{batch_id}.jsonl"
            _ensure_dir(err_path.parent)
            err_path.write_text(err_raw, encoding="utf-8")
            err_lines = [l for l in err_raw.strip().split("\n") if l.strip()]
            print(f"[batch] 下载 {len(err_lines)} 条错误记录 → {err_path}")
            # 显示第一条错误帮助诊断
            if err_lines:
                first_err = json.loads(err_lines[0])
                err_body = first_err.get("error", first_err.get("response", {}).get("body", {}).get("error", {}))
                print(f"[batch] 首条错误: code={err_body.get('code')}, message={err_body.get('message')}")
        except Exception as e:
            print(f"[batch] 下载错误文件失败: {e}")

    if not batch.output_file_id:
        print(f"[batch] 无输出文件，状态: {batch.status}")
        return []

    content = _with_retry(lambda: client.files.content(batch.output_file_id), f"download output file {batch.output_file_id}")
    # content 可能是 bytes 或 HttpxBinaryResponseContent
    raw = content.content if hasattr(content, "content") else content
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    else:
        raw = str(raw)

    _ensure_dir(output_path.parent)
    output_path.write_text(raw, encoding="utf-8")

    results = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        try:
            results.append(json.loads(line))
        except Exception:
            continue

    print(f"[batch] 下载 {len(results)} 条结果 → {output_path}")
    return results


# ============================================================
# 5. 解析结果 → 标准输出格式
# ============================================================
def parse_e0_batch_results(results: List[Dict]) -> List[Dict]:
    """将 E0 batch 结果转为标准输出格式。"""
    from pipeline import safe_parse_json, normalize_result, _build_output

    outputs = []
    for item in results:
        custom_id = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})
        status_code = response.get("status_code", 0)

        # 解析 paper_id: "e0_xml_2166" → "xml_2166"
        paper_id = custom_id.replace("e0_", "", 1)

        if status_code != 200:
            outputs.append({
                "paper_id": paper_id,
                "source_file": f"{paper_id}.md",
                "strategy": "E0",
                "error": f"Batch API error: status={status_code}",
            })
            continue

        choices = body.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage = body.get("usage", {})

        parsed, parse_err = safe_parse_json(content)
        result = normalize_result(parsed, paper_id)

        out = _build_output(
            paper_id=paper_id,
            source_file=f"{paper_id}.md",
            strategy="E0",
            model=body.get("model", config.LLM_PROVIDERS["zhipu"]["model"]),
            result=result,
            raw_output=content,
            parse_error=parse_err,
        )
        out["token_stats"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        outputs.append(out)

    return outputs


def prepare_e1_wiki_batch(
    md_files: List[Path],
    existing_ids: set = None,
    params: Dict[str, Any] = None,
) -> Path:
    """
    E1 Wiki-RAG Phase 1: Milvus 全局检索 → 生成 wiki_summarize batch JSONL。
    每篇论文一条请求，输入是检索到的全局 chunks 拼接文本。
    """
    from pipeline import load_prompt, render_prompt
    from retrieval import MilvusRetriever

    existing_ids = existing_ids or set()
    params = params or {}
    global_top_k = params.get("wiki_global_top_k", params.get("global_top_k", config.WIKI_GLOBAL_TOP_K))
    dedup_threshold = params.get("dedup_threshold", config.MILVUS_DEDUP_THRESHOLD)

    wiki_template = load_prompt(config.PROMPTS_DIR / "wiki_summarize.txt")

    out_dir = _batch_dir("E1")
    out_path = out_dir / "e1_wiki_batch_input.jsonl"
    global_chunks_map_path = out_dir / "e1_global_chunks_map.json"

    milvus = MilvusRetriever()
    global_chunks_map = {}  # paper_id -> [{chunk_id, text}]
    count = 0

    # 预先构建 _META_INDEX_CACHE 方便排除同一 DOI
    from pipeline import _lookup_doi

    print(f"[E1-wiki-prepare] 开始 Milvus 全局检索 ({len(md_files)} 篇)...")
    with out_path.open("w", encoding="utf-8") as f:
        for md_path in md_files:
            paper_id = md_path.stem
            if paper_id in existing_ids:
                continue

            field_queries = list(config.FIELD_QUERIES.values())
            exclude_doi = _lookup_doi(paper_id)
            try:
                global_hits = milvus.search(
                    queries=field_queries,
                    top_k=global_top_k,
                    exclude_paper_id=paper_id,
                    exclude_doi=exclude_doi,
                    dedup_threshold=dedup_threshold,
                )
            except Exception as e:
                print(f"  [{paper_id}] Milvus 检索失败: {e}，跳过")
                continue

            # 格式化为 wiki 蒸馏素材
            context_parts = [f"[Fragment-{i+1} | {c['chunk_id']}]\n{c['text']}\n" for i, c in enumerate(global_hits)]
            global_context = "\n".join(context_parts)

            global_chunks_map[paper_id] = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in global_hits]

            wiki_prompt = render_prompt(wiki_template, global_context=global_context)
            request = {
                "custom_id": f"e1wiki_{paper_id}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": config.LLM_PROVIDERS["zhipu"]["model"],
                    "messages": [{"role": "user", "content": wiki_prompt}],
                    "max_tokens": 1024,
                    "temperature": config.LLM_TEMPERATURE,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
            print(f"  [{count}] {paper_id}: {len(global_hits)} 个全局 chunks 写入 wiki batch")

    global_chunks_map_path.write_text(
        json.dumps(global_chunks_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[E1-wiki-prepare] 完成: {count} 篇，请求文件 → {out_path}")
    return out_path


def parse_e1_wiki_results(results: List[Dict]) -> Dict[str, str]:
    """将 E1 wiki_summarize batch 结果解析为 {paper_id: wiki_text}。"""
    wiki_map = {}
    for item in results:
        custom_id = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})
        status_code = response.get("status_code", 0)
        paper_id = custom_id.replace("e1wiki_", "", 1)

        if status_code != 200:
            print(f"  [wiki] {paper_id} batch error status={status_code}")
            continue

        choices = body.get("choices", [])
        wiki_text = (choices[0]["message"]["content"] if choices else "").strip()
        if wiki_text:
            wiki_map[paper_id] = wiki_text

    return wiki_map


def prepare_e1_extract_batch(
    md_files: List[Path],
    wiki_map: Dict[str, str],
    existing_ids: set = None,
    params: Dict[str, Any] = None,
) -> Path:
    """
    E1 Wiki-RAG Phase 2: 将 wiki + 目标论文全文拼接，生成 extract_wiki batch JSONL。
    只处理 wiki_map 中有条目的论文。
    """
    from pipeline import load_prompt, render_prompt

    existing_ids = existing_ids or set()
    params = params or {}
    max_chars = params.get("max_doc_chars", config.DEFAULT_PARAMS.get("max_doc_chars", 80000))

    extract_template = load_prompt(config.PROMPTS_DIR / "extract_wiki.txt")
    md_map = {md.stem: md for md in md_files}

    out_dir = _batch_dir("E1")
    out_path = out_dir / "e1_extract_batch_input.jsonl"
    count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for paper_id, wiki_text in wiki_map.items():
            if paper_id in existing_ids:
                continue
            md_path = md_map.get(paper_id)
            if md_path is None:
                print(f"  [E1-extract] 找不到 {paper_id} 的 md 文件，跳过")
                continue

            target_text = md_path.read_text(encoding="utf-8")[:max_chars]
            combined = f"[Domain Wiki]\n{wiki_text}\n\n[Target Paper]\n{target_text}"

            prompt = render_prompt(extract_template, content=combined)
            request = {
                "custom_id": f"e1ext_{paper_id}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": config.LLM_PROVIDERS["zhipu"]["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": int(config.BATCH_MAX_TOKENS),
                    "temperature": config.LLM_TEMPERATURE,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1

    print(f"[E1-extract-prepare] 完成: {count} 篇，请求文件 → {out_path}")
    return out_path


def parse_e1_extract_results(results: List[Dict]) -> List[Dict]:
    """将 E1 extract_wiki batch 结果转为标准输出格式。"""
    from pipeline import safe_parse_json, normalize_result, _build_output

    outputs = []
    for item in results:
        custom_id = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})
        status_code = response.get("status_code", 0)

        paper_id = custom_id.replace("e1ext_", "", 1)

        if status_code != 200:
            outputs.append({
                "paper_id": paper_id,
                "source_file": f"{paper_id}.md",
                "strategy": "E1",
                "error": f"Batch API error: status={status_code}",
            })
            continue

        choices = body.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage = body.get("usage", {})

        parsed, parse_err = safe_parse_json(content)
        result = normalize_result(parsed, paper_id)

        out = _build_output(
            paper_id=paper_id,
            source_file=f"{paper_id}.md",
            strategy="E1",
            model=body.get("model", config.LLM_PROVIDERS["zhipu"]["model"]),
            result=result,
            raw_output=content,
            parse_error=parse_err,
        )
        out["token_stats"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        outputs.append(out)

    return outputs




# ============================================================
# 6. 一键全流程
# ============================================================
def run_e0_batch(md_files: List[Path], limit: int = None, poll_interval: int = 30):
    """E0 Batch 全流程: 准备 → 提交 → 等待 → 下载 → 写入结果。"""
    if limit:
        md_files = md_files[:limit]

    # 检查已有结果
    strategy_dir = _ensure_dir(Path(config.RESULTS_DIR) / "e0")
    result_file = strategy_dir / "e0_results.jsonl"
    existing_ids = _load_existing_ids(result_file)

    batch_dir_path = _batch_dir("E0")
    # 检测是否有已提交但未收集结果的 batch（网络中断续接）
    pending = _find_pending_batch_info(batch_dir_path, "e0_batch")
    if pending:
        batch_id = pending["batch_id"]
        print(f"[batch] E0: 检测到未收集的 batch {batch_id}，续接...")
    else:
        # 1. 准备
        input_path = prepare_e0_batch(md_files, existing_ids)
        if input_path.stat().st_size == 0:
            print("[batch] E0: 所有文档已处理，无需提交")
            return
        # 2. 提交
        batch_id = submit_batch(input_path, description="E0 extraction batch")

    # 3. 等待
    batch_result = wait_batch(batch_id, poll_interval=poll_interval)
    if batch_result["status"] != "completed":
        print(f"[batch] E0 Batch 未完成: {batch_result['status']}")
        return

    # 4. 下载
    output_path = _batch_dir("E0") / f"e0_batch_output_{batch_id}.jsonl"
    results = download_batch_results(batch_id, output_path)

    # 5. 解析并写入标准结果文件
    outputs = parse_e0_batch_results(results)
    raw_dir = _ensure_dir(strategy_dir / "raw")
    success = failed = 0
    for out in outputs:
        if "error" in out:
            failed_file = strategy_dir / "e0_failed.jsonl"
            _append_jsonl(failed_file, out)
            failed += 1
        else:
            _append_jsonl(result_file, out)
            _write_json(raw_dir / f"{out['paper_id']}.json", out)
            success += 1

    print(f"[batch] E0 完成: 成功={success}, 失败={failed}")

    # 6. 自动清理平台上的 Batch 文件
    cleanup_batch_files(batch_id)


def run_e1_batch(md_files: List[Path], limit: int = None, poll_interval: int = 30):
    """
    E1 Wiki-RAG Batch 全流程 (两轮):
      Phase 1: Milvus 全局检索 → wiki_summarize batch → Domain Wiki 文本
      Phase 2: [Domain Wiki] + 目标论文全文 → extract_wiki batch → 最终结果
    """
    if limit:
        md_files = md_files[:limit]

    strategy_dir = _ensure_dir(Path(config.RESULTS_DIR) / "e1")
    result_file = strategy_dir / "e1_results.jsonl"
    existing_ids = _load_existing_ids(result_file)

    # === Phase 1: wiki summarization batch ===
    print("[batch] E1 Wiki-RAG Phase 1: Milvus 检索 + wiki_summarize batch")
    wiki_input_path = prepare_e1_wiki_batch(md_files, existing_ids)
    if wiki_input_path.stat().st_size == 0:
        print("[batch] E1: 所有文档已处理，无需提交")
        return

    batch_id_1 = submit_batch(wiki_input_path, description="E1 wiki summarization")
    result_1 = wait_batch(batch_id_1, poll_interval=poll_interval)
    if result_1["status"] != "completed":
        print(f"[batch] E1 Phase 1 未完成: {result_1['status']}")
        return

    output_path_1 = _batch_dir("E1") / f"e1_wiki_output_{batch_id_1}.jsonl"
    raw_results_1 = download_batch_results(batch_id_1, output_path_1)

    wiki_map = parse_e1_wiki_results(raw_results_1)
    print(f"[batch] E1 Phase 1 完成: {len(wiki_map)} 篇论文生成 Domain Wiki")

    if not wiki_map:
        print("[batch] E1: 无有效 wiki 结果，终止")
        return

    # === Phase 2: extraction batch ===
    print("[batch] E1 Wiki-RAG Phase 2: [Domain Wiki] + 全文 → extract_wiki batch")
    extract_input_path = prepare_e1_extract_batch(md_files, wiki_map, existing_ids)

    batch_id_2 = submit_batch(extract_input_path, description="E1 wiki+paper extraction")
    result_2 = wait_batch(batch_id_2, poll_interval=poll_interval)
    if result_2["status"] != "completed":
        print(f"[batch] E1 Phase 2 未完成: {result_2['status']}")
        return

    output_path_2 = _batch_dir("E1") / f"e1_extract_output_{batch_id_2}.jsonl"
    raw_results_2 = download_batch_results(batch_id_2, output_path_2)

    outputs = parse_e1_extract_results(raw_results_2)
    raw_dir = _ensure_dir(strategy_dir / "raw")
    success = failed = 0
    for out in outputs:
        if "error" in out:
            _append_jsonl(strategy_dir / "e1_failed.jsonl", out)
            failed += 1
        else:
            _append_jsonl(result_file, out)
            _write_json(raw_dir / f"{out['paper_id']}.json", out)
            success += 1

    print(f"[batch] E1 Wiki-RAG 完成: 成功={success}, 失败={failed}")

    cleanup_batch_files(batch_id_1)
    cleanup_batch_files(batch_id_2)


# ============================================================# E3 Batch 支持: 阶段1本地 embed+rerank 选块 → 阶段2 Batch 抽取
# ============================================================
def prepare_e3_batch(
    md_files: List[Path],
    existing_ids: set = None,
    params: Dict[str, Any] = None,
) -> Path:
    """
    阶段1 (本地): 对每篇论文用 embedding+reranker 选出 top-N chunks,
    将结果打包成 Batch JSONL 请求文件并保存 chunks 应射。
    """
    from pipeline import load_prompt, render_prompt
    from prepare import split_single_md
    from retrieval import LocalRetriever

    existing_ids = existing_ids or set()
    params = params or config.DEFAULT_PARAMS
    top_k = params.get("retrieval_top_k", config.DEFAULT_PARAMS["retrieval_top_k"])
    rerank_top_n = params.get("rerank_top_n", config.DEFAULT_PARAMS["rerank_top_n"])
    window_size = params.get("window_size", config.DEFAULT_PARAMS["window_size"])
    enable_reranker = params.get("enable_reranker", config.DEFAULT_PARAMS["enable_reranker"])
    enable_window = params.get("enable_window_expand", config.DEFAULT_PARAMS["enable_window_expand"])
    chunk_size = params.get("chunk_size", config.DEFAULT_PARAMS["chunk_size"])
    chunk_overlap = params.get("chunk_overlap", config.DEFAULT_PARAMS["chunk_overlap"])

    prompt_template = load_prompt(config.PROMPTS_DIR / "extract.txt")
    queries = list(config.FIELD_NAMES)
    joined_query = " ; ".join(queries)

    out_dir = _batch_dir("E3")
    out_path = out_dir / "e3_batch_input.jsonl"
    chunks_map_path = out_dir / "e3_chunks_map.json"

    retriever = LocalRetriever()
    chunks_map = {}  # paper_id -> [{chunk_id, text}]
    count = 0

    print(f"[E3-prepare] 开始本地 embed+rerank ({len(md_files)} 篇)...")
    with out_path.open("w", encoding="utf-8") as f:
        for md_path in md_files:
            paper_id = md_path.stem
            if paper_id in existing_ids:
                continue

            # 切分
            _, chunks = split_single_md(md_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                continue

            # 向量+BM25 混合检索（使用持久化缓存，避免重复向量化）
            bm25_weight = params.get("bm25_weight", config.DEFAULT_PARAMS.get("bm25_weight", 0.0))
            cache_key = f"{paper_id}_cs{chunk_size}_co{chunk_overlap}"
            hits = retriever.retrieve(chunks, queries, top_k=top_k, cache_key=cache_key, bm25_weight=bm25_weight)

            # 重排序（chunks 数量 ≤ rerank_top_n 时跳过，避免截断）
            if enable_reranker and len(hits) > rerank_top_n:
                hits = retriever.rerank(joined_query, hits, top_n=rerank_top_n)
            else:
                hits = hits[:rerank_top_n]

            # 窗口扩展
            if enable_window:
                hits = LocalRetriever.expand_window(hits, chunks, window_size)

            # 过滤纯占位符垃圾 chunks
            hits = [c for c in hits if not re.fullmatch(r'\s*((\[参见表格\s*\d+\]|\[.*?\])\s*)+', c['text'].strip())]

            # 保存 chunks 映射（方便调试）
            chunks_map[paper_id] = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in hits]

            # 拼接 content（简化标签格式）
            parts = [f"--- chunk {i+1} ---\n{c['text']}\n" for i, c in enumerate(hits)]
            content = "\n".join(parts)

            prompt = render_prompt(prompt_template, content=content)
            request = {
                "custom_id": f"e3_{paper_id}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": config.LLM_PROVIDERS["zhipu"]["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": int(config.BATCH_MAX_TOKENS),
                    "temperature": config.LLM_TEMPERATURE,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
            print(f"  [{count}] {paper_id}: {len(hits)} 个 chunks 已写入")

    # 保存 chunks 映射
    chunks_map_path.write_text(
        json.dumps(chunks_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[E3-prepare] 完成: {count} 篇 请求文件 → {out_path}")
    print(f"[E3-prepare] Chunks 映射已保存 → {chunks_map_path}")
    return out_path


def parse_e3_batch_results(results: List[Dict]) -> List[Dict]:
    """将 E3 batch 结果转为标准输出格式。"""
    from pipeline import safe_parse_json, normalize_result, _build_output

    outputs = []
    for item in results:
        custom_id = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})
        status_code = response.get("status_code", 0)

        paper_id = custom_id.replace("e3_", "", 1)

        if status_code != 200:
            outputs.append({
                "paper_id": paper_id,
                "source_file": f"{paper_id}.md",
                "strategy": "E3",
                "error": f"Batch API error: status={status_code}",
            })
            continue

        choices = body.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage = body.get("usage", {})

        parsed, parse_err = safe_parse_json(content)
        result = normalize_result(parsed, paper_id)

        out = _build_output(
            paper_id=paper_id,
            source_file=f"{paper_id}.md",
            strategy="E3",
            model=body.get("model", config.LLM_PROVIDERS["zhipu"]["model"]),
            result=result,
            raw_output=content,
            parse_error=parse_err,
        )
        out["token_stats"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        outputs.append(out)

    return outputs


def run_e3_batch(md_files: List[Path], limit: int = None, poll_interval: int = 30):
    """
    E3 Batch 全流程:
      阶段1 (本地): embed+rerank 选出 chunks 并生成 JSONL
      阶段2 (Batch API): 一次性提交所有论文 → 等待 → 下载 → 解析
    """
    if limit:
        md_files = md_files[:limit]

    strategy_dir = _ensure_dir(Path(config.RESULTS_DIR) / "e3")
    result_file = strategy_dir / "e3_results.jsonl"
    existing_ids = _load_existing_ids(result_file)

    batch_dir_path = _batch_dir("E3")
    # 检测是否有已提交但未收集结果的 batch（网络中断续接）
    pending = _find_pending_batch_info(batch_dir_path, "e3_batch")
    if pending:
        batch_id = pending["batch_id"]
        print(f"[batch] E3: 检测到未收集的 batch {batch_id}，续接...")
    else:
        # === 阶段1: 本地 embed+rerank ===
        input_path = prepare_e3_batch(md_files, existing_ids)
        if input_path.stat().st_size == 0:
            print("[batch] E3: 所有文档已处理，无需提交")
            return
        # === 阶段2: Batch API ===
        batch_id = submit_batch(input_path, description="E3 retrieval+extraction batch")

    batch_result = wait_batch(batch_id, poll_interval=poll_interval)
    if batch_result["status"] != "completed":
        print(f"[batch] E3 Batch 未完成: {batch_result['status']}")
        return

    output_path = _batch_dir("E3") / f"e3_batch_output_{batch_id}.jsonl"
    results = download_batch_results(batch_id, output_path)

    outputs = parse_e3_batch_results(results)
    raw_dir = _ensure_dir(strategy_dir / "raw")
    success = failed = 0
    for out in outputs:
        if "error" in out:
            _append_jsonl(strategy_dir / "e3_failed.jsonl", out)
            failed += 1
        else:
            _append_jsonl(result_file, out)
            _write_json(raw_dir / f"{out['paper_id']}.json", out)
            success += 1

    print(f"[batch] E3 完成: 成功={success}, 失败={failed}")
    cleanup_batch_files(batch_id)


def collect_e3_batch(batch_id: str, poll_interval: int = 30):
    """第二步: 等待 E3 Batch 完成 → 下载 → 解析写入结果文件。"""
    strategy_dir = _ensure_dir(Path(config.RESULTS_DIR) / "e3")
    result_file = strategy_dir / "e3_results.jsonl"

    batch_result = wait_batch(batch_id, poll_interval=poll_interval)
    if batch_result["status"] != "completed":
        print(f"[batch] E3 Batch 未完成: {batch_result['status']}")
        return

    output_path = _batch_dir("E3") / f"e3_batch_output_{batch_id}.jsonl"
    results = download_batch_results(batch_id, output_path)

    outputs = parse_e3_batch_results(results)
    raw_dir = _ensure_dir(strategy_dir / "raw")
    success = failed = 0
    for out in outputs:
        if "error" in out:
            _append_jsonl(strategy_dir / "e3_failed.jsonl", out)
            failed += 1
        else:
            _append_jsonl(result_file, out)
            _write_json(raw_dir / f"{out['paper_id']}.json", out)
            success += 1

    print(f"[batch] E3 collect 完成: 成功={success}, 失败={failed}")
    cleanup_batch_files(batch_id)


def collect_e0_batch(batch_id: str, poll_interval: int = 30):
    """第二步: 等待 E0 Batch 完成 → 下载 → 解析写入结果文件。"""
    strategy_dir = _ensure_dir(Path(config.RESULTS_DIR) / "e0")
    result_file = strategy_dir / "e0_results.jsonl"

    batch_result = wait_batch(batch_id, poll_interval=poll_interval)
    if batch_result["status"] != "completed":
        print(f"[batch] E0 Batch 未完成: {batch_result['status']}")
        return

    output_path = _batch_dir("E0") / f"e0_batch_output_{batch_id}.jsonl"
    results = download_batch_results(batch_id, output_path)

    outputs = parse_e0_batch_results(results)
    raw_dir = _ensure_dir(strategy_dir / "raw")
    success = failed = 0
    for out in outputs:
        if "error" in out:
            _append_jsonl(strategy_dir / "e0_failed.jsonl", out)
            failed += 1
        else:
            _append_jsonl(result_file, out)
            _write_json(raw_dir / f"{out['paper_id']}.json", out)
            success += 1

    print(f"[batch] E0 collect 完成: 成功={success}, 失败={failed}")
    cleanup_batch_files(batch_id)


# ============================================================
# E4 Batch 支持: 两轮 Batch (E3 抽取 → 证据验证)
# ============================================================
def prepare_e4_verification_batch(
    extraction_results: Dict[str, Dict],  # paper_id -> e3 result dict
    md_files: List[Path],
    params: Dict[str, Any] = None,
) -> Path:
    """
    根据 E3 batch 的抽取结果，对每篇论文做本地证据检索，生成验证 batch JSONL。
    """
    from pipeline import load_prompt, render_prompt, _extract_verify_queries
    from prepare import split_single_md
    from retrieval import LocalRetriever

    params = params or {}
    top_k = params.get("retrieval_top_k", config.DEFAULT_PARAMS["retrieval_top_k"])
    rerank_top_n = params.get("rerank_top_n", config.DEFAULT_PARAMS["rerank_top_n"])
    enable_reranker = params.get("enable_reranker", config.DEFAULT_PARAMS["enable_reranker"])
    chunk_size = params.get("chunk_size", config.DEFAULT_PARAMS["chunk_size"])
    chunk_overlap = params.get("chunk_overlap", config.DEFAULT_PARAMS["chunk_overlap"])

    verify_template = load_prompt(config.PROMPTS_DIR / "verify.txt")
    local = LocalRetriever()

    out_dir = _batch_dir("E4")
    out_path = out_dir / "e4_verify_batch_input.jsonl"

    md_map = {p.stem: p for p in md_files}
    count = 0

    print(f"[E4-verify-prepare] 开始证据检索 ({len(extraction_results)} 篇)...")
    with out_path.open("w", encoding="utf-8") as f:
        for paper_id, e3_result in extraction_results.items():
            records = e3_result.get("records", [])
            if not records:
                continue

            md_path = md_map.get(paper_id)
            if not md_path:
                continue

            # 本地切分 + 证据检索
            _, chunks = split_single_md(md_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            verify_queries = _extract_verify_queries(records)
            evidence_hits = local.retrieve(chunks, verify_queries, top_k=top_k)
            if enable_reranker:
                evidence_hits = local.rerank(" ; ".join(verify_queries), evidence_hits, top_n=rerank_top_n)
            else:
                evidence_hits = evidence_hits[:rerank_top_n]

            evidence_parts = [f"[Evidence-{i} | {ch['chunk_id']}]\n{ch['text']}\n"
                              for i, ch in enumerate(evidence_hits, start=1)]
            evidence_text = "\n".join(evidence_parts)
            extraction_json = json.dumps({"records": records}, ensure_ascii=False, indent=2)

            verify_prompt = render_prompt(
                verify_template,
                extraction_json=extraction_json,
                evidence_chunks=evidence_text,
            )
            request = {
                "custom_id": f"e4_{paper_id}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": config.LLM_PROVIDERS["zhipu"]["model"],
                    "messages": [{"role": "user", "content": verify_prompt}],
                    "max_tokens": int(config.BATCH_MAX_TOKENS),
                    "temperature": config.LLM_TEMPERATURE,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1

    print(f"[E4-verify-prepare] 完成: {count} 篇验证请求 → {out_path}")
    return out_path


def parse_e4_batch_results(results: List[Dict]) -> List[Dict]:
    """将 E4 验证 batch 结果转为标准输出格式。"""
    from pipeline import safe_parse_json, normalize_result, _build_output

    outputs = []
    for item in results:
        custom_id = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})
        status_code = response.get("status_code", 0)
        paper_id = custom_id.replace("e4_", "", 1)

        if status_code != 200:
            outputs.append({"paper_id": paper_id, "source_file": f"{paper_id}.md",
                            "strategy": "E4", "error": f"Batch API error: status={status_code}"})
            continue

        choices = body.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage = body.get("usage", {})
        parsed, parse_err = safe_parse_json(content)
        result = normalize_result(parsed, paper_id)
        out = _build_output(
            paper_id=paper_id, source_file=f"{paper_id}.md", strategy="E4",
            model=body.get("model", config.LLM_PROVIDERS["zhipu"]["model"]),
            result=result, raw_output=content, parse_error=parse_err,
        )
        out["token_stats"] = {"input_tokens": usage.get("prompt_tokens", 0),
                              "output_tokens": usage.get("completion_tokens", 0)}
        outputs.append(out)

    return outputs


def run_e4_batch(md_files: List[Path], limit: int = None, poll_interval: int = 30):
    """
    E4 Batch 全流程 (两轮):
      轮1: 本地 embed+rerank → E3-style batch LLM 抽取
      轮2: 本地证据检索 → batch LLM 验证
    """
    if limit:
        md_files = md_files[:limit]

    strategy_dir = _ensure_dir(Path(config.RESULTS_DIR) / "e4")
    result_file = strategy_dir / "e4_results.jsonl"
    existing_ids = _load_existing_ids(result_file)
    batch_dir_path = _batch_dir("E4")
    checkpoint_path = batch_dir_path / "e4_round1_checkpoint.json"

    # 检查是否有轮1的 checkpoint（轮1已完成，从轮2续接）
    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            e3_outputs = ckpt["e3_outputs"]
            batch_id_1 = ckpt.get("batch_id_1", "(recovered)")
            print(f"[batch] E4: 从轮1 checkpoint 恢复，跳过轮1 (batch_id_1={batch_id_1})")
        except Exception as e:
            print(f"[batch] E4: 读取 checkpoint 失败 ({e})，重跑轮1")
            checkpoint_path.unlink(missing_ok=True)
            e3_outputs = None
    else:
        e3_outputs = None

    if e3_outputs is None:
        # === 轮1: E3-style 抽取 batch（复用 prepare_e3_batch）===
        print("[batch] E4 轮1: E3-style 本地检索 → Batch 抽取")
        # 检测是否有已提交但未收集结果的 batch（网络中断续接）
        pending = _find_pending_batch_info(batch_dir_path, "e4_extraction")
        if pending:
            batch_id_1 = pending["batch_id"]
            print(f"[batch] E4 轮1: 续接已提交的 batch {batch_id_1}")
        else:
            e3_input_path = prepare_e3_batch(md_files, existing_ids)
            if e3_input_path.stat().st_size == 0:
                print("[batch] E4: 所有文档已处理，无需提交")
                return
            batch_id_1 = submit_batch(e3_input_path, description="E4 E3-style extraction batch (round 1)")

        result_1 = wait_batch(batch_id_1, poll_interval=poll_interval)
        if result_1["status"] != "completed":
            print(f"[batch] E4 轮1 未完成: {result_1['status']}")
            return

        e3_output_path = _batch_dir("E4") / f"e4_extraction_output_{batch_id_1}.jsonl"
        raw_results_1 = download_batch_results(batch_id_1, e3_output_path)
        cleanup_batch_files(batch_id_1)
        e3_outputs = parse_e3_batch_results(raw_results_1)
        # 保存轮1 checkpoint，防止轮2中断后重跑
        checkpoint_path.write_text(
            json.dumps({"batch_id_1": batch_id_1, "e3_outputs": e3_outputs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        # checkpoint 路径恢复，e3_outputs 已就绪
        pass

    extraction_results = {out["paper_id"]: out["result"] for out in e3_outputs if "result" in out}
    print(f"[batch] E4 轮1 完成: {len(extraction_results)} 篇抽取结果")

    if not extraction_results:
        print("[batch] E4: E3 抽取无结果，停止")
        return

    # === 轮2: 证据验证 batch ===
    print("[batch] E4 轮2: 本地证据检索 → Batch 验证")
    verify_input_path = prepare_e4_verification_batch(extraction_results, md_files)
    if verify_input_path.stat().st_size == 0:
        print("[batch] E4: 无需验证（所有抽取记录为空）")
        return

    batch_id_2 = submit_batch(verify_input_path, description="E4 verification batch (round 2)")
    result_2 = wait_batch(batch_id_2, poll_interval=poll_interval)
    if result_2["status"] != "completed":
        print(f"[batch] E4 轮2 未完成: {result_2['status']}")
        return

    verify_output_path = _batch_dir("E4") / f"e4_verify_output_{batch_id_2}.jsonl"
    raw_results_2 = download_batch_results(batch_id_2, verify_output_path)
    cleanup_batch_files(batch_id_2)

    # 合并：有验证结果的用验证结果，没的用 E3 抽取结果（记录 strategy=E4）
    verified_ids = set()
    outputs = parse_e4_batch_results(raw_results_2)
    raw_dir = _ensure_dir(strategy_dir / "raw")
    success = failed = 0
    for out in outputs:
        verified_ids.add(out["paper_id"])
        if "error" in out:
            _append_jsonl(strategy_dir / "e4_failed.jsonl", out)
            failed += 1
        else:
            _append_jsonl(result_file, out)
            _write_json(raw_dir / f"{out['paper_id']}.json", out)
            success += 1

    # 没有进入验证轮的论文（空 E3 结果），直接写空结果
    from pipeline import _build_output
    for out in e3_outputs:
        pid = out["paper_id"]
        if pid not in verified_ids:
            fallback = _build_output(
                paper_id=pid, source_file=f"{pid}.md", strategy="E4",
                model=out.get("model", ""), result=out.get("result", {"paper_id": pid, "records": []}),
                raw_output=out.get("raw_output", ""), parse_error=None,
            )
            _append_jsonl(result_file, fallback)
            _write_json(raw_dir / f"{pid}.json", fallback)
            success += 1

    print(f"[batch] E4 完成: 成功={success}, 失败={failed}")
    # 清理轮1 checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()


# ============================================================
# E6 Batch 支持: 本地 embed+rerank + Milvus → Batch 抽取
# ============================================================
def prepare_e6_batch(
    md_files: List[Path],
    existing_ids: set = None,
    params: Dict[str, Any] = None,
) -> Path:
    """
    阶段1 (本地): 本地 embed+rerank (E3-style) + Milvus 全局检索,
    合并后生成 Batch JSONL。
    """
    from pipeline import load_prompt, render_prompt
    from prepare import split_single_md
    from retrieval import LocalRetriever, MilvusRetriever

    existing_ids = existing_ids or set()
    params = params or {}
    top_k = params.get("retrieval_top_k", config.DEFAULT_PARAMS["retrieval_top_k"])
    rerank_top_n = params.get("rerank_top_n", config.DEFAULT_PARAMS["rerank_top_n"])
    window_size = params.get("window_size", config.DEFAULT_PARAMS["window_size"])
    enable_reranker = params.get("enable_reranker", config.DEFAULT_PARAMS["enable_reranker"])
    enable_window = params.get("enable_window_expand", config.DEFAULT_PARAMS["enable_window_expand"])
    field_top_k = params.get("field_retrieval_top_k", config.FIELD_RETRIEVAL_TOP_K)
    max_local = params.get("max_final_chunks", config.MAX_FINAL_CHUNKS)
    global_top_k = params.get("global_hybrid_top_k", params.get("global_top_k", config.MILVUS_EXTERNAL_TOP_K))
    chunk_size = params.get("chunk_size", config.DEFAULT_PARAMS["chunk_size"])
    chunk_overlap = params.get("chunk_overlap", config.DEFAULT_PARAMS["chunk_overlap"])

    prompt_template = load_prompt(config.PROMPTS_DIR / "extract_global.txt")
    field_queries = list(config.FIELD_QUERIES.values())
    joined_query = " ; ".join(field_queries)
    exclude_tables = params.get("exclude_tables", config.MILVUS_EXCLUDE_TABLES)

    out_dir = _batch_dir("E6")
    out_path = out_dir / "e6_batch_input.jsonl"
    chunks_map_path = out_dir / "e6_chunks_map.json"

    local = LocalRetriever()
    milvus = MilvusRetriever()
    chunks_map = {}
    count = 0

    print(f"[E6-prepare] 开始本地+Milvus 检索 ({len(md_files)} 篇)...")
    with out_path.open("w", encoding="utf-8") as f:
        for md_path in md_files:
            paper_id = md_path.stem
            if paper_id in existing_ids:
                continue

            _, chunks = split_single_md(md_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                continue

            # 本地字段级检索
            bm25_weight = params.get("bm25_weight", config.DEFAULT_PARAMS.get("bm25_weight", 0.0))
            cache_key = f"{paper_id}_cs{chunk_size}_co{chunk_overlap}"
            seen_ids: dict = {}
            for field_query in field_queries:
                hits = local.retrieve(chunks, [field_query], top_k=field_top_k,
                                      cache_key=cache_key, bm25_weight=bm25_weight)
                for h in hits:
                    cid = h.get("chunk_id", "")
                    if cid not in seen_ids or h.get("retrieval_score", 0) > seen_ids[cid].get("retrieval_score", 0):
                        seen_ids[cid] = h

            all_hits = sorted(seen_ids.values(), key=lambda x: x.get("retrieval_score", 0), reverse=True)[:top_k]
            # 重排序（chunks 数量 ≤ rerank_top_n 时跳过，避免截断）
            if enable_reranker and len(all_hits) > min(rerank_top_n, max_local):
                all_hits = local.rerank(joined_query, all_hits, top_n=min(rerank_top_n, max_local))
            else:
                all_hits = all_hits[:min(rerank_top_n, max_local)]
            if enable_window:
                all_hits = LocalRetriever.expand_window(all_hits, chunks, window_size)[:max_local]

            # Milvus 全局检索（论文特定查询 + 排除表格）
            target_text = md_path.read_text(encoding="utf-8")[:3000]
            paper_queries = MilvusRetriever.extract_paper_queries(target_text)
            if not paper_queries:
                paper_queries = field_queries
            try:
                global_hits = milvus.search(
                    queries=paper_queries,
                    top_k=global_top_k,
                    exclude_paper_id=paper_id,
                    dedup_threshold=config.MILVUS_DEDUP_THRESHOLD,
                    exclude_tables=exclude_tables,
                )
            except Exception as e:
                print(f"  [{paper_id}] Milvus 检索失败: {e}，仅用本地结果")
                global_hits = []

            # 保存映射
            chunks_map[paper_id] = {
                "local": [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in all_hits],
                "global": [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in global_hits],
            }

            # 过滤纯占位符垃圾 chunks
            all_hits = [c for c in all_hits if not re.fullmatch(r'\s*((\[参见表格\s*\d+\]|\[.*?\])\s*)+', c['text'].strip())]

            # 拼接 content（简化标签格式）
            parts = [f"[Local-{i+1}]\n{c['text']}\n" for i, c in enumerate(all_hits)]
            parts += [f"[Global-{i+1}]\n{c['text']}\n" for i, c in enumerate(global_hits)]
            content = "\n".join(parts)

            prompt = render_prompt(prompt_template, content=content)
            request = {
                "custom_id": f"e6_{paper_id}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": config.LLM_PROVIDERS["zhipu"]["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": int(config.BATCH_MAX_TOKENS),
                    "temperature": config.LLM_TEMPERATURE,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
            print(f"  [{count}] {paper_id}: local={len(all_hits)}, global={len(global_hits)} chunks")

    chunks_map_path.write_text(json.dumps(chunks_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E6-prepare] 完成: {count} 篇 → {out_path}")
    return out_path


def parse_e6_batch_results(results: List[Dict]) -> List[Dict]:
    """将 E6 batch 结果转为标准输出格式。"""
    from pipeline import safe_parse_json, normalize_result, _build_output

    outputs = []
    for item in results:
        custom_id = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})
        status_code = response.get("status_code", 0)
        paper_id = custom_id.replace("e6_", "", 1)

        if status_code != 200:
            outputs.append({"paper_id": paper_id, "source_file": f"{paper_id}.md",
                            "strategy": "E6", "error": f"Batch API error: status={status_code}"})
            continue

        choices = body.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage = body.get("usage", {})
        parsed, parse_err = safe_parse_json(content)
        result = normalize_result(parsed, paper_id)
        out = _build_output(
            paper_id=paper_id, source_file=f"{paper_id}.md", strategy="E6",
            model=body.get("model", config.LLM_PROVIDERS["zhipu"]["model"]),
            result=result, raw_output=content, parse_error=parse_err,
        )
        out["token_stats"] = {"input_tokens": usage.get("prompt_tokens", 0),
                              "output_tokens": usage.get("completion_tokens", 0)}
        outputs.append(out)

    return outputs


def run_e6_batch(md_files: List[Path], limit: int = None, poll_interval: int = 30):
    """E6 Batch 全流程: 本地 embed+rerank + Milvus → Batch LLM。"""
    if limit:
        md_files = md_files[:limit]

    strategy_dir = _ensure_dir(Path(config.RESULTS_DIR) / "E6")
    result_file = strategy_dir / "E6_results.jsonl"
    existing_ids = _load_existing_ids(result_file)

    batch_dir_path = _batch_dir("E6")
    # 检测是否有已提交但未收集结果的 batch（网络中断续接）
    pending = _find_pending_batch_info(batch_dir_path, "e6_batch")
    if pending:
        batch_id = pending["batch_id"]
        print(f"[batch] E6: 检测到未收集的 batch {batch_id}，续接...")
    else:
        input_path = prepare_e6_batch(md_files, existing_ids)
        if input_path.stat().st_size == 0:
            print("[batch] E6: 所有文档已处理，无需提交")
            return
        batch_id = submit_batch(input_path, description="E6 extraction batch")

    batch_result = wait_batch(batch_id, poll_interval=poll_interval)
    if batch_result["status"] != "completed":
        print(f"[batch] E6 Batch 未完成: {batch_result['status']}")
        return

    output_path = _batch_dir("E6") / f"e6_batch_output_{batch_id}.jsonl"
    results = download_batch_results(batch_id, output_path)
    outputs = parse_e6_batch_results(results)
    raw_dir = _ensure_dir(strategy_dir / "raw")
    success = failed = 0
    for out in outputs:
        if "error" in out:
            _append_jsonl(strategy_dir / "E6_failed.jsonl", out)
            failed += 1
        else:
            _append_jsonl(result_file, out)
            _write_json(raw_dir / f"{out['paper_id']}.json", out)
            success += 1

    print(f"[batch] E6 完成: 成功={success}, 失败={failed}")
    cleanup_batch_files(batch_id)


# ============================================================
# E5 Batch 支持: 阶段1 Milvus 检索 → 阶段2 Batch 抽取
# ============================================================
def prepare_e5_batch(
    md_files: List[Path],
    existing_ids: set = None,
    params: Dict[str, Any] = None,
) -> Path:
    """
    阶段1 (本地+Milvus): 读取原文 + Milvus 全局检索拼接 context，
    生成 Batch JSONL 请求文件。
    """
    from pipeline import load_prompt, render_prompt
    from retrieval import MilvusRetriever

    existing_ids = existing_ids or set()
    params = params or {}
    max_chars = params.get("max_doc_chars", config.DEFAULT_PARAMS.get("max_doc_chars", 80000))
    global_top_k = params.get("global_top_k", config.MILVUS_EXTERNAL_TOP_K)
    dedup_threshold = params.get("dedup_threshold", config.MILVUS_DEDUP_THRESHOLD)
    exclude_tables = params.get("exclude_tables", config.MILVUS_EXCLUDE_TABLES)

    prompt_template = load_prompt(config.PROMPTS_DIR / "extract_global.txt")

    out_dir = _batch_dir("E5")
    out_path = out_dir / "e5_batch_input.jsonl"
    chunks_map_path = out_dir / "e5_global_chunks_map.json"

    milvus = MilvusRetriever()
    chunks_map = {}  # paper_id -> [{chunk_id, text}]
    count = 0

    print(f"[E5-prepare] 开始 Milvus 全局检索 ({len(md_files)} 篇, exclude_tables={exclude_tables})...")
    with out_path.open("w", encoding="utf-8") as f:
        for md_path in md_files:
            paper_id = md_path.stem
            if paper_id in existing_ids:
                continue

            # 读取原文
            target_text = md_path.read_text(encoding="utf-8")[:max_chars]

            # 从目标论文提取关键实体，生成论文特定查询
            paper_queries = MilvusRetriever.extract_paper_queries(target_text)
            if not paper_queries:
                print(f"  [{paper_id}] 未能提取论文查询，使用静态查询")
                paper_queries = list(config.FIELD_QUERIES.values())

            # Milvus 全局检索（排除同论文 + 可选排除表格）
            try:
                global_hits = milvus.search(
                    queries=paper_queries,
                    top_k=global_top_k,
                    exclude_paper_id=paper_id,
                    dedup_threshold=dedup_threshold,
                    exclude_tables=exclude_tables,
                )
            except Exception as e:
                print(f"  [{paper_id}] Milvus 检索失败: {e}，跳过")
                continue

            # 保存 global chunks 映射
            chunks_map[paper_id] = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in global_hits]

            # 拼接 content
            context_parts = [f"[Global-{i+1} | {c['chunk_id']}]\n{c['text']}\n" for i, c in enumerate(global_hits)]
            context_text = "\n".join(context_parts)
            combined = (
                f"[Target Paper]\n{target_text}\n\n"
                f"[Domain Context — from other papers, do NOT extract data from these]\n{context_text}"
            )

            prompt = render_prompt(prompt_template, content=combined)
            request = {
                "custom_id": f"e5_{paper_id}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": config.LLM_PROVIDERS["zhipu"]["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": int(config.BATCH_MAX_TOKENS),
                    "temperature": config.LLM_TEMPERATURE,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
            print(f"  [{count}] {paper_id}: {len(global_hits)} 个全局 chunks 已写入")

    chunks_map_path.write_text(
        json.dumps(chunks_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[E5-prepare] 完成: {count} 篇 请求文件 → {out_path}")
    return out_path


def parse_e5_batch_results(results: List[Dict]) -> List[Dict]:
    """将 E5 batch 结果转为标准输出格式。"""
    from pipeline import safe_parse_json, normalize_result, _build_output

    outputs = []
    for item in results:
        custom_id = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})
        status_code = response.get("status_code", 0)

        paper_id = custom_id.replace("e5_", "", 1)

        if status_code != 200:
            outputs.append({
                "paper_id": paper_id,
                "source_file": f"{paper_id}.md",
                "strategy": "E5",
                "error": f"Batch API error: status={status_code}",
            })
            continue

        choices = body.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage = body.get("usage", {})

        parsed, parse_err = safe_parse_json(content)
        result = normalize_result(parsed, paper_id)

        out = _build_output(
            paper_id=paper_id,
            source_file=f"{paper_id}.md",
            strategy="E5",
            model=body.get("model", config.LLM_PROVIDERS["zhipu"]["model"]),
            result=result,
            raw_output=content,
            parse_error=parse_err,
        )
        out["token_stats"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        outputs.append(out)

    return outputs


def run_e5_batch(md_files: List[Path], limit: int = None, poll_interval: int = 30):
    """
    E5 Batch 全流程:
      阶段1 (本地+Milvus): 读原文 + 全局检索 → 生成 JSONL
      阶段2 (Batch API): 批量提交 → 等待 → 下载 → 解析
    """
    if limit:
        md_files = md_files[:limit]

    strategy_dir = _ensure_dir(Path(config.RESULTS_DIR) / "E5")
    result_file = strategy_dir / "E5_results.jsonl"
    existing_ids = _load_existing_ids(result_file)

    batch_dir_path = _batch_dir("E5")
    # 检测是否有已提交但未收集结果的 batch（网络中断续接）
    pending = _find_pending_batch_info(batch_dir_path, "e5_batch")
    if pending:
        batch_id = pending["batch_id"]
        print(f"[batch] E5: 检测到未收集的 batch {batch_id}，续接...")
    else:
        # === 阶段1: 本地+Milvus ===
        input_path = prepare_e5_batch(md_files, existing_ids)
        if input_path.stat().st_size == 0:
            print("[batch] E5: 所有文档已处理，无需提交")
            return
        # === 阶段2: Batch API ===
        batch_id = submit_batch(input_path, description="E5 retrieval+extraction batch")

    batch_result = wait_batch(batch_id, poll_interval=poll_interval)
    if batch_result["status"] != "completed":
        print(f"[batch] E5 Batch 未完成: {batch_result['status']}")
        return

    output_path = _batch_dir("E5") / f"e5_batch_output_{batch_id}.jsonl"
    results = download_batch_results(batch_id, output_path)

    outputs = parse_e5_batch_results(results)
    raw_dir = _ensure_dir(strategy_dir / "raw")
    success = failed = 0
    for out in outputs:
        if "error" in out:
            _append_jsonl(strategy_dir / "E5_failed.jsonl", out)
            failed += 1
        else:
            _append_jsonl(result_file, out)
            _write_json(raw_dir / f"{out['paper_id']}.json", out)
            success += 1

    print(f"[batch] E5 完成: 成功={success}, 失败={failed}")
    cleanup_batch_files(batch_id)


# ============================================================
# 工具函数
# ============================================================
def _load_existing_ids(result_file: Path) -> set:
    ids = set()
    if result_file.exists():
        with result_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("paper_id")
                    if pid:
                        ids.add(pid)
                except Exception:
                    pass
    return ids


def _find_pending_batch_info(batch_dir: Path, output_prefix: str) -> Optional[Dict]:
    """查找 batch 目录中已提交但尚未收集结果的 batch，返回 batch_info dict 或 None。"""
    if not batch_dir.exists():
        return None
    for info_file in sorted(batch_dir.glob("batch_info_*.json")):
        try:
            info = json.loads(info_file.read_text(encoding="utf-8"))
            batch_id = info.get("batch_id")
            if not batch_id:
                continue
            # 若对应输出文件已存在，说明结果已收集过
            if list(batch_dir.glob(f"{output_prefix}_output_{batch_id}.jsonl")):
                continue
            return info
        except Exception:
            continue
    return None


def _append_jsonl(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def check_batch_status(batch_id: str):
    """查询 batch 状态。"""
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=config.LLM_PROVIDERS["zhipu"]["api_key"])
    batch = client.batches.retrieve(batch_id)
    counts = getattr(batch, "request_counts", None)
    print(f"Batch ID: {batch_id}")
    print(f"  状态: {batch.status}")
    if counts:
        print(f"  总计: {getattr(counts, 'total', '?')}")
        print(f"  完成: {getattr(counts, 'completed', '?')}")
        print(f"  失败: {getattr(counts, 'failed', '?')}")
    print(f"  输出文件: {getattr(batch, 'output_file_id', 'N/A')}")
    print(f"  错误文件: {getattr(batch, 'error_file_id', 'N/A')}")


# ============================================================
# CLI
# ============================================================
# 清理远程 Batch 文件
# ============================================================
def cleanup_batch_files(batch_id: str = None):
    """
    清理智谱平台上的 Batch 文件。

    - 指定 batch_id: 删除该 batch 关联的 input/output/error 文件
    - 不指定: 列出所有文件供确认
    """
    from zhipuai import ZhipuAI

    api_key = config.LLM_PROVIDERS["zhipu"]["api_key"]
    client = ZhipuAI(api_key=api_key)

    if batch_id:
        # 删除指定 batch 关联的文件
        batch = client.batches.retrieve(batch_id)
        file_ids = set()
        for attr in ("input_file_id", "output_file_id", "error_file_id"):
            fid = getattr(batch, attr, None)
            if fid:
                file_ids.add(fid)

        for fid in file_ids:
            try:
                client.files.delete(file_id=fid)
                print(f"[cleanup] 已删除文件: {fid}")
            except Exception as e:
                print(f"[cleanup] 删除失败 {fid}: {e}")
        print(f"[cleanup] Batch {batch_id} 关联的 {len(file_ids)} 个文件已清理")
    else:
        # 列出所有文件
        try:
            file_list = client.files.list(purpose="batch")
            files = list(file_list.data) if hasattr(file_list, "data") else []
            if not files:
                print("[cleanup] 平台上无 Batch 文件")
                return
            print(f"[cleanup] 平台上共 {len(files)} 个 Batch 文件:")
            for f in files:
                print(f"  {f.id}  {f.filename}  {f.bytes / 1024:.1f}KB  {f.created_at}")
            print(f"\n[cleanup] 如需全部删除, 运行: python batch_client.py cleanup --all")
        except Exception as e:
            print(f"[cleanup] 列出文件失败: {e}")


def cleanup_all_batch_files():
    """删除平台上所有 Batch 相关文件。"""
    from zhipuai import ZhipuAI

    api_key = config.LLM_PROVIDERS["zhipu"]["api_key"]
    client = ZhipuAI(api_key=api_key)

    try:
        file_list = client.files.list(purpose="batch")
        files = list(file_list.data) if hasattr(file_list, "data") else []
    except Exception as e:
        print(f"[cleanup] 列出文件失败: {e}")
        return

    if not files:
        print("[cleanup] 平台上无 Batch 文件")
        return

    deleted = 0
    for f in files:
        try:
            client.files.delete(file_id=f.id)
            deleted += 1
            print(f"  已删除: {f.filename} ({f.id})")
        except Exception as e:
            print(f"  删除失败: {f.filename} -> {e}")
    print(f"[cleanup] 共删除 {deleted}/{len(files)} 个文件")


# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="智谱 Batch API 客户端 (50% 价格)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # E0 全流程
  python batch_client.py run --strategy E0

  # E3 分两步运行
  python batch_client.py prepare --strategy E3          # 第一步: 本地 embed+rerank 选块
    # → 查看 output/results/e3/batch/e3_chunks_map.json 确认选块质量
  python batch_client.py submit  --strategy E3          # 第二步: 提交 Batch (返回 batch_id)
  python batch_client.py collect --strategy E3 --batch-id batch_2039282233651245056  # 第三步: 等待+下载+写入结果

  # 其他
  python batch_client.py status  --batch-id batch_2039282233651245056   # 查询状态
  python batch_client.py cleanup                        # 列出平台上的 Batch 文件
  python batch_client.py cleanup --all                  # 删除全部 Batch 文件
  python batch_client.py cleanup --batch-id batch_2039282233651245056   # 删除指定 batch 的文件
        """,
    )
    parser.add_argument(
        "action",
        choices=["prepare", "submit", "collect", "status", "download", "run", "cleanup"],
        help="操作: prepare(选块/生成请求), submit(提交), collect(等待+下载+写入), status(查状态), download(下载), run(全流程), cleanup(清理文件)",
    )
    parser.add_argument("--strategy", type=str, default="E0", choices=["E0", "E1", "E3"], help="策略")
    parser.add_argument("--limit", type=int, default=None, help="处理文档数限制")
    parser.add_argument("--batch-id", type=str, default=None, help="Batch ID (status/download/cleanup)")
    parser.add_argument("--poll-interval", type=int, default=30, help="轮询间隔秒数")
    parser.add_argument("--all", action="store_true", help="cleanup 时删除全部文件")
    args = parser.parse_args()

    # 获取文档列表
    md_dir = config.MD_DIR
    if not md_dir.exists():
        md_dir = Path(config.DOCUMENTS_DIR)
    md_files = sorted(md_dir.glob("*.md")) if md_dir.exists() else []

    strategy = args.strategy.upper()

    if args.action == "run":
        if not md_files:
            print(f"[batch] 未找到 MD 文件: {md_dir}")
            return
        if strategy == "E0":
            run_e0_batch(md_files, limit=args.limit, poll_interval=args.poll_interval)
        elif strategy == "E1":
            run_e1_batch(md_files, limit=args.limit, poll_interval=args.poll_interval)
        elif strategy == "E3":
            run_e3_batch(md_files, limit=args.limit, poll_interval=args.poll_interval)

    elif args.action == "prepare":
        if not md_files:
            print(f"[batch] 未找到 MD 文件: {md_dir}")
            return
        if strategy == "E0":
            prepare_e0_batch(md_files)
        elif strategy == "E1":
            prepare_e1_extraction_batch(md_files)
        elif strategy == "E3":
            prepare_e3_batch(md_files)

    elif args.action == "submit":
        batch_dir = _batch_dir(strategy)
        if strategy == "E0":
            input_path = batch_dir / "e0_batch_input.jsonl"
        elif strategy == "E3":
            input_path = batch_dir / "e3_batch_input.jsonl"
        else:
            input_path = batch_dir / "e1_extraction_batch_input.jsonl"
        if not input_path.exists():
            print(f"[batch] 请先 prepare: {input_path}")
            return
        bid = submit_batch(input_path, description=f"{strategy} batch")
        print(f"[batch] 已提交, batch_id={bid}")
        print(f"[batch] 查询状态: python batch_client.py status --batch-id {bid}")
        print(f"[batch] 等待并写入结果: python batch_client.py collect --strategy {strategy} --batch-id {bid}")

    elif args.action == "collect":
        if not args.batch_id:
            print("[batch] 请指定 --batch-id")
            return
        if strategy == "E3":
            collect_e3_batch(args.batch_id, poll_interval=args.poll_interval)
        elif strategy == "E0":
            collect_e0_batch(args.batch_id, poll_interval=args.poll_interval)
        else:
            print(f"[batch] collect 暂不支持 {strategy}，请用 run 一键完成")

    elif args.action == "status":
        if not args.batch_id:
            print("[batch] 请指定 --batch-id")
            return
        check_batch_status(args.batch_id)

    elif args.action == "download":
        if not args.batch_id:
            print("[batch] 请指定 --batch-id")
            return
        output_path = _batch_dir(strategy) / f"batch_output_{args.batch_id}.jsonl"
        download_batch_results(args.batch_id, output_path)

    elif args.action == "cleanup":
        if getattr(args, "all", False):
            cleanup_all_batch_files()
        else:
            cleanup_batch_files(batch_id=args.batch_id)


if __name__ == "__main__":
    main()
