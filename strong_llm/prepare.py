"""
数据准备模块

功能:
  - PDF → Markdown 转换 (opendataloader_pdf)
  - Markdown → Chunks 切分 (splitter.py)
  - Gold 标注数据加载
"""

from __future__ import annotations

import importlib.util
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import config


# ============================================================
# 工具函数
# ============================================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _import_splitter():
    """动态导入 splitter 模块（避免硬依赖）。"""
    splitter_path = str(config.SPLITTER_PATH)
    spec = importlib.util.spec_from_file_location("user_splitter", splitter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入 splitter: {splitter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _docs_to_chunks(docs: list, paper_id: str) -> List[Dict[str, Any]]:
    """将 splitter 返回的 Document 对象列表转为 dict 列表。"""
    chunks: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs):
        meta = getattr(doc, "metadata", {}) or {}
        section_title = (
            meta.get("header_4")
            or meta.get("header_3")
            or meta.get("header_2")
            or meta.get("header_1")
            or meta.get("title")
            or ""
        )
        chunks.append({
            "paper_id": paper_id,
            "chunk_id": meta.get("chunk_id", f"{paper_id}#chunk_{i}"),
            "chunk_index_in_doc": meta.get("chunk_index_in_doc", i),
            "total_chunks_in_doc": meta.get("total_chunks_in_doc", len(docs)),
            "chunk_type": meta.get("content_type", "text"),
            "section_title": section_title,
            "text": getattr(doc, "page_content", ""),
            "metadata": meta,
        })
    return chunks


# ============================================================
# PDF → Markdown
# ============================================================
def convert_pdfs(pdf_dir: Path = None, md_dir: Path = None) -> None:
    """使用 opendataloader_pdf 将 PDF 转换为 Markdown。"""
    pdf_dir = pdf_dir or config.PDF_DIR
    md_dir = md_dir or config.MD_DIR

    if not pdf_dir.exists():
        print(f"[prepare] PDF 目录不存在: {pdf_dir}")
        return

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[prepare] PDF 目录为空: {pdf_dir}")
        return

    ensure_dir(md_dir)

    try:
        import opendataloader_pdf
    except ImportError:
        raise ImportError(
            "需要安装 opendataloader_pdf:\n  pip install opendataloader_pdf"
        )

    print(f"[prepare] 转换 {len(pdfs)} 个 PDF → {md_dir}")
    opendataloader_pdf.convert(
        input_path=[str(pdf_dir)],
        output_dir=str(md_dir),
        format="json，markdown",
    )
    print("[prepare] PDF 转换完成")


# ============================================================
# Markdown → Chunks
# ============================================================
def chunk_documents(
    md_dir: Path = None,
    chunks_dir: Path = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> int:
    """切分 Markdown 文档为 chunks 并保存为 JSON。返回处理的文档数。"""
    md_dir = md_dir or config.MD_DIR
    chunks_dir = chunks_dir or config.CHUNKS_DIR
    chunk_size = chunk_size or config.DEFAULT_PARAMS["chunk_size"]
    chunk_overlap = chunk_overlap or config.DEFAULT_PARAMS["chunk_overlap"]

    if not md_dir.exists():
        print(f"[prepare] MD 目录不存在: {md_dir}")
        return 0

    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        print(f"[prepare] MD 目录为空: {md_dir}")
        return 0

    ensure_dir(chunks_dir)
    splitter_module = _import_splitter()

    # 尝试带元数据的 splitter；若不支持则用基础版
    meta_path = config.METADATA_INDEX_PATH
    use_metadata = meta_path and Path(meta_path).exists()

    if use_metadata:
        splitter = splitter_module.create_metadata_aware_splitter(
            metadata_index_path=str(meta_path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        splitter = splitter_module.create_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    count = 0
    for md_path in md_files:
        paper_id = md_path.stem
        out_path = chunks_dir / f"{paper_id}_chunks.json"

        if use_metadata:
            docs = splitter.split_file(str(md_path))
        else:
            text = md_path.read_text(encoding="utf-8")
            docs = splitter.split_text(
                text,
                metadata={
                    "source": str(md_path),
                    "filename": md_path.name,
                    "doc_id": paper_id,
                    "title": paper_id,
                },
            )

        chunks = _docs_to_chunks(docs, paper_id)
        out_path.write_text(
            json.dumps({"paper_id": paper_id, "chunks": chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        count += 1

    print(f"[prepare] 完成切分: {count} 篇 → {chunks_dir}")
    return count


# ============================================================
# Markdown 切分（单文件, 供 pipeline 调用）
# ============================================================
def split_single_md(
    md_path: Path,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """切分单个 MD 文件，返回 (paper_id, chunks)。"""
    chunk_size = chunk_size or config.DEFAULT_PARAMS["chunk_size"]
    chunk_overlap = chunk_overlap or config.DEFAULT_PARAMS["chunk_overlap"]
    paper_id = md_path.stem

    splitter_module = _import_splitter()

    meta_path = config.METADATA_INDEX_PATH
    if meta_path and Path(meta_path).exists():
        splitter = splitter_module.create_metadata_aware_splitter(
            metadata_index_path=str(meta_path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs = splitter.split_file(str(md_path))
    else:
        splitter = splitter_module.create_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        text = md_path.read_text(encoding="utf-8")
        docs = splitter.split_text(
            text,
            metadata={
                "source": str(md_path),
                "filename": md_path.name,
                "doc_id": paper_id,
                "title": paper_id,
            },
        )

    return paper_id, _docs_to_chunks(docs, paper_id)


# ============================================================
# 加载 chunk JSON
# ============================================================
def load_chunk_json(json_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """从 JSON 文件加载 chunks。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    paper_id = data.get("paper_id") or json_path.stem.replace("_chunks", "")
    return paper_id, data.get("chunks", [])


# ============================================================
# Gold 标注加载
# ============================================================
def load_gold(path: Path = None) -> Dict[str, List[Dict[str, Any]]]:
    """加载 JSONL 格式的 gold 标注，返回 {paper_id: [records]}。"""
    path = path or config.GOLD_PATH
    if not path.exists():
        print(f"[prepare] Gold 文件不存在: {path}")
        return {}

    gold_map: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("paper_id")
            if not pid:
                continue
            records = obj.get("records", [])
            # 兼容嵌套格式
            if not records:
                result = obj.get("result", {})
                records = result.get("records", [])
                if not records:
                    cia = result.get("corrosion_inhibition_analysis", {})
                    records = cia.get("records", [])
            gold_map[str(pid)] = records
    return gold_map


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="数据准备")
    parser.add_argument("--convert", action="store_true", help="PDF → Markdown")
    parser.add_argument("--chunk", action="store_true", help="Markdown → Chunks")
    parser.add_argument("--all", action="store_true", help="执行全部步骤")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    if args.all or args.convert:
        convert_pdfs()
    if args.all or args.chunk:
        chunk_documents(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    if not (args.all or args.convert or args.chunk):
        parser.print_help()


if __name__ == "__main__":
    main()
