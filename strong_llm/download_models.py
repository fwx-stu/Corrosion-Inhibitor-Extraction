"""
模型下载脚本
============

支持 HuggingFace 和 ModelScope（魔搭）两种下载源。

用法:
  # 从魔搭下载全部模型（国内服务器推荐）
  python download_models.py --all --source modelscope

  # 从 HuggingFace 镜像下载
  python download_models.py --all --source hf-mirror

  # 仅下载 NuExtract + GLiNER + Embedding/Reranker（核心组合）
  python download_models.py --nuextract --gliner --embedding --source modelscope

  # 仅下载 embedding/reranker
  python download_models.py --embedding --source modelscope

  # 下载到自定义目录
  python download_models.py --all --source modelscope --cache-dir /data/models

模型清单:
  ┌──────────────────────────┬──────────────────────────────────┬──────────┬──────────────┐
  │ 模型                      │ ModelScope / HF ID                │ 大小      │ 用途          │
  ├──────────────────────────┼──────────────────────────────────┼──────────┼──────────────┤
  │ GLiNER medium v2.1        │ urchade/gliner_medium-v2.1        │ ~1.2 GB  │ 零样本 NER    │
  │ NuExtract-1.5-smol        │ numind/NuExtract-1.5-smol         │ ~1.0 GB  │ 结构化抽取     │
  │ BGE-M3                    │ BAAI/bge-m3                       │ ~2.2 GB  │ Embedding    │
  │ BGE-Reranker-v2-m3        │ BAAI/bge-reranker-v2-m3           │ ~2.2 GB  │ Reranker     │
  │ Qwen2.5-3B-Instruct       │ Qwen/Qwen2.5-3B-Instruct          │ ~6.0 GB  │ 通用指令(可选) │
  │ Qwen2.5-1.5B-Instruct     │ Qwen/Qwen2.5-1.5B-Instruct        │ ~3.0 GB  │ 通用指令(可选) │
  └──────────────────────────┴──────────────────────────────────┴──────────┴──────────────┘

  核心组合 (GLiNER + NuExtract + BGE-M3 + Reranker): ~6.6 GB
  全部模型: ~15.6 GB
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

import config


def get_cache_dir() -> Path:
    """获取模型缓存目录。"""
    if hasattr(config, "MODELS_DIR") and config.MODELS_DIR:
        return Path(config.MODELS_DIR)
    return _EXPERIMENT_DIR / "models"


def download_from_modelscope(model_id: str, cache_dir: Path):
    """从 ModelScope（魔搭）下载模型，失败抛异常。"""
    from modelscope import snapshot_download

    print(f"\n{'='*60}")
    print(f"[ModelScope] 下载: {model_id}")
    print(f"目录: {cache_dir / model_id.replace('/', '--')}")
    print(f"{'='*60}")

    local_dir = snapshot_download(
        model_id=model_id,
        cache_dir=str(cache_dir),
        revision="master",
    )
    print(f"完成: {local_dir}")
    return local_dir


def download_from_hf(model_id: str, cache_dir: Path, endpoint: str = None):
    """从 HuggingFace 下载模型到统一目录。"""
    from huggingface_hub import snapshot_download

    # 统一目录格式: models/org/model/（与 ModelScope 一致）
    org, model = model_id.split("/")
    local_dir = cache_dir / org / model
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[HuggingFace] 下载: {model_id}")
    print(f"目录: {local_dir}")
    if endpoint:
        print(f"镜像: {endpoint}")
    print(f"{'='*60}")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        endpoint=endpoint,
        resume_download=True,
        local_files_only=False,
    )
    print(f"完成: {local_dir}")
    return local_dir


# 魔搭上有的模型（国内可直接下载）
_MODELSCOPE_MODELS = {
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
}

# 魔搭没有的模型，需要从 HuggingFace 下载
_HF_ONLY_MODELS = {
    "urchade/gliner_medium-v2.1",
    "numind/NuExtract-1.5-smol",
}


def download_model(model_id: str, cache_dir: Path, source: str):
    """统一下载入口。魔搭没有的模型自动回退到 HuggingFace。"""
    cache_dir.mkdir(parents=True, exist_ok=True)

    if source == "modelscope" and model_id in _HF_ONLY_MODELS:
        print(f"\n[提示] {model_id} 魔搭上没有，改用 HuggingFace 镜像")
        download_from_hf(model_id, cache_dir, endpoint="https://hf-mirror.com")
    elif source == "modelscope":
        download_from_modelscope(model_id, cache_dir)
    elif source == "hf-mirror":
        download_from_hf(model_id, cache_dir, endpoint="https://hf-mirror.com")
    else:
        download_from_hf(model_id, cache_dir)


# ── 各模型下载 ──────────────────────────────────────────────────

def download_gliner(cache_dir: Path, source: str):
    download_model("urchade/gliner_medium-v2.1", cache_dir, source)


def download_nuextract(cache_dir: Path, source: str):
    download_model("numind/NuExtract-1.5-smol", cache_dir, source)


def download_bge_m3(cache_dir: Path, source: str):
    """BGE-M3 embedding 模型。"""
    download_model("BAAI/bge-m3", cache_dir, source)


def download_bge_reranker(cache_dir: Path, source: str):
    """BGE-Reranker-v2-m3 重排序模型。"""
    download_model("BAAI/bge-reranker-v2-m3", cache_dir, source)


def download_instruct(cache_dir: Path, source: str, size: str = "3B"):
    download_model(f"Qwen/Qwen2.5-{size}-Instruct", cache_dir, source)


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="下载小模型 + RAG 实验所需的全部模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="下载全部模型")
    parser.add_argument("--gliner", action="store_true", help="GLiNER medium v2.1")
    parser.add_argument("--nuextract", action="store_true", help="NuExtract-1.5-smol")
    parser.add_argument("--embedding", action="store_true",
                        help="BGE-M3 + BGE-Reranker-v2-m3（本地 Embedding/Reranker）")
    parser.add_argument("--instruct", action="store_true", help="Qwen2.5-3B-Instruct + 1.5B")
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="模型缓存目录（默认: experiment/models/）"
    )
    parser.add_argument(
        "--source", type=str, default="modelscope",
        choices=["modelscope", "hf-mirror", "huggingface"],
        help="下载源（国内服务器推荐 modelscope）"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else get_cache_dir()
    source = args.source
    print(f"模型缓存目录: {cache_dir}")
    print(f"下载源: {source}")

    # 检查依赖
    if source == "modelscope":
        try:
            import modelscope  # noqa: F401
        except ImportError:
            print("请先安装 modelscope: pip install modelscope")
            sys.exit(1)
        # 某些模型魔搭没有，需要 huggingface_hub 兜底
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print("请先安装 huggingface_hub: pip install huggingface_hub")
            sys.exit(1)
    else:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print("请先安装 huggingface_hub: pip install huggingface_hub")
            sys.exit(1)

    download_all = args.all or (not args.gliner and not args.nuextract
                                 and not args.embedding and not args.instruct)

    if download_all or args.embedding:
        download_bge_m3(cache_dir, source)
        download_bge_reranker(cache_dir, source)

    if download_all or args.gliner:
        download_gliner(cache_dir, source)

    if download_all or args.nuextract:
        download_nuextract(cache_dir, source)

    if download_all or args.instruct:
        download_instruct(cache_dir, source, size="3B")
        download_instruct(cache_dir, source, size="1.5B")

    # ── 打印后续步骤 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("全部下载完成！")
    print(f"模型位置: {cache_dir}")
    print()
    print("接下来配置 .env（使用本地 embedding）：")
    print("  EMBEDDING_BACKEND=local")
    print()
    print("然后运行:")
    print("  python run_full_comparison.py --limit 5          # 试跑 5 篇")
    print("  python run_full_comparison.py                    # 全量对比")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
