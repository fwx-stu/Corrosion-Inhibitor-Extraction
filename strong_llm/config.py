"""
实验框架配置中心

集中管理路径、Schema、LLM 参数、超参数、BO 搜索空间。
通过 python-dotenv 加载 API Key。

设计原则:
  - 阶段 A (E0/E1): 零基础设施依赖，只需 LLM API
  - 阶段 B (E3-E6): 需 Milvus + embedding/reranker 模型
"""

import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# ============================================================
# 路径体系（绝对路径，迁移项目时只需修改 EXPERIMENT_DIR）
# ============================================================
EXPERIMENT_DIR = Path(__file__).resolve().parent
IDEA_DIR       = EXPERIMENT_DIR.parent
PROJECT_ROOT   = IDEA_DIR.parent

# 加载 .env（按优先级搜索）
for _env in [
    EXPERIMENT_DIR / ".env",
    IDEA_DIR / ".env",
    PROJECT_ROOT / "data" / "metadata" / ".env",
]:
    if _env.exists():
        load_dotenv(_env)
        break

# --- 输入目录 ---
INPUT_DIR  = EXPERIMENT_DIR / "input"
PDF_DIR    = INPUT_DIR / "pdf"
MD_DIR     = INPUT_DIR / "md"
GOLD_DIR   = INPUT_DIR / "gold"
GOLD_PATH  = GOLD_DIR / "train.jsonl"
TEST_PATH  = GOLD_DIR / "test.jsonl"    # 泛化验证集（100 篇，标注中）

# --- 输出目录 ---
OUTPUT_DIR  = EXPERIMENT_DIR / "output"
CHUNKS_DIR  = OUTPUT_DIR / "chunks"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# --- 提示词目录（代码资源，非数据）---
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"

# --- 切分器（已在 experiment 目录下）---
SPLITTER_PATH = EXPERIMENT_DIR / "splitter.py"
METADATA_INDEX_PATH = PROJECT_ROOT / "data" / "metadata" / "output" / "metadata_index.json"

# 已有文档目录（作为 MD_DIR 的 fallback）
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "metadata" / "output" / "documents"

# --- 模型路径 (experiment/models/) ---
MODELS_DIR      = EXPERIMENT_DIR / "models"
GLINER_MODEL    = str(MODELS_DIR / "gliner_medium-v2.1")
NUEXTRACT_MODEL = str(MODELS_DIR / "NuExtract-1.5-smol")
INSTRUCT_MODEL  = str(MODELS_DIR / "Qwen2.5-3B-Instruct")

# --- embedding / reranker 本地模型 ---
EMBEDDING_MODEL = str(MODELS_DIR / "BAAI" / "bge-m3")
RERANKER_MODEL  = str(MODELS_DIR / "BAAI" / "bge-reranker-v2-m3")

# ============================================================
# SiliconFlow API 配置 (embedding/reranker 云端推理)
# EMBEDDING_BACKEND: "local" 使用本地模型, "siliconflow" 使用 API
# ============================================================
SILICONFLOW_API_KEY  = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_EMBED_MODEL   = "BAAI/bge-m3"
SILICONFLOW_RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "siliconflow")  # "local" | "siliconflow"

# Milvus (全局知识库，与 rag_server_v2.py 保持一致)
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = ""
MILVUS_COLLECTION = "corrosion_inhibitor_milvus"
MILVUS_DENSE_FIELD = "vector"           # Dense 向量字段 (COSINE)
MILVUS_SPARSE_FIELD = "sparse_vector"   # Sparse 向量字段 (IP)
MILVUS_TEXT_FIELD = "text"
MILVUS_PK_FIELD = "id"

# 混合检索权重 (Dense + Sparse via Milvus WeightedRanker)
MILVUS_DENSE_WEIGHT = 0.7
MILVUS_SPARSE_WEIGHT = 0.3

# Global-RAG 去重与过滤参数
MILVUS_EXCLUDE_SAME_DOI = True          # 检索时排除与目标论文相同 DOI 的 chunks
MILVUS_DEDUP_THRESHOLD  = 0.95          # 余弦相似度 > 此值视为重复，只保留一条
MILVUS_EXTERNAL_TOP_K   = 3            # 全局检索返回的外部 chunk 数（E5/E6 使用）
MILVUS_EXCLUDE_TABLES   = True          # E5 全局检索时排除表格 chunks，只保留文本补全知识
WIKI_GLOBAL_TOP_K       = 30           # E1 Wiki-RAG：wiki 蒸馏阶段检索的外部 chunk 数（建议 10-20，越多 wiki 越丰富）

# ChromaDB (局部 RAG，单文档临时集合)
CHROMA_LOCAL_DIR = OUTPUT_DIR / "chroma_local"   # 持久化目录，可选
CHROMA_DISTANCE  = "cosine"                       # cosine | l2 | ip

DEVICE = "cuda:0"

# ============================================================
# Schema: 13 字段 (9 核心 + 4 量子化学/热力学参数)
# ============================================================
FIELD_NAMES = [
    "inhibitor_name",
    "substrate",
    "corrosive_medium",
    "medium_concentration",
    "inhibitor_concentration",
    "temperature",
    "immersion_time",
    "test_method",
    "inhibition_efficiency",
    "homo",
    "lumo",
    "delta_g_ads",
    "adsorption_energy",
]

# 紧凑输出格式：短字段名 ↔ 标准字段名映射
FIELD_SHORT_MAP = {
    "inh":    "inhibitor_name",
    "sub":    "substrate",
    "med":    "corrosive_medium",
    "med_c":  "medium_concentration",
    "inh_c":  "inhibitor_concentration",
    "temp":   "temperature",
    "time":   "immersion_time",
    "method": "test_method",
    "ie":     "inhibition_efficiency",
    "homo":   "homo",
    "lumo":   "lumo",
    "dg":     "delta_g_ads",
    "eads":   "adsorption_energy",
}
FIELD_LONG_TO_SHORT = {v: k for k, v in FIELD_SHORT_MAP.items()}

# ============================================================
# 策略依赖声明
# ============================================================
STRATEGY_DEPS = {
    "E0": [],
    "E1": ["embedding", "milvus"],   # Wiki-RAG: 需要 Milvus 全局检索
    "E3": ["embedding", "reranker"],
    "E4": ["embedding", "reranker"],
    "E5": ["embedding", "milvus"],
    "E6": ["embedding", "reranker", "milvus"],
}

ALL_STRATEGIES = ["E0", "E1", "E3", "E4", "E5", "E6"]

# ============================================================
# E3/E4 字段级语义查询（替代裸字段名，提升检索相关度）
# ============================================================
FIELD_QUERIES: Dict[str, str] = {
    "inhibitor_name":          "inhibitor compound name abbreviation molecule Table inhibitor structure",
    "substrate":               "substrate metal steel alloy electrode material Table specimen",
    "corrosive_medium":        "corrosive medium acid solution electrolyte HCl H2SO4 NaCl Table experimental conditions",
    "medium_concentration":    "acid concentration solution molarity mol/L wt% Table experimental conditions",
    "inhibitor_concentration": "inhibitor concentration ppm mg/L dosage Table experimental conditions",
    "temperature":             "temperature degrees Celsius Kelvin Table experimental conditions",
    "immersion_time":          "immersion time hours exposure duration Table weight loss experimental conditions",
    "test_method":             "test method weight loss EIS polarization electrochemical Table electrochemical measurements",
    "inhibition_efficiency":   "inhibition efficiency IE% corrosion protection performance Table weight loss electrochemical results",
    "homo":                    "HOMO highest occupied molecular orbital energy DFT eV Table quantum chemical parameters",
    "lumo":                    "LUMO lowest unoccupied molecular orbital energy DFT eV Table quantum chemical parameters",
    "delta_g_ads":             "Gibbs free energy adsorption delta G kJ/mol Table thermodynamic parameters adsorption",
    "adsorption_energy":       "adsorption energy Eads kcal/mol molecular dynamics Table simulation parameters",
}

# E3 字段级检索参数
FIELD_RETRIEVAL_TOP_K = 15      # 每字段检索的初始候选数
MAX_FINAL_CHUNKS     = 60     # 合并去重后送入 LLM 的最大 chunk 数

# ============================================================
# LLM 配置 (兼容 idea/llm_client.py 的接口)
# ============================================================
LLM_PROVIDER = "zhipu"
LLM_TEMPERATURE = 0.15
LLM_MAX_TOKENS = 65536    # 实时 API 最大输出 token
BATCH_MAX_TOKENS = 8192   # Batch API 限制 (glm-4-plus 校验上限)
LLM_MAX_RETRIES = 5

# API rate limiting
RATE_LIMIT_DELAY_S = 35        # 每次 LLM 调用后等待（RPM=2 需要 ≥30s）
RATE_LIMIT_RECOVERY_S = 120    # 触发限速后额外等待
MAX_CONSECUTIVE_FAILS = 5      # 连续失败超过此次数则提前终止策略

# idea/llm_client.py 依赖此字典作为默认值
LLM_PROVIDERS = {
    "zhipu": {
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
        "model": os.getenv("ZHIPU_MODEL", "glm-4-plus"),
        "base_url": None,
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
    },
    "qwen": {
        "api_key": os.getenv("QWEN_API_KEY", ""),
        "model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
}

# ============================================================
# Pipeline 超参数默认值
# ============================================================
DEFAULT_PARAMS = {
    "chunk_size": 1000,
    "chunk_overlap": 50,
    "retrieval_top_k": 60,
    "rerank_top_n": 40,
    "window_size": 1,
    "dense_weight": 0.7,
    "bm25_weight": 0.3,       # BM25 混合检索权重：0=纯向量，1=纯BM25
    "enable_reranker": True,
    "enable_window_expand": True,
    "enable_query_rewrite": False,
    "reranker_backend": "cross_encoder",
    "max_doc_chars": 80000,
}

# ============================================================
# BO 搜索空间 (Optuna)
# ============================================================
BO_SEARCH_SPACE = {
    "chunk_size": {"type": "categorical", "choices": [400, 600, 800, 1000, 1200]},
    "chunk_overlap": {"type": "categorical", "choices": [0, 50, 100, 150, 200]},
    "local_top_k": {"type": "int", "low": 10, "high": 70},
    "global_top_k": {"type": "int", "low": 10, "high": 70},
    "rerank_top_n": {"type": "int", "low": 5, "high": 50},
    "enable_reranker": {"type": "categorical", "choices": [True, False]},
    "window_size": {"type": "int", "low": 0, "high": 2},
    "dense_weight": {"type": "float", "low": 0.5, "high": 0.9},
    "temperature": {"type": "float", "low": 0.0, "high": 0.15},
}

# E3 专用搜索空间（纯本地 ChromaDB，无 Milvus，去掉 dense_weight / global_top_k）
BO_SEARCH_SPACE_E3 = {
    "chunk_size":      {"type": "categorical", "choices": [400, 600, 800, 1000, 1200]},
    "chunk_overlap":   {"type": "categorical", "choices": [0, 50, 100, 150, 200]},
    "local_top_k":     {"type": "int", "low": 10, "high": 70},
    "rerank_top_n":    {"type": "int", "low": 5, "high": 50},
    "enable_reranker": {"type": "categorical", "choices": [True, False]},
    "window_size":     {"type": "int", "low": 0, "high": 2},
    "bm25_weight":     {"type": "float", "low": 0.0, "high": 0.5},
    "temperature":     {"type": "float", "low": 0.0, "high": 0.15},
}

# ============================================================
# 评估配置
# ============================================================
EVAL_FIELDS = FIELD_NAMES
RECORD_MATCH_THRESHOLD = 0.60          # 至少 60% 字段匹配才算配对成功
NUMERIC_REL_TOL = 0.02                 # 相对误差 2%
NUMERIC_ABS_TOL = 1e-6                 # 绝对误差 1e-6


def check_deps(strategy: str) -> List[str]:
    """检查策略所需依赖是否可用，返回缺失依赖列表。"""
    deps = STRATEGY_DEPS.get(strategy, [])
    missing = []
    for dep in deps:
        if dep == "milvus":
            try:
                from pymilvus import MilvusClient  # noqa: F401
            except ImportError:
                missing.append("pymilvus")
                continue
            # 检测 Milvus 服务可达性
            try:
                from pymilvus import connections
                connections.connect(alias="dep_check", uri=MILVUS_URI, token=MILVUS_TOKEN, timeout=5)
                connections.disconnect("dep_check")
            except Exception:
                missing.append(f"Milvus 服务不可达: {MILVUS_URI}")
        elif dep == "embedding":
            if EMBEDDING_BACKEND == "siliconflow":
                if not SILICONFLOW_API_KEY:
                    missing.append("SILICONFLOW_API_KEY not set")
            elif not Path(EMBEDDING_MODEL).exists():
                missing.append(f"embedding model: {EMBEDDING_MODEL}")
        elif dep == "reranker":
            if EMBEDDING_BACKEND == "siliconflow":
                if not SILICONFLOW_API_KEY:
                    missing.append("SILICONFLOW_API_KEY not set (reranker)")
            elif not Path(RERANKER_MODEL).exists():
                missing.append(f"reranker model: {RERANKER_MODEL}")
    return missing
