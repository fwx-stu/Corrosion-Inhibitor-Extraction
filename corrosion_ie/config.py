from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import expand_env_vars, read_yaml_or_json


def _env_nonempty(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def default_models_dir(config_path: str | Path | None = None) -> Path:
    """Auto-detect the shared local model directory.

    This mirrors the legacy small-model project behavior: model lookup is
    anchored to the project/config location, not to whatever shell directory
    happens to be current.  It supports the common layout used on the server:

        workdir/
          models/
            BAAI/bge-m3
            BAAI/bge-reranker-v2-m3
            Qwen2.5-3B-Instruct
          corrosion_experiment_llm_prompt_compare_t005/
            configs/
            scripts/
            corrosion_ie/

    Environment variables still take priority for reproducibility.
    """
    env = _env_nonempty("MODEL_CACHE_DIR") or _env_nonempty("MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()

    if config_path is not None:
        project_root = Path(config_path).resolve().parent.parent
    else:
        project_root = Path(__file__).resolve().parents[1]

    candidates = [
        project_root.parent / "models",  # sibling of project package, legacy default
        project_root / "models",         # inside project package
        Path.cwd() / "models",           # current shell dir
        Path.cwd().parent / "models",    # useful when running from scripts/ or subdirs
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (project_root.parent / "models").resolve()


def ensure_model_env(config_path: str | Path | None = None) -> Path:
    """Populate model-cache environment variables when they are unset.

    The config files use ${MODELS_DIR}; the retrieval code uses BGE defaults;
    HuggingFace also consults several cache variables.  Setting them here makes
    Qwen, BGE-M3 embedding, and BGE reranker resolve from the same model root.
    Existing explicit environment variables are never overwritten.
    """
    models_dir = default_models_dir(config_path)
    for name in (
        "MODELS_DIR",
        "MODEL_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        if not os.getenv(name):
            os.environ[name] = str(models_dir)

    # Keep offline behavior by default for server-local model experiments; users
    # can override by exporting these variables before running the script.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if not os.getenv("EMBEDDING_MODEL"):
        os.environ["EMBEDDING_MODEL"] = str(models_dir / "BAAI" / "bge-m3")
    if not os.getenv("RERANKER_MODEL"):
        os.environ["RERANKER_MODEL"] = str(models_dir / "BAAI" / "bge-reranker-v2-m3")
    if not os.getenv("EMBEDDING_BACKEND"):
        os.environ["EMBEDDING_BACKEND"] = "local"
    return models_dir


@dataclass
class ExperimentConfig:
    raw: Dict[str, Any]
    config_path: Path

    @property
    def root(self) -> Path:
        pr = Path(self.raw.get("project_root", "."))
        if not pr.is_absolute():
            pr = self.config_path.parent.parent / pr
        return pr.resolve()

    @property
    def experiment_id(self) -> str:
        return str(self.raw.get("experiment_id", "exp"))


    @property
    def dataset_name(self) -> str:
        return str(self.raw.get("dataset_name", self.raw.get("dataset_role", "dataset")))

    @property
    def dataset_role(self) -> str:
        return str(self.raw.get("dataset_role", "dev"))

    @property
    def expected_paper_count(self) -> int | None:
        value = self.raw.get("expected_paper_count")
        return int(value) if value is not None else None

    @property
    def table_norm_dir(self) -> Path | None:
        value = self.raw.get("input", {}).get("table_norm_dir")
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @property
    def manifest_path(self) -> Path | None:
        value = self.raw.get("input", {}).get("manifest_path")
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @property
    def require_table_norm(self) -> bool:
        return bool(self.raw.get("input", {}).get("require_table_norm", False))

    @property
    def md_dir(self) -> Path:
        p = Path(self.raw.get("input", {}).get("md_dir", "input/md"))
        return p if p.is_absolute() else self.root / p

    @property
    def gold_path(self) -> Path:
        p = Path(self.raw.get("input", {}).get("gold_path", "input/gold/train.jsonl"))
        return p if p.is_absolute() else self.root / p

    @property
    def output_root(self) -> Path:
        p = Path(self.raw.get("output_root", "output/exp_v2"))
        return p if p.is_absolute() else self.root / p

    @property
    def run_root(self) -> Path:
        return self.output_root / self.experiment_id

    @property
    def models(self) -> List[str]:
        return list(self.raw.get("models", []))

    @property
    def strategies(self) -> List[str]:
        return list(self.raw.get("strategies", []))

    @property
    def prompt_variants(self) -> List[str]:
        return list(self.raw.get("prompt_variants", ["balanced"]))

    @property
    def repeat_seeds(self) -> List[int]:
        return [int(x) for x in self.raw.get("repeat_seeds", [1])]

    @property
    def retrieval(self) -> Dict[str, Any]:
        """Return the effective retrieval protocol.

        The original ``CustomMarkdownSplitter`` is the framework-wide default
        for both proprietary LLM and local SLM experiments.  New structured
        chunking is opt-in via ``chunking_backend: v2_structured`` so that model
        comparisons never change evidence segmentation accidentally.
        """
        ensure_model_env(self.config_path)
        cfg = expand_env_vars(dict(self.raw.get("retrieval", {})))
        cfg.setdefault("backend", "legacy_exact")
        cfg.setdefault("chunking_backend", "legacy_exact")
        cfg.setdefault("scoring_protocol", "legacy_exact")
        if str(cfg.get("backend", "")).lower() not in {"lexical", "none"}:
            cfg.setdefault("strict_loading", True)
        cfg.setdefault("enable_grounding_filter", False)
        cfg.setdefault("enable_row_level_table_evidence", False)
        cfg.setdefault("merge_row_table_candidates", False)
        cfg.setdefault("max_final_chunks", 40)
        cfg.setdefault("small_model_max_chunks", 80)
        cfg.setdefault("retrieval_candidate_max_chunks", 80)
        cfg.setdefault("field_rerank_top_n", 20)
        cfg.setdefault("rerank_rrf_k", 60)
        cfg.setdefault("window_candidate_max_chunks", 16)
        cfg.setdefault("max_window_chunks", 8)
        return cfg

    @property
    def generation(self) -> Dict[str, Any]:
        """Experiment-level generation overrides applied after model registry."""
        return expand_env_vars(dict(self.raw.get("generation", {})))

    @property
    def evaluation(self) -> Dict[str, Any]:
        return dict(self.raw.get("evaluation", {}))

    def strategy_params(self, strategy: str) -> Dict[str, Any]:
        params = self.raw.get("strategy_params", {}) or {}
        if strategy in params:
            return dict(params.get(strategy, {}))
        compatibility = {
            "q0_safe_full": "q0_full",
            "q1_budgeted_rag": "q1_naive_rag",
        }
        legacy = compatibility.get(strategy)
        return dict(params.get(legacy, {})) if legacy else {}


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path).resolve()
    ensure_model_env(p)
    raw = read_yaml_or_json(p)
    return ExperimentConfig(raw=raw, config_path=p)


def load_model_registry(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    ensure_model_env(config_path)
    reg_path = config_path.parent / "model_registry.yaml"
    raw = read_yaml_or_json(reg_path)
    return expand_env_vars(raw.get("models", {}))


def _candidate_has_model(p: Path) -> bool:
    if (p / "config.json").exists():
        return True
    snap = p / "snapshots"
    if snap.exists() and any((x / "config.json").exists() for x in snap.iterdir() if x.is_dir()):
        return True
    return False


def resolve_model_config(model_key: str, config_path: str | Path) -> Dict[str, Any]:
    registry = load_model_registry(config_path)
    if model_key not in registry:
        raise KeyError(f"Unknown model key: {model_key}. Available: {', '.join(registry)}")
    cfg = dict(registry[model_key])
    local_env = cfg.get("local_path_env")
    if local_env and os.getenv(local_env):
        cfg["resolved_path"] = os.getenv(local_env)
    else:
        for cand in cfg.get("local_path_candidates", []) or []:
            p = Path(str(cand)).expanduser()
            if (p / "config.json").exists():
                cfg["resolved_path"] = str(p.resolve())
                break
            snap = p / "snapshots"
            if snap.exists():
                hits = sorted([x for x in snap.iterdir() if (x / "config.json").exists()])
                if hits:
                    cfg["resolved_path"] = str(hits[-1].resolve())
                    break
    cfg["models_dir"] = str(default_models_dir(config_path))
    return cfg
