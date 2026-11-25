from __future__ import annotations

"""Legacy high-recall local RAG for the multi-model runner.

The parity path reproduces the previous small-model protocol:
field-level queries -> BGE-M3 dense retrieval + BM25 fusion -> reranker ->
window expansion -> table-chunk priority.

Chunking and scoring are configured independently:
- ``chunking_backend=legacy_exact`` is the framework default for LLM and SLM.
- ``chunking_backend=v2_structured`` is an explicit ablation only.
- ``scoring_protocol=legacy_exact`` restores the old BM25/dense fusion details.
"""

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..schemas import EvidenceUnit
from .chunker_factory import resolve_chunking_backend, split_document


LEGACY_FIELD_QUERIES: Dict[str, str] = {
    "inhibitor_name": "inhibitor compound name abbreviation molecule Table inhibitor structure",
    "substrate": "substrate metal steel alloy electrode material Table specimen",
    "corrosive_medium": "corrosive medium acid solution electrolyte HCl H2SO4 NaCl Table experimental conditions",
    "medium_concentration": "acid concentration solution molarity mol/L wt% Table experimental conditions",
    "inhibitor_concentration": "inhibitor concentration ppm mg/L dosage Table experimental conditions",
    "temperature": "temperature degrees Celsius Kelvin Table experimental conditions",
    "immersion_time": "immersion time hours exposure duration Table weight loss experimental conditions",
    "test_method": "test method weight loss EIS polarization electrochemical Table electrochemical measurements",
    "inhibition_efficiency": "inhibition efficiency IE% corrosion protection performance Table weight loss electrochemical results",
    "homo": "HOMO highest occupied molecular orbital energy DFT eV Table quantum chemical parameters",
    "lumo": "LUMO lowest unoccupied molecular orbital energy DFT eV Table quantum chemical parameters",
    "delta_g_ads": "Gibbs free energy adsorption delta G kJ/mol Table thermodynamic parameters adsorption",
    "adsorption_energy": "adsorption energy Eads kcal/mol molecular dynamics Table simulation parameters",
}

FIELD_TO_GROUP: Dict[str, str] = {
    "inhibitor_name": "performance",
    "inhibitor_concentration": "performance",
    "test_method": "performance",
    "inhibition_efficiency": "performance",
    "substrate": "conditions",
    "corrosive_medium": "conditions",
    "medium_concentration": "conditions",
    "temperature": "conditions",
    "immersion_time": "conditions",
    "homo": "quantum",
    "lumo": "quantum",
    "delta_g_ads": "adsorption",
    "adsorption_energy": "adsorption",
}

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u0394\u03b4][a-zA-Z0-9_°]*|%|℃|η|−|-?\d+(?:\.\d+)?")
_JUNK_RE = re.compile(r"^\s*((\[参见表格\s*\d+\]|\[.*?\])\s*)+$", re.DOTALL)


class RetrievalDependencyError(RuntimeError):
    """Raised when strict parity retrieval cannot load or execute a model."""


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _legacy_tokenize(text: str) -> List[str]:
    # Exact tokenizer from old retrieval.py / rank_bm25 path.
    return re.sub(r"[^\w\s]", " ", (text or "").lower()).split()


def _is_table_chunk(chunk: Mapping[str, Any]) -> bool:
    if chunk.get("chunk_type") == "table" or chunk.get("content_type") == "table":
        return True
    meta = chunk.get("metadata") or {}
    if isinstance(meta, Mapping) and (meta.get("content_type") == "table" or meta.get("is_table")):
        return True
    text = chunk.get("text", "") or ""
    return bool(
        re.search(r"<table[\s\S]*?</table>", text, re.I)
        or "[Normalized HTML Table Rows]" in text
        or _looks_like_md_table(text)
    )


def _looks_like_md_table(text: str) -> bool:
    lines = [x.strip() for x in (text or "").splitlines()]
    return sum(1 for x in lines if "|" in x and len(x.split("|")) >= 3) >= 2


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if abs(span) < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / span for k, v in scores.items()}




def _normalize_scores_legacy(scores: Dict[str, float]) -> Dict[str, float]:
    """Match the original LocalRetriever min-max normalization exactly.

    The old code used ``span = max - min or 1.0``.  Therefore an all-equal
    score vector becomes all zeros, not all ones.  That detail matters when a
    field query has no BM25 match.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in scores.items()}

def _default_model_path(kind: str) -> str:
    try:
        from ..config import ensure_model_env

        models_dir = ensure_model_env()
    except Exception:
        models_dir = Path(os.getenv("MODEL_CACHE_DIR") or os.getenv("MODELS_DIR") or "models").expanduser()
    if kind == "embedding":
        return os.getenv("EMBEDDING_MODEL", str(Path(models_dir) / "BAAI" / "bge-m3"))
    return os.getenv("RERANKER_MODEL", str(Path(models_dir) / "BAAI" / "bge-reranker-v2-m3"))


class _BM25Index:
    def __init__(self, tokenized_docs: List[List[str]], ids: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = tokenized_docs
        self.ids = ids
        self.k1 = k1
        self.b = b
        self.N = len(tokenized_docs)
        self.avgdl = sum(len(d) for d in tokenized_docs) / max(1, self.N)
        self.df: Counter[str] = Counter()
        for doc in tokenized_docs:
            self.df.update(set(doc))

    def score(self, query: str) -> Dict[str, float]:
        q_terms = _tokenize(query)
        if not q_terms:
            return {}
        out: Dict[str, float] = {}
        for cid, doc in zip(self.ids, self.docs):
            if not doc:
                continue
            tf = Counter(doc)
            dl = len(doc)
            score = 0.0
            for term in dict.fromkeys(q_terms):
                f = tf.get(term, 0)
                if not f:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / max(1e-9, self.avgdl))
                score += idf * f * (self.k1 + 1.0) / max(1e-9, denom)
            if score > 0:
                out[cid] = float(score)
        return out


class _LegacyBM25Index:
    """Exact wrapper around the rank_bm25 implementation used by old Q1."""

    def __init__(self, tokenized_docs: List[List[str]], ids: List[str]) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore

        self.index = BM25Okapi(tokenized_docs)
        self.ids = ids

    def score(self, query: str) -> Dict[str, float]:
        scores = self.index.get_scores(_legacy_tokenize(query))
        # Old code populated every chunk id, including zero-score chunks.
        return {cid: float(score) for cid, score in zip(self.ids, scores)}


class LegacyHybridRetriever:
    """Single-paper hybrid retriever with strict dependency checks and caching."""

    _encoder_cache: Dict[str, Any] = {}
    _reranker_cache: Dict[str, Any] = {}

    def __init__(self, retrieval_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = retrieval_cfg or {}
        self.embedding_model = str(self.cfg.get("embedding_model") or _default_model_path("embedding"))
        self.reranker_model = str(self.cfg.get("reranker_model") or _default_model_path("reranker"))
        self.device = str(
            self.cfg.get("device")
            or os.getenv("RETRIEVAL_DEVICE")
            or os.getenv("DEVICE")
            or "cuda:0"
        )
        self.embedding_backend = str(
            self.cfg.get("embedding_backend") or os.getenv("EMBEDDING_BACKEND") or "local"
        )
        backend = str(self.cfg.get("backend") or "legacy_exact").lower().strip()
        scoring_protocol = str(
            self.cfg.get("scoring_protocol")
            or ("legacy_exact" if backend in {"legacy_exact", "old_q1_exact", "exact"} else "legacy_hybrid")
        ).lower().strip()
        self.scoring_protocol = scoring_protocol
        self.exact_scoring = scoring_protocol in {"legacy_exact", "old_q1_exact", "exact"}
        self.chunking_backend = resolve_chunking_backend(self.cfg)
        self.reranker_backend = str(self.cfg.get("reranker_backend") or "cross_encoder").lower().strip()
        self.strict_loading = _as_bool(self.cfg.get("strict_loading"), False)
        self._last_embedding_error: Optional[str] = None
        self._last_rerank_error: Optional[str] = None
        self._prepared_fingerprint: Optional[str] = None
        self._prepared_ids: List[str] = []
        self._prepared_texts: List[str] = []
        self._id_to_chunk: Dict[str, Dict[str, Any]] = {}
        self._bm25: Optional[_BM25Index] = None
        self._doc_embeddings: Any = None
        self._document_encode_calls = 0
        self._query_encode_calls = 0
        self._rerank_result_cache: Dict[Tuple[str, Tuple[str, ...], int], List[Dict[str, Any]]] = {}

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return {
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "device": self.device,
            "strict_loading": self.strict_loading,
            "exact_scoring": self.exact_scoring,
            "reranker_backend": self.reranker_backend,
            "embedding_error": self._last_embedding_error,
            "rerank_error": self._last_rerank_error,
            "document_encode_calls": self._document_encode_calls,
            "query_encode_calls": self._query_encode_calls,
            "prepared_chunk_count": len(self._prepared_ids),
        }

    def _embedding_failure(self, exc: BaseException):
        self._last_embedding_error = repr(exc)
        if self.strict_loading:
            raise RetrievalDependencyError(
                f"BGE embedding failed in strict mode: model={self.embedding_model!r}, "
                f"device={self.device!r}, error={exc!r}"
            ) from exc
        return None

    def _reranker_failure(self, exc: BaseException):
        self._last_rerank_error = repr(exc)
        if self.strict_loading:
            raise RetrievalDependencyError(
                f"BGE reranker failed in strict mode: model={self.reranker_model!r}, "
                f"device={self.device!r}, error={exc!r}"
            ) from exc
        return None

    def _get_encoder(self):
        if self.embedding_backend == "none":
            if self.strict_loading:
                raise RetrievalDependencyError("embedding_backend=none is not allowed in strict parity mode")
            return None
        key = f"{self.embedding_backend}:{self.embedding_model}:{self.device}"
        if key in self._encoder_cache:
            return self._encoder_cache[key]
        try:
            if self.embedding_backend == "siliconflow":
                from siliconflow_embedder import SiliconFlowEmbedder  # type: ignore

                enc = SiliconFlowEmbedder()
            else:
                from sentence_transformers import SentenceTransformer  # type: ignore

                enc = SentenceTransformer(self.embedding_model, device=self.device)
            self._encoder_cache[key] = enc
            self._last_embedding_error = None
            return enc
        except Exception as exc:
            return self._embedding_failure(exc)

    def _encode(self, texts: List[str], *, document_batch: bool = False):
        enc = self._get_encoder()
        if enc is None:
            return None
        try:
            import numpy as np  # type: ignore

            arr = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            arr = np.asarray(arr, dtype="float32")
            if arr.ndim != 2 or arr.shape[0] != len(texts):
                raise ValueError(f"unexpected embedding shape {arr.shape}; expected ({len(texts)}, dim)")
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            if document_batch:
                self._document_encode_calls += 1
            else:
                self._query_encode_calls += 1
            self._last_embedding_error = None
            return arr / norms
        except Exception as exc:
            return self._embedding_failure(exc)

    def _get_reranker(self):
        key = f"{self.reranker_backend}:{self.reranker_model}:{self.device}"
        if key in self._reranker_cache:
            return self._reranker_cache[key]

        if self.embedding_backend == "siliconflow":
            try:
                from siliconflow_embedder import SiliconFlowReranker  # type: ignore

                rr = SiliconFlowReranker()
                self._reranker_cache[key] = ("siliconflow", rr)
                self._last_rerank_error = None
                return self._reranker_cache[key]
            except Exception as exc:
                return self._reranker_failure(exc)

        if self.reranker_backend == "flagembedding":
            try:
                from FlagEmbedding import FlagReranker  # type: ignore

                rr = FlagReranker(self.reranker_model, use_fp16=self.device.startswith("cuda"))
                self._reranker_cache[key] = ("flagembedding", rr)
                self._last_rerank_error = None
                return self._reranker_cache[key]
            except Exception as exc:
                return self._reranker_failure(exc)

        # Strict legacy parity uses CrossEncoder only, matching old retrieval.py.
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            rr = CrossEncoder(self.reranker_model, device=self.device)
            self._reranker_cache[key] = ("cross_encoder", rr)
            self._last_rerank_error = None
            return self._reranker_cache[key]
        except Exception as cross_error:
            if self.exact_scoring or self.strict_loading:
                return self._reranker_failure(cross_error)
            # Non-parity mode may use FlagEmbedding as a compatibility fallback.
            try:
                from FlagEmbedding import FlagReranker  # type: ignore

                rr = FlagReranker(self.reranker_model, use_fp16=self.device.startswith("cuda"))
                self._reranker_cache[key] = ("flagembedding", rr)
                self._last_rerank_error = None
                return self._reranker_cache[key]
            except Exception as second_error:
                return self._reranker_failure(
                    RuntimeError(f"CrossEncoder={cross_error!r}; FlagReranker={second_error!r}")
                )

    @staticmethod
    def _fingerprint(ids: List[str], texts: List[str]) -> str:
        h = hashlib.sha1()
        for cid, text in zip(ids, texts):
            h.update(cid.encode("utf-8", errors="ignore"))
            h.update(b"\0")
            h.update(text.encode("utf-8", errors="ignore"))
            h.update(b"\0")
        return h.hexdigest()

    def prepare(self, chunks: List[Dict[str, Any]]) -> None:
        """Build the BM25 index and encode all document chunks exactly once."""
        ids = [str(c.get("chunk_id") or c.get("unit_id") or f"chunk_{i}") for i, c in enumerate(chunks)]
        texts = [str(c.get("text") or "") for c in chunks]
        fingerprint = self._fingerprint(ids, texts)
        if fingerprint == self._prepared_fingerprint:
            return
        self._prepared_ids = ids
        self._prepared_texts = texts
        self._id_to_chunk = {cid: c for cid, c in zip(ids, chunks)}
        if self.exact_scoring:
            self._bm25 = _LegacyBM25Index([_legacy_tokenize(t) for t in texts], ids)
        else:
            self._bm25 = _BM25Index([_tokenize(t) for t in texts], ids)
        self._doc_embeddings = self._encode(texts, document_batch=True)
        self._prepared_fingerprint = fingerprint
        self._rerank_result_cache.clear()
        if self.strict_loading and self._doc_embeddings is None:
            raise RetrievalDependencyError("dense document embeddings are unavailable in strict parity mode")

    def preflight(self, require_reranker: bool = True) -> Dict[str, Any]:
        """Load and execute both retrieval models; raise on any strict failure."""
        probe = self._encode(
            ["corrosion inhibitor efficiency", "electrochemical impedance spectroscopy"],
            document_batch=False,
        )
        if probe is None:
            raise RetrievalDependencyError("embedding preflight returned no vectors")
        if require_reranker:
            rr = self._get_reranker()
            if rr is None:
                raise RetrievalDependencyError("reranker preflight could not load a model")
            backend, model = rr
            pairs = [["inhibition efficiency", "The inhibition efficiency was 91.2% by EIS."]]
            try:
                if backend in {"flagembedding", "siliconflow"}:
                    scores = model.compute_score(pairs)
                else:
                    scores = model.predict(pairs)
                if scores is None:
                    raise ValueError("reranker returned None")
                if hasattr(scores, "__len__") and not isinstance(scores, (str, bytes)):
                    if len(scores) != 1:
                        raise ValueError(f"unexpected reranker output: {scores!r}")
                    float(scores[0])
                else:
                    float(scores)
                self._last_rerank_error = None
            except Exception as exc:
                self._reranker_failure(exc)
        return self.diagnostics

    def retrieve(
        self,
        chunks: List[Dict[str, Any]],
        queries: List[str],
        top_k: int,
        bm25_weight: float = 0.25,
    ) -> List[Dict[str, Any]]:
        if not chunks or not queries:
            return []
        self.prepare(chunks)
        assert self._bm25 is not None

        vec_scores: Dict[str, float] = {}
        if self._doc_embeddings is not None:
            try:
                import numpy as np  # type: ignore

                q_emb = self._encode(queries, document_batch=False)
                if q_emb is not None:
                    sims = np.matmul(q_emb, self._doc_embeddings.T)
                    if self.exact_scoring:
                        # Old Chroma path returned only n_results=top_k per query.
                        for row in sims:
                            indices = np.argsort(-row, kind="stable")[: min(top_k, len(self._prepared_ids))]
                            for index in indices:
                                cid = self._prepared_ids[int(index)]
                                score = float(row[int(index)])
                                if cid not in vec_scores or score > vec_scores[cid]:
                                    vec_scores[cid] = score
                    else:
                        max_sims = sims.max(axis=0)
                        vec_scores = {cid: float(score) for cid, score in zip(self._prepared_ids, max_sims)}
            except Exception as exc:
                self._embedding_failure(exc)
                vec_scores = {}

        bm25_scores: Dict[str, float] = {}
        for query in queries:
            for cid, score in self._bm25.score(query).items():
                if cid not in bm25_scores or score > bm25_scores[cid]:
                    bm25_scores[cid] = score

        if not vec_scores:
            if self.strict_loading:
                raise RetrievalDependencyError("dense retrieval produced no scores in strict parity mode")
            bm25_weight = 1.0

        normalizer = _normalize_scores_legacy if self.exact_scoring else _normalize_scores
        vec_norm = normalizer(vec_scores)
        bm25_norm = normalizer(bm25_scores)
        all_ids = set(vec_norm) | set(bm25_norm)
        fused = {
            cid: (1.0 - bm25_weight) * vec_norm.get(cid, 0.0)
            + bm25_weight * bm25_norm.get(cid, 0.0)
            for cid in all_ids
        }
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
        out: List[Dict[str, Any]] = []
        for cid in ranked_ids:
            chunk = dict(self._id_to_chunk[cid])
            chunk.setdefault("chunk_id", cid)
            chunk.setdefault("unit_id", cid)
            chunk["retrieval_score"] = float(fused[cid])
            out.append(chunk)
        return out

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        ids = tuple(str(c.get("chunk_id") or c.get("unit_id") or i) for i, c in enumerate(chunks))
        cache_key = (query, ids, int(top_n))
        if cache_key in self._rerank_result_cache:
            return [dict(x) for x in self._rerank_result_cache[cache_key]]
        rr = self._get_reranker()
        if rr is None:
            return chunks[:top_n]
        backend, model = rr
        pairs = [[query, str(c.get("text") or "")] for c in chunks]
        try:
            if backend in {"flagembedding", "siliconflow"}:
                scores = model.compute_score(pairs)
            else:
                scores = model.predict(pairs)
            ranked: List[Dict[str, Any]] = []
            for chunk, score in zip(chunks, scores):
                item = dict(chunk)
                item["rerank_score"] = float(score)
                ranked.append(item)
            ranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            result = ranked[:top_n]
            self._last_rerank_error = None
            self._rerank_result_cache[cache_key] = [dict(x) for x in result]
            return result
        except Exception as exc:
            self._reranker_failure(exc)
            return chunks[:top_n]

    def release_index(self) -> None:
        """Release per-paper vectors while retaining loaded models for reuse."""
        self._prepared_fingerprint = None
        self._prepared_ids = []
        self._prepared_texts = []
        self._id_to_chunk = {}
        self._bm25 = None
        self._doc_embeddings = None
        self._rerank_result_cache.clear()

    @staticmethod
    def clear_model_cache() -> None:
        """Explicitly unload retrieval models, useful before loading 7B/14B."""
        LegacyHybridRetriever._encoder_cache.clear()
        LegacyHybridRetriever._reranker_cache.clear()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def expand_window(
        chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]],
        window_size: int = 1,
        max_window_candidates: int | None = None,
        score_decay: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """Return neighbour candidates without destroying rerank order.

        Direct hits remain in their fused-rerank order.  Neighbours are appended
        afterwards, ranked by parent hit rank and distance.  This prevents tables
        stored at the end of converted Markdown from being discarded merely
        because their original document index is large.
        """
        direct = [dict(chunk) for chunk in chunks]
        if window_size <= 0 or not direct:
            return direct
        idx_map = {
            str(c.get("chunk_id") or c.get("unit_id") or f"chunk_{i}"): i
            for i, c in enumerate(all_chunks)
        }
        direct_ids = {
            str(c.get("chunk_id") or c.get("unit_id") or "") for c in direct
        }
        neighbours: Dict[str, Dict[str, Any]] = {}
        for parent_rank, parent in enumerate(direct, start=1):
            cid = str(parent.get("chunk_id") or parent.get("unit_id") or "")
            index = idx_map.get(cid)
            if index is None:
                continue
            parent_score = float(
                parent.get("selection_score", parent.get("fused_rerank_score", parent.get("rerank_score", 0.0)))
                or 0.0
            )
            for j in range(max(0, index - window_size), min(len(all_chunks), index + window_size + 1)):
                candidate = all_chunks[j]
                nid = str(candidate.get("chunk_id") or candidate.get("unit_id") or "")
                if not nid or nid in direct_ids:
                    continue
                distance = abs(j - index)
                inherited = parent_score * (float(score_decay) ** max(1, distance))
                previous = neighbours.get(nid)
                if previous is not None and float(previous.get("selection_score", 0.0)) >= inherited:
                    continue
                item = dict(candidate)
                item.update({
                    "query_group": "window",
                    "query": "neighbour context from a fused-rerank direct hit",
                    "direct_hit": False,
                    "window_hit": True,
                    "parent_unit_id": cid,
                    "parent_rerank_rank": parent_rank,
                    "window_distance": distance,
                    "selection_score": inherited,
                })
                neighbours[nid] = item
        ordered_neighbours = sorted(
            neighbours.values(),
            key=lambda x: (
                int(x.get("parent_rerank_rank", 10**9)),
                int(x.get("window_distance", 10**9)),
                -float(x.get("selection_score", 0.0) or 0.0),
                int(x.get("chunk_index_in_doc", x.get("chunk_index", 10**9)) or 10**9),
            ),
        )
        if max_window_candidates is not None:
            ordered_neighbours = ordered_neighbours[: max(0, int(max_window_candidates))]
        return direct + ordered_neighbours


def _field_queries_from_config(
    fields_cfg: Dict[str, Any],
    include_groups: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    include = set(include_groups or ["performance", "conditions", "quantum", "adsorption"])
    configured = fields_cfg.get("field_queries") or {}
    queries = configured if isinstance(configured, Mapping) and configured else LEGACY_FIELD_QUERIES
    out: List[Tuple[str, str, str]] = []
    for field, query in queries.items():
        group = FIELD_TO_GROUP.get(field, "performance")
        if group in include:
            out.append((field, group, str(query)))
    if out:
        return out
    for group, obj in (fields_cfg.get("field_groups") or {}).items():
        if group not in include:
            continue
        for query in obj.get("queries", []) or []:
            out.append((group, group, str(query)))
    return out


def _chunk_to_evidence(
    chunk: Dict[str, Any],
    paper_id: str,
    group: str,
    query: str,
    rank: int,
) -> EvidenceUnit:
    unit_id = str(chunk.get("unit_id") or chunk.get("chunk_id") or f"{paper_id}#chunk_{rank}")
    return EvidenceUnit(
        paper_id=paper_id,
        unit_id=unit_id,
        unit_type=chunk.get("unit_type", "chunk"),
        text=str(chunk.get("text") or ""),
        query_group=group,
        query=query,
        rank=rank,
        score=float(chunk.get("rerank_score", chunk.get("retrieval_score", 0.0)) or 0.0),
        section=chunk.get("section") or chunk.get("section_title") or "",
        caption=chunk.get("caption", ""),
        columns=chunk.get("columns", {}) or {},
        metadata={k: v for k, v in chunk.items() if k not in {"text", "columns"}},
    )


def _split_for_config(
    text: str,
    paper_id: str,
    retrieval_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    return split_document(text, paper_id=paper_id, retrieval_cfg=retrieval_cfg)


def build_legacy_high_recall_evidence(
    paper_id: str,
    text: str,
    fields_cfg: Dict[str, Any],
    retrieval_cfg: Dict[str, Any],
    include_groups: Optional[List[str]] = None,
) -> List[EvidenceUnit]:
    """Build a high-recall candidate pool with field-wise rerank fusion.

    Selection and presentation are intentionally separated later by Q1/Q2:
    this function keeps up to ``rerank_top_n`` direct candidates in fused
    relevance order, then appends a bounded number of neighbour and table-extra
    candidates.  It never restores global document order before the final top-k
    selection.
    """
    params = dict(retrieval_cfg or {})
    backend = str(params.get("backend") or "legacy_exact").lower().strip()
    chunking_backend = resolve_chunking_backend(params)
    scoring_protocol = str(params.get("scoring_protocol") or "legacy_exact").lower().strip()
    field_top_k = int(params.get("field_retrieval_top_k", 20))
    retrieval_top_k = int(params.get("retrieval_top_k", 120))
    rerank_top_n = int(params.get("rerank_top_n", 80))
    direct_candidate_cap = int(params.get("retrieval_candidate_max_chunks", rerank_top_n))
    field_rerank_top_n = int(params.get("field_rerank_top_n", field_top_k))
    rrf_k = float(params.get("rerank_rrf_k", 60.0))
    window_size = int(params.get("window_size", 1))
    max_window_candidates = int(params.get("window_candidate_max_chunks", 16))
    window_score_decay = float(params.get("window_score_decay", 0.85))
    enable_reranker = _as_bool(params.get("enable_reranker"), True)
    bm25_weight = float(params.get("bm25_weight", 0.25))
    table_extra_chunks = int(params.get("table_extra_chunks", 12))
    max_chars = int(params.get("max_evidence_chars", 0))

    chunks = _split_for_config(text, paper_id, params)
    queries = _field_queries_from_config(fields_cfg, include_groups)
    if not chunks or not queries:
        return []

    retriever = LegacyHybridRetriever(params)
    retriever.prepare(chunks)
    if enable_reranker and retriever.strict_loading:
        retriever._get_reranker()

    # Stage 1: union of field-level hybrid retrieval results.
    seen: Dict[str, Dict[str, Any]] = {}
    retrieval_ranks: Dict[str, Dict[str, int]] = {}
    for field_name, group, query in queries:
        hits = retriever.retrieve(chunks, [query], top_k=field_top_k, bm25_weight=bm25_weight)
        for rank, hit in enumerate(hits, start=1):
            cid = str(hit.get("chunk_id") or hit.get("unit_id") or "")
            if not cid:
                continue
            retrieval_ranks.setdefault(cid, {})[field_name] = rank
            if cid not in seen or float(hit.get("retrieval_score", 0.0) or 0.0) > float(seen[cid].get("retrieval_score", 0.0) or 0.0):
                item = dict(hit)
                item.update({"field_name": field_name, "query_group": group, "query": query})
                seen[cid] = item

    hybrid_pool = sorted(seen.values(), key=lambda x: float(x.get("retrieval_score", 0.0) or 0.0), reverse=True)
    hybrid_pool = hybrid_pool[:retrieval_top_k]

    # Stage 2: rerank per field query and fuse ranks with RRF.  Raw scores from
    # different queries are not assumed to be directly calibrated.
    fused: Dict[str, Dict[str, Any]] = {}
    if enable_reranker and hybrid_pool:
        for field_name, group, query in queries:
            ranked = retriever.rerank(query, hybrid_pool, top_n=min(field_rerank_top_n, len(hybrid_pool)))
            for rank, chunk in enumerate(ranked, start=1):
                cid = str(chunk.get("chunk_id") or chunk.get("unit_id") or "")
                if not cid:
                    continue
                entry = fused.setdefault(cid, {
                    "chunk": dict(chunk),
                    "rrf_score": 0.0,
                    "best_raw_score": float("-inf"),
                    "best_rank": 10**9,
                    "best_group": group,
                    "best_query": query,
                    "best_field": field_name,
                    "query_hits": [],
                })
                raw_score = float(chunk.get("rerank_score", chunk.get("retrieval_score", 0.0)) or 0.0)
                entry["rrf_score"] += 1.0 / (rrf_k + rank)
                entry["query_hits"].append({
                    "field": field_name,
                    "group": group,
                    "rank": rank,
                    "score": raw_score,
                })
                if rank < entry["best_rank"] or (rank == entry["best_rank"] and raw_score > entry["best_raw_score"]):
                    entry.update({
                        "best_raw_score": raw_score,
                        "best_rank": rank,
                        "best_group": group,
                        "best_query": query,
                        "best_field": field_name,
                    })
        fused_rows = sorted(
            fused.values(),
            key=lambda e: (-float(e["rrf_score"]), int(e["best_rank"]), -float(e["best_raw_score"]), str(e["chunk"].get("chunk_id", ""))),
        )
        direct_hits: List[Dict[str, Any]] = []
        for fused_rank, entry in enumerate(fused_rows[: min(rerank_top_n, direct_candidate_cap)], start=1):
            item = dict(entry["chunk"])
            item.update({
                "query_group": entry["best_group"],
                "query": entry["best_query"],
                "field_name": entry["best_field"],
                "direct_hit": True,
                "window_hit": False,
                "fused_rerank_score": float(entry["rrf_score"]),
                "selection_score": float(entry["rrf_score"]),
                "best_rerank_score": float(entry["best_raw_score"]),
                "best_rerank_rank": int(entry["best_rank"]),
                "fused_rerank_rank": fused_rank,
                "query_hits": entry["query_hits"],
                "retrieval_ranks": retrieval_ranks.get(str(item.get("chunk_id") or item.get("unit_id") or ""), {}),
            })
            direct_hits.append(item)
    else:
        direct_hits = []
        for rank, chunk in enumerate(hybrid_pool[: min(rerank_top_n, direct_candidate_cap)], start=1):
            item = dict(chunk)
            score = float(item.get("retrieval_score", 0.0) or 0.0)
            item.update({
                "direct_hit": True,
                "window_hit": False,
                "selection_score": score,
                "fused_rerank_score": score,
                "fused_rerank_rank": rank,
            })
            direct_hits.append(item)

    # Stage 3: append bounded neighbours while keeping direct rerank order intact.
    candidate_hits = retriever.expand_window(
        direct_hits,
        chunks,
        window_size=window_size,
        max_window_candidates=max_window_candidates,
        score_decay=window_score_decay,
    )
    candidate_hits = [chunk for chunk in candidate_hits if not _JUNK_RE.fullmatch(str(chunk.get("text") or "").strip())]

    # Stage 4: append likely target tables omitted by retrieval.  These are only
    # candidates; Q2 validates and prioritises them, while Q1 usually selects its
    # fused-rerank direct hits first.
    retrieved_ids = {str(chunk.get("chunk_id") or chunk.get("unit_id")) for chunk in candidate_hits}
    target_table_re = re.compile(
        r"(?:inhibition\s+efficien|\bIE\s*%|weight\s+loss|potentiodynamic|polarization|\bEIS\b|"
        r"\bRct\b|\bIcorr\b|\bHOMO\b|\bLUMO\b|delta\s+g|Δ\s*G|adsorption\s+energy|\bEads\b)",
        re.I,
    )
    extra_tables: List[Dict[str, Any]] = []
    for chunk in chunks:
        cid = str(chunk.get("chunk_id") or chunk.get("unit_id"))
        if cid in retrieved_ids or not _is_table_chunk(chunk):
            continue
        item = dict(chunk)
        text_blob = "\n".join(str(item.get(k) or "") for k in ("caption", "section", "text"))
        target_bonus = 0.0001 if target_table_re.search(text_blob) else 0.0
        item.update({
            "query_group": "table_extra",
            "query": "table candidate omitted by hybrid retrieval",
            "direct_hit": False,
            "window_hit": False,
            "table_extra": True,
            "selection_score": target_bonus,
        })
        extra_tables.append(item)
    extra_tables.sort(key=lambda x: (-float(x.get("selection_score", 0.0)), int(x.get("chunk_index_in_doc", x.get("chunk_index", 10**9)) or 10**9)))
    candidate_hits.extend(extra_tables[: max(0, table_extra_chunks)])

    evidence: List[EvidenceUnit] = []
    total_chars = 0
    for rank, chunk in enumerate(candidate_hits, start=1):
        group = str(chunk.get("query_group") or "legacy")
        query = str(chunk.get("query") or "field-wise fused high-recall local RAG")
        ev = _chunk_to_evidence(chunk, paper_id, group, query, rank)
        if max_chars > 0 and total_chars + len(ev.text) > max_chars:
            continue
        evidence.append(ev)
        total_chars += len(ev.text)

    diagnostics = retriever.diagnostics
    retriever.release_index()
    if evidence:
        evidence[0].metadata.setdefault("legacy_retrieval_diagnostics", diagnostics)
        evidence[0].metadata.setdefault("legacy_retrieval_params", {
            "backend": backend,
            "chunking_backend": chunking_backend,
            "scoring_protocol": scoring_protocol,
            "field_retrieval_top_k": field_top_k,
            "retrieval_top_k": retrieval_top_k,
            "field_rerank_top_n": field_rerank_top_n,
            "rerank_top_n": rerank_top_n,
            "rerank_rrf_k": rrf_k,
            "retrieval_candidate_max_chunks": direct_candidate_cap,
            "window_candidate_max_chunks": max_window_candidates,
            "max_final_chunks": int(params.get("max_final_chunks", 40)),
            "window_size": window_size,
            "bm25_weight": bm25_weight,
            "table_extra_chunks": table_extra_chunks,
            "direct_candidate_count": len(direct_hits),
            "total_candidate_count": len(evidence),
        })
    return evidence
