from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List


class InputTokenBudgetError(RuntimeError):
    """Raised when a strategy submits a chat prompt above the frozen input budget."""


@dataclass
class GenerationConfig:
    max_new_tokens: int = 4096
    temperature: float = 0.0
    top_p: float | None = None
    seed: int = 1
    max_input_tokens: int | None = None
    # Explicit protocol switch. The frozen local baseline is greedy.
    do_sample: bool = False
    # Stop deterministic local generation when compact JSON rows enter an
    # exact repetition loop. This does not suppress a single duplicate; it
    # requires a repeated suffix cycle and then keeps one copy of the cycle.
    json_repeat_guard: bool = True
    json_repeat_guard_min_rows: int = 6
    json_repeat_guard_max_cycle: int = 4
    json_repeat_guard_check_interval: int = 16



def _scan_complete_compact_rows(text: str) -> List[List[Any]]:
    """Return complete scalar JSON rows found inside an incomplete outer array."""
    rows: List[List[Any]] = []
    s = text or ""
    outer = s.find("[")
    if outer < 0:
        return rows
    depth = 0
    in_string = False
    escaped = False
    row_start: int | None = None
    for idx in range(outer, len(s)):
        ch = s[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
            if depth == 2:
                row_start = idx
            continue
        if ch == "]":
            if depth == 2 and row_start is not None:
                piece = s[row_start:idx + 1]
                try:
                    value = json.loads(piece)
                except Exception:
                    value = None
                if (
                    isinstance(value, list)
                    and 9 <= len(value) <= 13
                    and not any(isinstance(x, (dict, list)) for x in value)
                ):
                    rows.append(value)
                row_start = None
            depth -= 1
            if depth <= 0:
                break
    return rows


def _repeat_cycle_trim(rows: List[List[Any]], min_rows: int, max_cycle: int) -> tuple[List[List[Any]], int] | None:
    """Detect an exact repeated suffix cycle and return one-cycle-trimmed rows."""
    n = len(rows)
    if n < max(1, int(min_rows)):
        return None
    keys = [json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows]
    upper = min(max(1, int(max_cycle)), n // 2)
    for cycle in range(1, upper + 1):
        if cycle == 1:
            if n >= 3 and keys[-1] == keys[-2] == keys[-3]:
                return rows[:-2], 1
            continue
        if keys[-cycle:] == keys[-2 * cycle:-cycle]:
            return rows[:-cycle], cycle
    return None


def _pad_compact_rows(rows: List[List[Any]]) -> List[List[Any]]:
    return [list(row) + [None] * (13 - len(row)) for row in rows if 9 <= len(row) <= 13]


class LLMClient:
    """Backend-neutral generation and token-accounting interface."""

    token_count_is_exact: bool = False

    def __init__(self) -> None:
        self.last_generation_metadata: Dict[str, Any] = {}

    def generate(self, system: str, user: str, gen: GenerationConfig) -> str:
        raise NotImplementedError

    def count_text_tokens(self, text: str) -> int:
        # Conservative fallback for remote clients without a public tokenizer.
        return max(0, (len(text or "") + 2) // 3)

    def count_chat_tokens(self, system: str, user: str) -> int:
        return self.count_text_tokens((system or "") + "\n" + (user or "")) + 16

    def token_counter_metadata(self) -> Dict[str, Any]:
        return {
            "token_counter_exact": bool(self.token_count_is_exact),
            "token_counter_backend": self.__class__.__name__,
        }


class TransformersClient(LLMClient):
    """Lazy local HuggingFace client with exact tokenizer-level accounting."""

    token_count_is_exact = True
    _tokenizer_cache: Dict[str, Any] = {}
    _model_cache: Dict[str, Any] = {}

    def __init__(self, model_path: str, model_id: str = "", device_map: str = "auto") -> None:
        super().__init__()
        self.model_path = model_path or model_id
        self.model_id = model_id or model_path
        self.device_map = device_map

    def _load_tokenizer(self):
        if self.model_path in self._tokenizer_cache:
            return self._tokenizer_cache[self.model_path]
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._tokenizer_cache[self.model_path] = tokenizer
        return tokenizer

    def _load_model(self):
        if self.model_path in self._model_cache:
            return self._model_cache[self.model_path]
        from transformers import AutoModelForCausalLM  # type: ignore

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
        )
        model.eval()
        self._model_cache[self.model_path] = model
        return model

    def _format_chat(self, system: str, user: str) -> str:
        tokenizer = self._load_tokenizer()
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_text_tokens(self, text: str) -> int:
        tokenizer = self._load_tokenizer()
        return len(tokenizer.encode(text or "", add_special_tokens=False))

    def count_chat_tokens(self, system: str, user: str) -> int:
        tokenizer = self._load_tokenizer()
        text = self._format_chat(system, user)
        encoded = tokenizer([text], truncation=False)
        return len(encoded.get("input_ids", [[]])[0])

    @classmethod
    def clear_cache(cls) -> None:
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        try:
            import gc
            import torch  # type: ignore

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def generate(self, system: str, user: str, gen: GenerationConfig) -> str:
        import torch  # type: ignore

        if gen.seed is not None:
            random.seed(gen.seed)
            torch.manual_seed(int(gen.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(gen.seed))

        tokenizer = self._load_tokenizer()
        model = self._load_model()
        text = self._format_chat(system, user)
        inputs = tokenizer([text], return_tensors="pt", truncation=False).to(model.device)
        input_tokens = int(inputs.input_ids.shape[-1])
        max_input = int(gen.max_input_tokens or 0)
        if max_input > 0 and input_tokens > max_input:
            self.last_generation_metadata = {
                "input_tokens_before_truncation": input_tokens,
                "input_tokens_after_truncation": input_tokens,
                "truncated": False,
                "input_over_budget": True,
                "retained_fraction": 1.0,
                "max_input_tokens": max_input,
                **self.token_counter_metadata(),
            }
            raise InputTokenBudgetError(
                f"strategy submitted {input_tokens} input tokens, exceeding max_input_tokens={max_input}; "
                "silent whole-prompt truncation is disabled"
            )

        do_sample = bool(gen.do_sample)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(gen.max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(gen.temperature) if gen.temperature > 0 else 1.0
            if gen.top_p is not None:
                generate_kwargs["top_p"] = float(gen.top_p)

        repeat_guard = None
        if bool(gen.json_repeat_guard):
            from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

            class _JsonRepeatGuard(StoppingCriteria):
                def __init__(self) -> None:
                    self.prompt_tokens = input_tokens
                    self.last_checked_tokens = 0
                    self.triggered = False
                    self.safe_rows: List[List[Any]] = []
                    self.cycle_length = 0
                    self.rows_before_trim = 0

                def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
                    generated = int(input_ids.shape[-1]) - self.prompt_tokens
                    interval = max(1, int(gen.json_repeat_guard_check_interval))
                    if generated - self.last_checked_tokens < interval:
                        return False
                    self.last_checked_tokens = generated
                    partial = tokenizer.decode(
                        input_ids[0][self.prompt_tokens:],
                        skip_special_tokens=True,
                    )
                    rows = _scan_complete_compact_rows(partial)
                    detected = _repeat_cycle_trim(
                        rows,
                        min_rows=int(gen.json_repeat_guard_min_rows),
                        max_cycle=int(gen.json_repeat_guard_max_cycle),
                    )
                    if detected is None:
                        return False
                    safe_rows, cycle = detected
                    self.triggered = True
                    self.safe_rows = _pad_compact_rows(safe_rows)
                    self.cycle_length = int(cycle)
                    self.rows_before_trim = len(rows)
                    return True

            repeat_guard = _JsonRepeatGuard()
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([repeat_guard])

        with torch.no_grad():
            out = model.generate(**inputs, **generate_kwargs)
        new_tokens = out[0][inputs.input_ids.shape[-1]:]
        output_tokens = int(new_tokens.shape[-1])

        eos_candidates: set[int] = set()
        for value in [tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)]:
            if isinstance(value, (list, tuple, set)):
                eos_candidates.update(int(v) for v in value if v is not None)
            elif value is not None:
                eos_candidates.add(int(value))
        ended_with_eos = bool(output_tokens and int(new_tokens[-1].item()) in eos_candidates)
        repeat_guard_triggered = bool(repeat_guard is not None and repeat_guard.triggered)
        reached_max = bool(
            output_tokens >= int(gen.max_new_tokens)
            and not ended_with_eos
            and not repeat_guard_triggered
        )
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if repeat_guard_triggered:
            # The guard stops only after a complete repeated compact-row cycle.
            # Reconstruct one valid outer array, retain one copy of the cycle,
            # and pad any omitted trailing null fields to the 13-column schema.
            decoded = json.dumps(
                repeat_guard.safe_rows,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        stripped = decoded.strip()
        json_closed = bool(
            (stripped.startswith("[") and stripped.endswith("]"))
            or (stripped.startswith("{") and stripped.endswith("}"))
        )

        self.last_generation_metadata = {
            "input_tokens_before_truncation": input_tokens,
            "input_tokens_after_truncation": input_tokens,
            "truncated": False,
            "input_over_budget": False,
            "retained_fraction": 1.0,
            "max_input_tokens": gen.max_input_tokens,
            "generated_tokens": output_tokens,
            "max_new_tokens": int(gen.max_new_tokens),
            "ended_with_eos": ended_with_eos,
            "reached_max_new_tokens": reached_max,
            "finish_reason": (
                "repeat_guard"
                if repeat_guard_triggered
                else ("length" if reached_max else ("eos" if ended_with_eos else "stopped"))
            ),
            "json_closed_by_outer_delimiter": json_closed,
            "json_repeat_guard_enabled": bool(gen.json_repeat_guard),
            "json_repeat_guard_triggered": repeat_guard_triggered,
            "json_repeat_guard_cycle_length": (
                int(repeat_guard.cycle_length) if repeat_guard_triggered else 0
            ),
            "json_repeat_guard_rows_before_trim": (
                int(repeat_guard.rows_before_trim) if repeat_guard_triggered else 0
            ),
            "json_repeat_guard_rows_after_trim": (
                len(repeat_guard.safe_rows) if repeat_guard_triggered else 0
            ),
            **self.token_counter_metadata(),
        }
        return decoded


class OpenAICompatibleClient(LLMClient):
    def __init__(self, model_id: str, base_url: str, api_key: str) -> None:
        super().__init__()
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, system: str, user: str, gen: GenerationConfig) -> str:
        from openai import OpenAI  # type: ignore

        estimated_tokens = self.count_chat_tokens(system, user)
        max_input = int(gen.max_input_tokens or 0)
        if max_input > 0 and estimated_tokens > max_input:
            raise InputTokenBudgetError(
                f"estimated remote input tokens {estimated_tokens} exceed max_input_tokens={max_input}"
            )

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        effective_temperature = float(gen.temperature) if gen.do_sample else 0.0
        resp = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=effective_temperature,
            **({"top_p": float(gen.top_p)} if (gen.do_sample and gen.top_p is not None) else {}),
            max_tokens=int(gen.max_new_tokens),
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        finish_reason = getattr(choice, "finish_reason", None)
        stripped = raw.strip()
        self.last_generation_metadata = {
            "input_tokens_before_truncation": prompt_tokens or estimated_tokens,
            "input_tokens_after_truncation": prompt_tokens or estimated_tokens,
            "truncated": False,
            "input_over_budget": False,
            "retained_fraction": 1.0,
            "max_input_tokens": gen.max_input_tokens,
            "generated_tokens": completion_tokens,
            "max_new_tokens": int(gen.max_new_tokens),
            "finish_reason": finish_reason,
            "reached_max_new_tokens": finish_reason == "length",
            "ended_with_eos": finish_reason == "stop",
            "json_closed_by_outer_delimiter": bool(
                (stripped.startswith("[") and stripped.endswith("]"))
                or (stripped.startswith("{") and stripped.endswith("}"))
            ),
            **self.token_counter_metadata(),
        }
        return raw


class DryRunClient(LLMClient):
    def count_text_tokens(self, text: str) -> int:
        return len((text or "").split())

    def count_chat_tokens(self, system: str, user: str) -> int:
        return self.count_text_tokens(system) + self.count_text_tokens(user) + 8

    def generate(self, system: str, user: str, gen: GenerationConfig) -> str:
        tokens = self.count_chat_tokens(system, user)
        if gen.max_input_tokens and tokens > int(gen.max_input_tokens):
            raise InputTokenBudgetError(f"dry-run prompt over budget: {tokens} > {gen.max_input_tokens}")
        self.last_generation_metadata = {
            "input_tokens_before_truncation": tokens,
            "input_tokens_after_truncation": tokens,
            "truncated": False,
            "input_over_budget": False,
            "retained_fraction": 1.0,
            "max_input_tokens": gen.max_input_tokens,
            "generated_tokens": 1,
            "max_new_tokens": int(gen.max_new_tokens),
            "ended_with_eos": True,
            "reached_max_new_tokens": False,
            "finish_reason": "dry_run",
            "json_closed_by_outer_delimiter": True,
            "dry_run": True,
            **self.token_counter_metadata(),
        }
        return '{"paper_id":"dry_run","records":[]}'


def build_client(model_cfg: Dict[str, Any], dry_run: bool = False) -> LLMClient:
    if dry_run:
        return DryRunClient()
    backend = model_cfg.get("backend", "transformers")
    if backend == "transformers":
        path = model_cfg.get("resolved_path") or model_cfg.get("model_id")
        return TransformersClient(model_path=path, model_id=model_cfg.get("model_id", ""))
    if backend == "openai_compatible":
        base_url = os.getenv(model_cfg.get("base_url_env", ""), model_cfg.get("base_url", ""))
        api_key = os.getenv(model_cfg.get("api_key_env", ""), model_cfg.get("api_key", ""))
        return OpenAICompatibleClient(model_id=model_cfg.get("model_id"), base_url=base_url, api_key=api_key)
    raise ValueError(f"Unsupported backend: {backend}")


def clear_local_model_cache() -> None:
    TransformersClient.clear_cache()


def generation_from_model_cfg(model_cfg: Dict[str, Any], seed: int) -> GenerationConfig:
    g = dict(model_cfg.get("generation", {}) or {})
    temp = os.getenv("GEN_TEMPERATURE")
    top_p = os.getenv("GEN_TOP_P")
    max_new = os.getenv("GEN_MAX_NEW_TOKENS")
    max_in = os.getenv("GEN_MAX_INPUT_TOKENS")
    sample_env = os.getenv("GEN_DO_SAMPLE")

    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    effective_temperature = float(temp if temp is not None else g.get("temperature", 0.0))
    registry_default_sample = (
        _as_bool(g.get("do_sample"), False)
        if "do_sample" in g
        else effective_temperature > 0
    )
    do_sample = _as_bool(sample_env, registry_default_sample)
    return GenerationConfig(
        max_new_tokens=int(max_new or g.get("max_new_tokens", 4096)),
        temperature=effective_temperature,
        top_p=(float(top_p) if top_p is not None else (float(g["top_p"]) if g.get("top_p") is not None else None)),
        seed=int(seed),
        max_input_tokens=(int(max_in) if max_in is not None else (int(g["max_input_tokens"]) if g.get("max_input_tokens") is not None else None)),
        do_sample=do_sample,
        json_repeat_guard=_as_bool(g.get("json_repeat_guard"), True),
        json_repeat_guard_min_rows=int(g.get("json_repeat_guard_min_rows", 6)),
        json_repeat_guard_max_cycle=int(g.get("json_repeat_guard_max_cycle", 4)),
        json_repeat_guard_check_interval=int(g.get("json_repeat_guard_check_interval", 16)),
    )
