from __future__ import annotations

from pathlib import Path
from typing import Dict

from .io_utils import read_text


def load_prompt(prompt_variant: str, prompts_dir: Path) -> str:
    path = prompts_dir / f"{prompt_variant}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return read_text(path)


def _remove_one_leading_newline(text: str) -> str:
    if text.startswith("\r\n"):
        return text[2:]
    if text.startswith("\n"):
        return text[1:]
    return text


def _remove_one_trailing_newline(text: str) -> str:
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text


def render_prompt(template: str, **kwargs: str) -> Dict[str, str]:
    """Return chat messages while preserving prompt-significant newlines.

    The legacy Q1 constants ended with one newline.  ``str.strip()`` removed it
    and made the supposedly exact prompt one character different.  We now
    remove only the marker separators and preserve the prompt body verbatim.
    """
    text = template
    for k, v in kwargs.items():
        text = text.replace("{" + k + "}", v or "")
    if "USER:" not in text:
        return {"system": "", "user": text}

    system_part, user_part = text.split("USER:", 1)
    if system_part.startswith("SYSTEM:"):
        system_part = system_part[len("SYSTEM:"):]
    system_part = _remove_one_leading_newline(system_part)
    # The blank line separating the system body from the USER marker contributes
    # two newlines. Remove one separator newline, leaving the original body newline.
    system_part = _remove_one_trailing_newline(system_part)
    user_part = _remove_one_leading_newline(user_part)
    return {"system": system_part, "user": user_part}
