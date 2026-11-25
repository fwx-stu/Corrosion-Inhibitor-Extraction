"""
Shared utility functions used across the experiment framework.
"""
from pathlib import Path


def load_prompt(path) -> str:
    """Read a prompt template file as UTF-8 text."""
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs) -> str:
    """Replace {placeholder} tokens with keyword argument values."""
    for k, v in kwargs.items():
        template = template.replace("{" + k + "}", str(v))
    return template
