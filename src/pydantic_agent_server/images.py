"""Pure image rendering helper.

Generates a simple placeholder PNG with wrapped text drawn on it. Kept free of any
web/agent concerns so it is independently usable and testable; callers decide
where to save the file and how to expose it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

_SIZE = (640, 360)
_BACKGROUND = (28, 30, 38)
_FOREGROUND = (235, 235, 240)
_MARGIN = 24
_WRAP_COLUMNS = 48


def render_text_png(text: str, path: Path) -> Path:
    """Render `text` onto a placeholder PNG saved at `path`; return `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", _SIZE, _BACKGROUND)
    draw = ImageDraw.Draw(image)
    wrapped = textwrap.fill(text, width=_WRAP_COLUMNS)
    draw.multiline_text((_MARGIN, _MARGIN), wrapped, fill=_FOREGROUND, spacing=6)

    image.save(path, format="PNG")
    return path
