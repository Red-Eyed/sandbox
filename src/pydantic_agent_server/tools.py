"""Agent tools.

Kept separate from the agent wiring so each tool is independently testable. The
image tool needs runtime config (where to save files, what URL to expose them at),
so it is produced by a factory that captures that config.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic_agent_server.images import render_text_png


def get_current_datetime(format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return the current local date and time formatted with `format_string`.

    `format_string` is a standard `strftime` pattern (e.g. "%B %d, %Y" for
    "January 04, 2026"). Defaults to "YYYY-MM-DD HH:MM:SS".
    """
    return datetime.now().strftime(format_string)


def make_generate_image_tool(images_dir: Path, public_base_url: str) -> Callable[[str], str]:
    """Build the `generate_image` tool bound to where images are saved and served."""

    def generate_image(description: str) -> str:
        """Generate an image from `description` and return Markdown that renders it.

        Call this whenever the user asks to create, draw, or generate an image.
        Include the returned Markdown verbatim in your reply so the image renders
        inline for the user.
        """
        name = f"{uuid.uuid4().hex}.png"
        render_text_png(description, images_dir / name)
        url = f"{public_base_url.rstrip('/')}/images/{name}"
        return f"![{description}]({url})"

    return generate_image
