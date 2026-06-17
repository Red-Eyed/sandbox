"""OpenAI-compatible server for any pydantic-ai `Agent`.

Hand it an agent and serve it; no host-application knowledge inside. See README.md.
"""

from .app import create_openai_app, register_openai_routes, serve

__all__ = ["create_openai_app", "register_openai_routes", "serve"]
