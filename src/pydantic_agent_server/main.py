"""Demo wiring: build the agent and serve it over the OpenAI-compatible API.

This is the imperative shell / glue. All the OpenAI protocol work lives in the
reusable `openai_server` component; here we only:

- build a demo `Agent` (Gemini by default) with a datetime tool and an
  image-generation tool,
- hand it to `create_openai_app`,
- mount the demo-only `/images` route that serves generated images.

Image input is handled entirely by the component (OpenAI multimodal content maps
to pydantic-ai content). Image output is demo glue: the `generate_image` tool
writes a PNG served from `/images`, and the agent embeds it as Markdown.
"""

from __future__ import annotations

from fastapi.staticfiles import StaticFiles
from pydantic_ai import Agent

from pydantic_agent_server.config import Settings
from pydantic_agent_server.openai_server import create_openai_app
from pydantic_agent_server.tools import get_current_datetime, make_generate_image_tool

INSTRUCTIONS = (
    "You are a concise, helpful assistant.\n"
    "- For the current date or time, call the get_current_datetime tool.\n"
    "- To create, draw, or generate an image, call the generate_image tool and "
    "include its returned Markdown verbatim in your reply so it renders inline.\n"
    "- When the user sends an image, look at it and answer their question about it."
)


def build_agent(settings: Settings) -> Agent:
    """Construct the demo agent with a datetime tool and an image-generation tool."""
    generate_image = make_generate_image_tool(settings.images_dir, settings.public_base_url)
    return Agent(model=settings.agent_model, instructions=INSTRUCTIONS, tools=[get_current_datetime, generate_image])


settings = Settings()
settings.images_dir.mkdir(parents=True, exist_ok=True)

agent = build_agent(settings)
app = create_openai_app(agent, model_id=settings.agent_model)
app.mount("/images", StaticFiles(directory=settings.images_dir), name="images")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.agent_model}
