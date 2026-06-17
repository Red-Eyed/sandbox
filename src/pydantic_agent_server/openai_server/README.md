# openai_server

Serve any [pydantic-ai](https://ai.pydantic.dev) `Agent` over an
**OpenAI-compatible API** (`/v1/chat/completions` + `/v1/models`), so OpenAI
clients — OpenWebUI, the `openai` SDK, LibreChat, etc. — can talk to it directly.

The component is **standalone**: it knows the OpenAI wire format and pydantic-ai,
and nothing about any host application (no model choice, no tools, no file
hosting). All of that is the caller's job.

## Usage

```python
from pydantic_ai import Agent
from pydantic_agent_server.openai_server import create_openai_app

agent = Agent("google:gemini-2.5-flash", instructions="Be helpful.")

# An ASGI app you can serve with uvicorn:
app = create_openai_app(agent, model_id="google:gemini-2.5-flash")
```

```sh
uvicorn my_module:app --host 127.0.0.1 --port 8000
```

Or run it directly (imperative-shell convenience):

```python
from pydantic_agent_server.openai_server import serve

serve(agent, model_id="google:gemini-2.5-flash", host="127.0.0.1", port=8000)
```

To add your own routes (health check, static files, ...), either mount onto the
returned app, or build your own and use the composition variant:

```python
from fastapi import FastAPI
from pydantic_agent_server.openai_server import register_openai_routes

app = FastAPI()
register_openai_routes(app, agent, model_id="...")
app.mount("/images", ...)
```

## Design (functional core / imperative shell)

| Module | Role | Side effects |
|--------|------|--------------|
| [`schema.py`](schema.py) | Typed OpenAI request/response models + SSE builders | none |
| [`messages.py`](messages.py) | Pure mapping: OpenAI messages → pydantic-ai content (text + images) | none |
| [`app.py`](app.py) | The shell: FastAPI routes, streaming over `agent.run_stream` | HTTP, runs the agent |

- **Image input** is handled here: OpenAI multimodal `content` parts map to
  pydantic-ai `BinaryContent` (for `data:` URIs) / `ImageUrl`. A multimodal agent
  (Gemini, gpt-4o) then sees the image.
- **Image output** is *not* this component's concern — the OpenAI stream is text.
  A caller that wants to return images can have a tool emit a Markdown image URL
  (see the demo's `generate_image` tool + `/images` mount in `main.py`).

## Contract

- `create_openai_app(agent, *, model_id) -> FastAPI`
- `register_openai_routes(app, agent, *, model_id) -> None`
- `serve(agent, *, model_id, host="127.0.0.1", port=8000) -> None`

`model_id` is what clients see in `/v1/models` and select; requests' own `model`
field is ignored (the given `agent` always handles the request).
