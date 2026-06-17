# pydantic-ai ↔ OpenWebUI (OpenAI-compatible)

A minimal, complete example of talking to a [pydantic-ai](https://ai.pydantic.dev)
agent from [OpenWebUI](https://openwebui.com), with **image input and output**.

The agent is a standalone service that exposes an **OpenAI-compatible API**
(`/v1/chat/completions`, `/v1/models`). OpenWebUI connects through its built-in
OpenAI connector — no plugin, no pipe, no extra configuration in the UI.

```
┌──────────────┐   OpenAI /v1/chat/completions (SSE)   ┌──────────────────────────┐
│  OpenWebUI    │  ───────────────────────────────────▶ │  pydantic-ai agent       │
│  :9000        │                                        │  FastAPI + Agent         │
│  (native      │  ◀───────────────────────────────────  │  :8000  (/v1, /images)   │
│   OpenAI)     │            text + Markdown              └──────────────────────────┘
└──────────────┘
```

## Why OpenAI-compatible (and not AG-UI / a pipe)

- OpenWebUI's OpenAI connector is **first-class and zero-config**: set the base
  URL and the model appears automatically. No pasting a Pipe Function, no valves.
- **Image input** uses the OpenAI multimodal format OpenWebUI already speaks
  (`content: [{type: image_url, image_url: {url: "data:..."}}]`), mapped straight to
  pydantic-ai `BinaryContent`/`ImageUrl`.
- No third-party bridge to vendor and patch.

The cost is owning the `/v1` surface (pydantic-ai has no built-in OpenAI-compatible
server). It's kept small and typed: pydantic request/response models, a pure
OpenAI↔pydantic-ai message mapper, and `agent.run_stream`.

## Layout

The OpenAI API is a **reusable component** you hand an `Agent`; everything
demo-specific (model choice, tools, image hosting) is glue in `main.py`.

```
src/pydantic_agent_server/
  openai_server/        # reusable: serve any pydantic-ai Agent over OpenAI's API
    app.py              #   create_openai_app(agent, model_id) / serve(...) — the shell
    schema.py           #   typed OpenAI request/response models + SSE builders
    messages.py         #   pure: OpenAI messages -> pydantic-ai content (incl. images)
    README.md           #   component docs (core/shell split, usage, contract)
  main.py               # glue: build demo agent, create_openai_app, mount /images, /health
  tools.py              # get_current_datetime + generate_image tool factory
  images.py             # pure: render a placeholder PNG
  config.py             # pydantic-settings (model, host/port, public_base_url, images_dir)
openwebui/
  serve.sh              # launch OpenWebUI wired to the agent (offline, no HF downloads)
  README.md             # connection details + image usage
justfile                # serve-agent / run-openwebui / check
```

Reuse it elsewhere in one line:

```python
from pydantic_agent_server.openai_server import create_openai_app
app = create_openai_app(my_agent, model_id="google:gemini-2.5-flash")
```

## Run it

Prerequisites: [`uv`](https://docs.astral.sh/uv/), [`just`](https://github.com/casey/just),
and a Gemini API key (the default model is `google:gemini-2.5-flash`).

1. **Configure secrets**

   ```sh
   cp .env.example .env
   # edit .env and set GOOGLE_API_KEY
   ```

2. **Start the agent** (terminal 1)

   ```sh
   just serve-agent          # OpenAI API on http://localhost:8000/v1
   ```

   Verify: `curl http://localhost:8000/health` and `curl http://localhost:8000/v1/models`.

3. **Start OpenWebUI** (terminal 2)

   ```sh
   just run-openwebui        # http://localhost:9000, pre-wired to the agent
   ```

4. **Chat** — open `http://localhost:9000`, pick the `google:gemini-2.5-flash`
   model (auto-discovered), and try:
   - **Text:** *"What's the current date and time?"* (calls a tool)
   - **Image input:** attach a photo and ask *"what's in this image?"*
   - **Image output:** *"generate an image of a sunset over mountains"*

## Switching the model

The default is Gemini. Override `AGENT_MODEL` in `.env` for any
[pydantic-ai model string](https://ai.pydantic.dev/models/), e.g.:

```sh
AGENT_MODEL=openai:gpt-4o-mini   # then set OPENAI_API_KEY instead of GOOGLE_API_KEY
```

For image input/output the model must be multimodal (Gemini and gpt-4o are).

## Develop

```sh
just check    # ruff format + ruff check --fix + pyrefly (src)
```
