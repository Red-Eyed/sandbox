# Connecting OpenWebUI to the agent

The agent ([`../src/pydantic_agent_server/`](../src/pydantic_agent_server/)) exposes
an **OpenAI-compatible API**, so OpenWebUI talks to it through its built-in OpenAI
connector — **there is no plugin or pipe to install**.

[`serve.sh`](serve.sh) launches OpenWebUI already wired to the agent:

- `OPENAI_API_BASE_URL=http://localhost:8000/v1` — points at the agent.
- `OPENAI_API_KEY=sk-no-key-needed` — the agent ignores it, but OpenWebUI wants a value.
- `ENABLE_OLLAMA_API=False` — only the agent shows up as a model.
- `OFFLINE_MODE=true` (+ HF offline flags) — no embedding/Whisper downloads from
  HuggingFace; this demo uses none of OpenWebUI's RAG/STT features.

## Use it

1. **Start the agent:** `just serve-agent` (OpenAI API on `http://localhost:8000/v1`).
2. **Start OpenWebUI:** `just run-openwebui` (UI on `http://localhost:9000`).
3. Open `http://localhost:9000`, start a new chat, and pick the model
   (`google:gemini-2.5-flash`) — it appears automatically via `/v1/models`.

### Override the connection

The script reads these env vars if set:

- `AGENT_BASE_URL` — agent OpenAI base URL (default `http://localhost:8000/v1`).
- `PORT` — OpenWebUI port (default `9000`).
- `OPEN_WEBUI_VERSION` — pinned OpenWebUI version (default `v0.9.6`).

## Try images

- **Input:** attach an image to a message and ask *"what's in this image?"* —
  OpenWebUI sends it as OpenAI multimodal content, which the agent (Gemini) reads.
- **Output:** ask *"generate an image of a sunset over mountains"* — the agent
  calls its `generate_image` tool, which writes a PNG served from the agent's
  `/images` route, and returns Markdown that OpenWebUI renders inline.

> Output images are delivered as a Markdown image URL pointing back at the agent
> (`public_base_url`, default `http://localhost:8000`). The browser must be able to
> reach that URL — fine for a local host setup; adjust `PUBLIC_BASE_URL` if the
> agent is elsewhere.
