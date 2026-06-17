# Run the pydantic-ai AG-UI agent. OpenWebUI's bridge pipe POSTs to this service.
# Binds 127.0.0.1 for a host-local OpenWebUI (run via `just run-openwebui`).
# If OpenWebUI runs in Docker instead, bind 0.0.0.0 and point the pipe valve at
# http://host.docker.internal:8000 (see README).
serve-agent:
    uv run --env-file=.env uvicorn pydantic_agent_server.main:app --host 127.0.0.1 --port 8000

# Run OpenWebUI locally on :9000 (offline; no HuggingFace downloads). After it
# starts, install the bridge pipe and set its endpoint — see openwebui/README.md.
run-openwebui:
    ./openwebui/serve.sh

# Format, lint, and type-check. pyrefly is scoped to src/ because openwebui/pipe.py
# is vendored and imports OpenWebUI internals that only exist inside OpenWebUI.
check:
    uv run ruff format .
    uv run ruff check --fix .
    uv run pyrefly check src
