#!/usr/bin/env bash
# Launch OpenWebUI locally and point its native OpenAI connector at the agent.
#
# This demo never uses OpenWebUI's own RAG/embeddings or speech-to-text — the
# agent does everything — so we run fully offline to avoid pulling models from
# HuggingFace on first boot.
#
# The agent (`just serve-agent`) exposes an OpenAI-compatible API, so there is no
# plugin to install: OpenWebUI talks to it directly. See openwebui/README.md.
set -euo pipefail

# Resolve the repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OPEN_WEBUI_VERSION="${OPEN_WEBUI_VERSION:-v0.9.6}"
PORT="${PORT:-9000}"
AGENT_BASE_URL="${AGENT_BASE_URL:-http://localhost:8000/v1}"

export WEBUI_AUTH=False
export DATA_DIR="$REPO_ROOT/work_dir/open-webui"

# Connect to the agent's OpenAI-compatible API; disable the Ollama connector.
export ENABLE_OPENAI_API=True
export ENABLE_OLLAMA_API=False
export OPENAI_API_BASE_URL="$AGENT_BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-no-key-needed}"

# Fully offline: no embedding/reranking/Whisper model downloads from HuggingFace.
export OFFLINE_MODE=true
export HF_HUB_OFFLINE=1
export RAG_EMBEDDING_MODEL_AUTO_UPDATE=false
export RAG_RERANKING_MODEL_AUTO_UPDATE=false
export WHISPER_MODEL_AUTO_UPDATE=false

exec uvx --python 3.11 \
  --from "git+https://github.com/open-webui/open-webui@${OPEN_WEBUI_VERSION}" \
  open-webui serve --port "$PORT"
