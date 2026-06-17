"""Serve any pydantic-ai `Agent` over an OpenAI-compatible API.

This is the reusable component: hand it an `Agent` and a `model_id`, get back a
FastAPI app exposing `GET /v1/models` and `POST /v1/chat/completions` (streaming
and non-streaming). It carries no host-application knowledge — no model choice, no
tools, no file hosting — so it drops into any project. All such glue lives in the
caller (see `main.py`).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, UserContent

from .messages import split_history_and_prompt
from .schema import (
    SSE_DONE,
    ChatCompletionRequest,
    completion_response,
    content_chunk,
    final_chunk,
    models_response,
    new_completion_id,
    sse,
)


def create_openai_app(agent: Agent, *, model_id: str) -> FastAPI:
    """Return a FastAPI app exposing `agent` over an OpenAI-compatible API.

    `model_id` is the identifier clients see in `/v1/models` and select. The caller
    may add more routes/mounts (health checks, static files) to the returned app.
    """
    app = FastAPI(title="pydantic-ai OpenAI-compatible server")
    register_openai_routes(app, agent, model_id=model_id)
    return app


def register_openai_routes(app: FastAPI, agent: Agent, *, model_id: str) -> None:
    """Attach the OpenAI routes to an existing FastAPI app (composition variant)."""

    @app.get("/v1/models")
    async def list_models() -> dict:
        return models_response(model_id)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        history, prompt = split_history_and_prompt(request.messages)
        if request.stream:
            stream = _stream_completion(agent, model_id, prompt, history)
            return StreamingResponse(stream, media_type="text/event-stream")
        result = await agent.run(prompt, message_history=history)
        return completion_response(model_id, str(result.output))


def serve(agent: Agent, *, model_id: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Build the app and run it with uvicorn (the imperative-shell convenience)."""
    import uvicorn

    uvicorn.run(create_openai_app(agent, model_id=model_id), host=host, port=port)


async def _stream_completion(
    agent: Agent,
    model_id: str,
    prompt: str | list[UserContent],
    history: list[ModelMessage],
) -> AsyncIterator[str]:
    completion_id = new_completion_id()
    created = int(time.time())

    async with agent.run_stream(prompt, message_history=history) as stream:
        async for delta in stream.stream_text(delta=True):
            yield sse(content_chunk(model_id, completion_id, created, delta))

    yield sse(final_chunk(model_id, completion_id, created))
    yield SSE_DONE
