"""Typed OpenAI-compatible request/response models and SSE builders.

Only the slice of the OpenAI Chat Completions API that clients (e.g. OpenWebUI)
actually use is modelled here; unknown fields are ignored. Keeping the wire format
in typed pydantic models (rather than loose dicts) means validation happens once at
the edge and the rest of the code is total.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Request models ---------------------------------------------------------


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str


class ImageUrlPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Annotated[TextPart | ImageUrlPart, Field(discriminator="type")]


class ChatMessage(BaseModel):
    """An OpenAI chat message. `content` is a string or a list of typed parts."""

    role: str
    content: str | list[ContentPart] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    stream: bool = False


# --- Response builders ------------------------------------------------------


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def models_response(model_id: str) -> dict:
    """Body for `GET /v1/models` — lets a client discover the agent."""
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "owned_by": "pydantic-ai"}],
    }


def completion_response(model: str, content: str) -> dict:
    """Body for a non-streaming `POST /v1/chat/completions`."""
    return {
        "id": new_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def _chunk(model: str, completion_id: str, created: int, delta: dict, finish_reason: str | None) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def content_chunk(model: str, completion_id: str, created: int, text: str) -> dict:
    return _chunk(model, completion_id, created, {"content": text}, None)


def final_chunk(model: str, completion_id: str, created: int) -> dict:
    return _chunk(model, completion_id, created, {}, "stop")


def sse(payload: dict) -> str:
    """Serialize one Server-Sent Event line in OpenAI's streaming format."""
    return f"data: {json.dumps(payload)}\n\n"


SSE_DONE = "data: [DONE]\n\n"
