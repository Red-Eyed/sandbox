"""Pure mapping from OpenAI chat messages to pydantic-ai message types.

This is the functional core of the server: no I/O, no agent, just a translation
between two message schemas. Clients send the full conversation on every request,
so we split it into prior turns (`message_history`) plus the final user turn (the
new `user_prompt`).
"""

from __future__ import annotations

from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserContent,
    UserPromptPart,
)

from .schema import ChatMessage, ImageUrlPart
from .schema import TextPart as OpenAITextPart


def _to_user_content(content: str | list | None) -> str | list[UserContent]:
    """Convert OpenAI message content into pydantic-ai user content.

    A plain string stays a string. A list of parts becomes a list of text and
    image content; `data:` URIs become inline `BinaryContent`, http(s) URLs become
    `ImageUrl`. A list holding a single text part collapses back to a string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[UserContent] = []
    for part in content:
        if isinstance(part, OpenAITextPart):
            parts.append(part.text)
        elif isinstance(part, ImageUrlPart):
            parts.append(_image_to_content(part.image_url.url))

    if len(parts) == 1 and isinstance(parts[0], str):
        return parts[0]
    return parts


def _image_to_content(url: str) -> UserContent:
    if url.startswith("data:"):
        return BinaryContent.from_data_uri(url)
    return ImageUrl(url=url)


def _to_text(content: str | list | None) -> str:
    """Flatten content to plain text (for system/assistant history turns)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.text for p in content if isinstance(p, OpenAITextPart))


def split_history_and_prompt(
    messages: list[ChatMessage],
) -> tuple[list[ModelMessage], str | list[UserContent]]:
    """Split OpenAI messages into (`message_history`, final user prompt).

    The last user message is the new prompt; everything before it becomes history.
    System messages map to a `SystemPromptPart`, assistant messages to a
    `ModelResponse`, and earlier user messages to a `UserPromptPart`.
    """
    last_user_index = _last_index_with_role(messages, "user")

    history: list[ModelMessage] = []
    prompt: str | list[UserContent] = ""

    for index, message in enumerate(messages):
        if index == last_user_index:
            prompt = _to_user_content(message.content)
        elif message.role == "system":
            history.append(ModelRequest(parts=[SystemPromptPart(content=_to_text(message.content))]))
        elif message.role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=_to_text(message.content))]))
        elif message.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=_to_user_content(message.content))]))

    return history, prompt


def _last_index_with_role(messages: list[ChatMessage], role: str) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == role:
            return index
    return None
