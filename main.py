import asyncio
import logging
from dataclasses import dataclass, field
from itertools import takewhile
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from google.genai import errors as genai_errors
from pydantic import BaseModel, StringConstraints

import config
from shop import seed
from shop.store import Shop
import support_agent

logger = logging.getLogger(__name__)

app = FastAPI(title="Mystery AI Shopper")

_UI_PAGE = Path(__file__).parent / "static" / "index.html"


class ChatRequest(BaseModel):
    """One customer message in a conversation."""

    conversation_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class ChatResponse(BaseModel):
    """The agent's reply to the latest message."""

    conversation_id: str
    reply: str


@dataclass
class Conversation:
    """A conversation's `history` and the lock that serialises its turns."""

    history: list[dict[str, Any]] = field(default_factory=list)
    shop: Shop = field(default_factory=seed.get_seed_shop)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# In memory only: lost on restart, never trimmed.
_conversations: dict[str, Conversation] = {}


def _final_reply(history: list[dict[str, Any]]) -> str:
    """Return the text of the agent's answer to the latest message.

    `support_agent` hands everything back through `history`, thoughts, tool
    calls and tool results included. The answer is the `model_output` step(s)
    at the end; the earlier steps of the turn are not for the customer. One
    model response can end with several such steps, and its text parts are
    joined as they are, the way the SDK builds `Interaction.output_text`.

    Args:
        history: A conversation whose last turn ended with a `model_output` step.

    Returns:
        The reply text, or "" if those steps carry no text.
    """
    tail = takewhile(lambda step: step.get("type") == "model_output", reversed(history))
    return "".join(
        part.get("text", "")
        for step in reversed(list(tail))
        for part in step.get("content", [])
        if part.get("type") == "text"
    )


@app.get("/")
async def root():
    """Landing message."""
    return {"message": "Mystery AI Shopper is running. Chat at /ui, API docs at /docs."}


@app.get("/ui", include_in_schema=False)
async def ui() -> FileResponse:
    """Serve the demo chat page."""
    return FileResponse(_UI_PAGE)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat")
async def chat(payload: ChatRequest) -> ChatResponse:
    """Send a customer message to the support agent and return its reply.

    The first message with a new `conversation_id` starts the conversation;
    later messages with the same id continue it. Messages of one conversation
    are handled one at a time, in arrival order.

    Raises:
        HTTPException: 502 if the agent could not give a final answer, gave an
            empty one, or the model call failed; 503 if the API key is not
            configured. After a failure the conversation is kept, so the
            customer can write again.
    """
    conversation = _conversations.get(payload.conversation_id)
    if conversation is None:
        conversation = _conversations[payload.conversation_id] = Conversation()
    async with conversation.lock:
        try:
            await support_agent.get_complete_reply(conversation.shop, conversation.history, payload.message)
        except config.MissingAPIKeyError:
            logger.exception("The Gemini API key is not configured")
            raise HTTPException(status_code=503, detail="The support agent is not configured.")
        except support_agent.TurnLimitError as e:
            logger.warning("Conversation %r: %s", payload.conversation_id, e)
            raise HTTPException(status_code=502, detail=str(e))
        except (genai_errors.APIError, httpx.HTTPError):
            logger.exception("Conversation %r: the model call failed", payload.conversation_id)
            raise HTTPException(
                status_code=502,
                detail="The support agent is unavailable right now. Please try again.",
            )
        reply = _final_reply(conversation.history)
        if not reply.strip():
            logger.error("Conversation %r: the final answer has no text", payload.conversation_id)
            raise HTTPException(
                status_code=502,
                detail="The support agent gave an empty answer. Please try again.",
            )
        return ChatResponse(conversation_id=payload.conversation_id, reply=reply)
