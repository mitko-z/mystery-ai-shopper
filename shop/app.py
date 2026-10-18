import logging
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from google.genai import errors as genai_errors
from pydantic import BaseModel, StringConstraints

from core import agent_loop, config
from shop.session import EmptyReplyError, SupportSession

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


# In memory only: lost on restart, never trimmed.
_conversations: dict[str, SupportSession] = {}


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
    session = _conversations.get(payload.conversation_id)
    if session is None:
        session = _conversations[payload.conversation_id] = SupportSession()
    try:
        reply = await session.send(payload.message)
    except config.MissingAPIKeyError:
        logger.exception("The Gemini API key is not configured")
        raise HTTPException(status_code=503, detail="The support agent is not configured.")
    except agent_loop.TurnLimitError as e:
        logger.warning("Conversation %r: %s", payload.conversation_id, e)
        raise HTTPException(status_code=502, detail=str(e))
    except (genai_errors.APIError, httpx.HTTPError):
        logger.exception("Conversation %r: the model call failed", payload.conversation_id)
        raise HTTPException(
            status_code=502,
            detail="The support agent is unavailable right now. Please try again.",
        )
    except EmptyReplyError:
        logger.error("Conversation %r: the final answer has no text", payload.conversation_id)
        raise HTTPException(
            status_code=502,
            detail="The support agent gave an empty answer. Please try again.",
        )
    return ChatResponse(conversation_id=payload.conversation_id, reply=reply)
