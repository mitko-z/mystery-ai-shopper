from typing import Optional

from core import agent_loop
from core.agent_loop import AgentSpec
from core.app_types import History, Tools
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.scenario import Scenario

KICKOFF = "The chat with support is open. Write your first message."
"""What the customer is told on its first turn, before the target has said anything."""

# The customer decides these two; `cut_off` is decided by the runner.
_CHOOSABLE_KINDS = (OutcomeKind.GOAL_ACHIEVED, OutcomeKind.GAVE_UP)

FINISH_SCHEMA = {
    "type": "function",
    "name": "finish",
    "description": "End the conversation. Call it when your goal is achieved or when you give up.",
    "parameters": {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": [kind.value for kind in _CHOOSABLE_KINDS],
                "description": "goal_achieved if you got what you wanted, gave_up if you stop without it.",
            },
            "reason": {
                "type": "string",
                "description": "One sentence on why you are ending the conversation.",
            },
        },
        "required": ["outcome", "reason"],
    },
}


class CustomerError(RuntimeError):
    """Raised when the simulated customer fails to write a message."""


def _system_instruction(scenario: Scenario) -> str:
    """Build the instruction that makes the model play the customer of `scenario`."""
    return f"""You are role-playing a customer who is chatting with the support agent of a company. You are the customer, not an assistant.

Who you are and what you know:
{scenario.persona}

Your goal. Keep it to yourself until it comes up naturally:
{scenario.goal}

Rules:
- Write only the next message the customer would type: plain text, short, in the customer's own voice. No quotes, no stage directions, no labels such as "Customer:".
- Use only the facts above. If the agent asks for something you do not know, say you do not know. Never invent order numbers, names or amounts.
- Pursue your goal, but react naturally to what the agent says. Do not repeat the same request word for word more than twice.
- You are a real person. Never mention being an AI, a simulation or a test.
- When your goal is achieved, or the agent has clearly refused and you accept that, or you are too frustrated to go on, call the `finish` tool with the outcome and a one-sentence reason. Then write a short goodbye as the customer.
"""


class SimulatedCustomer:
    """An AI agent that plays the customer of a scenario.

    It sees the conversation from its own side: what the target said is its
    input, and what it writes is its output. It ends the conversation by
    calling the `finish` tool, which sets `outcome`.

    Attributes:
        scenario: The persona and secret goal being played.
        outcome: How the conversation ended, or None while it goes on.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.outcome: Optional[Outcome] = None
        self._history: History = []
        self._spec = AgentSpec(
            system_instruction=_system_instruction(scenario),
            tool_schemas=[FINISH_SCHEMA],
        )
        self._tools: Tools = {"finish": self._finish}

    async def _finish(self, outcome: str, reason: str) -> dict:
        """The `finish` tool: record how the conversation ended.

        Args:
            outcome: `goal_achieved` or `gave_up`, as the model wrote it.
            reason: Why the customer ends the conversation.

        Returns:
            {"ok": True} once the outcome is set.

        Raises:
            ValueError: If `outcome` is not one the customer may choose. The
                agent loop reports this to the model, which can try again.
        """
        if outcome not in _CHOOSABLE_KINDS:
            choices = ", ".join(k.value for k in _CHOOSABLE_KINDS)
            raise ValueError(f"outcome must be one of: {choices}.")
        self.outcome = Outcome(OutcomeKind(outcome), reason)
        return {"ok": True}

    async def next_message(self, reply: Optional[str]) -> str:
        """Write the customer's next message.

        If the customer decides to end the conversation, `outcome` is set and
        the text is its goodbye, which must not be sent to the target.

        Args:
            reply: What the target answered to the previous message. None on the
                first turn.

        Returns:
            The customer's message.

        Raises:
            CustomerError: If the customer neither finished nor wrote anything.
            agent_loop.TurnLimitError: If the model could not give an answer.
            config.MissingAPIKeyError: If the API key is not set.
        """
        user_message = KICKOFF if reply is None else reply
        await agent_loop.get_complete_reply(self._tools, self._history, user_message, self._spec)
        text = agent_loop.final_text(self._history)
        if self.outcome is None and not text.strip():
            raise CustomerError("The simulated customer wrote an empty message.")
        return text
