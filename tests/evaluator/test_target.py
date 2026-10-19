import pytest

from evaluator.target import Target, ToolCallSource


class PlainTarget(Target):
    """A black-box target: it can be talked to and nothing else."""

    async def send(self, message):
        return "ok"


class LoggedTarget(PlainTarget):
    """A target that also keeps a log of its agent's tool calls."""

    def tool_calls(self):
        return []


def test_a_target_without_a_log_is_not_a_tool_call_source():
    """Having no log is a state of its own, so the check must say no rather than offer an empty list."""
    assert not isinstance(PlainTarget(), ToolCallSource)


def test_a_target_with_a_log_is_a_tool_call_source():
    """The runner can find the log by asking `isinstance`, without the target inheriting anything."""
    assert isinstance(LoggedTarget(), ToolCallSource)


def test_the_log_is_optional_for_a_target():
    """`Target` itself asks for `send` only."""
    assert not hasattr(Target, "tool_calls")


def test_a_class_that_forgets_send_cannot_be_created():
    """`Target` is abstract in `send`, so a subclass that skips it fails when it is built."""
    class Forgetful(Target):
        pass

    with pytest.raises(TypeError, match="send"):
        Forgetful()
