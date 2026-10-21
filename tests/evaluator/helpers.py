"""Scenarios and a model with a script for each of them, for the tests of runs side by side.

Scenario i is told apart by the marker "<<persona i>>" in its persona. The simulated
customer's instruction contains the persona, so the model fake can pick the script
of the scenario whose customer is calling.
"""

from evaluator.scenario import Scenario
from tests.fakes import (
    RoutedInteractions,
    ScriptedInteractions,
    Turn,
    call_step,
    reply_step,
    reported_usage,
    use_fake,
)


def marker(i):
    return f"<<persona {i}>>"


def make_scenarios(count, timeouts=None):
    """Scenarios s0, s1, ...; `timeouts` maps an index to a timeout_seconds."""
    timeouts = timeouts or {}
    return [
        Scenario(
            name=f"s{i}",
            persona=f"{marker(i)} You are customer {i}.",
            goal="Get a refund.",
            timeout_seconds=timeouts.get(i),
        )
        for i in range(count)
    ]


def customer_script(i):
    """Customer i says "Hi from i", then finishes and says goodbye. Each model call costs 10*(i+1) tokens."""
    usage = reported_usage(input=10 * (i + 1), total=10 * (i + 1))
    finish = [call_step("f1", "finish", {"outcome": "gave_up", "reason": "Refused."})]
    return ScriptedInteractions(
        Turn([reply_step(f"Hi from {i}")], usage),
        Turn(finish, usage),
        Turn([reply_step("Thanks, bye.")], usage),
    )


def use_scripted_customers(monkeypatch, count):
    """Make the model answer as customers 0 to count-1, each with the script of its own."""
    use_fake(monkeypatch, RoutedInteractions({marker(i): customer_script(i) for i in range(count)}))
