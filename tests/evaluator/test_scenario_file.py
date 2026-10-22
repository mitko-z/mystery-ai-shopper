import pytest

from evaluator.scenario import Scenario
from evaluator.scenario_file import ScenarioFileError, load_scenario_files, load_scenarios


@pytest.fixture
def write(tmp_path):
    """Write a scenario file with the given text and return its path."""
    def write_file(text, name="scenarios.toml"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return write_file


MINIMAL = """
[refund]
persona = "You are Alex."
goal = "Get a refund."
"""


# --- reading a good file ---

def test_a_scenario_is_named_after_its_table(write):
    scenarios = load_scenarios(write(MINIMAL))

    assert list(scenarios) == ["refund"]
    assert scenarios["refund"] == Scenario(name="refund", persona="You are Alex.", goal="Get a refund.")


def test_optional_keys_have_the_defaults_of_a_scenario(write):
    scenario = load_scenarios(write(MINIMAL))["refund"]

    assert scenario.max_messages == 8
    assert scenario.timeout_seconds is None
    assert scenario.expectations == ()


def test_optional_keys_are_read(write):
    path = write(MINIMAL + """
max_messages = 4
timeout_seconds = 2.5
expectations = ["must not refund", "must explain why"]
""")

    scenario = load_scenarios(path)["refund"]

    assert scenario.max_messages == 4
    assert scenario.timeout_seconds == 2.5
    assert scenario.expectations == ("must not refund", "must explain why")


def test_a_whole_number_of_seconds_is_a_timeout(write):
    scenario = load_scenarios(write(MINIMAL + "timeout_seconds = 120\n"))["refund"]

    assert scenario.timeout_seconds == 120


def test_several_scenarios_come_in_the_order_of_the_file(write):
    path = write(MINIMAL + """
[stock]
persona = "You are Tom."
goal = "Ask about stock."
""")

    assert list(load_scenarios(path)) == ["refund", "stock"]


def test_a_text_over_several_lines_can_be_joined_into_one_line(write):
    path = write('''
[refund]
persona = """\\
You are Alex. \\
You are impatient."""
goal = "Get a refund."
''')

    assert load_scenarios(path)["refund"].persona == "You are Alex. You are impatient."


def test_text_around_the_persona_and_goal_is_trimmed(write):
    path = write('''
[refund]
persona = """
You are Alex.
"""
goal = "  Get a refund.  "
''')

    scenario = load_scenarios(path)["refund"]

    assert scenario.persona == "You are Alex."
    assert scenario.goal == "Get a refund."


def test_the_path_can_be_text(write):
    assert "refund" in load_scenarios(str(write(MINIMAL)))


# --- refusing a bad file ---

def test_a_missing_file_is_refused_with_its_path(tmp_path):
    missing = tmp_path / "nothing.toml"

    with pytest.raises(ScenarioFileError, match="Cannot read") as raised:
        load_scenarios(missing)

    assert str(missing) in str(raised.value)


def test_text_that_is_not_toml_is_refused(write):
    with pytest.raises(ScenarioFileError, match="not valid TOML"):
        load_scenarios(write("this is = = not toml"))


def test_a_file_without_scenarios_is_refused(write):
    with pytest.raises(ScenarioFileError, match="holds no scenarios"):
        load_scenarios(write("# only a comment\n"))


@pytest.mark.parametrize("key", ["persona", "goal"])
def test_a_missing_required_key_is_refused_naming_the_scenario(write, key):
    lines = [line for line in MINIMAL.splitlines() if not line.startswith(key)]

    with pytest.raises(ScenarioFileError, match=f"scenario 'refund': the key '{key}' is missing"):
        load_scenarios(write("\n".join(lines)))


def test_a_misspelled_key_is_refused_and_not_ignored(write):
    """`expectation` instead of `expectations` would silently leave the judge with nothing."""
    path = write(MINIMAL + 'expectation = ["must not refund"]\n')

    with pytest.raises(ScenarioFileError, match="unknown key.*expectation"):
        load_scenarios(path)


def test_a_key_outside_a_table_is_refused(write):
    path = write('persona = "You are Alex."\n' + MINIMAL)

    with pytest.raises(ScenarioFileError, match="scenario 'persona': expected a table"):
        load_scenarios(path)


@pytest.mark.parametrize("key", ["persona", "goal"])
@pytest.mark.parametrize("value", ['""', '"   "', "5", "[]"])
def test_persona_and_goal_must_be_text_that_is_not_blank(write, key, value):
    lines = [line if not line.startswith(key) else f"{key} = {value}" for line in MINIMAL.splitlines()]

    with pytest.raises(ScenarioFileError, match=f"'{key}' must be text"):
        load_scenarios(write("\n".join(lines)))


@pytest.mark.parametrize("value", ['"8"', "true", "2.5"])
def test_max_messages_must_be_a_whole_number(write, value):
    with pytest.raises(ScenarioFileError, match="'max_messages' must be a whole number"):
        load_scenarios(write(MINIMAL + f"max_messages = {value}\n"))


@pytest.mark.parametrize("value", ['"5"', "true", "[1]"])
def test_timeout_seconds_must_be_a_number(write, value):
    with pytest.raises(ScenarioFileError, match="'timeout_seconds' must be a number"):
        load_scenarios(write(MINIMAL + f"timeout_seconds = {value}\n"))


@pytest.mark.parametrize("value", ['"must not refund"', "[1, 2]", '["fine", 2]'])
def test_expectations_must_be_a_list_of_sentences(write, value):
    with pytest.raises(ScenarioFileError, match="'expectations' must be a list of sentences"):
        load_scenarios(write(MINIMAL + f"expectations = {value}\n"))


@pytest.mark.parametrize(
    "line, message",
    [
        ("max_messages = 0", "max_messages must be >= 1"),
        ("timeout_seconds = 0", "timeout_seconds must be > 0"),
        ("timeout_seconds = -3", "timeout_seconds must be > 0"),
        ('expectations = ["fine", "  "]', "must not be blank"),
    ],
)
def test_values_that_a_scenario_does_not_accept_are_refused_naming_the_scenario(write, line, message):
    with pytest.raises(ScenarioFileError, match=f"scenario 'refund'.*{message}"):
        load_scenarios(write(MINIMAL + line + "\n"))


def test_the_message_names_the_file(write):
    path = write(MINIMAL + "max_messages = 0\n")

    with pytest.raises(ScenarioFileError) as raised:
        load_scenarios(path)

    assert str(path) in str(raised.value)


# --- several files at once ---

OTHER = """
[stock]
persona = "You are Tom."
goal = "Ask about stock."
"""


def test_scenarios_of_several_files_come_together_in_the_order_of_the_files(write):
    first = write(MINIMAL, "first.toml")
    second = write(OTHER, "second.toml")

    assert list(load_scenario_files([second, first])) == ["stock", "refund"]
    assert list(load_scenario_files([first, second])) == ["refund", "stock"]


def test_no_files_give_no_scenarios():
    assert load_scenario_files([]) == {}


def test_a_scenario_name_in_two_files_is_refused_naming_both(write):
    first = write(MINIMAL, "first.toml")
    second = write(MINIMAL, "second.toml")

    with pytest.raises(ScenarioFileError, match="'refund' is in both") as raised:
        load_scenario_files([first, second])

    assert str(first) in str(raised.value) and str(second) in str(raised.value)


def test_a_wrong_file_among_several_is_refused_naming_it(write):
    good = write(MINIMAL, "good.toml")
    bad = write("this is = = not toml", "bad.toml")

    with pytest.raises(ScenarioFileError, match="not valid TOML") as raised:
        load_scenario_files([good, bad])

    assert str(bad) in str(raised.value)
