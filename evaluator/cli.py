"""The command line of the evaluator.

    uv run python -m evaluator single <target> <scenario>
    uv run python -m evaluator parallel <target>
    uv run python -m evaluator judge <results file>
    uv run python -m evaluator agreement <key file> <judge file>

`single` runs one scenario and prints the conversation as it happens. `parallel`
runs many scenarios at once, prints a line for each as it finishes, and writes
all the results to a JSON file. `judge` reads such a file and judges, for every
finished run, whether the target met each expectation of its scenario. `agreement`
puts the judge's verdicts beside our own, in a table.

`<target>` is a module in the `targets` package, e.g. `shop`. It provides
`open_target()`, an async context manager that yields a `Target`, and
`SCENARIOS`, a dict of scenarios by name.
"""

import argparse
import asyncio
import importlib
import pkgutil
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Optional

import httpx

from core import agent_loop, config
from core.usage import Usage, grand_total, track_usage
from evaluator import agreement, judge
from evaluator.customer import CustomerError
from evaluator.outcome import Outcome
from evaluator.runner import (
    RESULTS_FOLDER,
    Message,
    ResultsFileError,
    RunResult,
    RunStatus,
    Transcript,
    build_report,
    count_statuses,
    default_results_path,
    run_in_parallel,
    run_scenario,
    write_report,
)
from evaluator.scenario import Scenario
from evaluator.scenario_file import ScenarioFileError, load_scenario_files
from evaluator.target import TargetError

TARGETS_PACKAGE = "targets"
DEFAULT_MAX_PARALLEL = 5
DEFAULT_TIMEOUT_SECONDS = 120.0

# Exit codes. 2 is taken by argparse, for a command line it cannot use.
EXIT_COULD_NOT_RUN = 1
EXIT_SOME_FAILED = 3  # a run failed or timed out, or a judgement got no verdict
EXIT_INTERRUPTED = 130


def format_header(scenario: Scenario) -> str:
    """The lines that open a printed run: who the customer is and what they want."""
    return (
        f"Scenario: {scenario.name}\n"
        f"Persona: {scenario.persona.strip()}\n"
        f"Secret goal: {scenario.goal.strip()}\n"
    )


def format_message(message: Message) -> str:
    """One numbered line of the conversation."""
    return f"{message.number}. {message.speaker}: {message.text}"


def format_outcome(outcome: Outcome) -> str:
    """The line that closes a printed run."""
    return f"Outcome: {outcome.kind.value} - {outcome.reason}"


def format_usage(usage: Usage) -> str:
    """The line that reports what a run cost: model calls and tokens.

    Calls that cannot be in the token counts are named: cancelled calls, which
    never got a response, and calls whose response carried no usage report.
    """
    line = (
        f"Model calls: {usage.calls} | Tokens: input {usage.input_tokens:,}, "
        f"output {usage.output_tokens:,}, thought {usage.thought_tokens:,}, "
        f"total {usage.total_tokens:,}"
    )
    if usage.calls_cancelled:
        line += f" | Calls cancelled (tokens unknown): {usage.calls_cancelled}"
    if usage.calls_without_usage:
        line += f" | Calls without a usage report: {usage.calls_without_usage} (tokens are incomplete)"
    return line


def format_run_line(result: RunResult, number: int, total: int) -> str:
    """The line that reports one run of a batch as it finishes.

    Example: `[2/5] refund_o1002: finished (gave_up) in 41.3 s, 10 model calls`.
    A call that was cancelled in flight, e.g. by the timeout, is named too: it is
    not among the model calls, which got an answer. A run that did not finish also
    gives the reason.
    """
    status = result.status.value
    if result.outcome is not None:
        status += f" ({result.outcome.kind.value})"
    line = (
        f"[{number}/{total}] {result.scenario.name}: {status} "
        f"in {result.duration_seconds:.1f} s, {result.usage.calls} model calls"
    )
    if result.usage.calls_cancelled:
        line += f", {result.usage.calls_cancelled} cancelled"
    if result.error is not None:
        line += f" - {result.error}"
    return line


def format_summary(results: Sequence[RunResult], totals: Usage, results_path: Path) -> str:
    """The lines that close a batch: runs by status, what it all cost, and where the results are."""
    statuses = " | ".join(f"{status}: {count}" for status, count in count_statuses(results).items())
    return "\n".join([
        f"Runs: {len(results)} | {statuses}",
        format_usage(totals),
        f"Results written to {results_path}",
    ])


def format_judge_line(judgement: judge.Judgement, number: int, total: int) -> str:
    """The line that reports one judgement as it finishes.

    Example: `[3/18] refund_o1002 #3: fail`. A judgement that reached no verdict says
    why. An error that came after the verdict is named too: the verdict stands.
    """
    verdict = judgement.verdict.value if judgement.verdict is not None else "no verdict"
    line = f"[{number}/{total}] {judgement.scenario} #{judgement.expectation_number}: {verdict}"
    if judgement.error_message is not None:
        line += f" - {judgement.error_message}"
    return line


def format_judge_summary(
    judgements: Sequence[judge.Judgement],
    skipped_runs: Sequence[dict[str, str]],
    totals: Usage,
    path: Path,
) -> str:
    """The lines that close a judging: the verdicts, the runs left out, what it cost, and where it is."""
    verdicts = Counter(j.verdict.value if j.verdict is not None else "no verdict" for j in judgements)
    counts = " | ".join(f"{name}: {verdicts[name]}" for name in (*(v.value for v in judge.Verdict), "no verdict"))
    lines = [f"Judgements: {len(judgements)} | {counts}"]
    if skipped_runs:
        lines.append("Runs not judged: " + "; ".join(f"{run['scenario']} ({run['reason']})" for run in skipped_runs))
    lines += [format_usage(totals), f"Judgements written to {path}"]
    return "\n".join(lines)


class UsageError(Exception):
    """Raised when the command line names a target or scenario that does not exist."""


def load_target(name: str) -> ModuleType:
    """Import the module of target `name` from the targets package.

    Raises:
        UsageError: If there is no such target; the message lists the ones there are.
    """
    package = importlib.import_module(TARGETS_PACKAGE)
    module_name = f"{TARGETS_PACKAGE}.{name}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name != module_name:
            raise  # the target exists but is broken
        available = ", ".join(sorted(m.name for m in pkgutil.iter_modules(package.__path__))) or "none"
        raise UsageError(f"Unknown target {name!r}. Available: {available}.") from e


def get_scenario(target_module: ModuleType, name: str) -> Scenario:
    """Look up scenario `name` in a target module.

    Raises:
        UsageError: If there is no such scenario; the message lists the ones there are.
    """
    scenarios = target_module.SCENARIOS
    if name not in scenarios:
        raise UsageError(f"Unknown scenario {name!r}. Available: {', '.join(sorted(scenarios))}.")
    return scenarios[name]


async def _single_command(target_name: str, scenario_name: str, max_calls: int) -> None:
    """The `single` command: run one scenario and print the conversation as it happens.

    The outcome, if the run reached one, and what the run cost are printed at the
    end, also when the run fails.
    """
    target_module = load_target(target_name)
    scenario = get_scenario(target_module, scenario_name)
    print(format_header(scenario))
    transcript = Transcript()
    with track_usage(max_calls) as usage:
        try:
            async with target_module.open_target() as target:
                await run_scenario(target, scenario, transcript, on_message=lambda m: print(format_message(m)))
        finally:
            print()
            if transcript.outcome is not None:
                print(format_outcome(transcript.outcome))
            print(format_usage(usage))


async def _parallel_command(
    target_name: str,
    scenario_files: Optional[list[str]],
    *,
    max_parallel: int,
    timeout: float,
    max_calls: int,
    results_path: Optional[str],
) -> list[RunResult]:
    """The `parallel` command: run many scenarios at once, write the results file and print a summary.

    A line is printed for each run as it finishes. The scenarios are those of
    `scenario_files`, or else those of the target. A run that times out or fails
    is a result like any other: it does not stop the batch, and the file is
    written all the same.

    Returns:
        The result of each run, in scenario order. `main` turns them into the exit code.

    Raises:
        UsageError: If the target is unknown or has no scenarios.
        ScenarioFileError: If the scenario file is wrong.
        config.MissingAPIKeyError: If the API key is not set. This is checked before
            any run starts, so a missing key is one error and not one per run.
        ResultsFileError: If the results file cannot be written.
    """
    target_module = load_target(target_name)
    source = load_scenario_files(scenario_files) if scenario_files else target_module.SCENARIOS
    scenarios = list(source.values())
    if not scenarios:
        raise UsageError(f"There are no scenarios to run on target {target_name!r}.")
    agent_loop.check_api_key()

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    path = Path(results_path) if results_path else default_results_path(target_name, started_at)
    count = f"{len(scenarios)} scenario" + ("" if len(scenarios) == 1 else "s")
    print(f"Running {count} on {target_name}, at most {max_parallel} at a time.", flush=True)

    finished = 0

    def report_progress(result: RunResult) -> None:
        nonlocal finished
        finished += 1
        print(format_run_line(result, finished, len(scenarios)), flush=True)

    results = await run_in_parallel(
        target_module,
        scenarios,
        max_parallel=max_parallel,
        default_timeout=timeout,
        max_calls=max_calls,
        on_finish=report_progress,
    )

    totals = grand_total()
    report = build_report(
        target_name,
        results,
        settings={
            "max_parallel": max_parallel,
            "default_timeout_seconds": timeout,
            "max_calls": max_calls,
        },
        started_at=started_at,
        duration_seconds=time.perf_counter() - started,
        totals=totals,
    )
    write_report(path, report)
    print()
    print(format_summary(results, totals, path))
    return results


async def _judge_command(
    results_path: str,
    out_path: Optional[str],
    *,
    use_tool_log: bool,
    judge_prompt_version: str,
    max_parallel: int,
    timeout: float,
    max_calls: int,
) -> list[judge.Judgement]:
    """The `judge` command: judge every expectation of a results file, write the judgements and print a summary.

    A line is printed for each judgement as it finishes. A judgement that fails is a
    result like any other: it does not stop the others, and the file is written all the same.

    Returns:
        The judgements, in the order of the file. `main` turns them into the exit code.

    Raises:
        judge.JudgeInputError: If the results file cannot be read.
        UsageError: If no run is to be judged, or the judge file exists already. Both are
            found before any model call is made.
        config.MissingAPIKeyError: If the API key is not set, also before any run starts.
        ResultsFileError: If the judge file cannot be written.
    """
    runs, skipped = judge.select_runs(judge.load_results(results_path))
    if not runs:
        why = "; ".join(f"{run['scenario']} ({run['reason']})" for run in skipped)
        raise UsageError(f"There is nothing to judge in {results_path}. " + (f"Not judged: {why}." if why else ""))
    path = Path(out_path) if out_path else judge.default_out_path(results_path, use_tool_log, judge_prompt_version)
    if path.exists():
        raise UsageError(
            f"{path} exists already. A judge file is kept and not overwritten: "
            "give another file with --out, or move this one away."
        )
    agent_loop.check_api_key()

    total = sum(len(run["expectations"]) for run in runs)
    tool_log = "with" if use_tool_log else "without"
    print(
        f"Judging {total} expectation{'' if total == 1 else 's'} of {len(runs)} run{'' if len(runs) == 1 else 's'} "
        f"with rules {judge_prompt_version}, {tool_log} the tool log, at most {max_parallel} at a time.",
        flush=True,
    )

    finished = 0

    def report_progress(judgement: judge.Judgement) -> None:
        nonlocal finished
        finished += 1
        print(format_judge_line(judgement, finished, total), flush=True)

    judgements = await judge.judge_runs(
        runs,
        use_tool_log=use_tool_log,
        prompt_version=judge_prompt_version,
        max_parallel=max_parallel,
        timeout=timeout,
        max_calls=max_calls,
        on_judged=report_progress,
    )

    totals = grand_total()
    write_report(path, judge.build_judge_report(
        judgements, skipped, use_tool_log=use_tool_log, totals=totals, prompt_version=judge_prompt_version,
    ))
    print()
    print(format_judge_summary(judgements, skipped, totals, path))
    return judgements


def _agreement_command(key_path: str, judge_path: str, out_path: Optional[str], overwrite: bool) -> list[agreement.Row]:
    """The `agreement` command: put the judge's verdicts beside the key's, write the table and print a summary.

    It makes no model call, and reads and writes files only.

    Returns:
        The rows of the table.

    Raises:
        agreement.AgreementError: If a file cannot be read or the two do not belong together,
            or the table exists already and `overwrite` is false (it may hold causes
            written by hand), or cannot be written.
    """
    rows, judge_info = agreement.compare_files(key_path, judge_path)
    text = agreement.render_markdown(
        rows, key_name=Path(key_path).name, judge_name=Path(judge_path).name, judge_info=judge_info
    )
    path = Path(out_path) if out_path else agreement.default_out_path(judge_path)
    agreement.write_table(path, text, overwrite=overwrite)

    print(agreement.format_summary(rows))
    needing_a_cause = sum(row.needs_cause for row in rows)
    if needing_a_cause:
        rows_text = "1 row needs" if needing_a_cause == 1 else f"{needing_a_cause} rows need"
        print(f"{rows_text} a cause: {agreement.CAUSE_TODO} in the table.")
    print(f"Agreement table written to {path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    """The command line: a mode (`single`, `parallel`, `judge` or `agreement`), then what that mode needs."""
    # allow_abbrev=False: a mistyped option is an error, and not a silent guess at
    # another one (e.g. `--max` for `--max-calls`). It has to be set on every parser.
    parser = argparse.ArgumentParser(
        prog="python -m evaluator",
        description="Run scenarios with a simulated customer against a target.",
        allow_abbrev=False,
    )
    modes = parser.add_subparsers(dest="mode", required=True, metavar="{single,parallel,judge,agreement}")

    def add_shared_arguments(mode: argparse.ArgumentParser) -> None:
        mode.add_argument("target", help=f"name of a module in the `{TARGETS_PACKAGE}` package, e.g. shop")
        mode.add_argument(
            "--max-calls",
            type=int,
            default=config.DEFAULT_CALLS_PER_RUN,
            help=f"most model calls one run may make (default: {config.DEFAULT_CALLS_PER_RUN})",
        )

    single = modes.add_parser(
        "single",
        help="run one scenario and print the conversation as it happens",
        description="Run one scenario and print the conversation as it happens.",
        allow_abbrev=False,
    )
    add_shared_arguments(single)
    single.add_argument("scenario", help="name of one of the target's scenarios, e.g. refund_o1002")

    parallel = modes.add_parser(
        "parallel",
        help="run many scenarios at once and write the results to a JSON file",
        description="Run many scenarios at once. A line is printed as each one finishes, "
                    "and all the results are written to a JSON file.",
        allow_abbrev=False,
    )
    add_shared_arguments(parallel)
    parallel.add_argument(
        "--scenario-file",
        metavar="FILE",
        action="append",
        help="TOML file with the scenarios to run, instead of the target's own. "
             "Repeat the option to use several files.",
    )
    parallel.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help=f"most runs active at the same time (default: {DEFAULT_MAX_PARALLEL})",
    )
    parallel.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="seconds one conversation may take, unless its scenario says otherwise "
             f"(default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    parallel.add_argument(
        "--results",
        metavar="FILE",
        help=f"where to write the results (default: {RESULTS_FOLDER}/<target>-<time>.json)",
    )

    judging = modes.add_parser(
        "judge",
        help="judge whether the target met the expectations, for the runs of a results file",
        description="Judge, for every finished run of a results file, whether the target met each "
                    "expectation of its scenario. A line is printed as each judgement finishes, "
                    "and all of them are written to a JSON file, which is never overwritten.",
        allow_abbrev=False,
    )
    judging.add_argument("results", help="the results file of a batch, e.g. results/shop-20261023-185334.json")
    judging.add_argument(
        "--tool-log",
        action="store_true",
        help="also give the judge the target's log of tool calls, for the runs that have one",
    )
    judging.add_argument(
        "--judge-prompt-version",
        choices=judge.PROMPT_VERSIONS,
        default=judge.PROMPT_VERSION,
        help=f"which version of the judge's rules to use (default: {judge.PROMPT_VERSION}, the newest)",
    )
    judging.add_argument(
        "--out",
        metavar="FILE",
        help=f"where to write the judgements (default: {judge.EVALUATIONS_FOLDER}/<results name>"
             f".judge-<version>.json, or .judge-<version>-toollog.json with --tool-log)",
    )
    judging.add_argument(
        "--max-parallel",
        type=int,
        default=judge.MAX_PARALLEL,
        help=f"most judgements active at the same time (default: {judge.MAX_PARALLEL})",
    )
    judging.add_argument(
        "--timeout",
        type=float,
        default=judge.TIMEOUT,
        help=f"seconds one judgement may take (default: {judge.TIMEOUT:g})",
    )
    judging.add_argument(
        "--max-calls",
        type=int,
        default=judge.MAX_CALLS,
        help=f"most model calls one judgement may make (default: {judge.MAX_CALLS})",
    )

    comparing = modes.add_parser(
        "agreement",
        help="put the judge's verdicts beside our own, in a table",
        description="Compare a judge file with the key (our own verdicts for the same results), and write "
                    "a Markdown table with a row for each expectation. It makes no model call.",
        allow_abbrev=False,
    )
    comparing.add_argument("key", help="the key: our verdicts, e.g. evaluations/shop-20261023-185334.ground-truth.json")
    comparing.add_argument("judge", help="the judge file, e.g. evaluations/shop-20261023-185334.judge-v1.json")
    comparing.add_argument(
        "--out",
        metavar="FILE",
        help="where to write the table (default: beside the judge file, with .judge- replaced by .agreement-)",
    )
    comparing.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the table if it exists. Without this it is kept, because the causes in it are written by hand",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Command line entry point.

    Returns:
        0 if the command did its job: the run finished, or every run of the batch did
        (whatever the customer's outcome), or every expectation got a verdict. 1 if it
        could not be run at all. 3 if a batch or a judging ran, and its file was written,
        but a run failed or timed out, or a judgement got no verdict. 130 if it was
        interrupted.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # A limit is checked in each mode that has it.
    if hasattr(args, "max_calls") and args.max_calls < 1:
        parser.error("--max-calls must be at least 1")
    if hasattr(args, "max_parallel") and args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    if hasattr(args, "timeout") and args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    # Model text can hold characters the Windows console codepage cannot print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        if args.mode == "single":
            asyncio.run(_single_command(args.target, args.scenario, args.max_calls))
            return 0
        elif args.mode == "parallel":
            results = asyncio.run(_parallel_command(
                args.target,
                args.scenario_file,
                max_parallel=args.max_parallel,
                timeout=args.timeout,
                max_calls=args.max_calls,
                results_path=args.results,
            ))
            if any(result.status is not RunStatus.FINISHED for result in results):
                return EXIT_SOME_FAILED
            return 0
        elif args.mode == "judge":
            judgements = asyncio.run(_judge_command(
                args.results,
                args.out,
                use_tool_log=args.tool_log,
                judge_prompt_version=args.judge_prompt_version,
                max_parallel=args.max_parallel,
                timeout=args.timeout,
                max_calls=args.max_calls,
            ))
            if any(judgement.failed for judgement in judgements):
                return EXIT_SOME_FAILED
            return 0
        else:
            _agreement_command(args.key, args.judge, args.out, args.overwrite)
            return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (
        UsageError,
        ScenarioFileError,
        ResultsFileError,
        TargetError,
        CustomerError,
        judge.JudgeInputError,
        agreement.AgreementError,
        config.MissingAPIKeyError,
        agent_loop.TurnLimitError,
        agent_loop.ModelError,
        httpx.HTTPError,
        agent_loop.CallLimitError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main())
