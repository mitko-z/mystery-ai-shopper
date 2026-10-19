"""The rules about which package may import which, checked on the source code.

    core       imports none of the others
    evaluator  imports core only
    shop       imports core only
    targets    may import anything (it is where a system and the evaluator meet)

This keeps the evaluator usable against a system that is not the fake shop.
Only static `import` statements are seen. Something like
`importlib.import_module(...)` with a name built at run time is not.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN = {
    "core": {"shop", "evaluator", "targets"},
    "evaluator": {"shop", "targets"},
    "shop": {"evaluator", "targets"},
}


def imported_packages(path: Path) -> dict[str, int]:
    """The top-level packages a module imports, each with the line of its first import.

    Relative imports stay inside one package, so they are not counted.
    """
    found = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            found.setdefault(name.split(".")[0], node.lineno)
    return found


def violations(package_dir: Path, forbidden: set[str]) -> list[str]:
    """One line per forbidden import under `package_dir`, as `file:line imports package`."""
    problems = []
    for path in sorted(package_dir.rglob("*.py")):
        for package, line in imported_packages(path).items():
            if package in forbidden:
                problems.append(f"{path.relative_to(package_dir.parent).as_posix()}:{line} imports {package}")
    return problems


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_imports_only_what_it_may(package):
    """No module of the package, tests included, imports a package it must not depend on."""
    assert violations(ROOT / package, FORBIDDEN[package]) == []


def test_the_scan_really_reads_the_packages():
    """If the scan found nothing anywhere, the rule test above would pass for the wrong reason."""
    assert "core" in imported_packages(ROOT / "evaluator" / "customer.py")
    assert "shop" in imported_packages(ROOT / "targets" / "shop.py")


def test_a_forbidden_import_is_reported(tmp_path):
    """Plain, from- and nested imports of a forbidden package are found, with their line; allowed ones are not."""
    (tmp_path / "plain.py").write_text("import os\nimport shop.store\n", encoding="utf-8")
    (tmp_path / "from_import.py").write_text("from targets import shop\n", encoding="utf-8")
    (tmp_path / "nested.py").write_text("def f():\n    from shop import app\n", encoding="utf-8")
    (tmp_path / "fine.py").write_text("import os\nfrom . import sibling\nfrom core import config\n", encoding="utf-8")

    problems = violations(tmp_path, {"shop", "targets"})

    # Each line starts with the folder's name; the rest is what matters.
    assert [problem.split("/", 1)[1] for problem in problems] == [
        "from_import.py:1 imports targets",
        "nested.py:2 imports shop",
        "plain.py:2 imports shop",
    ]
