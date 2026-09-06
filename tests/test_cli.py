import http.server
import json
import os
import subprocess
import sys
import threading
import urllib.parse
import zipfile
from pathlib import Path

import pytest

from backend.compiler import NOTEBOOK_TO_API_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_notebook(path):
    path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": (
                            "def add(a: int, b: int) -> int:\n"
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_notebook_with_function(path, function_source):
    path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": function_source,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _run_cli(args, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "backend.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_version_flag_prints_the_tools_own_version_and_exits_zero(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["--version"], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == f"notebook-to-api {NOTEBOOK_TO_API_VERSION}"


def test_version_short_flag_prints_the_same_thing_as_the_long_flag(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    long_proc = _run_cli(["--version"], cwd=workdir)
    short_proc = _run_cli(["-V"], cwd=workdir)

    assert short_proc.returncode == 0, short_proc.stdout + short_proc.stderr
    assert short_proc.stdout == long_proc.stdout


def test_version_flag_works_with_no_subcommand_given(tmp_path):
    """Unlike every other flag, --version must not trip the top-level
    parser's own required=True subparsers -- confirmed this doesn't
    regress into the same "required: command" argparse error a bare
    `notebook-to-api` (no args at all) already gets.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["--version"], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "required" not in proc.stderr.lower()


def test_compile_command_writes_the_generated_app(tmp_path):
    """The `compile`, `inspect`, `export-openapi`, and `export-sdk`
    subcommands were previously only exercised by calling their
    underlying functions directly (see test_compiler.py,
    test_openapi_exporter.py, test_sdk_generator.py) -- never through the
    actual `backend.cli` argparse entry point, unlike `deploy`
    (test_cli_deploy.py). That left the subparser wiring itself
    untested: test_deploy_command_is_registered in test_cli_deploy.py
    documents a real bug this exact gap already let through once (a
    dispatch branch in main() with no matching add_parser(...)).
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "built" / "app.py").exists()
    assert (workdir / "built" / "requirements.txt").exists()
    assert (workdir / "built" / "Dockerfile").exists()
    assert "Compilation finished" in proc.stdout


def test_compile_command_defaults_output_to_generated(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(["compile", str(notebook_path)], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "generated" / "app.py").exists()


def test_compile_command_smoke_test_passes_for_a_healthy_app(tmp_path):
    """`compile --smoke-test` actually imports the just-compiled app in
    this process and calls its own GET /health -- the same diagnostic
    POST /api/compile's own "smoke_test" already performs against a
    notebook compiled on a running dashboard, applied here to a local
    compile instead of one wherever the CLI happens to be invoked.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        [
            "compile", str(notebook_path), "--output", "built",
            "--smoke-test",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Smoke test: passed (GET /health responded 200)" in proc.stdout


def test_compile_command_smoke_test_works_with_a_nested_output_directory(
    tmp_path
):
    """`package_name` (output_dir's own basename) is only importable once
    output_dir's own *parent* directory is on sys.path -- true by
    coincidence for the documented default (--output "generated", a
    direct child of this process's own cwd, already on sys.path when
    `python -m backend.cli` runs it), but not for a nested --output like
    this one, whose own parent ("subdir", not the invocation directory
    itself) was never on sys.path at all before this fix.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        [
            "compile", str(notebook_path), "--output",
            str(Path("subdir") / "built"), "--smoke-test",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Smoke test: passed (GET /health responded 200)" in proc.stdout


def test_compile_command_json_flag_includes_smoke_test_field(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        [
            "compile", str(notebook_path), "--output", "built",
            "--smoke-test", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["smoke_test"] == {
        "passed": True,
        "status_code": 200,
        "detail": None,
    }


def test_compile_command_without_smoke_test_omits_the_field(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built", "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert "smoke_test" not in data
    assert "Smoke test" not in proc.stdout


def test_run_local_compile_smoke_test_fails_cleanly_when_the_compiled_app_cannot_import(
    tmp_path
):
    """A codegen bug that writes a syntactically-broken app.py must be
    reported back as a failed smoke test, not raise -- the compile itself
    already succeeded (every file is really on disk), so this is a
    diagnostic, not a fatal error. The identical case
    test_compile_smoke_test_fails_cleanly_when_the_compiled_app_cannot_import
    (tests/test_upload_routes.py) already covers for the dashboard's own
    _run_compile_smoke_test.
    """
    from backend.cli import _run_local_compile_smoke_test
    from backend.compiler import compile_notebook, package_name_for_output_dir

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    output_dir = workdir / "built"
    compile_notebook(str(notebook_path), str(output_dir))

    app_path = output_dir / "app.py"
    app_path.write_text(
        app_path.read_text(encoding="utf-8") + "\nthis is not valid python(((\n",
        encoding="utf-8",
    )

    package_name = package_name_for_output_dir(str(output_dir))
    parent_dir = str(output_dir.resolve().parent)

    assert parent_dir not in sys.path

    try:

        result = _run_local_compile_smoke_test(package_name, str(output_dir))

        assert result["passed"] is False
        assert result["status_code"] is None
        assert "failed to import" in result["detail"]

        # The parent directory this needed on sys.path to import the
        # package is removed again afterward, whether the import
        # succeeded or not -- a failed smoke test must not leave this
        # process's own sys.path permanently altered.
        assert parent_dir not in sys.path

    finally:

        # Called in-process (unlike a real `compile --smoke-test`, whose
        # own fresh-process-per-invocation guarantee is exactly what
        # _run_local_compile_smoke_test's own docstring says makes this
        # unnecessary there) -- this test's own "built" package must not
        # linger in this pytest process' sys.modules for a later test
        # that happens to reuse the same --output basename.
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                del sys.modules[name]


def _write_add_subtract_notebook(path):
    _write_notebook_with_function(
        path,
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    return a - b\n",
    )


def test_compile_command_only_compiles_just_the_named_function(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_add_subtract_notebook(notebook_path)

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built", "--only", "add"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated_app = (workdir / "built" / "app.py").read_text(encoding="utf-8")
    assert '"/add"' in generated_app
    assert '"/subtract"' not in generated_app
    assert "Generated 1 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert "POST /subtract" not in proc.stdout


def test_compile_command_exclude_omits_just_the_named_function(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_add_subtract_notebook(notebook_path)

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built", "--exclude", "subtract"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated_app = (workdir / "built" / "app.py").read_text(encoding="utf-8")
    assert '"/add"' in generated_app
    assert '"/subtract"' not in generated_app


def test_compile_command_only_accepts_a_comma_separated_list(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": (
                            "def add(a: int, b: int) -> int:\n    return a + b\n\n"
                            "def subtract(a: int, b: int) -> int:\n    return a - b\n\n"
                            "def multiply(a: int, b: int) -> int:\n    return a * b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = _run_cli(
        [
            "compile", str(notebook_path), "--output", "built",
            "--only", "add, multiply",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated_app = (workdir / "built" / "app.py").read_text(encoding="utf-8")
    assert '"/add"' in generated_app
    assert '"/multiply"' in generated_app
    assert '"/subtract"' not in generated_app


def test_compile_command_reports_a_clean_error_for_a_non_python_notebook(tmp_path):
    """Confirmed exploitable before this fix: every cell of a genuinely
    non-Python notebook (its own kernelspec.language declaring "R", say)
    simply failed is_parseable_python and was silently dropped --
    `compile` "succeeded" with zero extracted functions, producing a
    working-but-endpoint-less app with nothing anywhere explaining why it
    exposed nothing.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    notebook_path.write_text(
        json.dumps({
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "name": "ir", "display_name": "R", "language": "R",
                },
            },
            "cells": [{
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": "f <- function(x) x + 1",
            }],
        }),
        encoding="utf-8",
    )

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "not Python")


def test_compile_command_only_and_exclude_together_reports_a_clean_error(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_add_subtract_notebook(notebook_path)

    proc = _run_cli(
        [
            "compile", str(notebook_path), "--output", "built",
            "--only", "add", "--exclude", "subtract",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "can't both be given")


def test_compile_command_only_reports_a_clean_error_for_an_unknown_function(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_add_subtract_notebook(notebook_path)

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built", "--only", "nope"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "not defined in this notebook")


def test_compile_command_json_with_only_reflects_just_the_compiled_functions(tmp_path):
    """compile --json's own output must not claim an endpoint exists for a
    function --only just excluded from the actual compile -- confirmed
    this would otherwise happen, since inspect_notebook_data re-parses the
    notebook fresh with no idea --only/--exclude restricted anything.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_add_subtract_notebook(notebook_path)

    proc = _run_cli(
        [
            "compile", str(notebook_path), "--output", "built",
            "--only", "add", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)

    assert [f["name"] for f in data["functions"]] == ["add"]
    assert [ep["path"] for ep in data["endpoints"]] == ["/add"]


def test_compile_command_prints_a_summary_of_generated_endpoints(tmp_path):
    """Before this, `compile` printed a single "Compilation finished"
    line and nothing else -- seeing what had actually been generated
    (the endpoint list, which ones are background/task_id-based,
    dependencies) required a separate `inspect` call, even though
    POST /api/compile's response already returns exactly this
    information.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": (
                            "def add(a: int, b: int) -> int:\n"
                            "    return a + b\n\n"
                            "def train_model(epochs: int) -> str:\n"
                            "    return 'done'\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Generated 2 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert "POST /train_model  [background]" in proc.stdout
    # add itself must not be flagged as background.
    add_line = next(
        line for line in proc.stdout.splitlines() if line.strip() == "POST /add"
    )
    assert "[background]" not in add_line
    # No third-party imports in this notebook -- the "Dependencies:" line
    # is only printed when there's something to report.
    assert "Dependencies:" not in proc.stdout


def test_compile_command_summary_lists_third_party_dependencies(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": (
                            "import pandas as pd\n\n"
                            "def summarize(count: int) -> int:\n"
                            "    return count * 2\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Dependencies: pandas" in proc.stdout


def test_compile_command_json_flag_emits_machine_readable_output(tmp_path):
    """Before --json existed on `compile`, the only way to get a
    compile's outcome (functions, dependencies, generated_files,
    endpoints, skipped_functions) as structured data was a separate
    `inspect --json` call afterwards -- `compile` itself only ever printed
    the human-readable summary (print_compile_summary), even though
    POST /api/compile's REST response already returns exactly this kind
    of data for the same operation. Reuses inspect_notebook_data (the
    same function `inspect --json` calls) so the two can't drift, now
    reflecting the app this compile call just actually produced.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built", "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    # compile_notebook (backend/compiler.py) itself unconditionally prints
    # progress lines ("Starting compilation for: ...", "Runtime module
    # generated.", ...) on top of print_compile_summary's human-readable
    # summary -- none of that may leak onto stdout in --json mode, or a
    # script doing json.loads(stdout) would choke on it. The whole of
    # stdout must be nothing but the JSON document itself.
    data = json.loads(proc.stdout)
    assert data["functions"][0]["name"] == "add"
    assert "app.py" in data["generated_files"]
    assert "requirements.txt" in data["generated_files"]
    assert data["dependencies"] == []
    assert data["reserved_name_conflicts"] == []
    assert data["endpoints"] == [
        {"path": "/add", "method": "POST", "is_async": False}
    ]
    assert data["skipped_functions"] == []


def test_compile_command_json_flag_reports_a_background_endpoint(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": (
                            "def train_model(epochs: int) -> str:\n"
                            "    return 'done'\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built", "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["endpoints"] == [
        {"path": "/train_model", "method": "POST", "is_async": True}
    ]


def test_inspect_command_reports_the_notebooks_function(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["inspect", str(notebook_path), "--output", "built"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "add(a: int, b: int) -> int" in proc.stdout
    assert "Route: POST /add" in proc.stdout


def test_inspect_command_does_not_create_the_output_directory(tmp_path):
    """`inspect` is documented as a read-only "preview what compiling this
    notebook will do" step (see its own --help), but the dispatch branch
    handling it used to unconditionally `mkdir(parents=True,
    exist_ok=True)` on --output before ever reading anything -- so it
    left an empty directory tree on disk purely as a side effect, even
    against a notebook that had never been compiled and even for a
    multi-segment --output path that didn't exist yet.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["inspect", str(notebook_path), "--output", "some/nested/output_dir"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (workdir / "some").exists()


def test_inspect_command_json_flag_does_not_create_the_output_directory(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["inspect", str(notebook_path), "--output", "built", "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["generated_files"] == []
    assert not (workdir / "built").exists()


def test_inspect_command_still_lists_generated_files_when_the_directory_already_exists(
    tmp_path
):
    """The fix for the mkdir side effect above must not regress the
    ordinary case: `inspect` after a real `compile` still reports the
    files that compile actually wrote.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    proc = _run_cli(
        ["inspect", str(notebook_path), "--output", "built"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "app.py" in proc.stdout
    assert "requirements.txt" in proc.stdout


def test_compile_command_still_creates_the_output_directory(tmp_path):
    """Unlike `inspect`, `compile` genuinely writes output there, so its
    own mkdir (and compile_notebook_to_api's own os.makedirs, backend/
    compiler.py) must be unaffected by inspect's fix above.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "some/nested/output_dir"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "some" / "nested" / "output_dir" / "app.py").is_file()


def test_inspect_command_reports_a_functions_own_docstring(tmp_path):
    """`inspect --json` (see inspect_notebook_data) already carried a
    function's own docstring, but the plain human-readable `inspect`
    report never printed it at all.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": (
                            "def add(a: int, b: int) -> int:\n"
                            '    """Add two numbers and return their sum."""\n'
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = _run_cli(["inspect", str(notebook_path)], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Add two numbers and return their sum." in proc.stdout


def test_inspect_command_json_flag_emits_machine_readable_output(tmp_path):
    """Before --json existed, `inspect` only ever printed the
    human-readable report (inspect_notebook) -- inspect_notebook_data,
    which returns the same functions/dependencies/generated_files as
    structured data, was only ever wired up to the REST API
    (/api/inspect), never to the CLI. A script parsing `inspect`'s stdout
    had nothing but that free-form text report to work with.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    proc = _run_cli(
        ["inspect", str(notebook_path), "--output", "built", "--json"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    data = json.loads(proc.stdout)
    assert data["functions"][0]["name"] == "add"
    assert "app.py" in data["generated_files"]
    assert "requirements.txt" in data["generated_files"]
    assert data["dependencies"] == []
    assert data["reserved_name_conflicts"] == []


def test_inspect_command_reports_a_reserved_name_conflict(tmp_path):
    """`inspect` is the CLI's own preview of what `compile` will do, but
    had no idea a function named "health_check" collides with an
    identifier the generated app itself defines (see
    RESERVED_INFRASTRUCTURE_NAMES in generator/api_generator.py) until
    `compile` actually failed on it.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": "def health_check() -> dict:\n    return {}\n",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    text_proc = _run_cli(["inspect", str(notebook_path)], cwd=workdir)
    assert text_proc.returncode == 0, text_proc.stdout + text_proc.stderr
    assert "Reserved Name Conflicts" in text_proc.stdout
    assert "health_check" in text_proc.stdout

    json_proc = _run_cli(["inspect", str(notebook_path), "--json"], cwd=workdir)
    assert json_proc.returncode == 0, json_proc.stdout + json_proc.stderr
    data = json.loads(json_proc.stdout)
    assert data["reserved_name_conflicts"] == ["health_check"]


def test_export_openapi_command_writes_json_schema_by_default(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    proc = _run_cli(
        ["export-openapi", "--app-dir", "built", "--output", "built/openapi.json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    schema_path = workdir / "built" / "openapi.json"
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "/add" in schema["paths"]


def test_export_openapi_command_writes_yaml_when_requested(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    proc = _run_cli(
        [
            "export-openapi", "--app-dir", "built", "--format", "yaml",
            "--output", "built/openapi.yaml",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    yaml_path = workdir / "built" / "openapi.yaml"
    assert yaml_path.exists()
    assert "/add:" in yaml_path.read_text(encoding="utf-8")


def test_export_openapi_command_defaults_output_next_to_the_app_dir(tmp_path):
    """Confirmed broken before this fix: without an explicit --output,
    export-openapi wrote to a literal "generated/openapi.json" regardless
    of --app-dir -- so compiling into any directory other than the
    default "generated" and then exporting from it (a completely normal
    workflow) silently wrote the schema somewhere unrelated to, and
    possibly not even containing, the app it was exported from.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    proc = _run_cli(["export-openapi", "--app-dir", "built"], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    schema_path = workdir / "built" / "openapi.json"
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "/add" in schema["paths"]
    # Must not have fallen back to the old hardcoded default.
    assert not (workdir / "generated").exists()


def test_export_openapi_command_defaults_yaml_output_next_to_the_app_dir(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    proc = _run_cli(
        ["export-openapi", "--app-dir", "built", "--format", "yaml"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    yaml_path = workdir / "built" / "openapi.yaml"
    assert yaml_path.exists()
    assert not (workdir / "generated").exists()


def test_export_openapi_command_rejects_invalid_format_choice(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["export-openapi", "--format", "xml"], cwd=workdir)

    assert proc.returncode != 0
    assert "invalid choice: 'xml'" in proc.stderr


def test_export_sdk_command_writes_python_client_by_default(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    openapi_proc = _run_cli(
        ["export-openapi", "--app-dir", "built", "--output", "built/openapi.json"],
        cwd=workdir,
    )
    assert openapi_proc.returncode == 0, openapi_proc.stdout + openapi_proc.stderr

    proc = _run_cli(
        [
            "export-sdk", "--openapi", "built/openapi.json",
            "--output", "built/sdk/python_client.py",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    client_path = workdir / "built" / "sdk" / "python_client.py"
    assert client_path.exists()
    client_source = client_path.read_text(encoding="utf-8")
    assert "class NotebookAPIClient" in client_source
    assert "def add(" in client_source


def test_export_sdk_command_writes_typescript_client_when_requested(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    openapi_proc = _run_cli(
        ["export-openapi", "--app-dir", "built", "--output", "built/openapi.json"],
        cwd=workdir,
    )
    assert openapi_proc.returncode == 0, openapi_proc.stdout + openapi_proc.stderr

    proc = _run_cli(
        [
            "export-sdk", "--openapi", "built/openapi.json", "--language", "typescript",
            "--output", "built/sdk/typescript_client.ts",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    client_path = workdir / "built" / "sdk" / "typescript_client.ts"
    assert client_path.exists()
    assert "class NotebookAPIClient" in client_path.read_text(encoding="utf-8")


def test_export_sdk_command_defaults_output_next_to_the_openapi_file(tmp_path):
    """Confirmed broken before this fix: without an explicit --output,
    export-sdk wrote to a literal "generated/sdk/python_client.py"
    regardless of where --openapi actually pointed -- so exporting an SDK
    from a schema compiled/exported anywhere other than the default
    "generated" (a completely normal workflow) silently wrote the client
    somewhere unrelated to the schema it was generated from.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    openapi_proc = _run_cli(
        ["export-openapi", "--app-dir", "built", "--output", "built/openapi.json"],
        cwd=workdir,
    )
    assert openapi_proc.returncode == 0, openapi_proc.stdout + openapi_proc.stderr

    proc = _run_cli(
        ["export-sdk", "--openapi", "built/openapi.json"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    client_path = workdir / "built" / "sdk" / "python_client.py"
    assert client_path.exists()
    assert "def add(" in client_path.read_text(encoding="utf-8")
    # Must not have fallen back to the old hardcoded default.
    assert not (workdir / "generated").exists()


def test_export_sdk_command_defaults_typescript_output_next_to_the_openapi_file(
    tmp_path
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    openapi_proc = _run_cli(
        ["export-openapi", "--app-dir", "built", "--output", "built/openapi.json"],
        cwd=workdir,
    )
    assert openapi_proc.returncode == 0, openapi_proc.stdout + openapi_proc.stderr

    proc = _run_cli(
        ["export-sdk", "--openapi", "built/openapi.json", "--language", "typescript"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    client_path = workdir / "built" / "sdk" / "typescript_client.ts"
    assert client_path.exists()
    assert not (workdir / "generated").exists()


def test_export_sdk_command_app_dir_locates_the_matching_openapi_export(tmp_path):
    """Confirmed broken before this fix: --openapi's default was a flat
    "generated/openapi.json" literal with no --app-dir concept at all --
    unlike export-openapi, which already derives its own default --output
    from --app-dir. `export-sdk --app-dir built` (mirroring `compile
    --output built` + `export-openapi --app-dir built`, an entirely
    normal non-default workflow) crashed looking for a nonexistent
    generated/openapi.json instead of finding built/openapi.json, the
    schema that command's own prerequisite step just wrote.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    openapi_proc = _run_cli(["export-openapi", "--app-dir", "built"], cwd=workdir)
    assert openapi_proc.returncode == 0, openapi_proc.stdout + openapi_proc.stderr

    proc = _run_cli(["export-sdk", "--app-dir", "built"], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    client_path = workdir / "built" / "sdk" / "python_client.py"
    assert client_path.exists()
    assert "def add(" in client_path.read_text(encoding="utf-8")


def test_export_sdk_command_does_not_silently_use_a_stale_default_app_dir_export(
    tmp_path
):
    """The worse failure mode this fix closes: without --app-dir/--openapi
    pointing export-sdk at the right schema, it fell back to a flat
    "generated/openapi.json" literal -- so if an *unrelated* notebook had
    ever been compiled into the default "generated" directory and
    exported there too, export-sdk silently generated a client for that
    stale, unrelated schema instead, with no error or warning at all.
    Confirmed reproduced: compiling two different notebooks into "built"
    and "generated" respectively and exporting both schemas, then running
    `export-sdk --app-dir built` produced a client exposing "add" (the
    "built" notebook actually being worked with), not "multiply" (the
    unrelated notebook that happened to be sitting in the default
    "generated" directory).
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    built_notebook = workdir / "nb_add.ipynb"
    _write_notebook_with_function(
        built_notebook, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    stale_notebook = workdir / "nb_multiply.ipynb"
    _write_notebook_with_function(
        stale_notebook,
        "def multiply(a: int, b: int) -> int:\n    return a * b\n",
    )

    # The unrelated notebook compiled and exported into the *default*
    # "generated" directory first -- simulating a previous, unrelated
    # `compile`/`export-openapi` run against this same working directory.
    compile_stale = _run_cli(
        ["compile", str(stale_notebook), "--output", "generated"], cwd=workdir
    )
    assert compile_stale.returncode == 0, compile_stale.stdout + compile_stale.stderr
    openapi_stale = _run_cli(["export-openapi", "--app-dir", "generated"], cwd=workdir)
    assert openapi_stale.returncode == 0, openapi_stale.stdout + openapi_stale.stderr

    # The notebook actually being worked with now, compiled and exported
    # into a different directory.
    compile_built = _run_cli(
        ["compile", str(built_notebook), "--output", "built"], cwd=workdir
    )
    assert compile_built.returncode == 0, compile_built.stdout + compile_built.stderr
    openapi_built = _run_cli(["export-openapi", "--app-dir", "built"], cwd=workdir)
    assert openapi_built.returncode == 0, openapi_built.stdout + openapi_built.stderr

    proc = _run_cli(["export-sdk", "--app-dir", "built"], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    client_source = (workdir / "built" / "sdk" / "python_client.py").read_text(
        encoding="utf-8"
    )
    assert "def add(" in client_source
    assert "def multiply(" not in client_source


def test_export_sdk_command_rejects_invalid_language_choice(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["export-sdk", "--language", "rust"], cwd=workdir)

    assert proc.returncode != 0
    assert "invalid choice: 'rust'" in proc.stderr


def test_export_openapi_command_json_flag_emits_machine_readable_output(tmp_path):
    """Before --json existed on `export-openapi`, the only way to get its
    outcome was reading the schema file it wrote back off disk -- the
    command itself only ever printed a single human-readable "OpenAPI
    schema written to ..." line (export_openapi_schema's own print), even
    though POST /api/export-openapi's REST response already returns the
    schema inline as structured data for the same operation.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    proc = _run_cli(
        [
            "export-openapi", "--app-dir", "built",
            "--output", "built/openapi.json", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    # export_openapi_schema unconditionally prints its own "OpenAPI schema
    # written to ..." progress line -- none of that may leak onto stdout
    # in --json mode, or a script doing json.loads(stdout) would choke on
    # it. The whole of stdout must be nothing but the JSON document
    # itself.
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert data["format"] == "json"
    assert data["path"] == "built/openapi.json"
    assert "/add" in data["schema"]["paths"]


def test_export_openapi_command_json_flag_reports_yaml_content_inline(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    proc = _run_cli(
        [
            "export-openapi", "--app-dir", "built", "--format", "yaml",
            "--output", "built/openapi.yaml", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert data["format"] == "yaml"
    assert "/add:" in data["content"]
    assert "schema" not in data


def test_export_sdk_command_json_flag_emits_machine_readable_output(tmp_path):
    """Same gap as `export-openapi --json` above, for `export-sdk`:
    generate_python_sdk/generate_typescript_sdk only ever print a single
    "Python SDK generated at ..."/"TypeScript SDK generated at ..." line,
    even though POST /api/export-sdk's REST response already returns the
    generated client source inline for the same operation.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    openapi_proc = _run_cli(
        ["export-openapi", "--app-dir", "built", "--output", "built/openapi.json"],
        cwd=workdir,
    )
    assert openapi_proc.returncode == 0, openapi_proc.stdout + openapi_proc.stderr

    proc = _run_cli(
        [
            "export-sdk", "--openapi", "built/openapi.json",
            "--output", "built/sdk/python_client.py", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert data["language"] == "python"
    assert data["path"] == "built/sdk/python_client.py"
    assert "class NotebookAPIClient" in data["code"]
    assert "def add(" in data["code"]


def test_export_sdk_command_json_flag_reports_typescript_language(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    compile_proc = _run_cli(
        ["compile", str(notebook_path), "--output", "built"], cwd=workdir
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    openapi_proc = _run_cli(
        ["export-openapi", "--app-dir", "built", "--output", "built/openapi.json"],
        cwd=workdir,
    )
    assert openapi_proc.returncode == 0, openapi_proc.stdout + openapi_proc.stderr

    proc = _run_cli(
        [
            "export-sdk", "--openapi", "built/openapi.json", "--language", "typescript",
            "--output", "built/sdk/typescript_client.ts", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert data["language"] == "typescript"
    assert "class NotebookAPIClient" in data["code"]


def _assert_clean_cli_error(proc, expected_message_fragment):
    """A core command's expected failure modes (missing file, invalid
    notebook, etc.) must produce a single-line "Error: ..." message on
    stderr with exit code 1 -- not a raw multi-frame Python traceback.
    Before CLI_USER_FACING_ERRORS existed, every one of these scenarios
    dumped a full traceback instead (confirmed by running each command
    directly against a missing/invalid notebook).
    """

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr
    assert expected_message_fragment in proc.stderr
    assert any(
        line.startswith("Error: ") for line in proc.stderr.splitlines()
    )


def test_compile_command_reports_a_clean_error_for_a_missing_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["compile", str(workdir / "does-not-exist.ipynb"), "--output", "built"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_compile_command_reports_a_clean_error_for_an_invalid_package_name(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["compile", str(notebook_path), "--output", "not-a-valid-package!"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "can't be used as a Python package name")


def test_compile_command_reports_a_clean_error_for_a_non_notebook_file(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    bad_notebook = workdir / "bad.ipynb"
    bad_notebook.write_text("this is not json at all", encoding="utf-8")

    proc = _run_cli(["compile", str(bad_notebook), "--output", "built"], cwd=workdir)

    _assert_clean_cli_error(proc, "does not appear to be JSON")


def test_inspect_command_reports_a_clean_error_for_a_missing_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["inspect", str(workdir / "does-not-exist.ipynb")], cwd=workdir
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_export_openapi_command_reports_a_clean_error_when_nothing_compiled(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["export-openapi", "--app-dir", "nope"], cwd=workdir)

    _assert_clean_cli_error(proc, "No module named 'nope'")


def test_export_sdk_command_reports_a_clean_error_for_a_missing_openapi_file(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-sdk", "--openapi", "missing-openapi.json"], cwd=workdir
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_export_sdk_command_hints_at_a_yaml_export_when_the_json_default_is_missing(
    tmp_path,
):
    """Confirmed exploitable before this fix: `export-sdk --openapi
    generated/openapi.json` (its own documented default) against a
    notebook only ever exported as yaml -- via `export-openapi --format
    yaml` -- crashed with a bare FileNotFoundError ("No such file or
    directory: 'openapi.json'"), giving no indication that the export the
    caller actually ran wrote openapi.yaml right next to it. Now falls
    back to reading that sibling file, which still can't be turned into
    an SDK (export-sdk only reads JSON schemas) but surfaces
    _load_openapi_schema's own specific hint (exporters/sdk_generator.py)
    instead of the generic OS error.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "openapi.yaml").write_text(
        "openapi: 3.0.0\ninfo:\n  title: Test\n  version: '1.0'\npaths: {}\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        ["export-sdk", "--openapi", "openapi.json"], cwd=workdir
    )

    _assert_clean_cli_error(proc, "This looks like a YAML export")


def test_export_sdk_command_still_reports_a_clean_missing_file_error_with_no_yaml_sibling(
    tmp_path,
):
    """The fallback above must not mask a genuinely missing export --
    with no openapi.json *or* a sibling openapi.yaml/.yml anywhere to
    fall back to, this must behave exactly as before: a clean "No such
    file or directory" error, not a confusing reference to a yaml file
    that doesn't exist either.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-sdk", "--openapi", "openapi.json"], cwd=workdir
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_export_sdk_command_reports_a_clean_error_for_a_corrupt_openapi_file(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "openapi.json").write_text("not valid json", encoding="utf-8")

    proc = _run_cli(
        ["export-sdk", "--openapi", "openapi.json"], cwd=workdir
    )

    _assert_clean_cli_error(proc, "Expecting value")


def test_serve_command_reports_a_clean_error_for_a_missing_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["serve", str(workdir / "does-not-exist.ipynb")], cwd=workdir
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_watch_command_is_registered():
    """`watch` (like `deploy` -- see test_deploy_command_is_registered in
    test_cli_deploy.py) needs both a subparsers.add_parser("watch", ...)
    call and a matching `elif args.command == "watch":` dispatch branch in
    _dispatch_core_command -- one without the other either makes argparse
    reject "watch" outright, or dispatches successfully into a command
    that was never actually declared. Exercised through the real
    `backend.cli` argparse entry point rather than calling
    watch_notebook directly, the same gap test_compile_command_writes_the_generated_app's
    own docstring documents for the other core commands.
    """

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "watch" in proc.stdout


def test_watch_command_reports_a_clean_error_for_a_missing_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["watch", str(workdir / "does-not-exist.ipynb")], cwd=workdir
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_watch_command_requires_a_notebook_argument():

    proc = _run_cli(["watch"], cwd=Path.cwd())

    assert proc.returncode != 0
    assert "notebook" in proc.stderr


def test_serve_command_accepts_only_and_exclude_flags(tmp_path):
    """`serve` (and `watch`, below) previously had no --only/--exclude at
    all, unlike `compile`/`deploy` -- confirmed here by checking argparse
    itself accepts the flags (reaching the missing-notebook error, not an
    "unrecognized arguments" one), the same wiring-only check
    test_watch_command_is_registered already applies to the subcommand
    itself.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["serve", str(workdir / "does-not-exist.ipynb"), "--only", "add"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_serve_command_only_and_exclude_are_mutually_exclusive(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    proc = _run_cli(
        [
            "serve", str(notebook_path),
            "--only", "add", "--exclude", "add", "--port", "0",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "only and exclude can't both be given")


def test_watch_command_accepts_only_and_exclude_flags(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["watch", str(workdir / "does-not-exist.ipynb"), "--exclude", "helper"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_watch_command_only_and_exclude_are_mutually_exclusive(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    proc = _run_cli(
        ["watch", str(notebook_path), "--only", "add", "--exclude", "add"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "only and exclude can't both be given")


def test_serve_command_accepts_debounce_flag(tmp_path):
    """Before this, the debounce window between recompiles was hardcoded
    in NotebookChangeHandler (backend/serve.py) with no CLI flag at all --
    confirmed here by checking argparse itself accepts --debounce
    (reaching the missing-notebook error, not an "unrecognized
    arguments" one), the same wiring-only check
    test_serve_command_accepts_only_and_exclude_flags already applies to
    --only/--exclude.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["serve", str(workdir / "does-not-exist.ipynb"), "--debounce", "2.5"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_serve_command_rejects_a_negative_debounce(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    proc = _run_cli(
        ["serve", str(notebook_path), "--debounce", "-1"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "--debounce must be zero or positive")


def test_watch_command_accepts_debounce_flag(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["watch", str(workdir / "does-not-exist.ipynb"), "--debounce", "0.1"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_watch_command_rejects_a_negative_debounce(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    proc = _run_cli(
        ["watch", str(notebook_path), "--debounce", "-0.5"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "--debounce must be zero or positive")


def test_serve_command_accepts_on_change_flag(tmp_path):
    """Wiring-only check, mirroring test_serve_command_accepts_debounce_flag
    above: argparse itself accepts --on-change (reaching the
    missing-notebook error, not an "unrecognized arguments" one).
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "serve", str(workdir / "does-not-exist.ipynb"),
            "--on-change", "pytest -x",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_watch_command_accepts_on_change_flag(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "watch", str(workdir / "does-not-exist.ipynb"),
            "--on-change", "pytest -x",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_diff_command_is_registered():
    """Same subparser/dispatch-branch mismatch gap test_deploy_command_is_registered
    (test_cli_deploy.py) and test_watch_command_is_registered (above)
    already guard against for `deploy`/`watch`.
    """

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "diff" in proc.stdout


def test_diff_command_reports_added_removed_and_changed_functions(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    old_path = workdir / "old.ipynb"
    new_path = workdir / "new.ipynb"

    _write_notebook_with_function(
        old_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )
    _write_notebook_with_function(
        new_path,
        "def add(a: int, b: int, c: int = 0) -> int:\n    return a + b + c\n\n"
        "def multiply(a: int, b: int) -> int:\n    return a * b\n",
    )

    proc = _run_cli(["diff", str(old_path), str(new_path)], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Added 1 endpoint(s):" in proc.stdout
    assert "POST /multiply" in proc.stdout
    assert "Removed 1 endpoint(s):" in proc.stdout
    assert "POST /subtract" in proc.stdout
    assert "Changed 1 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout


def test_diff_command_reports_no_changes_for_identical_notebooks(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["diff", str(notebook_path), str(notebook_path)], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No changes to the compiled API surface." in proc.stdout


def test_diff_command_json_flag_emits_machine_readable_output(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    old_path = workdir / "old.ipynb"
    new_path = workdir / "new.ipynb"
    _write_notebook_with_function(
        old_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    _write_notebook_with_function(
        new_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n    return a * b\n",
    )

    proc = _run_cli(
        ["diff", str(old_path), str(new_path), "--json"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)

    assert [f["name"] for f in data["added"]] == ["multiply"]
    assert data["removed"] == []
    assert data["changed"] == []
    assert data["unchanged"] == ["add"]
    assert data["compatible"] is True
    assert data["breaking_changes"] == []


def test_diff_command_content_flag_prints_a_line_level_diff(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    old_path = workdir / "old.ipynb"
    new_path = workdir / "new.ipynb"

    # A body-only edit -- the compiled API surface (signature) is
    # unchanged, so the structural report alone shows nothing, but the
    # actual code did change.
    _write_notebook_with_function(
        old_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    _write_notebook_with_function(
        new_path, "def add(a: int, b: int) -> int:\n    return a + b + 1\n"
    )

    proc = _run_cli(
        ["diff", str(old_path), str(new_path), "--content"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No changes to the compiled API surface." in proc.stdout
    assert "-    return a + b" in proc.stdout
    assert "+    return a + b + 1" in proc.stdout


def test_diff_command_content_flag_json_output_includes_content_diff(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    old_path = workdir / "old.ipynb"
    new_path = workdir / "new.ipynb"

    _write_notebook_with_function(
        old_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    _write_notebook_with_function(
        new_path, "def add(a: int, b: int) -> int:\n    return a + b + 1\n"
    )

    proc = _run_cli(
        ["diff", str(old_path), str(new_path), "--content", "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert any("return a + b + 1" in line for line in data["content_diff"])


def test_diff_command_without_content_flag_omits_content_diff(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    old_path = workdir / "old.ipynb"
    new_path = workdir / "new.ipynb"
    _write_notebook_with_function(
        old_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    _write_notebook_with_function(
        new_path, "def add(a: int, b: int) -> int:\n    return a + b + 1\n"
    )

    proc = _run_cli(["diff", str(old_path), str(new_path), "--json"], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert "content_diff" not in data


def test_diff_command_fail_on_breaking_exits_nonzero_for_a_breaking_change(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    old_path = workdir / "old.ipynb"
    new_path = workdir / "new.ipynb"
    _write_notebook_with_function(
        old_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    _write_notebook_with_function(
        new_path, "def add(a: int, b: int, c: int) -> int:\n    return a + b + c\n"
    )

    proc = _run_cli(
        ["diff", str(old_path), str(new_path), "--fail-on-breaking"], cwd=workdir
    )

    assert proc.returncode == 1
    assert "breaking change(s)" in proc.stdout
    assert "New required parameter 'c' was added to 'add'." in proc.stdout


def test_diff_command_fail_on_breaking_exits_zero_when_compatible(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    old_path = workdir / "old.ipynb"
    new_path = workdir / "new.ipynb"
    _write_notebook_with_function(
        old_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    _write_notebook_with_function(
        new_path,
        "def add(a: int, b: int, c: int = 0) -> int:\n    return a + b + c\n",
    )

    proc = _run_cli(
        ["diff", str(old_path), str(new_path), "--fail-on-breaking"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No breaking changes to the compiled API's contract." in proc.stdout


def test_diff_command_reports_a_clean_error_for_a_missing_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["diff", str(notebook_path), str(workdir / "does-not-exist.ipynb")],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_diff_command_requires_both_notebook_arguments(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(["diff", str(notebook_path)], cwd=workdir)

    assert proc.returncode != 0
    assert "new_notebook" in proc.stderr


def test_export_curl_command_is_registered():
    """Same subparser/dispatch-branch mismatch gap test_deploy_command_is_registered
    (test_cli_deploy.py) and test_watch_command_is_registered (above)
    already guard against for `deploy`/`watch`.
    """

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "export-curl" in proc.stdout


def test_export_curl_command_writes_a_script_with_a_command_per_function(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    proc = _run_cli(["export-curl", str(notebook_path)], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cURL script written to: requests.sh (2 request(s))" in proc.stdout

    script = (workdir / "requests.sh").read_text(encoding="utf-8")
    assert "curl -X POST http://localhost:8000/add" in script
    assert "curl -X POST http://localhost:8000/subtract" in script
    assert "X-API-Key: notebook-to-api-dev-key" in script


def test_export_curl_command_respects_host_port_api_key_and_output(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    proc = _run_cli(
        [
            "export-curl", str(notebook_path),
            "--host", "api.example.com", "--port", "9000",
            "--api-key", "mykey123", "--output", "custom.sh",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    script = (workdir / "custom.sh").read_text(encoding="utf-8")
    assert "curl -X POST http://api.example.com:9000/add" in script
    assert "X-API-Key: mykey123" in script
    assert not (workdir / "requests.sh").exists()


def test_export_curl_command_json_flag_emits_machine_readable_output(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    proc = _run_cli(
        ["export-curl", str(notebook_path), "--json"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)

    assert data["status"] == "success"
    assert data["path"] == "requests.sh"
    assert len(data["commands"]) == 1
    assert "curl -X POST http://localhost:8000/add" in data["commands"][0]


def test_export_curl_command_reports_a_clean_error_for_a_missing_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-curl", str(workdir / "does-not-exist.ipynb")], cwd=workdir
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_export_curl_command_only_restricts_to_the_named_functions(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    proc = _run_cli(
        ["export-curl", str(notebook_path), "--only", "add"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "(1 request(s))" in proc.stdout

    script = (workdir / "requests.sh").read_text(encoding="utf-8")
    assert "curl -X POST http://localhost:8000/add" in script
    assert "curl -X POST http://localhost:8000/subtract" not in script


def test_export_curl_command_rejects_only_and_exclude_together(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    proc = _run_cli(
        [
            "export-curl", str(notebook_path),
            "--only", "add", "--exclude", "add",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "only and exclude can't both be given")


def test_export_postman_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "export-postman" in proc.stdout


def test_export_postman_command_writes_a_collection_with_an_item_per_function(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    proc = _run_cli(["export-postman", str(notebook_path)], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "Postman collection written to: postman_collection.json (2 request(s))"
        in proc.stdout
    )

    collection = json.loads(
        (workdir / "postman_collection.json").read_text(encoding="utf-8")
    )
    assert [item["name"] for item in collection["item"]] == ["add", "subtract"]
    assert collection["info"]["name"] == "nb"


def test_export_postman_command_respects_host_port_api_key_name_and_output(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    proc = _run_cli(
        [
            "export-postman", str(notebook_path),
            "--host", "api.example.com", "--port", "9000",
            "--api-key", "mykey123", "--collection-name", "My API",
            "--output", "custom.json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    collection = json.loads((workdir / "custom.json").read_text(encoding="utf-8"))
    assert collection["info"]["name"] == "My API"
    variables = {v["key"]: v["value"] for v in collection["variable"]}
    assert variables["base_url"] == "http://api.example.com:9000"
    assert variables["api_key"] == "mykey123"
    assert not (workdir / "postman_collection.json").exists()


def test_export_postman_command_json_flag_emits_machine_readable_output(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    proc = _run_cli(
        ["export-postman", str(notebook_path), "--json"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)

    assert data["status"] == "success"
    assert data["path"] == "postman_collection.json"
    assert [item["name"] for item in data["collection"]["item"]] == ["add"]


def test_export_postman_command_reports_a_clean_error_for_a_missing_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-postman", str(workdir / "does-not-exist.ipynb")], cwd=workdir
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_export_postman_command_only_restricts_to_the_named_functions(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    proc = _run_cli(
        ["export-postman", str(notebook_path), "--only", "add"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "(1 request(s))" in proc.stdout

    collection = json.loads(
        (workdir / "postman_collection.json").read_text(encoding="utf-8")
    )
    assert [item["name"] for item in collection["item"]] == ["add"]


def test_export_postman_command_rejects_only_and_exclude_together(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    proc = _run_cli(
        [
            "export-postman", str(notebook_path),
            "--only", "add", "--exclude", "add",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "only and exclude can't both be given")


def test_remote_curl_command_only_restricts_to_the_named_functions(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n\n"
                "def subtract(a: int, b: int) -> int:\n    return a - b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-curl", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--only", "add",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    script = (workdir / "requests.sh").read_text(encoding="utf-8")
    assert "curl -X POST http://localhost:8000/add" in script
    assert "curl -X POST http://localhost:8000/subtract" not in script


def _json_response(status_code, body):
    """Queue-entry helper for _FakeDashboardHandler.responses: a JSON
    body, encoded and content-typed the way every real dashboard JSON
    response already is.
    """
    return (status_code, json.dumps(body).encode("utf-8"), "application/json")


def _raw_response(status_code, content, content_type="application/x-ipynb+json"):
    """Queue-entry helper for _FakeDashboardHandler.responses: raw bytes,
    the same shape GET /api/notebooks/{filename}'s own FileResponse
    actually returns (a notebook's content, not a JSON envelope).
    """
    return (status_code, content, content_type)


class _FakeDashboardHandler(http.server.BaseHTTPRequestHandler):
    """A minimal stand-in for a running dashboard, used only to exercise
    the `upload`/`list`/`download` CLI commands' own HTTP handling
    (request construction, success/error response handling,
    connection-failure handling) -- not to re-verify the dashboard's own
    route behavior (multipart parsing, validation, atomic writes, the
    actual notebook listing/lookup, ...), which is already exhaustively
    covered directly in tests/test_upload_routes.py.

    `responses` is consumed FIFO, one (status_code, payload_bytes,
    content_type) entry per request received (see _json_response/
    _raw_response above); `requests` records each request's raw path
    (including its query string) so a test can confirm e.g. "search" or
    "overwrite" was actually passed through.

    `response_headers` is an optional, separately-consumed FIFO queue of
    extra {header_name: value} dicts, one per request, for tests that
    need to control a response header `responses`' own (status_code,
    payload, content_type) shape has no room for -- e.g. Content-
    Disposition, which GET /api/download's own remote-build CLI command
    reads to pick a default --output filename. A response with nothing
    queued here just gets no extra headers, the same as before this
    existed.
    """

    responses = []
    requests = []
    bodies = []
    response_headers = []

    def _handle(self):

        if self.command in ("POST", "PATCH", "PUT"):
            content_length = int(self.headers.get("Content-Length", 0))
            type(self).bodies.append(self.rfile.read(content_length))
        else:
            type(self).bodies.append(b"")

        type(self).requests.append(self.path)

        status_code, payload, content_type = type(self).responses.pop(0)

        extra_headers = (
            type(self).response_headers.pop(0)
            if type(self).response_headers else {}
        )

        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for header_name, value in extra_headers.items():
            self.send_header(header_name, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        self._handle()

    def do_GET(self):
        self._handle()

    def do_DELETE(self):
        self._handle()

    def do_PATCH(self):
        self._handle()

    def do_PUT(self):
        self._handle()

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_dashboard():
    _FakeDashboardHandler.responses = []
    _FakeDashboardHandler.requests = []
    _FakeDashboardHandler.bodies = []
    _FakeDashboardHandler.response_headers = []

    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeDashboardHandler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        yield f"http://127.0.0.1:{port}", _FakeDashboardHandler
    finally:
        server.shutdown()
        server_thread.join(timeout=5)


def test_upload_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "upload" in proc.stdout


def test_upload_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb",
            "overwritten": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["upload", str(notebook_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Uploaded 'nb.ipynb'" in proc.stdout
    assert "overwritten: False" in proc.stdout
    assert handler.requests == ["/api/upload?overwrite=false"]


def test_upload_command_notes_when_it_overwrote_the_compiled_notebook(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb",
            "overwritten": True,
            "was_currently_compiled": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["upload", str(notebook_path), "--overwrite", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "note: this was the notebook backing the currently compiled app." in proc.stdout


def test_upload_command_passes_the_overwrite_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb", "overwritten": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        [
            "upload", str(notebook_path),
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/upload?overwrite=true"]


def test_upload_command_dry_run_passes_the_flag_through_and_prints_would_upload(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb", "overwritten": False,
            "sha256": "abc123", "dry_run": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["upload", str(notebook_path), "--dashboard-url", dashboard_url, "--dry-run"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/upload?overwrite=false&dry_run=true"]
    assert "Would upload 'nb.ipynb'" in proc.stdout


def test_upload_command_with_multiple_notebooks_dry_run_passes_the_flag_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [
                {"filename": "a.ipynb", "status": "success", "overwritten": False},
                {"filename": "b.ipynb", "status": "success", "overwritten": False},
            ],
            "succeeded_count": 2, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_a = workdir / "a.ipynb"
    notebook_b = workdir / "b.ipynb"
    _write_notebook(notebook_a)
    _write_notebook(notebook_b)

    proc = _run_cli(
        [
            "upload", str(notebook_a), str(notebook_b),
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/upload/batch?overwrite=false&dry_run=true"]
    assert "Would upload 'a.ipynb'" in proc.stdout
    assert "Would upload 'b.ipynb'" in proc.stdout


def test_upload_command_reports_the_sha256(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb", "overwritten": False,
            "sha256": "a" * 64,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["upload", str(notebook_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"sha256: {'a' * 64}" in proc.stdout


def test_upload_command_passes_the_expected_sha256_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb", "overwritten": False,
            "sha256": "a" * 64,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        [
            "upload", str(notebook_path),
            "--dashboard-url", dashboard_url, "--expected-sha256", "a" * 64,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        f"/api/upload?overwrite=false&expected_sha256={'a' * 64}"
    ]


def test_upload_command_reports_a_clean_error_for_an_expected_sha256_mismatch(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {
            "detail": "Uploaded content does not match expected_sha256: expected a, got b",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        [
            "upload", str(notebook_path),
            "--dashboard-url", dashboard_url, "--expected-sha256", "a",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "does not match expected_sha256")


def test_upload_command_rejects_expected_sha256_with_multiple_notebooks(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_a = workdir / "a.ipynb"
    notebook_b = workdir / "b.ipynb"
    _write_notebook(notebook_a)
    _write_notebook(notebook_b)

    proc = _run_cli(
        [
            "upload", str(notebook_a), str(notebook_b),
            "--expected-sha256", "a" * 64,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "single notebook")


def test_upload_command_passes_the_tags_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb", "overwritten": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        [
            "upload", str(notebook_path),
            "--dashboard-url", dashboard_url, "--tags", "prod,reviewed",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/upload?overwrite=false&tags=prod%2Creviewed"]


def test_upload_command_passes_the_description_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb", "overwritten": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        [
            "upload", str(notebook_path),
            "--dashboard-url", dashboard_url, "--description", "adds two numbers",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["description"] == ["adds two numbers"]


def test_upload_command_omits_the_description_query_param_by_default(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb", "overwritten": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["upload", str(notebook_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/upload?overwrite=false"]


def test_upload_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb", "overwritten": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["upload", str(notebook_path), "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == {
        "status": "success", "filename": "nb.ipynb",
        "path": "/srv/uploads/nb.ipynb", "overwritten": False,
    }


def test_upload_command_reports_a_clean_error_for_a_rejected_upload(
    tmp_path, fake_dashboard
):
    """A 409 (same-name collision without --overwrite), or any other
    non-2xx the dashboard returns, must surface as the same clean
    "Error: ..." single-line message every other core command's expected
    failure modes already get, using the dashboard's own {"detail": ...}
    body -- not a raw HTTP response dump.
    """

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(409, {"detail": "A notebook named 'nb.ipynb' already exists."})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        ["upload", str(notebook_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "already exists")


def test_upload_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    proc = _run_cli(
        [
            "upload", str(notebook_path),
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_upload_command_reports_a_clean_error_for_a_missing_notebook(tmp_path):
    """The local file must be checked before ever attempting to reach the
    dashboard -- no server is running for this test at all, so a
    connection-error message here (instead of the missing-file one) would
    mean the CLI tried to open a request before validating its own input.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "upload", str(workdir / "does-not-exist.ipynb"),
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_upload_command_with_multiple_notebooks_hits_the_batch_endpoint(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "status": "success", "filename": "a.ipynb",
                    "path": "/srv/uploads/a.ipynb", "overwritten": False,
                },
                {
                    "status": "success", "filename": "b.ipynb",
                    "path": "/srv/uploads/b.ipynb", "overwritten": False,
                },
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_a = workdir / "a.ipynb"
    notebook_b = workdir / "b.ipynb"
    _write_notebook(notebook_a)
    _write_notebook(notebook_b)

    proc = _run_cli(
        [
            "upload", str(notebook_a), str(notebook_b),
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Uploaded 'a.ipynb'" in proc.stdout
    assert "Uploaded 'b.ipynb'" in proc.stdout
    assert "2 succeeded, 0 failed." in proc.stdout
    assert handler.requests == ["/api/upload/batch?overwrite=false"]


def test_upload_command_with_multiple_notebooks_reports_per_file_failures(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "status": "success", "filename": "a.ipynb",
                    "path": "/srv/uploads/a.ipynb", "overwritten": False,
                },
                {
                    "status": "error", "filename": "b.ipynb",
                    "detail": "A notebook named 'b.ipynb' already exists.",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_a = workdir / "a.ipynb"
    notebook_b = workdir / "b.ipynb"
    _write_notebook(notebook_a)
    _write_notebook(notebook_b)

    proc = _run_cli(
        [
            "upload", str(notebook_a), str(notebook_b),
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Uploaded 'a.ipynb'" in proc.stdout
    assert "Failed 'b.ipynb': A notebook named 'b.ipynb' already exists." in proc.stdout
    assert "1 succeeded, 1 failed." in proc.stdout


def test_upload_command_with_multiple_notebooks_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "status": "success", "filename": "a.ipynb",
                    "path": "/srv/uploads/a.ipynb", "overwritten": False,
                },
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_a = workdir / "a.ipynb"
    _write_notebook(notebook_a)

    proc = _run_cli(
        [
            "upload", str(notebook_a),
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["succeeded_count"] == 1


def test_upload_command_with_multiple_notebooks_passes_the_overwrite_flag_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_a = workdir / "a.ipynb"
    notebook_b = workdir / "b.ipynb"
    _write_notebook(notebook_a)
    _write_notebook(notebook_b)

    proc = _run_cli(
        [
            "upload", str(notebook_a), str(notebook_b),
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/upload/batch?overwrite=true"]


def test_upload_command_with_multiple_notebooks_passes_the_tags_flag_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_a = workdir / "a.ipynb"
    notebook_b = workdir / "b.ipynb"
    _write_notebook(notebook_a)
    _write_notebook(notebook_b)

    proc = _run_cli(
        [
            "upload", str(notebook_a), str(notebook_b),
            "--dashboard-url", dashboard_url, "--tags", "prod",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/upload/batch?overwrite=false&tags=prod"]


def test_upload_command_with_multiple_notebooks_passes_the_description_flag_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_a = workdir / "a.ipynb"
    notebook_b = workdir / "b.ipynb"
    _write_notebook(notebook_a)
    _write_notebook(notebook_b)

    proc = _run_cli(
        [
            "upload", str(notebook_a), str(notebook_b),
            "--dashboard-url", dashboard_url, "--description", "batch-uploaded",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["description"] == ["batch-uploaded"]


def test_upload_command_with_multiple_notebooks_reports_a_clean_error_for_a_missing_notebook(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_a = workdir / "a.ipynb"
    _write_notebook(notebook_a)

    proc = _run_cli(
        [
            "upload", str(notebook_a), str(workdir / "does-not-exist.ipynb"),
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_import_url_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "import-url" in proc.stdout


def test_import_url_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "path": "/srv/uploads/nb.ipynb",
            "overwritten": False,
            "sha256": "a" * 64,
            "dry_run": False,
            "source_url": "https://example.com/nb.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "import-url", "https://example.com/nb.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Imported 'nb.ipynb' from https://example.com/nb.ipynb" in proc.stdout
    assert "overwritten: False" in proc.stdout
    assert handler.requests == ["/api/notebooks/import-url"]

    body = json.loads(handler.bodies[0])
    assert body == {"url": "https://example.com/nb.ipynb", "overwrite": False}


def test_import_url_command_passes_optional_fields_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "custom.ipynb",
            "path": "/srv/uploads/custom.ipynb",
            "overwritten": True,
            "sha256": "b" * 64,
            "dry_run": True,
            "source_url": "https://example.com/nb.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "import-url", "https://example.com/nb.ipynb",
            "--dashboard-url", dashboard_url,
            "--filename", "custom.ipynb",
            "--overwrite",
            "--tags", "a,b",
            "--description", "fetched notebook",
            "--expected-sha256", "c" * 64,
            "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would import 'custom.ipynb'" in proc.stdout

    body = json.loads(handler.bodies[0])
    assert body == {
        "url": "https://example.com/nb.ipynb",
        "overwrite": True,
        "filename": "custom.ipynb",
        "tags": "a,b",
        "description": "fetched notebook",
        "expected_sha256": "c" * 64,
        "dry_run": True,
    }


def test_import_url_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "filename": "nb.ipynb",
        "path": "/srv/uploads/nb.ipynb",
        "overwritten": False,
        "sha256": "a" * 64,
        "dry_run": False,
        "source_url": "https://example.com/nb.ipynb",
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "import-url", "https://example.com/nb.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_import_url_command_reports_a_dashboard_error(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {"detail": "url is required and must be a non-empty string"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "import-url", "https://example.com/nb.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "url is required and must be a non-empty string")


def test_import_url_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "import-url", "https://example.com/nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def _write_zip(path, entries):
    """Write a local .zip archive at `path` from {entry_name: content_bytes}."""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry_name, content in entries.items():
            archive.writestr(entry_name, content)


def test_import_notebooks_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "import-notebooks" in proc.stdout


def test_import_notebooks_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"status": "success", "filename": "a.ipynb", "path": "/srv/a.ipynb", "overwritten": False},
                {"status": "success", "filename": "b.ipynb", "path": "/srv/b.ipynb", "overwritten": False},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}", "b.ipynb": b"{}"})

    proc = _run_cli(
        ["import-notebooks", str(zip_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Imported 'a.ipynb' (overwritten: False)" in proc.stdout
    assert "Imported 'b.ipynb' (overwritten: False)" in proc.stdout
    assert "2 succeeded, 0 failed." in proc.stdout
    assert handler.requests == ["/api/notebooks/import?overwrite=false"]


def test_import_notebooks_command_reports_restored_version_count(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "status": "success", "filename": "a.ipynb", "path": "/srv/a.ipynb",
                    "overwritten": False, "restored_version_count": 2,
                },
                {
                    "status": "success", "filename": "b.ipynb", "path": "/srv/b.ipynb",
                    "overwritten": False, "restored_version_count": 0,
                },
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}", "b.ipynb": b"{}"})

    proc = _run_cli(
        ["import-notebooks", str(zip_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Imported 'a.ipynb' (overwritten: False, restored 2 version(s))" in proc.stdout
    assert "Imported 'b.ipynb' (overwritten: False)" in proc.stdout


def test_import_notebooks_command_passes_the_overwrite_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}"})

    proc = _run_cli(
        [
            "import-notebooks", str(zip_path),
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/import?overwrite=true"]


def test_import_notebooks_command_passes_the_expected_sha256_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}"})

    proc = _run_cli(
        [
            "import-notebooks", str(zip_path),
            "--dashboard-url", dashboard_url, "--expected-sha256", "abc123",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks/import?overwrite=false&expected_sha256=abc123"
    ]


def test_import_notebooks_command_reports_a_clean_error_for_a_mismatched_expected_sha256(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {
            "detail": "Uploaded archive does not match expected_sha256: expected abc, got def",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}"})

    proc = _run_cli(
        [
            "import-notebooks", str(zip_path),
            "--dashboard-url", dashboard_url, "--expected-sha256", "abc",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "does not match expected_sha256")


def test_import_notebooks_command_passes_the_tags_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}"})

    proc = _run_cli(
        [
            "import-notebooks", str(zip_path),
            "--dashboard-url", dashboard_url, "--tags", "imported,reviewed",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks/import?overwrite=false&tags=imported%2Creviewed"
    ]


def test_import_notebooks_command_passes_the_description_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}"})

    proc = _run_cli(
        [
            "import-notebooks", str(zip_path),
            "--dashboard-url", dashboard_url, "--description", "imported from backup",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["description"] == ["imported from backup"]


def test_import_notebooks_command_reports_per_file_failures_and_exits_nonzero(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"status": "success", "filename": "a.ipynb", "path": "/srv/a.ipynb", "overwritten": False},
                {"status": "error", "filename": "bad.ipynb", "detail": "Uploaded file is not a valid Jupyter notebook"},
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}", "bad.ipynb": b"not a notebook"})

    proc = _run_cli(
        ["import-notebooks", str(zip_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Imported 'a.ipynb'" in proc.stdout
    assert "Failed 'bad.ipynb': Uploaded file is not a valid Jupyter notebook" in proc.stdout
    assert "1 succeeded, 1 failed." in proc.stdout


def test_import_notebooks_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [
            {"status": "success", "filename": "a.ipynb", "path": "/srv/a.ipynb", "overwritten": False},
        ],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}"})

    proc = _run_cli(
        ["import-notebooks", str(zip_path), "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_import_notebooks_command_reports_a_clean_error_for_a_rejected_import(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {"detail": "Zip archive contains no .ipynb files"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"README.md": b"nothing here"})

    proc = _run_cli(
        ["import-notebooks", str(zip_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Zip archive contains no .ipynb files")


def test_import_notebooks_command_reports_a_clean_error_for_a_missing_zip(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "import-notebooks", str(workdir / "does-not-exist.zip"),
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_import_notebooks_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "bundle.zip"
    _write_zip(zip_path, {"a.ipynb": b"{}"})

    proc = _run_cli(
        [
            "import-notebooks", str(zip_path),
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_list_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "list" in proc.stdout


def test_list_command_prints_notebooks_from_the_dashboard(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebooks": [
                {
                    "filename": "add.ipynb", "size_bytes": 123,
                    "modified_at": "2026-01-01T00:00:00+00:00",
                    "currently_compiled": True, "tags": ["prod"],
                },
                {
                    "filename": "scratch.ipynb", "size_bytes": 45,
                    "modified_at": "2026-01-02T00:00:00+00:00",
                    "currently_compiled": False, "tags": [],
                },
            ],
            "total_count": 2, "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(["list", "--dashboard-url", dashboard_url], cwd=Path.cwd())

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "add.ipynb  (123 bytes)  [currently compiled; tags: prod]" in proc.stdout
    assert "scratch.ipynb  (45 bytes)" in proc.stdout
    assert "2 notebook(s) total." in proc.stdout
    assert handler.requests == ["/api/notebooks?sort=name&order=asc&offset=0"]


def test_dashboard_url_defaults_from_environment_variable(fake_dashboard, monkeypatch):
    """Every dashboard-facing command's own --dashboard-url now defaults
    to $NOTEBOOK_API_DASHBOARD_URL when set, the same "already
    independently configurable via its own NOTEBOOK_API_* environment
    variable" convention dashboard_host()/dashboard_port() (backend/
    dashboard.py) already establish server-side -- `list` (already
    exercised against a real --dashboard-url just above) is exercised
    here with no --dashboard-url flag at all, relying entirely on the
    environment variable to reach the fake dashboard.
    """

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    monkeypatch.setenv("NOTEBOOK_API_DASHBOARD_URL", dashboard_url)

    proc = _run_cli(["list"], cwd=Path.cwd())

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks?sort=name&order=asc&offset=0"]


def test_dashboard_url_explicit_flag_overrides_environment_variable(
    fake_dashboard, monkeypatch
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    # Points the environment default at a port nothing is listening on --
    # if the explicit --dashboard-url below were ignored in favor of
    # this, the request would fail to connect instead of reaching the
    # fake dashboard.
    monkeypatch.setenv("NOTEBOOK_API_DASHBOARD_URL", "http://127.0.0.1:1")

    proc = _run_cli(["list", "--dashboard-url", dashboard_url], cwd=Path.cwd())

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks?sort=name&order=asc&offset=0"]


def test_list_command_shows_the_compiled_version_id_when_present(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebooks": [
                {
                    "filename": "add.ipynb", "size_bytes": 123,
                    "modified_at": "2026-01-01T00:00:00+00:00",
                    "currently_compiled": True, "tags": [],
                    "compiled_version_id": "20260101T000000000000_abcd.ipynb",
                },
            ],
            "total_count": 1, "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(["list", "--dashboard-url", dashboard_url], cwd=Path.cwd())

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "add.ipynb  (123 bytes)  [currently compiled from version "
        "'20260101T000000000000_abcd.ipynb']"
    ) in proc.stdout


def test_list_command_passes_the_search_flag_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(
        ["list", "--dashboard-url", dashboard_url, "--search", "add"],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks?sort=name&order=asc&offset=0&search=add"]


def test_list_command_passes_sort_order_tag_and_limit_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": 5, "offset": 0,
        })
    ]

    proc = _run_cli(
        [
            "list", "--dashboard-url", dashboard_url,
            "--sort", "modified", "--order", "desc", "--tag", "prod", "--limit", "5",
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks?sort=modified&order=desc&offset=0&tag=prod&limit=5"
    ]


def test_list_command_passes_the_description_search_flag_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(
        [
            "list", "--dashboard-url", dashboard_url,
            "--description-search", "churn",
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks?sort=name&order=asc&offset=0&description_search=churn"
    ]


def test_list_command_passes_the_regex_flag_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(
        [
            "list", "--dashboard-url", dashboard_url,
            "--search", r"^report_\d+$", "--regex",
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["regex"] == ["true"]
    assert query["search"] == [r"^report_\d+$"]


def test_list_command_passes_the_sha256_flag_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(
        [
            "list", "--dashboard-url", dashboard_url,
            "--sha256", "abc123",
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks?sort=name&order=asc&offset=0&sha256=abc123"
    ]


def test_list_command_passes_the_checksums_flag_through_and_prints_the_hash(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebooks": [
                {
                    "filename": "nb.ipynb", "size_bytes": 123,
                    "currently_compiled": False, "tags": [],
                    "sha256": "abc123",
                }
            ],
            "total_count": 1, "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(
        ["list", "--dashboard-url", dashboard_url, "--checksums"],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks?sort=name&order=asc&offset=0&checksums=true"
    ]
    assert "sha256:abc123" in proc.stdout


def test_list_command_passes_modified_after_and_before_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(
        [
            "list", "--dashboard-url", dashboard_url,
            "--modified-after", "2026-01-01T00:00:00+00:00",
            "--modified-before", "2026-06-01T00:00:00+00:00",
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks?sort=name&order=asc&offset=0"
        "&modified_after=2026-01-01T00%3A00%3A00%2B00%3A00"
        "&modified_before=2026-06-01T00%3A00%3A00%2B00%3A00"
    ]


def test_list_command_reports_the_dashboards_error_for_an_invalid_modified_after(
    fake_dashboard,
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {"detail": "modified_after must be an ISO 8601 datetime"})
    ]

    proc = _run_cli(
        ["list", "--dashboard-url", dashboard_url, "--modified-after", "not-a-date"],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "modified_after must be an ISO 8601 datetime")


def test_list_command_passes_offset_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": None, "offset": 10,
        })
    ]

    proc = _run_cli(
        ["list", "--dashboard-url", dashboard_url, "--offset", "10"],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks?sort=name&order=asc&offset=10"]


def test_list_command_reports_a_partial_page_when_limited(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebooks": [
                {
                    "filename": "a.ipynb", "size_bytes": 1,
                    "modified_at": "2026-01-01T00:00:00+00:00",
                    "currently_compiled": False, "tags": [],
                },
            ],
            "total_count": 5, "limit": 1, "offset": 0,
        })
    ]

    proc = _run_cli(
        ["list", "--dashboard-url", dashboard_url, "--limit", "1"],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Showing 1 of 5 notebook(s) (offset 0)." in proc.stdout


def test_list_command_rejects_an_invalid_sort_value(fake_dashboard):

    dashboard_url, handler = fake_dashboard

    proc = _run_cli(
        ["list", "--dashboard-url", dashboard_url, "--sort", "bogus"],
        cwd=Path.cwd(),
    )

    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr


def test_list_command_reports_no_notebooks_found(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebooks": [], "total_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(["list", "--dashboard-url", dashboard_url], cwd=Path.cwd())

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No notebooks found." in proc.stdout


def test_list_command_json_flag_emits_the_dashboards_own_response(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "notebooks": [{
            "filename": "add.ipynb", "size_bytes": 123,
            "modified_at": "2026-01-01T00:00:00+00:00",
            "currently_compiled": False, "tags": [],
        }],
        "total_count": 1, "limit": None, "offset": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["list", "--dashboard-url", dashboard_url, "--json"], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_list_command_format_csv_prints_the_dashboards_raw_csv_response(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    csv_body = (
        "filename,size_bytes,modified_at,currently_compiled,tags,"
        "description,notebook_changed_since_compile,compiled_at\r\n"
        "nb.ipynb,100,2026-01-01T00:00:00+00:00,False,,,,\r\n"
    )
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    proc = _run_cli(
        ["list", "--format", "csv", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == [
        "/api/notebooks?sort=name&order=asc&offset=0&format=csv"
    ]


def test_list_command_omits_format_query_param_by_default(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "notebooks": [], "total_count": 0,
        "limit": None, "offset": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(["list", "--dashboard-url", dashboard_url], cwd=Path.cwd())

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks?sort=name&order=asc&offset=0"]


def test_list_command_reports_a_clean_error_when_the_dashboard_is_unreachable():

    proc = _run_cli(
        ["list", "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5"],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_info_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "info" in proc.stdout


def test_info_command_prints_a_notebooks_metadata(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "filename": "add.ipynb",
        "size_bytes": 123,
        "modified_at": "2026-01-01T00:00:00+00:00",
        "currently_compiled": False,
        "tags": ["production", "v2"],
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["info", "add.ipynb", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "add.ipynb  (123 bytes)" in proc.stdout
    assert "tags: production, v2" in proc.stdout
    assert "currently compiled: False" in proc.stdout
    assert handler.requests == ["/api/notebooks/add.ipynb/info"]


def test_info_command_prints_source_url_when_present(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "filename": "add.ipynb",
        "size_bytes": 123,
        "modified_at": "2026-01-01T00:00:00+00:00",
        "currently_compiled": False,
        "tags": [],
        "source_url": "https://example.com/notebooks/add.ipynb",
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["info", "add.ipynb", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "source url: https://example.com/notebooks/add.ipynb" in proc.stdout


def test_info_command_omits_source_url_line_when_absent(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "filename": "add.ipynb",
        "size_bytes": 123,
        "modified_at": "2026-01-01T00:00:00+00:00",
        "currently_compiled": False,
        "tags": [],
        "source_url": None,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["info", "add.ipynb", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "source url" not in proc.stdout


def test_info_command_reports_no_tags(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "filename": "add.ipynb",
        "size_bytes": 123,
        "modified_at": "2026-01-01T00:00:00+00:00",
        "currently_compiled": False,
        "tags": [],
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["info", "add.ipynb", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tags: (none)" in proc.stdout


def test_info_command_reports_currently_compiled_fields(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "filename": "add.ipynb",
        "size_bytes": 123,
        "modified_at": "2026-01-01T00:00:00+00:00",
        "currently_compiled": True,
        "tags": [],
        "notebook_changed_since_compile": False,
        "compiled_at": "2026-01-01T00:05:00+00:00",
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["info", "add.ipynb", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "currently compiled: True" in proc.stdout
    assert "compiled at: 2026-01-01T00:05:00+00:00" in proc.stdout
    assert "changed since compile: False" in proc.stdout


def test_info_command_json_flag_emits_the_dashboards_own_response(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "filename": "add.ipynb",
        "size_bytes": 123,
        "modified_at": "2026-01-01T00:00:00+00:00",
        "currently_compiled": False,
        "tags": [],
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["info", "add.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_info_command_reports_a_clean_error_for_a_missing_notebook(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    proc = _run_cli(
        ["info", "does-not-exist.ipynb", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_info_command_reports_a_clean_error_when_the_dashboard_is_unreachable():

    proc = _run_cli(
        [
            "info", "add.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_info_batch_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "info-batch" in proc.stdout


def test_info_batch_command_prints_each_notebooks_metadata(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [
            {
                "status": "success", "filename": "a.ipynb", "size_bytes": 100,
                "modified_at": "2026-01-01T00:00:00+00:00",
                "currently_compiled": False, "tags": ["scratch"],
            },
            {
                "status": "success", "filename": "b.ipynb", "size_bytes": 200,
                "modified_at": "2026-01-01T00:00:00+00:00",
                "currently_compiled": False, "tags": [],
            },
        ],
        "succeeded_count": 2,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["info-batch", "a.ipynb", "b.ipynb", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb  (100 bytes) tags: scratch" in proc.stdout
    assert "b.ipynb  (200 bytes) tags: (none)" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/info-batch"]
    assert json.loads(handler.bodies[0]) == {"filenames": ["a.ipynb", "b.ipynb"]}


def test_info_batch_command_reports_a_partial_failure(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [
            {
                "status": "success", "filename": "a.ipynb", "size_bytes": 100,
                "modified_at": "2026-01-01T00:00:00+00:00",
                "currently_compiled": False, "tags": [],
            },
            {
                "status": "error", "filename": "missing.ipynb",
                "detail": "Notebook file not found",
            },
        ],
        "succeeded_count": 1,
        "failed_count": 1,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["info-batch", "a.ipynb", "missing.ipynb", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb  (100 bytes)" in proc.stdout
    assert "Failed 'missing.ipynb': Notebook file not found" in proc.stdout
    assert "1 succeeded, 1 failed" in proc.stdout


def test_info_batch_command_json_flag_emits_the_dashboards_own_response(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [
            {
                "status": "success", "filename": "a.ipynb", "size_bytes": 100,
                "modified_at": "2026-01-01T00:00:00+00:00",
                "currently_compiled": False, "tags": [],
            },
        ],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["info-batch", "a.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_info_batch_command_reports_a_clean_error_when_the_dashboard_is_unreachable():

    proc = _run_cli(
        [
            "info-batch", "a.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_search_functions_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "search-functions" in proc.stdout


def test_search_functions_command_prints_matching_notebooks(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "search": "train",
        "matches": [
            {
                "filename": "a.ipynb",
                "functions": [{"name": "train_model"}],
            },
            {
                "filename": "b.ipynb",
                "functions": [{"name": "retrain"}],
            },
        ],
        "notebook_count": 2,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["search-functions", "train", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb: train_model" in proc.stdout
    assert "b.ipynb: retrain" in proc.stdout
    assert "2 notebook(s) matched." in proc.stdout
    assert handler.requests == ["/api/functions?search=train&offset=0"]


def test_search_functions_command_sends_tag_query_param(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "search": "train", "matches": [], "notebook_count": 0}
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["search-functions", "train", "--tag", "prod", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/functions?search=train&offset=0&tag=prod"]


def test_search_functions_command_sends_regex_query_param(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "search": r"_v\d+$", "regex": True,
        "matches": [], "notebook_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        [
            "search-functions", r"_v\d+$", "--regex",
            "--dashboard-url", dashboard_url,
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/functions?search=_v%5Cd%2B%24&offset=0&regex=true"
    ]


def test_search_functions_command_omits_regex_query_param_by_default(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "search": "train", "matches": [], "notebook_count": 0}
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["search-functions", "train", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/functions?search=train&offset=0"]


def test_search_functions_command_reports_no_matches(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "search": "nonexistent",
        "matches": [],
        "notebook_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["search-functions", "nonexistent", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No notebooks define a function matching 'nonexistent'." in proc.stdout


def test_search_functions_command_passes_limit_and_offset_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "search": "train",
        "matches": [{"filename": "b.ipynb", "functions": [{"name": "retrain"}]}],
        "notebook_count": 3, "limit": 1, "offset": 1,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        [
            "search-functions", "train", "--limit", "1", "--offset", "1",
            "--dashboard-url", dashboard_url,
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "b.ipynb: retrain" in proc.stdout
    assert "Showing 1 of 3 notebook(s) (offset 1)." in proc.stdout
    assert handler.requests == ["/api/functions?search=train&offset=1&limit=1"]


def test_search_functions_command_json_flag_emits_the_dashboards_own_response(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "search": "train",
        "matches": [{"filename": "a.ipynb", "functions": [{"name": "train_model"}]}],
        "notebook_count": 1,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["search-functions", "train", "--dashboard-url", dashboard_url, "--json"],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_search_functions_command_format_csv_prints_the_dashboards_raw_csv_response(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    csv_body = (
        "filename,function_name,args,return_type,is_async\r\n"
        "nb.ipynb,train_model,epochs:int,str,False\r\n"
    )
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    proc = _run_cli(
        [
            "search-functions", "train", "--format", "csv",
            "--dashboard-url", dashboard_url,
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == [
        "/api/functions?search=train&offset=0&format=csv"
    ]


def test_search_functions_command_omits_format_query_param_by_default(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "search": "train", "regex": False,
            "matches": [], "notebook_count": 0, "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(
        ["search-functions", "train", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/functions?search=train&offset=0"]


def test_search_functions_command_reports_a_clean_error_for_an_empty_search(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {"detail": "search is required"})
    ]

    proc = _run_cli(
        ["search-functions", "", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "search is required")


def test_search_functions_command_reports_a_clean_error_when_the_dashboard_is_unreachable():

    proc = _run_cli(
        [
            "search-functions", "train",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_search_content_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "search-content" in proc.stdout


def test_search_content_command_prints_matching_notebooks(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "search": "read_csv",
        "matches": [
            {
                "filename": "a.ipynb",
                "matches": [{"cell_index": 0, "snippet": "df = pd.read_csv('x.csv')"}],
            },
        ],
        "notebook_count": 1,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["search-content", "read_csv", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb:" in proc.stdout
    assert "[0] df = pd.read_csv('x.csv')" in proc.stdout
    assert "1 notebook(s) matched." in proc.stdout
    assert handler.requests == ["/api/notebooks/search-content?search=read_csv&offset=0"]


def test_search_content_command_sends_tag_query_param(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "search": "read_csv", "matches": [], "notebook_count": 0}
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        [
            "search-content", "read_csv", "--tag", "prod",
            "--dashboard-url", dashboard_url,
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/search-content?search=read_csv&offset=0&tag=prod"]


def test_search_content_command_sends_regex_query_param(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "search": r"read_csv\(.*index_col=", "regex": True,
        "matches": [], "notebook_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        [
            "search-content", r"read_csv\(.*index_col=", "--regex",
            "--dashboard-url", dashboard_url,
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks/search-content?search=read_csv%5C%28.%2Aindex_col%3D&offset=0&regex=true"
    ]


def test_search_content_command_omits_regex_query_param_by_default(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "search": "read_csv", "matches": [], "notebook_count": 0}
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["search-content", "read_csv", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/search-content?search=read_csv&offset=0"]


def test_search_content_command_reports_no_matches(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "search": "nonexistent",
        "matches": [], "notebook_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["search-content", "nonexistent", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No notebooks have a code cell matching 'nonexistent'." in proc.stdout


def test_search_content_command_passes_limit_and_offset_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "search": "read_csv",
        "matches": [{"filename": "b.ipynb", "matches": [{"cell_index": 0, "snippet": "read_csv(x)"}]}],
        "notebook_count": 3, "limit": 1, "offset": 1,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        [
            "search-content", "read_csv", "--limit", "1", "--offset", "1",
            "--dashboard-url", dashboard_url,
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "b.ipynb:" in proc.stdout
    assert "Showing 1 of 3 notebook(s) (offset 1)." in proc.stdout
    assert handler.requests == [
        "/api/notebooks/search-content?search=read_csv&offset=1&limit=1"
    ]


def test_search_content_command_json_flag_emits_the_dashboards_own_response(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "search": "read_csv",
        "matches": [
            {"filename": "a.ipynb", "matches": [{"cell_index": 0, "snippet": "x"}]},
        ],
        "notebook_count": 1,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["search-content", "read_csv", "--dashboard-url", dashboard_url, "--json"],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_search_content_command_format_csv_prints_the_dashboards_raw_csv_response(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    csv_body = (
        "filename,cell_index,snippet\r\n"
        "nb.ipynb,0,df = pd.read_csv('a.csv')\r\n"
    )
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    proc = _run_cli(
        [
            "search-content", "pd.", "--format", "csv",
            "--dashboard-url", dashboard_url,
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == [
        "/api/notebooks/search-content?search=pd.&offset=0&format=csv"
    ]


def test_search_content_command_reports_a_clean_error_for_an_empty_search(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {"detail": "search is required"})
    ]

    proc = _run_cli(
        ["search-content", "", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "search is required")


def test_search_content_command_reports_a_clean_error_when_the_dashboard_is_unreachable():

    proc = _run_cli(
        [
            "search-content", "read_csv",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_find_duplicates_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "find-duplicates" in proc.stdout


def test_find_duplicates_command_prints_duplicate_groups(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "duplicate_groups": [
            {
                "sha256": "abc123",
                "filenames": ["a.ipynb", "b.ipynb"],
                "size_bytes": 512,
            },
        ],
        "group_count": 1,
        "duplicate_notebook_count": 2,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["find-duplicates", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "abc123: a.ipynb, b.ipynb" in proc.stdout
    assert "1 duplicate group(s), 2 notebook(s) total" in proc.stdout
    assert handler.requests == ["/api/notebooks/duplicates"]


def test_find_duplicates_command_reports_no_duplicates(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "duplicate_groups": [],
        "group_count": 0,
        "duplicate_notebook_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["find-duplicates", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No duplicate notebooks found" in proc.stdout


def test_find_duplicates_command_json_flag_emits_the_dashboards_own_response(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "duplicate_groups": [
            {"sha256": "abc123", "filenames": ["a.ipynb", "b.ipynb"], "size_bytes": 512},
        ],
        "group_count": 1,
        "duplicate_notebook_count": 2,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["find-duplicates", "--dashboard-url", dashboard_url, "--json"],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_find_duplicates_command_sends_tag_query_param(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "duplicate_groups": [],
            "group_count": 0, "duplicate_notebook_count": 0,
        })
    ]

    proc = _run_cli(
        ["find-duplicates", "--tag", "production", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["tag"] == ["production"]


def test_find_duplicates_command_omits_tag_query_param_by_default(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "duplicate_groups": [],
            "group_count": 0, "duplicate_notebook_count": 0,
        })
    ]

    proc = _run_cli(
        ["find-duplicates", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests[0] == "/api/notebooks/duplicates"


def test_find_duplicates_command_sends_sha256_query_param(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "duplicate_groups": [
                {"sha256": "abc123", "filenames": ["a.ipynb", "b.ipynb"], "size_bytes": 10},
            ],
            "group_count": 1, "duplicate_notebook_count": 2,
        })
    ]

    proc = _run_cli(
        ["find-duplicates", "--sha256", "abc123", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "abc123: a.ipynb, b.ipynb" in proc.stdout
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["sha256"] == ["abc123"]


def test_find_duplicates_command_sends_limit_and_offset_query_params(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "duplicate_groups": [
                {"sha256": "abc123", "filenames": ["a.ipynb", "b.ipynb"], "size_bytes": 10},
            ],
            "group_count": 3, "duplicate_notebook_count": 6,
            "limit": 1, "offset": 1,
        })
    ]

    proc = _run_cli(
        [
            "find-duplicates", "--limit", "1", "--offset", "1",
            "--dashboard-url", dashboard_url,
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "3 duplicate group(s), 6 notebook(s) total" in proc.stdout
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["limit"] == ["1"]
    assert query["offset"] == ["1"]


def test_find_duplicates_command_omits_limit_and_offset_query_params_by_default(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "duplicate_groups": [],
            "group_count": 0, "duplicate_notebook_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(
        ["find-duplicates", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/duplicates"]


def test_find_duplicates_command_format_csv_prints_the_dashboards_raw_csv_response(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    csv_body = (
        "sha256,filename,size_bytes\r\n"
        "abc123,nb1.ipynb,42\r\n"
        "abc123,nb2.ipynb,42\r\n"
    )
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    proc = _run_cli(
        ["find-duplicates", "--format", "csv", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == ["/api/notebooks/duplicates?format=csv"]


def test_find_duplicates_command_omits_format_query_param_by_default(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "duplicate_groups": [],
            "group_count": 0, "duplicate_notebook_count": 0,
            "limit": None, "offset": 0,
        })
    ]

    proc = _run_cli(
        ["find-duplicates", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/duplicates"]


def test_find_duplicates_command_reports_a_clean_error_when_the_dashboard_is_unreachable():

    proc = _run_cli(
        [
            "find-duplicates",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_resolve_duplicates_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "resolve-duplicates" in proc.stdout


def test_resolve_duplicates_command_reports_success_with_yes_flag(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "sha256": "abc123",
                    "status": "success",
                    "kept_filename": "a.ipynb",
                    "deleted_filenames": [
                        {"filename": "b.ipynb", "was_currently_compiled": False},
                    ],
                },
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["resolve-duplicates", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Kept a.ipynb, deleted b.ipynb" in proc.stdout
    assert "1 group(s) resolved, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/duplicates/resolve"]
    assert json.loads(handler.bodies[0]) == {"keep": {}}


def test_resolve_duplicates_command_dry_run_skips_confirmation_and_sends_dry_run_field(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [
                {
                    "sha256": "abc123",
                    "status": "success",
                    "kept_filename": "a.ipynb",
                    "deleted_filenames": [
                        {"filename": "b.ipynb", "was_currently_compiled": False},
                    ],
                },
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        ["resolve-duplicates", "--dry-run", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Kept a.ipynb, would delete b.ipynb" in proc.stdout
    assert "1 group(s) previewed, 0 failed" in proc.stdout
    assert json.loads(handler.bodies[0]) == {"keep": {}, "dry_run": True}


def test_resolve_duplicates_command_sends_keep_overrides(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "resolve-duplicates", "--yes",
            "--keep", "abc123=b.ipynb",
            "--keep", "def456=c.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "keep": {"abc123": "b.ipynb", "def456": "c.ipynb"},
    }


def test_resolve_duplicates_command_sends_tag_body_field(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "resolve-duplicates", "--yes",
            "--tag", "production",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {"keep": {}, "tag": "production"}


def test_resolve_duplicates_command_sends_sha256_body_field(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "resolve-duplicates", "--yes",
            "--sha256", "abc123",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {"keep": {}, "sha256": "abc123"}


def test_resolve_duplicates_command_omits_tag_body_field_by_default(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "succeeded_count": 0, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["resolve-duplicates", "--yes", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {"keep": {}}


def test_resolve_duplicates_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "sha256": "abc123", "status": "error",
                    "detail": "'x.ipynb' is not a member of duplicate group abc123",
                },
            ],
            "succeeded_count": 0,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["resolve-duplicates", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Failed to resolve group abc123" in proc.stdout
    assert "0 group(s) resolved, 1 failed" in proc.stdout


def test_resolve_duplicates_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "results": [], "succeeded_count": 0, "failed_count": 0}
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["resolve-duplicates", "--dashboard-url", dashboard_url, "--yes", "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_resolve_duplicates_command_aborts_without_yes_when_declined(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "resolve-duplicates", "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        input="n\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_resolve_duplicates_command_rejects_a_malformed_keep_entry(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "resolve-duplicates", "--yes", "--keep", "no-equals-sign-here",
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Invalid --keep")


def test_resolve_duplicates_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "resolve-duplicates",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_storage_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "storage" in proc.stdout


def test_storage_command_prints_per_notebook_and_total_usage(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "notebooks": [
            {
                "filename": "a.ipynb",
                "notebook_bytes": 900,
                "version_bytes": 300,
                "version_count": 2,
                "total_bytes": 1200,
            },
        ],
        "notebook_count": 1,
        "total_notebook_bytes": 900,
        "total_version_bytes": 300,
        "total_version_count": 2,
        "total_bytes": 1200,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["storage", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb: 1200 bytes (900 notebook + 300 bytes across 2 version(s))" in proc.stdout
    assert "1 notebook(s), 1200 bytes total" in proc.stdout
    assert handler.requests == ["/api/notebooks/storage?offset=0"]


def test_storage_command_prints_the_catalog_cap_when_configured(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "notebooks": [],
        "notebook_count": 0,
        "total_notebook_bytes": 0,
        "total_version_bytes": 0,
        "total_version_count": 0,
        "total_bytes": 0,
        "max_notebooks": 500,
        "notebooks_remaining": 497,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["storage", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Catalog cap: 500 notebook(s), 497 remaining" in proc.stdout


def test_storage_command_omits_the_catalog_cap_line_when_disabled(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "notebooks": [],
        "notebook_count": 0,
        "total_notebook_bytes": 0,
        "total_version_bytes": 0,
        "total_version_count": 0,
        "total_bytes": 0,
        "max_notebooks": 0,
        "notebooks_remaining": None,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["storage", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Catalog cap" not in proc.stdout


def test_storage_command_sends_tag_query_param(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "notebooks": [], "notebook_count": 0,
        "total_notebook_bytes": 0, "total_version_bytes": 0,
        "total_version_count": 0, "total_bytes": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["storage", "--tag", "prod", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/storage?offset=0&tag=prod"]


def test_storage_command_format_csv_prints_the_dashboards_raw_csv_response(
    fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    csv_body = (
        "filename,notebook_bytes,version_bytes,version_count,total_bytes\r\n"
        "nb.ipynb,100,0,0,100\r\n"
    )
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    proc = _run_cli(
        ["storage", "--format", "csv", "--dashboard-url", dashboard_url],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == ["/api/notebooks/storage?offset=0&format=csv"]


def test_storage_command_omits_format_query_param_by_default(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "notebooks": [], "notebook_count": 0,
        "total_notebook_bytes": 0, "total_version_bytes": 0,
        "total_version_count": 0, "total_bytes": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["storage", "--dashboard-url", dashboard_url], cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/storage?offset=0"]


def test_storage_command_passes_limit_and_offset_through(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "notebooks": [
            {
                "filename": "b.ipynb", "notebook_bytes": 500, "version_bytes": 0,
                "version_count": 0, "total_bytes": 500,
            },
        ],
        "notebook_count": 3, "limit": 1, "offset": 1,
        "total_notebook_bytes": 1800, "total_version_bytes": 0,
        "total_version_count": 0, "total_bytes": 1800,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        [
            "storage", "--limit", "1", "--offset", "1",
            "--dashboard-url", dashboard_url,
        ],
        cwd=Path.cwd(),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "b.ipynb: 500 bytes" in proc.stdout
    assert "Showing 1 of 3 notebook(s) (offset 1)." in proc.stdout
    assert "1800 bytes total" in proc.stdout
    assert handler.requests == ["/api/notebooks/storage?offset=1&limit=1"]


def test_storage_command_reports_no_notebooks(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "notebooks": [],
        "notebook_count": 0,
        "total_notebook_bytes": 0,
        "total_version_bytes": 0,
        "total_version_count": 0,
        "total_bytes": 0,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["storage", "--dashboard-url", dashboard_url], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No notebooks uploaded" in proc.stdout


def test_storage_command_json_flag_emits_the_dashboards_own_response(fake_dashboard):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "notebooks": [
            {
                "filename": "a.ipynb", "notebook_bytes": 900,
                "version_bytes": 0, "version_count": 0, "total_bytes": 900,
            },
        ],
        "notebook_count": 1,
        "total_notebook_bytes": 900,
        "total_version_bytes": 0,
        "total_version_count": 0,
        "total_bytes": 900,
    }
    handler.responses = [_json_response(200, body)]

    proc = _run_cli(
        ["storage", "--dashboard-url", dashboard_url, "--json"], cwd=Path.cwd()
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_storage_command_reports_a_clean_error_when_the_dashboard_is_unreachable():

    proc = _run_cli(
        [
            "storage",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=Path.cwd(),
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_download_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "download" in proc.stdout


def test_download_command_saves_the_notebook_to_the_default_path(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    notebook_bytes = b'{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}'
    handler.responses = [_raw_response(200, notebook_bytes)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["download", "nb.ipynb", "--dashboard-url", dashboard_url], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"({len(notebook_bytes)} bytes)" in proc.stdout
    assert (workdir / "nb.ipynb").read_bytes() == notebook_bytes
    assert handler.requests == ["/api/notebooks/nb.ipynb"]


def test_download_command_respects_a_custom_output_path(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    notebook_bytes = b'{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}'
    handler.responses = [_raw_response(200, notebook_bytes)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "download", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--output", "saved_here.ipynb",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "saved_here.ipynb").read_bytes() == notebook_bytes
    assert not (workdir / "nb.ipynb").exists()


def test_download_command_json_flag_emits_a_machine_readable_result(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    notebook_bytes = b'{"cells": []}'
    handler.responses = [_raw_response(200, notebook_bytes)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["download", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == {
        "status": "success",
        "filename": "nb.ipynb",
        "path": "nb.ipynb",
        "size_bytes": len(notebook_bytes),
        "sha256": None,
    }


def test_download_command_prints_and_reports_the_content_sha256_header(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    notebook_bytes = b'{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}'
    handler.responses = [_raw_response(200, notebook_bytes)]
    handler.response_headers = [{"X-Content-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["download", "nb.ipynb", "--dashboard-url", dashboard_url], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sha256: abc123" in proc.stdout

    handler.responses = [_raw_response(200, notebook_bytes)]
    handler.response_headers = [{"X-Content-SHA256": "abc123"}]

    proc = _run_cli(
        ["download", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["sha256"] == "abc123"


def test_download_command_expected_sha256_succeeds_on_a_match(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    notebook_bytes = b'{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}'
    handler.responses = [_raw_response(200, notebook_bytes)]
    handler.response_headers = [{"X-Content-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "download", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--expected-sha256", "abc123",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "nb.ipynb").read_bytes() == notebook_bytes


def test_download_command_expected_sha256_fails_on_a_mismatch_and_writes_nothing(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    notebook_bytes = b'{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}'
    handler.responses = [_raw_response(200, notebook_bytes)]
    handler.response_headers = [{"X-Content-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "download", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--expected-sha256", "does-not-match",
        ],
        cwd=workdir,
    )

    assert proc.returncode != 0
    assert "does not match" in (proc.stdout + proc.stderr)
    assert not (workdir / "nb.ipynb").exists()


def test_download_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["download", "does-not-exist.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")
    assert not (workdir / "does-not-exist.ipynb").exists()


def test_download_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "download", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_export_notebooks_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "export-notebooks" in proc.stdout


def test_export_notebooks_command_passes_filenames_through_and_saves_the_zip(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18  # minimal empty-zip end-of-central-directory
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "export-notebooks", "a.ipynb", "b.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"({len(zip_bytes)} bytes)" in proc.stdout
    assert (workdir / "notebooks_export.zip").read_bytes() == zip_bytes
    assert handler.requests == ["/api/notebooks/export?filenames=a.ipynb%2Cb.ipynb"]


def test_export_notebooks_command_expected_sha256_succeeds_on_a_match(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]
    handler.response_headers = [{"X-Bundle-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "export-notebooks", "a.ipynb", "--dashboard-url", dashboard_url,
            "--expected-sha256", "abc123",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "notebooks_export.zip").read_bytes() == zip_bytes
    assert "bundle sha256: abc123" in proc.stdout


def test_export_notebooks_command_expected_sha256_fails_on_a_mismatch_and_writes_nothing(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]
    handler.response_headers = [{"X-Bundle-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "export-notebooks", "a.ipynb", "--dashboard-url", dashboard_url,
            "--expected-sha256", "does-not-match",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "does not match the expected value")
    assert not (workdir / "notebooks_export.zip").exists()


def test_export_notebooks_command_with_no_filenames_omits_the_query_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-notebooks", "--dashboard-url", dashboard_url], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/export"]


def test_export_notebooks_command_sends_tag_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-notebooks", "--tag", "prod", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/export?tag=prod"]


def test_export_notebooks_command_sends_include_versions_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-notebooks", "--include-versions", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/export?include_versions=true"]


def test_export_notebooks_command_omits_include_versions_query_param_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-notebooks", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/export"]


def test_export_notebooks_command_rejects_filename_and_tag_together(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "export-notebooks", "a.ipynb", "--tag", "prod",
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "either a filename or --tag, not both")


def test_export_notebooks_command_respects_a_custom_output_path(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "export-notebooks", "--dashboard-url", dashboard_url,
            "--output", "backup.zip",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "backup.zip").read_bytes() == zip_bytes
    assert not (workdir / "notebooks_export.zip").exists()


def test_export_notebooks_command_json_flag_emits_a_machine_readable_result(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-notebooks", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == {
        "status": "success",
        "path": "notebooks_export.zip",
        "size_bytes": len(zip_bytes),
        "bundle_sha256": None,
    }


def test_export_notebooks_command_reports_a_clean_error_for_a_missing_filename(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file(s) not found: missing.ipynb"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["export-notebooks", "missing.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file(s) not found: missing.ipynb")
    assert not (workdir / "notebooks_export.zip").exists()


def test_export_notebooks_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "export-notebooks",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_delete_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "delete" in proc.stdout


def test_delete_command_reports_success_with_yes_flag(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "was_currently_compiled": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "nb.ipynb", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Deleted 'nb.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb"]


def test_delete_command_dry_run_skips_the_prompt_and_prints_would_delete(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True,
            "filename": "nb.ipynb", "was_currently_compiled": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "nb.ipynb", "--dashboard-url", dashboard_url, "--dry-run"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would delete 'nb.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb?dry_run=true"]


def test_delete_command_flags_when_the_currently_compiled_notebook_was_deleted(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "was_currently_compiled": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "nb.ipynb", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "currently compiled app" in proc.stdout


def test_delete_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "was_currently_compiled": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "nb.ipynb", "--dashboard-url", dashboard_url, "--yes", "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["filename"] == "nb.ipynb"


def test_delete_command_aborts_without_yes_when_declined(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "delete", "nb.ipynb", "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        input="n\n",
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_delete_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "nb.ipynb", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_delete_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "delete", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_delete_command_all_flag_reports_success_with_yes_flag(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "deleted_count": 2,
            "deleted_filenames": ["a.ipynb", "b.ipynb"],
            "currently_compiled_notebook_deleted": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "--all", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Deleted 'a.ipynb'" in proc.stdout
    assert "Deleted 'b.ipynb'" in proc.stdout
    assert "2 notebook(s) deleted" in proc.stdout
    assert handler.requests == ["/api/notebooks?confirm=true"]


def test_delete_command_all_flag_dry_run_skips_the_prompt_and_prints_would_delete(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True,
            "deleted_count": 2,
            "deleted_filenames": ["a.ipynb", "b.ipynb"],
            "currently_compiled_notebook_deleted": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "--all", "--dashboard-url", dashboard_url, "--dry-run"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would delete 'a.ipynb'" in proc.stdout
    assert "Would delete 'b.ipynb'" in proc.stdout
    assert "2 notebook(s) would be deleted" in proc.stdout
    assert handler.requests == ["/api/notebooks?dry_run=true"]


def test_delete_command_all_flag_reports_nothing_to_delete(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "deleted_count": 0,
            "deleted_filenames": [], "currently_compiled_notebook_deleted": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "--all", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No notebooks to delete" in proc.stdout


def test_delete_command_all_flag_flags_the_currently_compiled_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "deleted_count": 1,
            "deleted_filenames": ["a.ipynb"],
            "currently_compiled_notebook_deleted": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "--all", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "currently compiled app" in proc.stdout


def test_delete_command_all_flag_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "deleted_count": 1,
            "deleted_filenames": ["a.ipynb"],
            "currently_compiled_notebook_deleted": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "--all", "--dashboard-url", dashboard_url, "--yes", "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["deleted_count"] == 1


def test_delete_command_all_flag_aborts_without_yes_when_declined(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "delete", "--all", "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        input="n\n",
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_delete_command_all_flag_sends_tag_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "deleted_count": 1,
            "deleted_filenames": ["a.ipynb"],
            "currently_compiled_notebook_deleted": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "delete", "--all", "--tag", "scratch",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks?confirm=true&tag=scratch"]


def test_delete_command_all_flag_with_tag_prompts_with_the_tag_named(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "delete", "--all", "--tag", "scratch",
            "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        input="n\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "every notebook tagged 'scratch'" in proc.stdout
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_delete_command_rejects_tag_without_all(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "delete", "nb.ipynb", "--tag", "scratch",
            "--dashboard-url", "http://127.0.0.1:1", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "--tag only applies together with --all.")


def test_delete_command_all_flag_sends_sha256_query_param(tmp_path, fake_dashboard):
    """Mirrors test_delete_command_all_flag_sends_tag_query_param: DELETE
    /api/notebooks's own "sha256" filter (the same exact-content-match
    "list"/"find-duplicates" already support) had no --sha256 flag of
    its own here at all, even though "delete --all --tag" already
    threads the identical sibling filter through.
    """

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "deleted_count": 1,
            "deleted_filenames": ["a.ipynb"],
            "currently_compiled_notebook_deleted": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "delete", "--all", "--sha256", "a" * 64,
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [f"/api/notebooks?confirm=true&sha256={'a' * 64}"]


def test_delete_command_all_flag_sends_both_tag_and_sha256_query_params(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "deleted_count": 1,
            "deleted_filenames": ["a.ipynb"],
            "currently_compiled_notebook_deleted": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "delete", "--all", "--tag", "scratch", "--sha256", "a" * 64,
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        f"/api/notebooks?confirm=true&tag=scratch&sha256={'a' * 64}"
    ]


def test_delete_command_all_flag_with_sha256_prompts_with_the_hash_named(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "delete", "--all", "--sha256", "a" * 64,
            "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        input="n\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"with sha256 '{'a' * 64}'" in proc.stdout
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_delete_command_rejects_sha256_without_all(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "delete", "nb.ipynb", "--sha256", "a" * 64,
            "--dashboard-url", "http://127.0.0.1:1", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "--sha256 only applies together with --all.")


def test_delete_command_rejects_both_filename_and_all(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "nb.ipynb", "--all", "--dashboard-url", "http://127.0.0.1:1", "--yes"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Pass either a filename or --all, not both.")


def test_delete_command_rejects_neither_filename_nor_all(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete", "--dashboard-url", "http://127.0.0.1:1", "--yes"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Pass a filename to delete, or --all")


def test_delete_command_all_flag_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "delete", "--all",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_delete_batch_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "delete-batch" in proc.stdout


def test_delete_batch_command_reports_success_with_yes_flag(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "status": "success", "was_currently_compiled": False},
                {"filename": "b.ipynb", "status": "success", "was_currently_compiled": True},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete-batch", "a.ipynb", "b.ipynb", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Deleted 'a.ipynb'" in proc.stdout
    assert "Deleted 'b.ipynb'" in proc.stdout
    assert "currently compiled app" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/delete-batch"]


def test_delete_batch_command_dry_run_skips_confirmation_and_sends_dry_run_field(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [
                {"filename": "a.ipynb", "status": "success", "was_currently_compiled": False},
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        ["delete-batch", "a.ipynb", "--dry-run", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would delete 'a.ipynb'" in proc.stdout
    assert "1 succeeded, 0 failed" in proc.stdout
    assert json.loads(handler.bodies[0]) == {"filenames": ["a.ipynb"], "dry_run": True}


def test_delete_batch_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "status": "success", "was_currently_compiled": False},
                {"filename": "missing.ipynb", "status": "error", "detail": "Notebook file not found"},
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["delete-batch", "a.ipynb", "missing.ipynb", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Deleted 'a.ipynb'" in proc.stdout
    assert "Failed to delete missing.ipynb: Notebook file not found" in proc.stdout
    assert "1 succeeded, 1 failed" in proc.stdout


def test_delete_batch_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "status": "success", "was_currently_compiled": False},
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "delete-batch", "a.ipynb",
            "--dashboard-url", dashboard_url, "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["succeeded_count"] == 1


def test_delete_batch_command_aborts_without_yes_when_declined(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "delete-batch", "a.ipynb", "b.ipynb", "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        input="n\n",
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_delete_batch_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "delete-batch", "a.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_prune_versions_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "prune-versions" in proc.stdout


def test_prune_versions_command_reports_success_with_yes_flag(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "older_than_days": 30,
            "results": [
                {
                    "filename": "a.ipynb",
                    "deleted_version_ids": ["v1.ipynb"],
                    "deleted_count": 1,
                },
            ],
            "notebook_count_affected": 1,
            "total_deleted_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-versions", "--older-than-days", "30",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb: discarded 1 version(s)" in proc.stdout
    assert "1 version(s) discarded across 1 notebook(s)" in proc.stdout
    assert handler.requests == ["/api/notebooks/versions?older_than_days=30"]


def test_prune_versions_command_dry_run_skips_confirmation_and_sends_dry_run_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "older_than_days": 30,
            "results": [
                {
                    "filename": "a.ipynb",
                    "deleted_version_ids": ["v1.ipynb"],
                    "deleted_count": 1,
                },
            ],
            "notebook_count_affected": 1,
            "total_deleted_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        [
            "prune-versions", "--older-than-days", "30", "--dry-run",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb: would discard 1 version(s)" in proc.stdout
    assert "1 version(s) would discard across 1 notebook(s)" in proc.stdout
    assert handler.requests == [
        "/api/notebooks/versions?older_than_days=30&dry_run=true"
    ]


def test_prune_versions_command_sends_tag_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "older_than_days": 30, "results": [],
            "notebook_count_affected": 0, "total_deleted_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-versions", "--older-than-days", "30", "--tag", "prod",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks/versions?older_than_days=30&tag=prod"
    ]


def test_prune_versions_command_reports_nothing_to_prune(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "older_than_days": 30,
            "results": [],
            "notebook_count_affected": 0,
            "total_deleted_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-versions", "--older-than-days", "30",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No versions" in proc.stdout


def test_prune_versions_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "older_than_days": 30,
        "results": [],
        "notebook_count_affected": 0,
        "total_deleted_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-versions", "--older-than-days", "30",
            "--dashboard-url", dashboard_url, "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_prune_versions_command_aborts_without_yes_when_declined(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "prune-versions", "--older-than-days", "30",
            "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        input="n\n",
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_prune_versions_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-versions", "--older-than-days", "30",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_prune_temp_files_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "prune-temp-files" in proc.stdout


def test_prune_temp_files_command_reports_success_with_yes_flag(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": False,
            "older_than_seconds": 3600,
            "deleted_files": [
                {"filename": ".a.ipynb.deadbeef.part", "size_bytes": 42, "age_seconds": 7200},
            ],
            "deleted_count": 1,
            "reclaimed_bytes": 42,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-temp-files",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Removed '.a.ipynb.deadbeef.part' (42 bytes, 7200s old)" in proc.stdout
    assert "1 file(s), 42 byte(s) reclaimed" in proc.stdout
    assert handler.requests == ["/api/upload/temp-files"]


def test_prune_temp_files_command_dry_run_skips_confirmation_and_sends_dry_run_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "older_than_seconds": 3600,
            "deleted_files": [
                {"filename": ".a.ipynb.deadbeef.part", "size_bytes": 42, "age_seconds": 7200},
            ],
            "deleted_count": 1,
            "reclaimed_bytes": 42,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        [
            "prune-temp-files", "--dry-run",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would remove '.a.ipynb.deadbeef.part' (42 bytes, 7200s old)" in proc.stdout
    assert "1 file(s), 42 byte(s) would be reclaimed" in proc.stdout
    assert handler.requests == ["/api/upload/temp-files?dry_run=true"]


def test_prune_temp_files_command_sends_older_than_seconds_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": False, "older_than_seconds": 120,
            "deleted_files": [], "deleted_count": 0, "reclaimed_bytes": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-temp-files", "--older-than-seconds", "120",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/upload/temp-files?older_than_seconds=120"]


def test_prune_temp_files_command_reports_nothing_to_prune(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": False,
            "older_than_seconds": 3600,
            "deleted_files": [],
            "deleted_count": 0,
            "reclaimed_bytes": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-temp-files",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No orphaned upload temp files found" in proc.stdout


def test_prune_temp_files_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "dry_run": False,
        "older_than_seconds": 3600,
        "deleted_files": [],
        "deleted_count": 0,
        "reclaimed_bytes": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-temp-files",
            "--dashboard-url", dashboard_url, "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_prune_temp_files_command_aborts_without_yes_when_declined(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "prune-temp-files",
            "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        input="n\n",
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_prune_temp_files_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "prune-temp-files",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_rename_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "rename" in proc.stdout


def test_rename_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "new_filename": "renamed.ipynb",
            "was_currently_compiled": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["rename", "nb.ipynb", "renamed.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Renamed 'nb.ipynb' to 'renamed.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb"]
    assert json.loads(handler.bodies[0]) == {
        "new_filename": "renamed.ipynb", "overwrite": False,
    }


def test_rename_command_passes_the_overwrite_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "new_filename": "renamed.ipynb", "was_currently_compiled": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename", "nb.ipynb", "renamed.ipynb",
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0])["overwrite"] is True


def test_rename_command_dry_run_sends_dry_run_and_prints_would_rename(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True, "filename": "nb.ipynb",
            "new_filename": "renamed.ipynb", "was_currently_compiled": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename", "nb.ipynb", "renamed.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0])["dry_run"] is True
    assert "Would rename 'nb.ipynb' to 'renamed.ipynb'" in proc.stdout


def test_rename_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "new_filename": "renamed.ipynb", "was_currently_compiled": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename", "nb.ipynb", "renamed.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["new_filename"] == "renamed.ipynb"


def test_rename_command_reports_a_clean_error_for_a_rejected_rename(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(409, {
            "detail": "A notebook named 'renamed.ipynb' already exists."
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["rename", "nb.ipynb", "renamed.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "already exists")


def test_rename_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename", "nb.ipynb", "renamed.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_rename_many_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "rename-many" in proc.stdout


def test_rename_many_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "new_filename": "a2.ipynb", "status": "success", "was_currently_compiled": False},
                {"filename": "b.ipynb", "new_filename": "b2.ipynb", "status": "success", "was_currently_compiled": True},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename-many", "a.ipynb:a2.ipynb", "b.ipynb:b2.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Renamed 'a.ipynb' to 'a2.ipynb'" in proc.stdout
    assert "Renamed 'b.ipynb' to 'b2.ipynb'" in proc.stdout
    assert "note: this was the notebook backing the currently compiled app." in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/rename-batch"]
    assert json.loads(handler.bodies[0]) == {
        "entries": [
            {"filename": "a.ipynb", "new_filename": "a2.ipynb", "overwrite": False},
            {"filename": "b.ipynb", "new_filename": "b2.ipynb", "overwrite": False},
        ]
    }


def test_rename_many_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "new_filename": "a2.ipynb", "status": "success", "was_currently_compiled": False},
                {
                    "filename": "missing.ipynb", "new_filename": "m2.ipynb",
                    "status": "error", "detail": "Notebook file not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename-many", "a.ipynb:a2.ipynb", "missing.ipynb:m2.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Renamed 'a.ipynb' to 'a2.ipynb'" in proc.stdout
    assert (
        "Failed to rename 'missing.ipynb' to 'm2.ipynb': "
        "Notebook file not found" in proc.stdout
    )
    assert "1 succeeded, 1 failed" in proc.stdout


def test_rename_many_command_passes_the_overwrite_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [{"filename": "a.ipynb", "new_filename": "a2.ipynb", "status": "success", "was_currently_compiled": False}],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename-many", "a.ipynb:a2.ipynb",
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"filename": "a.ipynb", "new_filename": "a2.ipynb", "overwrite": True}]
    }


def test_rename_many_command_dry_run_sends_dry_run_field(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [{"filename": "a.ipynb", "new_filename": "a2.ipynb", "status": "success", "was_currently_compiled": False}],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename-many", "a.ipynb:a2.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would rename 'a.ipynb' to 'a2.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"filename": "a.ipynb", "new_filename": "a2.ipynb", "overwrite": False}],
        "dry_run": True,
    }


def test_rename_many_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [{"filename": "a.ipynb", "new_filename": "a2.ipynb", "status": "success", "was_currently_compiled": False}],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename-many", "a.ipynb:a2.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_rename_many_command_rejects_a_malformed_entry(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["rename-many", "no-colon-here"],
        cwd=workdir,
    )

    assert proc.returncode != 0
    assert "filename:new_filename" in proc.stderr


def test_rename_many_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "rename-many", "a.ipynb:a2.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_copy_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "copy" in proc.stdout


def test_copy_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "new_filename": "nb_copy.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["copy", "nb.ipynb", "nb_copy.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Copied 'nb.ipynb' to 'nb_copy.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/copy"]
    assert json.loads(handler.bodies[0]) == {
        "new_filename": "nb_copy.ipynb", "overwrite": False,
    }


def test_copy_command_passes_the_overwrite_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "new_filename": "nb_copy.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy", "nb.ipynb", "nb_copy.ipynb",
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0])["overwrite"] is True


def test_copy_command_dry_run_sends_dry_run_and_prints_would_copy(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True, "filename": "nb.ipynb",
            "new_filename": "nb_copy.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy", "nb.ipynb", "nb_copy.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0])["dry_run"] is True
    assert "Would copy 'nb.ipynb' to 'nb_copy.ipynb'" in proc.stdout


def test_copy_command_passes_tags_and_description_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "new_filename": "nb_copy.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy", "nb.ipynb", "nb_copy.ipynb",
            "--dashboard-url", dashboard_url,
            "--tags", "a,b", "--description", "a scratch copy",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    body = json.loads(handler.bodies[0])
    assert body["tags"] == ["a", "b"]
    assert body["description"] == "a scratch copy"


def test_copy_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "new_filename": "nb_copy.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy", "nb.ipynb", "nb_copy.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["new_filename"] == "nb_copy.ipynb"


def test_copy_command_reports_a_clean_error_for_a_rejected_copy(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(409, {
            "detail": "A notebook named 'nb_copy.ipynb' already exists."
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["copy", "nb.ipynb", "nb_copy.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "already exists")


def test_copy_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy", "nb.ipynb", "nb_copy.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_copy_batch_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "copy-batch" in proc.stdout


def test_copy_batch_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "results": [
                {"new_filename": "a.ipynb", "status": "success"},
                {"new_filename": "b.ipynb", "status": "success"},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-batch", "nb.ipynb", "a.ipynb", "b.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Copied 'nb.ipynb' to 'a.ipynb'" in proc.stdout
    assert "Copied 'nb.ipynb' to 'b.ipynb'" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/copy-batch"]
    assert json.loads(handler.bodies[0]) == {
        "new_filenames": ["a.ipynb", "b.ipynb"], "overwrite": False,
    }


def test_copy_batch_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "results": [
                {"new_filename": "a.ipynb", "status": "success"},
                {"new_filename": "existing.ipynb", "status": "error", "detail": "already exists"},
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-batch", "nb.ipynb", "a.ipynb", "existing.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Copied 'nb.ipynb' to 'a.ipynb'" in proc.stdout
    assert "Failed to copy to 'existing.ipynb': already exists" in proc.stdout
    assert "1 succeeded, 1 failed" in proc.stdout


def test_copy_batch_command_passes_the_overwrite_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "results": [{"new_filename": "a.ipynb", "status": "success"}],
            "succeeded_count": 1, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-batch", "nb.ipynb", "a.ipynb",
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "new_filenames": ["a.ipynb"], "overwrite": True,
    }


def test_copy_batch_command_passes_tags_and_description_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "results": [{"new_filename": "a.ipynb", "status": "success"}],
            "succeeded_count": 1, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-batch", "nb.ipynb", "a.ipynb",
            "--dashboard-url", dashboard_url,
            "--tags", "scratch", "--description", "batch copy",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    body = json.loads(handler.bodies[0])
    assert body["tags"] == ["scratch"]
    assert body["description"] == "batch copy"


def test_copy_batch_command_dry_run_sends_dry_run_field(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True, "filename": "nb.ipynb",
            "results": [{"new_filename": "a.ipynb", "status": "success"}],
            "succeeded_count": 1, "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-batch", "nb.ipynb", "a.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would copy 'nb.ipynb' to 'a.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "new_filenames": ["a.ipynb"], "overwrite": False, "dry_run": True,
    }


def test_copy_batch_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "filename": "nb.ipynb",
        "results": [{"new_filename": "a.ipynb", "status": "success"}],
        "succeeded_count": 1, "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-batch", "nb.ipynb", "a.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_copy_batch_command_reports_a_clean_error_for_a_missing_source(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-batch", "nb.ipynb", "a.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_copy_batch_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-batch", "nb.ipynb", "a.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_copy_many_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "copy-many" in proc.stdout


def test_copy_many_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "new_filename": "a-copy.ipynb", "status": "success"},
                {"filename": "b.ipynb", "new_filename": "b-copy.ipynb", "status": "success"},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-many", "a.ipynb:a-copy.ipynb", "b.ipynb:b-copy.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Copied 'a.ipynb' to 'a-copy.ipynb'" in proc.stdout
    assert "Copied 'b.ipynb' to 'b-copy.ipynb'" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/copy-batch"]
    assert json.loads(handler.bodies[0]) == {
        "entries": [
            {"filename": "a.ipynb", "new_filename": "a-copy.ipynb", "overwrite": False},
            {"filename": "b.ipynb", "new_filename": "b-copy.ipynb", "overwrite": False},
        ]
    }


def test_copy_many_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "new_filename": "a-copy.ipynb", "status": "success"},
                {
                    "filename": "missing.ipynb", "new_filename": "m-copy.ipynb",
                    "status": "error", "detail": "Notebook file not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-many", "a.ipynb:a-copy.ipynb", "missing.ipynb:m-copy.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Copied 'a.ipynb' to 'a-copy.ipynb'" in proc.stdout
    assert (
        "Failed to copy 'missing.ipynb' to 'm-copy.ipynb': "
        "Notebook file not found" in proc.stdout
    )
    assert "1 succeeded, 1 failed" in proc.stdout


def test_copy_many_command_passes_the_overwrite_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [{"filename": "a.ipynb", "new_filename": "a-copy.ipynb", "status": "success"}],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-many", "a.ipynb:a-copy.ipynb",
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"filename": "a.ipynb", "new_filename": "a-copy.ipynb", "overwrite": True}]
    }


def test_copy_many_command_dry_run_sends_dry_run_field(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [{"filename": "a.ipynb", "new_filename": "a-copy.ipynb", "status": "success"}],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-many", "a.ipynb:a-copy.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would copy 'a.ipynb' to 'a-copy.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"filename": "a.ipynb", "new_filename": "a-copy.ipynb", "overwrite": False}],
        "dry_run": True,
    }


def test_copy_many_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [{"filename": "a.ipynb", "new_filename": "a-copy.ipynb", "status": "success"}],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-many", "a.ipynb:a-copy.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_copy_many_command_rejects_a_malformed_entry(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["copy-many", "no-colon-here"],
        cwd=workdir,
    )

    assert proc.returncode != 0
    assert "filename:new_filename" in proc.stderr


def test_copy_many_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "copy-many", "a.ipynb:a-copy.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_tags_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "tags" in proc.stdout


def test_tags_get_command_prints_the_notebooks_tags(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb", "tags": ["prod", "v2"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "get", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb: prod, v2" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/tags"]


def test_tags_get_command_reports_no_tags(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "filename": "nb.ipynb", "tags": []})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "get", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb: (no tags)" in proc.stdout


def test_tags_get_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb", "tags": ["prod"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "get", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["tags"] == ["prod"]


def test_tags_list_command_prints_the_tag_catalog(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tags": [
                {"tag": "bug", "notebook_count": 1},
                {"tag": "production", "notebook_count": 2},
            ],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "list", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "bug  (1 notebook)" in proc.stdout
    assert "production  (2 notebooks)" in proc.stdout
    assert handler.requests == ["/api/tags"]


def test_tags_list_command_reports_no_tags(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "tags": []})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "list", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No tags in use on any notebook." in proc.stdout


def test_tags_list_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tags": [{"tag": "prod", "notebook_count": 3}],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "list", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["tags"] == [{"tag": "prod", "notebook_count": 3}]


def test_tags_list_command_format_csv_prints_the_dashboards_raw_csv_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    csv_body = "tag,notebook_count\r\nprod,3\r\n"
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "list", "--format", "csv", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == ["/api/tags?format=csv"]


def test_tags_delete_command_reports_success_with_yes_flag(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "scratch",
            "affected_notebooks": ["a.ipynb", "b.ipynb"],
            "notebook_count": 2,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "delete", "scratch", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Removed 'scratch' from a.ipynb" in proc.stdout
    assert "Removed 'scratch' from b.ipynb" in proc.stdout
    assert "2 notebook(s) updated" in proc.stdout
    assert handler.requests == ["/api/tags/scratch"]


def test_tags_delete_command_dry_run_skips_confirmation_and_sends_dry_run_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "tag": "scratch",
            "affected_notebooks": ["a.ipynb"],
            "notebook_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        ["tags", "delete", "scratch", "--dashboard-url", dashboard_url, "--dry-run"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would remove 'scratch' from a.ipynb" in proc.stdout
    assert handler.requests == ["/api/tags/scratch?dry_run=true"]


def test_tags_delete_command_reports_no_notebooks_affected(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "nonexistent",
            "affected_notebooks": [],
            "notebook_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "delete", "nonexistent", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No notebooks" in proc.stdout
    assert "carry tag 'nonexistent'" in proc.stdout


def test_tags_delete_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "scratch",
            "affected_notebooks": ["a.ipynb"],
            "notebook_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "delete", "scratch",
            "--dashboard-url", dashboard_url, "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["affected_notebooks"] == ["a.ipynb"]


def test_tags_delete_command_aborts_without_yes_when_declined(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "tags", "delete", "scratch", "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        input="n\n",
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_tags_delete_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "delete", "scratch",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_tags_apply_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "production",
            "results": [
                {"filename": "a.ipynb", "status": "success", "tags": ["production"]},
                {"filename": "b.ipynb", "status": "success", "tags": ["bug", "production"]},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "apply", "production", "a.ipynb", "b.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Tagged a.ipynb with 'production'" in proc.stdout
    assert "Tagged b.ipynb with 'production'" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/tags/production/apply"]
    assert json.loads(handler.bodies[0]) == {
        "filenames": ["a.ipynb", "b.ipynb"],
    }


def test_tags_apply_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "urgent",
            "results": [
                {"filename": "a.ipynb", "status": "success", "tags": ["urgent"]},
                {
                    "filename": "missing.ipynb", "status": "error",
                    "detail": "Notebook file not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "apply", "urgent", "a.ipynb", "missing.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Tagged a.ipynb with 'urgent'" in proc.stdout
    assert "Failed to tag missing.ipynb: Notebook file not found" in proc.stdout
    assert "1 succeeded, 1 failed" in proc.stdout


def test_tags_apply_command_dry_run_passes_the_flag_through_and_prints_would_tag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "tag": "production",
            "results": [
                {"filename": "a.ipynb", "status": "success", "tags": ["production"]},
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "apply", "production", "a.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would tag a.ipynb with 'production'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "filenames": ["a.ipynb"],
        "dry_run": True,
    }


def test_tags_apply_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "production",
            "results": [{"filename": "a.ipynb", "status": "success", "tags": ["production"]}],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "apply", "production", "a.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["succeeded_count"] == 1


def test_tags_apply_command_reports_a_clean_error_for_an_invalid_tag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {
            "detail": "tags must not be empty or whitespace-only strings"
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "apply", " ", "a.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "empty or whitespace-only")


def test_tags_apply_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "apply", "production", "a.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_tags_remove_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "production",
            "results": [
                {"filename": "a.ipynb", "status": "success", "tags": []},
                {"filename": "b.ipynb", "status": "success", "tags": ["bug"]},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "remove", "production", "a.ipynb", "b.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Removed 'production' from a.ipynb" in proc.stdout
    assert "Removed 'production' from b.ipynb" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/tags/production/remove"]
    assert json.loads(handler.bodies[0]) == {
        "filenames": ["a.ipynb", "b.ipynb"],
    }


def test_tags_remove_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "urgent",
            "results": [
                {"filename": "a.ipynb", "status": "success", "tags": []},
                {
                    "filename": "missing.ipynb", "status": "error",
                    "detail": "Notebook file not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "remove", "urgent", "a.ipynb", "missing.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Removed 'urgent' from a.ipynb" in proc.stdout
    assert "Failed to update missing.ipynb: Notebook file not found" in proc.stdout
    assert "1 succeeded, 1 failed" in proc.stdout


def test_tags_remove_command_dry_run_passes_the_flag_through_and_prints_would_remove(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "tag": "production",
            "results": [
                {"filename": "a.ipynb", "status": "success", "tags": []},
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "remove", "production", "a.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would remove 'production' from a.ipynb" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "filenames": ["a.ipynb"],
        "dry_run": True,
    }


def test_tags_remove_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "production",
            "results": [{"filename": "a.ipynb", "status": "success", "tags": []}],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "remove", "production", "a.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["succeeded_count"] == 1


def test_tags_remove_command_reports_a_clean_error_for_an_invalid_tag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {
            "detail": "tags must not be empty or whitespace-only strings"
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "remove", " ", "a.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "empty or whitespace-only")


def test_tags_remove_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "remove", "production", "a.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_tags_rename_command_reports_success_with_yes_flag(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "prod",
            "new_tag": "production",
            "affected_notebooks": ["a.ipynb", "b.ipynb"],
            "notebook_count": 2,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "rename", "prod", "production",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Renamed 'prod' to 'production' on a.ipynb" in proc.stdout
    assert "Renamed 'prod' to 'production' on b.ipynb" in proc.stdout
    assert "2 notebook(s) updated" in proc.stdout
    assert handler.requests == ["/api/tags/prod"]


def test_tags_rename_command_dry_run_skips_confirmation_and_sends_dry_run_field(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "tag": "prod",
            "new_tag": "production",
            "affected_notebooks": ["a.ipynb"],
            "notebook_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        [
            "tags", "rename", "prod", "production",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would rename 'prod' to 'production' on a.ipynb" in proc.stdout
    assert json.loads(handler.bodies[0]) == {"new_tag": "production", "dry_run": True}


def test_tags_rename_command_reports_no_notebooks_affected(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "nonexistent",
            "new_tag": "renamed",
            "affected_notebooks": [],
            "notebook_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "rename", "nonexistent", "renamed",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No notebooks" in proc.stdout
    assert "carry tag 'nonexistent'" in proc.stdout


def test_tags_rename_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "tag": "prod",
            "new_tag": "production",
            "affected_notebooks": ["a.ipynb"],
            "notebook_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "rename", "prod", "production",
            "--dashboard-url", dashboard_url, "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["affected_notebooks"] == ["a.ipynb"]
    assert data["new_tag"] == "production"


def test_tags_rename_command_aborts_without_yes_when_declined(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "tags", "rename", "prod", "production", "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        input="n\n",
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_tags_rename_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "rename", "prod", "production",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_tags_set_command_replaces_the_notebooks_tags(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb", "tags": ["prod", "v2"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "set", "nb.ipynb", "prod", "v2", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb tags set to: prod, v2" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/tags"]
    assert json.loads(handler.bodies[0]) == {"tags": ["prod", "v2"]}


def test_tags_set_command_dry_run_sends_dry_run_and_prints_would_be_set_to(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True,
            "filename": "nb.ipynb", "tags": ["prod", "v2"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "set", "nb.ipynb", "prod", "v2",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb tags would be set to: prod, v2" in proc.stdout
    assert json.loads(handler.bodies[0]) == {"tags": ["prod", "v2"], "dry_run": True}


def test_tags_set_command_with_no_tags_clears_them(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "filename": "nb.ipynb", "tags": []})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "set", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb tags set to: (none)" in proc.stdout
    assert json.loads(handler.bodies[0]) == {"tags": []}


def test_tags_set_command_reports_a_clean_error_for_an_invalid_tag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {"detail": "Each tag must be a non-empty string"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["tags", "set", "nb.ipynb", "", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "non-empty string")


def test_tags_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "get", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_tags_set_batch_command_is_registered():

    proc = _run_cli(["tags", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "set-batch" in proc.stdout


def test_tags_set_batch_command_sends_one_entry_per_notebook(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "status": "success", "tags": ["prod", "v2"]},
                {"filename": "b.ipynb", "status": "success", "tags": []},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "set-batch",
            "--entry", "a.ipynb=prod,v2",
            "--entry", "b.ipynb=",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb tags set to: prod, v2" in proc.stdout
    assert "b.ipynb tags set to: (none)" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/tags-batch"]
    assert json.loads(handler.bodies[0]) == {
        "entries": [
            {"filename": "a.ipynb", "tags": ["prod", "v2"]},
            {"filename": "b.ipynb", "tags": []},
        ],
    }


def test_tags_set_batch_command_dry_run_sends_dry_run_field(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [
                {"filename": "a.ipynb", "status": "success", "tags": ["prod"]},
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "set-batch",
            "--entry", "a.ipynb=prod",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb tags would be set to: prod" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"filename": "a.ipynb", "tags": ["prod"]}],
        "dry_run": True,
    }


def test_tags_set_batch_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "status": "success", "tags": ["ok"]},
                {
                    "filename": "b.ipynb", "status": "error",
                    "detail": "Notebook file not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "set-batch",
            "--entry", "a.ipynb=ok",
            "--entry", "b.ipynb=ok",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb tags set to: ok" in proc.stdout
    assert "Failed to set tags for b.ipynb: Notebook file not found" in proc.stdout
    assert "1 succeeded, 1 failed" in proc.stdout


def test_tags_set_batch_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [{"filename": "a.ipynb", "status": "success", "tags": ["ok"]}],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "set-batch", "--entry", "a.ipynb=ok",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_tags_set_batch_command_reports_a_clean_error_for_a_malformed_entry(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "set-batch", "--entry", "no-equals-sign-here",
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Invalid --entry")


def test_tags_set_batch_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "tags", "set-batch", "--entry", "a.ipynb=ok",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_description_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "description" in proc.stdout


def test_description_get_command_prints_the_notebooks_description(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "description": "the quarterly churn model",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["description", "get", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb: the quarterly churn model" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/description"]


def test_description_get_command_reports_no_description(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "filename": "nb.ipynb", "description": ""})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["description", "get", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb: (no description)" in proc.stdout


def test_description_get_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "filename": "nb.ipynb", "description": "hello"}
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["description", "get", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_description_get_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["description", "get", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_description_set_command_replaces_the_notebooks_description(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb", "description": "new description",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set", "nb.ipynb", "new description",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb description set to: new description" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/description"]
    assert json.loads(handler.bodies[0]) == {"description": "new description"}


def test_description_set_command_dry_run_sends_dry_run_and_prints_would_be_set_to(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True,
            "filename": "nb.ipynb", "description": "new description",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set", "nb.ipynb", "new description",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb description would be set to: new description" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "description": "new description", "dry_run": True,
    }


def test_description_set_command_with_empty_string_clears_it(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "filename": "nb.ipynb", "description": ""})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["description", "set", "nb.ipynb", "", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb description set to: (cleared)" in proc.stdout


def test_description_set_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "filename": "nb.ipynb", "description": "hello"}
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set", "nb.ipynb", "hello",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_description_set_command_reports_a_clean_error_for_an_invalid_value(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {"detail": "description must be at most 2000 characters long"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set", "nb.ipynb", "x" * 2001,
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "at most 2000 characters")


def test_description_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "get", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_source_url_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "source-url" in proc.stdout


def test_source_url_get_command_prints_the_notebooks_source_url(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "source_url": "https://example.com/nb.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["source-url", "get", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb: https://example.com/nb.ipynb" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/source-url"]


def test_source_url_get_command_reports_no_source_url(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "filename": "nb.ipynb", "source_url": None})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["source-url", "get", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb: (no source url)" in proc.stdout


def test_source_url_get_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "filename": "nb.ipynb", "source_url": "https://example.com/nb.ipynb"}
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["source-url", "get", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_source_url_get_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["source-url", "get", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_source_url_set_command_replaces_the_notebooks_source_url(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "source_url": "https://example.com/new.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set", "nb.ipynb", "https://example.com/new.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb source url set to: https://example.com/new.ipynb" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/source-url"]
    assert json.loads(handler.bodies[0]) == {"source_url": "https://example.com/new.ipynb"}


def test_source_url_set_command_dry_run_sends_dry_run_and_prints_would_be_set_to(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True,
            "filename": "nb.ipynb", "source_url": "https://example.com/new.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set", "nb.ipynb", "https://example.com/new.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb source url would be set to: https://example.com/new.ipynb" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "source_url": "https://example.com/new.ipynb", "dry_run": True,
    }


def test_source_url_set_command_with_empty_string_clears_it(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "filename": "nb.ipynb", "source_url": None})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["source-url", "set", "nb.ipynb", "", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb source url set to: (cleared)" in proc.stdout


def test_source_url_set_command_with_no_value_clears_it(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "filename": "nb.ipynb", "source_url": None})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["source-url", "set", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb source url set to: (cleared)" in proc.stdout
    assert json.loads(handler.bodies[0]) == {"source_url": ""}


def test_source_url_set_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "filename": "nb.ipynb", "source_url": "https://example.com/nb.ipynb"}
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set", "nb.ipynb", "https://example.com/nb.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_source_url_set_command_reports_a_clean_error_for_an_invalid_value(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {"detail": "Unsupported source_url 'ftp://x': only http:// and https:// URLs are accepted"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set", "nb.ipynb", "ftp://x",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "only http:// and https:// URLs are accepted")


def test_source_url_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "get", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_source_url_set_batch_command_is_registered():

    proc = _run_cli(["source-url", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "set-batch" in proc.stdout


def test_source_url_set_batch_command_sends_one_entry_per_notebook(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "status": "success", "source_url": "https://example.com/a.ipynb"},
                {"filename": "b.ipynb", "status": "success", "source_url": None},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set-batch",
            "--entry", "a.ipynb=https://example.com/a.ipynb",
            "--entry", "b.ipynb=",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb source url set to: https://example.com/a.ipynb" in proc.stdout
    assert "b.ipynb source url set to: (cleared)" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/source-url-batch"]
    assert json.loads(handler.bodies[0]) == {
        "entries": [
            {"filename": "a.ipynb", "source_url": "https://example.com/a.ipynb"},
            {"filename": "b.ipynb", "source_url": ""},
        ],
    }


def test_source_url_set_batch_command_dry_run_sends_dry_run_field(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [
                {"filename": "a.ipynb", "status": "success", "source_url": "https://example.com/new.ipynb"},
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set-batch",
            "--entry", "a.ipynb=https://example.com/new.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb source url would be set to: https://example.com/new.ipynb" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"filename": "a.ipynb", "source_url": "https://example.com/new.ipynb"}],
        "dry_run": True,
    }


def test_source_url_set_batch_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "status": "success", "source_url": "https://example.com/ok.ipynb"},
                {
                    "filename": "b.ipynb", "status": "error",
                    "detail": "Notebook file not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set-batch",
            "--entry", "a.ipynb=https://example.com/ok.ipynb",
            "--entry", "b.ipynb=https://example.com/ok.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb source url set to: https://example.com/ok.ipynb" in proc.stdout
    assert "Failed to set source url for b.ipynb: Notebook file not found" in proc.stdout
    assert "1 succeeded, 1 failed" in proc.stdout


def test_source_url_set_batch_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [{"filename": "a.ipynb", "status": "success", "source_url": "https://example.com/ok.ipynb"}],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set-batch", "--entry", "a.ipynb=https://example.com/ok.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_source_url_set_batch_command_reports_a_clean_error_for_a_malformed_entry(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set-batch", "--entry", "no-equals-sign-here",
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Invalid --entry")


def test_source_url_set_batch_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "source-url", "set-batch", "--entry", "a.ipynb=https://example.com/ok.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_description_set_batch_command_is_registered():

    proc = _run_cli(["description", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "set-batch" in proc.stdout


def test_description_set_batch_command_sends_one_entry_per_notebook(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "status": "success", "description": "the churn model"},
                {"filename": "b.ipynb", "status": "success", "description": ""},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set-batch",
            "--entry", "a.ipynb=the churn model",
            "--entry", "b.ipynb=",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb description set to: the churn model" in proc.stdout
    assert "b.ipynb description set to: (cleared)" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/description-batch"]
    assert json.loads(handler.bodies[0]) == {
        "entries": [
            {"filename": "a.ipynb", "description": "the churn model"},
            {"filename": "b.ipynb", "description": ""},
        ],
    }


def test_description_set_batch_command_dry_run_sends_dry_run_field(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [
                {"filename": "a.ipynb", "status": "success", "description": "new description"},
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set-batch",
            "--entry", "a.ipynb=new description",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb description would be set to: new description" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"filename": "a.ipynb", "description": "new description"}],
        "dry_run": True,
    }


def test_description_set_batch_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "status": "success", "description": "ok"},
                {
                    "filename": "b.ipynb", "status": "error",
                    "detail": "Notebook file not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set-batch",
            "--entry", "a.ipynb=ok",
            "--entry", "b.ipynb=ok",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "a.ipynb description set to: ok" in proc.stdout
    assert "Failed to set description for b.ipynb: Notebook file not found" in proc.stdout
    assert "1 succeeded, 1 failed" in proc.stdout


def test_description_set_batch_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [{"filename": "a.ipynb", "status": "success", "description": "ok"}],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set-batch", "--entry", "a.ipynb=ok",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_description_set_batch_command_reports_a_clean_error_for_a_malformed_entry(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set-batch", "--entry", "no-equals-sign-here",
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Invalid --entry")


def test_description_set_batch_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "description", "set-batch", "--entry", "a.ipynb=ok",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_validate_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "validate" in proc.stdout


def test_validate_command_passes_a_clean_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    proc = _run_cli(["validate", str(notebook_path)], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No issues found." in proc.stdout


def test_validate_command_warns_but_does_not_fail_on_skipped_functions(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def unsupported(a, **kwargs):\n    return a\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    proc = _run_cli(["validate", str(notebook_path)], cwd=workdir)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Skipped functions" in proc.stdout
    assert "unsupported" in proc.stdout
    assert "still compile cleanly" in proc.stdout


def test_validate_command_strict_flag_fails_on_skipped_functions(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def unsupported(a, **kwargs):\n    return a\n"
    )

    proc = _run_cli(["validate", str(notebook_path), "--strict"], cwd=workdir)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Validation failed." in proc.stdout


def test_validate_command_warns_but_does_not_fail_on_duplicate_functions(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def add(a: int, b: int) -> int:\n    return a * b\n",
    )

    proc = _run_cli(["validate", str(notebook_path)], cwd=workdir)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Duplicate functions" in proc.stdout
    assert "add" in proc.stdout
    assert "still compile cleanly" in proc.stdout


def test_validate_command_strict_flag_fails_on_duplicate_functions(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def add(a: int, b: int) -> int:\n    return a * b\n",
    )

    proc = _run_cli(["validate", str(notebook_path), "--strict"], cwd=workdir)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Validation failed." in proc.stdout


def test_validate_command_fails_on_a_reserved_name_conflict_even_without_strict(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def health_check() -> dict:\n    return {}\n"
    )

    proc = _run_cli(["validate", str(notebook_path)], cwd=workdir)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Reserved name conflicts" in proc.stdout
    assert "health_check" in proc.stdout
    assert "Validation failed." in proc.stdout


def test_validate_command_exclude_of_the_conflicting_function_passes(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def health_check() -> dict:\n    return {}\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    proc = _run_cli(
        ["validate", str(notebook_path), "--exclude", "health_check"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No issues found." in proc.stdout


def test_validate_command_only_including_the_conflicting_function_still_fails(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path,
        "def health_check() -> dict:\n    return {}\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    proc = _run_cli(
        ["validate", str(notebook_path), "--only", "health_check,add"],
        cwd=workdir,
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "health_check" in proc.stdout


def test_validate_command_json_flag_emits_a_machine_readable_status(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def health_check() -> dict:\n    return {}\n"
    )

    proc = _run_cli(["validate", str(notebook_path), "--json"], cwd=workdir)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "fail"
    assert data["reserved_name_conflicts"] == ["health_check"]
    assert data["skipped_functions"] == []


def test_validate_command_json_flag_reports_pass_status_for_a_clean_notebook(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    proc = _run_cli(["validate", str(notebook_path), "--json"], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "pass"


def test_validate_command_does_not_create_any_output_directory(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook_with_function(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    proc = _run_cli(["validate", str(notebook_path)], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (workdir / "generated").exists()


def test_validate_command_reports_a_clean_error_for_a_missing_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["validate", str(workdir / "does-not-exist.ipynb")], cwd=workdir
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_remote_compile_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "remote-compile" in proc.stdout


def test_remote_compile_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "functions": [{"name": "add"}],
            "endpoints": [{"path": "/add", "method": "POST", "is_async": False}],
            "skipped_functions": [],
            "dependencies": ["fastapi"],
            "generated_files": ["app.py"],
            "message": "Notebook compiled successfully",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-compile", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Compiled 'nb.ipynb'" in proc.stdout
    assert "POST /add" in proc.stdout
    assert "Dependencies: fastapi" in proc.stdout
    assert handler.requests == ["/api/compile"]
    assert json.loads(handler.bodies[0]) == {"notebook_path": "nb.ipynb"}


def test_remote_compile_command_passes_smoke_test_through_to_the_dashboard(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "functions": [{"name": "add"}],
            "endpoints": [{"path": "/add", "method": "POST", "is_async": False}],
            "skipped_functions": [],
            "dependencies": [],
            "generated_files": ["app.py"],
            "message": "Notebook compiled successfully",
            "smoke_test": {"passed": True, "status_code": 200, "detail": None},
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-compile", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--smoke-test",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Smoke test: passed" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "smoke_test": True,
    }


def test_remote_compile_command_exits_1_when_the_smoke_test_fails(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "functions": [{"name": "add"}],
            "endpoints": [{"path": "/add", "method": "POST", "is_async": False}],
            "skipped_functions": [],
            "dependencies": [],
            "generated_files": ["app.py"],
            "message": "Notebook compiled successfully",
            "smoke_test": {
                "passed": False, "status_code": None,
                "detail": "Compiled app failed to import: boom",
            },
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-compile", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--smoke-test",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Smoke test: FAILED" in proc.stdout
    assert "boom" in proc.stdout


def test_remote_compile_command_passes_only_through_to_the_dashboard(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "functions": [{"name": "add"}],
            "endpoints": [{"path": "/add", "method": "POST", "is_async": False}],
            "skipped_functions": [],
            "dependencies": [],
            "generated_files": [],
            "message": "Notebook compiled successfully",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-compile", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--only", "add, subtract",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "only": ["add", "subtract"],
    }


def test_remote_compile_command_passes_exclude_through_to_the_dashboard(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "functions": [{"name": "add"}],
            "endpoints": [{"path": "/add", "method": "POST", "is_async": False}],
            "skipped_functions": [],
            "dependencies": [],
            "generated_files": [],
            "message": "Notebook compiled successfully",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-compile", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--exclude", "subtract",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "exclude": ["subtract"],
    }


def test_remote_compile_command_passes_the_version_id_flag_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "version_id": "v1.ipynb",
            "functions": [{"name": "add"}],
            "endpoints": [{"path": "/add", "method": "POST", "is_async": False}],
            "skipped_functions": [],
            "dependencies": [],
            "generated_files": [],
            "message": "Notebook compiled successfully",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-compile", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--version-id", "v1.ipynb",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "'nb.ipynb' version 'v1.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "version_id": "v1.ipynb",
    }


def test_remote_compile_command_reports_the_dashboards_error_for_conflicting_only_and_exclude(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(
            400,
            {"detail": "only and exclude can't both be given -- choose one."},
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-compile", "nb.ipynb",
            "--dashboard-url", dashboard_url,
            "--only", "add", "--exclude", "subtract",
        ],
        cwd=workdir,
    )

    assert proc.returncode != 0
    assert "only and exclude" in proc.stdout + proc.stderr


def test_remote_compile_command_flags_background_endpoints(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "functions": [],
            "endpoints": [{"path": "/train_model", "method": "POST", "is_async": True}],
            "skipped_functions": [],
            "dependencies": [],
            "generated_files": [],
            "message": "Notebook compiled successfully",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-compile", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "POST /train_model  [background]" in proc.stdout


def test_remote_compile_command_reports_skipped_functions(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "functions": [],
            "endpoints": [],
            "skipped_functions": [{"name": "unsupported", "reason": "uses **kwargs"}],
            "dependencies": [],
            "generated_files": [],
            "message": "Notebook compiled successfully",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-compile", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 skipped function(s):" in proc.stdout
    assert "unsupported: uses **kwargs" in proc.stdout


def test_remote_compile_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebook": "nb.ipynb", "functions": [],
            "endpoints": [], "skipped_functions": [], "dependencies": [],
            "generated_files": [], "message": "Notebook compiled successfully",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-compile", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["notebook"] == "nb.ipynb"


def test_remote_compile_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-compile", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_remote_compile_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-compile", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_remote_inspect_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "remote-inspect" in proc.stdout


def test_remote_inspect_command_prints_the_full_report(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "functions": [{"name": "add"}, {"name": "multiply"}],
            "dependencies": ["numpy", "pandas"],
            "generated_files": ["app.py", "requirements.txt"],
            "reserved_name_conflicts": [],
            "endpoints": [
                {"path": "/add", "method": "POST", "is_async": False},
                {"path": "/multiply", "method": "POST", "is_async": True},
            ],
            "skipped_functions": [],
            "private_functions": [],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-inspect", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Inspecting 'nb.ipynb'" in proc.stdout
    assert "2 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert "POST /multiply  [background]" in proc.stdout
    assert "Dependencies: numpy, pandas" in proc.stdout
    assert "Generated files: app.py, requirements.txt" in proc.stdout
    assert handler.requests == ["/api/inspect"]
    assert json.loads(handler.bodies[0]) == {"notebook_path": "nb.ipynb"}


def test_remote_inspect_command_reports_reserved_name_conflicts_skipped_and_private_functions(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "functions": [],
            "dependencies": [],
            "generated_files": [],
            "reserved_name_conflicts": ["health_check"],
            "endpoints": [],
            "skipped_functions": [{"name": "load_data", "reason": "no return type"}],
            "private_functions": ["_helper"],
            "excluded_imports": ["pandas"],
            "functions_without_docstrings": ["undocumented"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-inspect", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Reserved name conflicts (compilation will fail):" in proc.stdout
    assert "- health_check" in proc.stdout
    assert "Private functions (never exposed as an endpoint):" in proc.stdout
    assert "- _helper" in proc.stdout
    assert "Excluded imports (opted out of requirements.txt):" in proc.stdout
    assert "- pandas" in proc.stdout
    assert "Functions without a docstring (will get a generic OpenAPI description):" in proc.stdout
    assert "- undocumented" in proc.stdout
    assert "Skipped functions (no endpoint will be generated):" in proc.stdout
    assert "- load_data: no return type" in proc.stdout


def test_remote_inspect_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "functions": [], "dependencies": [], "generated_files": [],
        "reserved_name_conflicts": [], "endpoints": [],
        "skipped_functions": [], "private_functions": [],
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-inspect", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_remote_inspect_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-inspect", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_remote_inspect_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-inspect", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_remote_validate_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "remote-validate" in proc.stdout


def test_remote_validate_command_passes_a_clean_notebook(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "pass",
            "notebook": "nb.ipynb",
            "reserved_name_conflicts": [],
            "skipped_functions": [],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-validate", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No issues found." in proc.stdout
    assert handler.requests == ["/api/validate"]
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "strict": False,
    }


def test_remote_validate_command_passes_the_version_id_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "fail",
            "notebook": "nb.ipynb",
            "version_id": "v1.ipynb",
            "reserved_name_conflicts": ["health_check"],
            "skipped_functions": [],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-validate", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--version-id", "v1.ipynb",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "'nb.ipynb' version 'v1.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "strict": False, "version_id": "v1.ipynb",
    }


def test_remote_validate_command_warns_but_does_not_fail_on_skipped_functions(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "warn",
            "notebook": "nb.ipynb",
            "reserved_name_conflicts": [],
            "skipped_functions": [{"name": "unsupported", "reason": "**kwargs"}],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-validate", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Skipped functions" in proc.stdout
    assert "unsupported" in proc.stdout
    assert "still compile cleanly" in proc.stdout


def test_remote_validate_command_warns_but_does_not_fail_on_duplicate_functions(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "warn",
            "notebook": "nb.ipynb",
            "reserved_name_conflicts": [],
            "skipped_functions": [],
            "duplicate_functions": ["add"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-validate", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Duplicate functions" in proc.stdout
    assert "add" in proc.stdout
    assert "still compile cleanly" in proc.stdout


def test_remote_validate_command_strict_flag_fails_on_skipped_functions(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "fail",
            "notebook": "nb.ipynb",
            "reserved_name_conflicts": [],
            "skipped_functions": [{"name": "unsupported", "reason": "**kwargs"}],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-validate", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--strict",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Validation failed." in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "strict": True,
    }


def test_remote_validate_command_fails_on_a_reserved_name_conflict(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "fail",
            "notebook": "nb.ipynb",
            "reserved_name_conflicts": ["health_check"],
            "skipped_functions": [],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-validate", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Reserved name conflicts" in proc.stdout
    assert "health_check" in proc.stdout
    assert "Validation failed." in proc.stdout


def test_remote_validate_command_sends_only_and_exclude_body_fields(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "pass",
            "notebook": "nb.ipynb",
            "reserved_name_conflicts": [],
            "skipped_functions": [],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-validate", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--exclude", "health_check",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "strict": False,
        "exclude": ["health_check"],
    }


def test_remote_validate_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "pass",
            "notebook": "nb.ipynb",
            "reserved_name_conflicts": [],
            "skipped_functions": [],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-validate", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["status"] == "pass"


def test_remote_validate_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-validate", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_remote_validate_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-validate", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_validate_all_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "validate-all" in proc.stdout


def test_validate_all_command_passes_when_every_notebook_is_clean(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "filename": "a.ipynb", "status": "pass",
                    "reserved_name_conflicts": [], "skipped_functions": [], "detail": None,
                },
            ],
            "pass_count": 1,
            "warn_count": 0,
            "fail_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["validate-all", "--dashboard-url", dashboard_url], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "✓ a.ipynb: pass" in proc.stdout
    assert "1 passed, 0 warned, 0 failed" in proc.stdout
    assert handler.requests == ["/api/validate-all?strict=false&offset=0"]


def test_validate_all_command_sends_tag_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [], "pass_count": 0,
            "warn_count": 0, "fail_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["validate-all", "--tag", "prod", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query == {"strict": ["false"], "tag": ["prod"], "offset": ["0"]}


def test_validate_all_command_exits_1_on_warnings_without_failing(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "filename": "warn.ipynb", "status": "warn",
                    "reserved_name_conflicts": [],
                    "skipped_functions": [{"name": "unsupported", "reason": "**kwargs"}],
                    "detail": None,
                },
            ],
            "pass_count": 0,
            "warn_count": 1,
            "fail_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["validate-all", "--dashboard-url", dashboard_url], cwd=workdir
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "⚠ warn.ipynb: warn" in proc.stdout
    assert "skipped: unsupported: **kwargs" in proc.stdout


def test_validate_all_command_exits_2_on_a_failure_and_passes_strict_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "filename": "bad.ipynb", "status": "fail",
                    "reserved_name_conflicts": ["health_check"],
                    "skipped_functions": [], "detail": None,
                },
            ],
            "pass_count": 0,
            "warn_count": 0,
            "fail_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["validate-all", "--dashboard-url", dashboard_url, "--strict"], cwd=workdir
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "✗ bad.ipynb: fail" in proc.stdout
    assert "reserved name conflict: health_check" in proc.stdout
    assert handler.requests == ["/api/validate-all?strict=true&offset=0"]


def test_validate_all_command_reports_no_notebooks(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "results": [],
            "pass_count": 0, "warn_count": 0, "fail_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["validate-all", "--dashboard-url", dashboard_url], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No notebooks to validate" in proc.stdout


def test_validate_all_command_format_csv_prints_the_dashboards_raw_csv_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    csv_body = (
        "filename,status,reserved_name_conflicts,skipped_functions,detail\r\n"
        "a.ipynb,pass,,,\r\n"
    )
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["validate-all", "--format", "csv", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    # --format csv always exits 0 -- it's for archiving/reporting, not
    # the default JSON/human mode's own CI-gating exit code.
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == ["/api/validate-all?strict=false&offset=0&format=csv"]


def test_validate_all_command_passes_limit_and_offset_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {
                    "filename": "b.ipynb", "status": "pass",
                    "reserved_name_conflicts": [], "skipped_functions": [], "detail": None,
                },
            ],
            "result_count": 3, "limit": 1, "offset": 1,
            "pass_count": 3, "warn_count": 0, "fail_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "validate-all", "--limit", "1", "--offset", "1",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "✓ b.ipynb: pass" in proc.stdout
    assert "Showing 1 of 3 result(s) (offset 1)." in proc.stdout
    assert "3 passed, 0 warned, 0 failed" in proc.stdout
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query == {"strict": ["false"], "offset": ["1"], "limit": ["1"]}


def test_validate_all_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "results": [],
        "pass_count": 0, "warn_count": 0, "fail_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["validate-all", "--dashboard-url", dashboard_url, "--json"], cwd=workdir
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_validate_all_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "validate-all",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_requirements_preview_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "requirements-preview" in proc.stdout


def test_requirements_preview_command_prints_the_requirements(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "requirements": ["fastapi==0.100.0", "pandas==2.1.0"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["requirements-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "fastapi==0.100.0" in proc.stdout
    assert "pandas==2.1.0" in proc.stdout
    assert handler.requests == ["/api/requirements-preview"]
    assert json.loads(handler.bodies[0]) == {"notebook_path": "nb.ipynb"}


def test_requirements_preview_command_marks_explicit_requirements(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "requirements": ["fastapi==0.100.0", "a-private-pkg==1.0.0"],
            "explicit_requirements": ["a-private-pkg==1.0.0"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["requirements-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    lines = {line.strip() for line in proc.stdout.splitlines()}
    assert "a-private-pkg==1.0.0  (explicit)" in lines
    assert "fastapi==0.100.0" in lines


def test_requirements_preview_command_lists_excluded_imports(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "requirements": ["fastapi==0.100.0"],
            "explicit_requirements": [],
            "excluded_imports": ["pandas"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["requirements-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Excluded imports (opted out of requirements.txt):" in proc.stdout
    assert "pandas" in proc.stdout


def test_requirements_preview_command_passes_the_version_id_flag_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "version_id": "v1.ipynb",
            "requirements": ["pandas==2.1.0"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "requirements-preview", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--version-id", "v1.ipynb",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "'nb.ipynb' version 'v1.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "version_id": "v1.ipynb",
    }


def test_requirements_preview_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "notebook": "nb.ipynb",
        "requirements": ["fastapi==0.100.0"],
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["requirements-preview", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_requirements_preview_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["requirements-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_requirements_preview_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "requirements-preview", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_app_preview_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "app-preview" in proc.stdout


def test_app_preview_command_prints_the_generated_source(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "package_name": "generated",
            "app_code": "from fastapi import FastAPI\n\napp = FastAPI()\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["app-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "app.py preview for 'nb.ipynb'" in proc.stdout
    assert "package 'generated'" in proc.stdout
    assert "app = FastAPI()" in proc.stdout
    assert handler.requests == ["/api/app-preview"]
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "only": None, "exclude": None,
    }


def test_app_preview_command_passes_only_and_exclude(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "package_name": "generated",
            "app_code": "app = FastAPI()\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "app-preview", "nb.ipynb", "--only", "add, subtract",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "only": ["add", "subtract"], "exclude": None,
    }


def test_app_preview_command_passes_the_version_id_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "version_id": "v1.ipynb",
            "package_name": "generated",
            "app_code": "app = FastAPI()\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "app-preview", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--version-id", "v1.ipynb",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "'nb.ipynb' version 'v1.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb", "only": None, "exclude": None,
        "version_id": "v1.ipynb",
    }


def test_app_preview_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "notebook": "nb.ipynb",
        "package_name": "generated", "app_code": "app = FastAPI()\n",
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["app-preview", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_app_preview_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["app-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_app_preview_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "app-preview", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_curl_preview_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "curl-preview" in proc.stdout


def test_curl_preview_command_prints_the_commands(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "commands": [
                "# add\ncurl -X POST http://localhost:8000/add \\\n"
                '  -H "Content-Type: application/json" \\\n'
                '  -H "X-API-Key: notebook-to-api-dev-key" \\\n'
                "  -d '{}'",
            ],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["curl-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "curl -X POST http://localhost:8000/add" in proc.stdout
    assert handler.requests == ["/api/curl-preview"]
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb",
        "host": "localhost",
        "port": 8000,
        "api_key": "notebook-to-api-dev-key",
    }


def test_curl_preview_command_passes_host_port_and_api_key_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebook": "nb.ipynb", "commands": [],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "curl-preview", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--host", "api.example.com", "--port", "9000", "--api-key", "mykey123",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb",
        "host": "api.example.com",
        "port": 9000,
        "api_key": "mykey123",
    }


def test_curl_preview_command_passes_the_version_id_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebook": "nb.ipynb", "version_id": "v1.ipynb",
            "commands": ["curl -X POST http://localhost:8000/old_func"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "curl-preview", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--version-id", "v1.ipynb",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "'nb.ipynb' version 'v1.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb",
        "host": "localhost",
        "port": 8000,
        "api_key": "notebook-to-api-dev-key",
        "version_id": "v1.ipynb",
    }


def test_curl_preview_command_reports_no_endpoints(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebook": "nb.ipynb", "commands": [],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["curl-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No endpoints would be generated" in proc.stdout


def test_curl_preview_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "notebook": "nb.ipynb",
        "commands": ["curl -X POST http://localhost:8000/add"],
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["curl-preview", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_curl_preview_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["curl-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_curl_preview_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "curl-preview", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_postman_preview_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "postman-preview" in proc.stdout


def test_postman_preview_command_lists_the_request_names(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "notebook": "nb.ipynb",
            "collection": {
                "info": {"name": "nb"},
                "variable": [],
                "item": [{"name": "add"}, {"name": "subtract"}],
            },
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["postman-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "- add" in proc.stdout
    assert "- subtract" in proc.stdout
    assert "2 request(s) total" in proc.stdout
    assert handler.requests == ["/api/postman-preview"]
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb",
        "host": "localhost",
        "port": 8000,
        "api_key": "notebook-to-api-dev-key",
        "only": None,
        "exclude": None,
        "collection_name": None,
    }


def test_postman_preview_command_passes_host_port_api_key_and_name_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebook": "nb.ipynb",
            "collection": {"info": {"name": "My API"}, "variable": [], "item": []},
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "postman-preview", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--host", "api.example.com", "--port", "9000", "--api-key", "mykey123",
            "--collection-name", "My API",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb",
        "host": "api.example.com",
        "port": 9000,
        "api_key": "mykey123",
        "only": None,
        "exclude": None,
        "collection_name": "My API",
    }


def test_postman_preview_command_passes_only_exclude_and_version_id_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebook": "nb.ipynb", "version_id": "v1.ipynb",
            "collection": {"info": {"name": "nb"}, "variable": [], "item": []},
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "postman-preview", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--version-id", "v1.ipynb",
            "--only", "add",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "'nb.ipynb' version 'v1.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "notebook_path": "nb.ipynb",
        "host": "localhost",
        "port": 8000,
        "api_key": "notebook-to-api-dev-key",
        "only": ["add"],
        "exclude": None,
        "collection_name": None,
        "version_id": "v1.ipynb",
    }


def test_postman_preview_command_reports_no_endpoints(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "notebook": "nb.ipynb",
            "collection": {"info": {"name": "nb"}, "variable": [], "item": []},
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["postman-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No endpoints would be generated" in proc.stdout


def test_postman_preview_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "notebook": "nb.ipynb",
        "collection": {"info": {"name": "nb"}, "variable": [], "item": [{"name": "add"}]},
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["postman-preview", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_postman_preview_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["postman-preview", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_postman_preview_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "postman-preview", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_dockerfile_preview_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "dockerfile-preview" in proc.stdout


def test_dockerfile_preview_command_prints_the_dockerfile(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "package_name": "generated",
            "compiling_python_version": "3.12",
            "dockerfile": "FROM python:3.12-slim\n",
            "dockerignore": ".git/\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["dockerfile-preview", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FROM python:3.12-slim" in proc.stdout
    assert ".git/" in proc.stdout
    assert "package 'generated'" in proc.stdout
    assert "Python 3.12" in proc.stdout
    assert handler.requests == ["/api/dockerfile-preview"]


def test_dockerfile_preview_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "package_name": "generated",
        "compiling_python_version": "3.12",
        "dockerfile": "FROM python:3.12-slim\n",
        "dockerignore": ".git/\n",
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["dockerfile-preview", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_dockerfile_preview_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "dockerfile-preview",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_docker_compose_preview_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "docker-compose-preview" in proc.stdout


def test_docker_compose_preview_command_prints_the_docker_compose_file(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "package_name": "generated",
            "docker_compose": "services:\n  generated:\n    build: .\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["docker-compose-preview", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "services:" in proc.stdout
    assert "package 'generated'" in proc.stdout
    assert handler.requests == ["/api/docker-compose-preview"]


def test_docker_compose_preview_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "package_name": "generated",
        "docker_compose": "services:\n  generated:\n    build: .\n",
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["docker-compose-preview", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_docker_compose_preview_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "docker-compose-preview",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_k8s_preview_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "k8s-preview" in proc.stdout


def test_k8s_preview_command_prints_the_manifest(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "package_name": "generated",
            "kubernetes_manifest": "apiVersion: apps/v1\nkind: Deployment\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["k8s-preview", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "kind: Deployment" in proc.stdout
    assert "package 'generated'" in proc.stdout
    assert handler.requests == ["/api/k8s-preview"]


def test_k8s_preview_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "package_name": "generated",
        "kubernetes_manifest": "apiVersion: apps/v1\nkind: Deployment\n",
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["k8s-preview", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_k8s_preview_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "k8s-preview",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_env_example_preview_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "env-example-preview" in proc.stdout


def test_env_example_preview_command_prints_the_env_example_file(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "env_example": "PORT=8000\nNOTEBOOK_API_KEY=notebook-to-api-dev-key\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["env-example-preview", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PORT=8000" in proc.stdout
    assert "NOTEBOOK_API_KEY=notebook-to-api-dev-key" in proc.stdout
    assert handler.requests == ["/api/env-example-preview"]


def test_env_example_preview_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "env_example": "PORT=8000\n",
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["env-example-preview", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_env_example_preview_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "env-example-preview",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_env_vars_preview_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "env-vars-preview" in proc.stdout


def test_env_vars_preview_command_prints_each_env_var(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "environment_variables": [
                {
                    "name": "NOTEBOOK_API_KEY",
                    "default": "notebook-to-api-dev-key",
                    "description": "Accepted X-API-Key values.",
                },
                {
                    "name": "NOTEBOOK_API_MAX_TASKS",
                    "default": "10000",
                    "description": "Maximum pending background tasks.",
                },
            ],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["env-vars-preview", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NOTEBOOK_API_KEY (default: 'notebook-to-api-dev-key')" in proc.stdout
    assert "Accepted X-API-Key values." in proc.stdout
    assert "NOTEBOOK_API_MAX_TASKS (default: '10000')" in proc.stdout
    assert "Maximum pending background tasks." in proc.stdout
    assert handler.requests == ["/api/env-vars-preview"]


def test_env_vars_preview_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "environment_variables": [
            {
                "name": "NOTEBOOK_API_KEY",
                "default": "notebook-to-api-dev-key",
                "description": "Accepted X-API-Key values.",
            },
        ],
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["env-vars-preview", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_env_vars_preview_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "env-vars-preview",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_remote_build_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "remote-build" in proc.stdout


def test_remote_build_command_saves_the_zip_using_the_content_disposition_name(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x03\x04fake-zip-content"
    handler.responses = [
        (
            200,
            zip_bytes,
            "application/zip",
        )
    ]
    handler.response_headers = [{"Content-Disposition": 'attachment; filename="generated.zip"'}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-build", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "generated.zip" in proc.stdout
    assert (workdir / "generated.zip").read_bytes() == zip_bytes
    assert handler.requests == ["/api/download"]


def test_remote_build_command_respects_a_custom_output_path(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x03\x04fake-zip-content"
    handler.responses = [(200, zip_bytes, "application/zip")]
    handler.response_headers = [{"Content-Disposition": 'attachment; filename="generated.zip"'}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-build", "--dashboard-url", dashboard_url,
            "--output", "my-build.zip",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "my-build.zip").read_bytes() == zip_bytes
    assert not (workdir / "generated.zip").exists()


def test_remote_build_command_expected_sha256_succeeds_on_a_match(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x03\x04fake-zip-content"
    handler.responses = [(200, zip_bytes, "application/zip")]
    handler.response_headers = [{
        "Content-Disposition": 'attachment; filename="generated.zip"',
        "X-Bundle-SHA256": "abc123",
    }]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-build", "--dashboard-url", dashboard_url,
            "--expected-sha256", "abc123",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "generated.zip").read_bytes() == zip_bytes
    assert "bundle sha256: abc123" in proc.stdout


def test_remote_build_command_expected_sha256_fails_on_a_mismatch_and_writes_nothing(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x03\x04fake-zip-content"
    handler.responses = [(200, zip_bytes, "application/zip")]
    handler.response_headers = [{
        "Content-Disposition": 'attachment; filename="generated.zip"',
        "X-Bundle-SHA256": "abc123",
    }]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-build", "--dashboard-url", dashboard_url,
            "--expected-sha256", "does-not-match",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "does not match the expected value")
    assert not (workdir / "generated.zip").exists()


def test_remote_build_command_json_flag_emits_a_machine_readable_result(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x03\x04fake-zip-content"
    handler.responses = [(200, zip_bytes, "application/zip")]
    handler.response_headers = [{"Content-Disposition": 'attachment; filename="generated.zip"'}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-build", "--dashboard-url", dashboard_url, "--json"], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert data["size_bytes"] == len(zip_bytes)
    assert data["notebook_changed_since_compile"] is False


def test_remote_build_command_warns_when_the_notebook_changed_since_compile(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x03\x04fake-zip-content"
    handler.responses = [(200, zip_bytes, "application/zip")]
    handler.response_headers = [{
        "Content-Disposition": 'attachment; filename="generated.zip"',
        "X-Notebook-Changed-Since-Compile": "true",
    }]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-build", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "warning" in proc.stdout.lower()
    assert "changed since" in proc.stdout


def test_remote_build_command_json_flag_reports_staleness(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x03\x04fake-zip-content"
    handler.responses = [(200, zip_bytes, "application/zip")]
    handler.response_headers = [{
        "Content-Disposition": 'attachment; filename="generated.zip"',
        "X-Notebook-Changed-Since-Compile": "true",
    }]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-build", "--dashboard-url", dashboard_url, "--json"], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["notebook_changed_since_compile"] is True


def test_remote_build_command_does_not_warn_when_not_stale(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x03\x04fake-zip-content"
    handler.responses = [(200, zip_bytes, "application/zip")]
    handler.response_headers = [{
        "Content-Disposition": 'attachment; filename="generated.zip"',
        "X-Notebook-Changed-Since-Compile": "false",
    }]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-build", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "warning" not in proc.stdout.lower()


def test_remote_build_command_reports_a_clean_error_when_no_app_is_compiled(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "No compiled app found. Run /api/compile first."})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-build", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No compiled app found")


def test_remote_build_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-build",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_versions_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "versions" in proc.stdout


def test_versions_list_command_prints_the_notebooks_versions(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "versions": [
                {
                    "version_id": "20260101T000000Z-abcdef.ipynb",
                    "size_bytes": 512,
                    "saved_at": "2026-01-01T00:00:00+00:00",
                },
            ],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["versions", "list", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "20260101T000000Z-abcdef.ipynb" in proc.stdout
    assert "512 bytes" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions?offset=0"]


def test_versions_list_command_checksums_flag_sends_the_query_param_and_prints_sha256(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "versions": [
                {
                    "version_id": "20260101T000000Z-abcdef.ipynb",
                    "size_bytes": 512,
                    "saved_at": "2026-01-01T00:00:00+00:00",
                    "sha256": "abc123",
                },
            ],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "list", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--checksums",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sha256:abc123" in proc.stdout
    assert handler.requests == [
        "/api/notebooks/nb.ipynb/versions?offset=0&checksums=true"
    ]


def test_versions_list_command_reports_no_saved_versions(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "filename": "nb.ipynb", "versions": []})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["versions", "list", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No saved versions for 'nb.ipynb'." in proc.stdout


def test_versions_list_command_format_csv_prints_the_dashboards_raw_csv_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    csv_body = "version_id,size_bytes,saved_at\r\nv1.ipynb,10,2026-01-01T00:00:00+00:00\r\n"
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "list", "nb.ipynb",
            "--format", "csv", "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions?offset=0&format=csv"]


def test_versions_list_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "versions": [
                {"version_id": "v1.ipynb", "size_bytes": 10, "saved_at": "2026-01-01T00:00:00+00:00"}
            ],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["versions", "list", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["versions"][0]["version_id"] == "v1.ipynb"


def test_versions_list_command_passes_limit_and_offset_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "versions": [
                {"version_id": "v2.ipynb", "size_bytes": 20, "saved_at": "2026-01-02T00:00:00+00:00"},
            ],
            "total_count": 3, "limit": 1, "offset": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "list", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--limit", "1", "--offset", "1",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "v2.ipynb" in proc.stdout
    assert "1 of 3 total version(s) shown" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions?offset=1&limit=1"]


def test_versions_list_command_sends_saved_after_and_before_query_params(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "filename": "nb.ipynb", "versions": []})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "list", "nb.ipynb",
            "--saved-after", "2026-01-01T00:00:00+00:00",
            "--saved-before", "2026-06-01T00:00:00+00:00",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query == {
        "offset": ["0"],
        "saved_after": ["2026-01-01T00:00:00+00:00"],
        "saved_before": ["2026-06-01T00:00:00+00:00"],
    }


def test_versions_list_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["versions", "list", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_versions_get_command_downloads_a_version_to_the_default_path(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(200, b'{"nbformat": 4, "cells": []}')
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "get", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "v1.ipynb").read_bytes() == b'{"nbformat": 4, "cells": []}'
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/v1.ipynb"]


def test_versions_get_command_prints_and_reports_the_content_sha256_header(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    notebook_bytes = b'{"nbformat": 4, "cells": []}'
    handler.responses = [_raw_response(200, notebook_bytes)]
    handler.response_headers = [{"X-Content-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "get", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sha256: abc123" in proc.stdout

    handler.responses = [_raw_response(200, notebook_bytes)]
    handler.response_headers = [{"X-Content-SHA256": "abc123"}]

    proc = _run_cli(
        [
            "versions", "get", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["sha256"] == "abc123"


def test_versions_get_command_expected_sha256_succeeds_on_a_match(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    notebook_bytes = b'{"nbformat": 4, "cells": []}'
    handler.responses = [_raw_response(200, notebook_bytes)]
    handler.response_headers = [{"X-Content-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "get", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--expected-sha256", "abc123",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "v1.ipynb").read_bytes() == notebook_bytes


def test_versions_get_command_expected_sha256_fails_on_a_mismatch_and_writes_nothing(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    notebook_bytes = b'{"nbformat": 4, "cells": []}'
    handler.responses = [_raw_response(200, notebook_bytes)]
    handler.response_headers = [{"X-Content-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "get", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
            "--expected-sha256", "does-not-match",
        ],
        cwd=workdir,
    )

    assert proc.returncode != 0
    assert "does not match" in (proc.stdout + proc.stderr)
    assert not (workdir / "v1.ipynb").exists()


def test_versions_get_command_respects_a_custom_output_path(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [_raw_response(200, b"notebook-bytes")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "get", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--output", "restored.ipynb",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "restored.ipynb").read_bytes() == b"notebook-bytes"
    assert not (workdir / "v1.ipynb").exists()


def test_versions_get_command_json_flag_emits_a_machine_readable_result(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [_raw_response(200, b"notebook-bytes")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "get", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["version_id"] == "v1.ipynb"
    assert data["size_bytes"] == len(b"notebook-bytes")


def test_versions_get_command_reports_a_clean_error_for_a_missing_version(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook version not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "get", "nb.ipynb", "does-not-exist.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook version not found")


def test_versions_inspect_command_is_registered():

    proc = _run_cli(["versions", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "inspect" in proc.stdout


def test_versions_inspect_command_prints_endpoints_and_dependencies(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "version_id": "v1.ipynb",
            "functions": [{"name": "add"}],
            "dependencies": ["pandas==2.1.0"],
            "generated_files": [],
            "reserved_name_conflicts": [],
            "endpoints": [{"path": "/add", "method": "POST", "is_async": False}],
            "skipped_functions": [],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "inspect", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Inspecting version 'v1.ipynb' of 'nb.ipynb'" in proc.stdout
    assert "1 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert "Dependencies: pandas==2.1.0" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/v1.ipynb/inspect"]


def test_versions_inspect_command_prints_reserved_name_conflicts_and_skipped_functions(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "version_id": "v1.ipynb",
            "functions": [],
            "dependencies": [],
            "generated_files": [],
            "reserved_name_conflicts": ["health_check"],
            "endpoints": [],
            "skipped_functions": [{"name": "helper", "reason": "uses **kwargs"}],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "inspect", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Reserved name conflicts" in proc.stdout
    assert "health_check" in proc.stdout
    assert "1 skipped function(s):" in proc.stdout
    assert "helper: uses **kwargs" in proc.stdout


def test_versions_inspect_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "filename": "nb.ipynb",
        "version_id": "v1.ipynb",
        "functions": [],
        "dependencies": [],
        "generated_files": [],
        "reserved_name_conflicts": [],
        "endpoints": [],
        "skipped_functions": [],
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "inspect", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_versions_inspect_command_reports_a_clean_error_for_a_missing_version(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook version not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "inspect", "nb.ipynb", "does-not-exist.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook version not found")


def test_versions_export_command_is_registered():

    proc = _run_cli(["versions", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "export" in proc.stdout


def test_versions_export_command_saves_the_zip_to_the_default_path(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["versions", "export", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"({len(zip_bytes)} bytes)" in proc.stdout
    assert (workdir / "nb.ipynb.versions.zip").read_bytes() == zip_bytes
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/export"]


def test_versions_export_command_expected_sha256_succeeds_on_a_match(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]
    handler.response_headers = [{"X-Bundle-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "export", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--expected-sha256", "abc123",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "nb.ipynb.versions.zip").read_bytes() == zip_bytes
    assert "bundle sha256: abc123" in proc.stdout


def test_versions_export_command_expected_sha256_fails_on_a_mismatch_and_writes_nothing(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]
    handler.response_headers = [{"X-Bundle-SHA256": "abc123"}]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "export", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--expected-sha256", "does-not-match",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "does not match the expected value")
    assert not (workdir / "nb.ipynb.versions.zip").exists()


def test_versions_export_command_version_id_sends_a_comma_separated_query_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "export", "nb.ipynb",
            "--version-id", "v1.ipynb", "v2.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks/nb.ipynb/versions/export?version_ids=v1.ipynb%2Cv2.ipynb"
    ]


def test_versions_export_command_omits_version_ids_query_param_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["versions", "export", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/export"]


def test_versions_export_command_respects_a_custom_output_path(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "export", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--output", "backup.zip",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "backup.zip").read_bytes() == zip_bytes
    assert not (workdir / "nb.ipynb.versions.zip").exists()


def test_versions_export_command_json_flag_emits_a_machine_readable_result(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    zip_bytes = b"PK\x05\x06" + b"\x00" * 18
    handler.responses = [_raw_response(200, zip_bytes, content_type="application/zip")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "export", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["filename"] == "nb.ipynb"
    assert data["size_bytes"] == len(zip_bytes)


def test_versions_export_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["versions", "export", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_versions_import_command_is_registered():

    proc = _run_cli(["versions", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "import" in proc.stdout


def test_versions_import_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "overwritten": False,
            "imported_version_ids": ["v1.ipynb", "v2.ipynb"],
            "imported_version_count": 2,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "backup.zip"
    _write_zip(zip_path, {"nb.ipynb": b"{}", "versions/v1.ipynb": b"{}", "versions/v2.ipynb": b"{}"})

    proc = _run_cli(
        ["versions", "import", "nb.ipynb", str(zip_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Restored 'nb.ipynb' (overwritten: False) with 2 version(s)" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/import?overwrite=false"]


def test_versions_import_command_passes_the_overwrite_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb", "overwritten": True,
            "imported_version_ids": [], "imported_version_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "backup.zip"
    _write_zip(zip_path, {"nb.ipynb": b"{}"})

    proc = _run_cli(
        [
            "versions", "import", "nb.ipynb", str(zip_path),
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/import?overwrite=true"]


def test_versions_import_command_passes_the_expected_sha256_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb", "overwritten": False,
            "imported_version_ids": [], "imported_version_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "backup.zip"
    _write_zip(zip_path, {"nb.ipynb": b"{}"})

    proc = _run_cli(
        [
            "versions", "import", "nb.ipynb", str(zip_path),
            "--dashboard-url", dashboard_url, "--expected-sha256", "abc123",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/notebooks/nb.ipynb/versions/import?overwrite=false&expected_sha256=abc123"
    ]


def test_versions_import_command_reports_a_clean_error_for_a_mismatched_expected_sha256(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(400, {
            "detail": "Uploaded archive does not match expected_sha256: expected abc, got def",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "backup.zip"
    _write_zip(zip_path, {"nb.ipynb": b"{}"})

    proc = _run_cli(
        [
            "versions", "import", "nb.ipynb", str(zip_path),
            "--dashboard-url", dashboard_url, "--expected-sha256", "abc",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "does not match expected_sha256")


def test_versions_import_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "filename": "nb.ipynb", "overwritten": False,
        "imported_version_ids": ["v1.ipynb"], "imported_version_count": 1,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "backup.zip"
    _write_zip(zip_path, {"nb.ipynb": b"{}"})

    proc = _run_cli(
        [
            "versions", "import", "nb.ipynb", str(zip_path),
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_versions_import_command_reports_a_clean_error_for_a_rejected_import(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(409, {"detail": "A notebook named 'nb.ipynb' already exists. Pass ?overwrite=true to replace it."})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "backup.zip"
    _write_zip(zip_path, {"nb.ipynb": b"{}"})

    proc = _run_cli(
        ["versions", "import", "nb.ipynb", str(zip_path), "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "already exists")


def test_versions_import_command_reports_a_clean_error_for_a_missing_zip(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "import", "nb.ipynb", str(workdir / "does-not-exist.zip"),
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_versions_import_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    zip_path = workdir / "backup.zip"
    _write_zip(zip_path, {"nb.ipynb": b"{}"})

    proc = _run_cli(
        [
            "versions", "import", "nb.ipynb", str(zip_path),
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_versions_copy_command_is_registered():

    proc = _run_cli(["versions", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "copy" in proc.stdout


def test_versions_copy_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "version_id": "v1.ipynb", "new_filename": "nb-old.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy", "nb.ipynb", "v1.ipynb", "nb-old.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Copied version 'v1.ipynb' of 'nb.ipynb' to 'nb-old.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/v1.ipynb/copy"]
    assert json.loads(handler.bodies[0]) == {
        "new_filename": "nb-old.ipynb", "overwrite": False,
    }


def test_versions_copy_command_passes_overwrite_flag(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "version_id": "v1.ipynb", "new_filename": "nb-old.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy", "nb.ipynb", "v1.ipynb", "nb-old.ipynb",
            "--overwrite", "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "new_filename": "nb-old.ipynb", "overwrite": True,
    }


def test_versions_copy_command_dry_run_sends_dry_run_and_prints_would_copy(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True, "filename": "nb.ipynb",
            "version_id": "v1.ipynb", "new_filename": "nb-old.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy", "nb.ipynb", "v1.ipynb", "nb-old.ipynb",
            "--dry-run", "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0])["dry_run"] is True
    assert "Would copy version 'v1.ipynb' of 'nb.ipynb' to 'nb-old.ipynb'" in proc.stdout


def test_versions_copy_command_passes_tags_and_description_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "version_id": "v1.ipynb", "new_filename": "nb-recovered.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy", "nb.ipynb", "v1.ipynb", "nb-recovered.ipynb",
            "--dashboard-url", dashboard_url,
            "--tags", "recovered", "--description", "recovered snapshot",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    body = json.loads(handler.bodies[0])
    assert body["tags"] == ["recovered"]
    assert body["description"] == "recovered snapshot"


def test_versions_copy_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "filename": "nb.ipynb",
        "version_id": "v1.ipynb", "new_filename": "nb-old.ipynb",
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy", "nb.ipynb", "v1.ipynb", "nb-old.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_versions_copy_command_reports_a_clean_error_for_a_missing_version(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook version not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy", "nb.ipynb", "does-not-exist.ipynb", "nb-old.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook version not found")


def test_versions_copy_batch_command_is_registered():

    proc = _run_cli(["versions", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "copy-batch" in proc.stdout


def test_versions_copy_batch_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": False,
            "filename": "nb.ipynb",
            "results": [
                {"version_id": "v1.ipynb", "new_filename": "a.ipynb", "status": "success"},
                {"version_id": "v2.ipynb", "new_filename": "b.ipynb", "status": "success"},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy-batch", "nb.ipynb", "v1.ipynb:a.ipynb", "v2.ipynb:b.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Copied version 'v1.ipynb' to 'a.ipynb'" in proc.stdout
    assert "Copied version 'v2.ipynb' to 'b.ipynb'" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/copy-batch"]
    assert json.loads(handler.bodies[0]) == {
        "entries": [
            {"version_id": "v1.ipynb", "new_filename": "a.ipynb", "overwrite": False},
            {"version_id": "v2.ipynb", "new_filename": "b.ipynb", "overwrite": False},
        ]
    }


def test_versions_copy_batch_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "results": [
                {"version_id": "v1.ipynb", "new_filename": "a.ipynb", "status": "success"},
                {
                    "version_id": "missing.ipynb", "new_filename": "b.ipynb",
                    "status": "error", "detail": "Notebook version not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy-batch", "nb.ipynb", "v1.ipynb:a.ipynb", "missing.ipynb:b.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Copied version 'v1.ipynb' to 'a.ipynb'" in proc.stdout
    assert (
        "Failed to copy version 'missing.ipynb' to 'b.ipynb': "
        "Notebook version not found" in proc.stdout
    )
    assert "1 succeeded, 1 failed" in proc.stdout


def test_versions_copy_batch_command_passes_the_overwrite_flag_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [{"version_id": "v1.ipynb", "new_filename": "a.ipynb", "status": "success"}],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy-batch", "nb.ipynb", "v1.ipynb:a.ipynb",
            "--dashboard-url", dashboard_url, "--overwrite",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"version_id": "v1.ipynb", "new_filename": "a.ipynb", "overwrite": True}]
    }


def test_versions_copy_batch_command_dry_run_sends_dry_run_field(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [{"version_id": "v1.ipynb", "new_filename": "a.ipynb", "status": "success"}],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy-batch", "nb.ipynb", "v1.ipynb:a.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would copy version 'v1.ipynb' to 'a.ipynb'" in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"version_id": "v1.ipynb", "new_filename": "a.ipynb", "overwrite": False}],
        "dry_run": True,
    }


def test_versions_copy_batch_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "results": [{"version_id": "v1.ipynb", "new_filename": "a.ipynb", "status": "success"}],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy-batch", "nb.ipynb", "v1.ipynb:a.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_versions_copy_batch_command_rejects_a_malformed_entry(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["versions", "copy-batch", "nb.ipynb", "no-colon-here"],
        cwd=workdir,
    )

    assert proc.returncode != 0
    assert "version_id:new_filename" in proc.stderr


def test_versions_copy_batch_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "copy-batch", "nb.ipynb", "v1.ipynb:a.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_versions_restore_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "restored_version_id": "v1.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Restored 'nb.ipynb' to version 'v1.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/v1.ipynb/restore"]


def test_versions_restore_command_notes_when_it_restored_the_compiled_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "restored_version_id": "v1.ipynb",
            "was_currently_compiled": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "note: this was the notebook backing the currently compiled app." in proc.stdout


def test_versions_restore_command_dry_run_reports_would_restore_and_sends_dry_run_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "restored_version_id": "v1.ipynb", "dry_run": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would restore 'nb.ipynb' to version 'v1.ipynb'" in proc.stdout
    assert handler.requests == [
        "/api/notebooks/nb.ipynb/versions/v1.ipynb/restore?dry_run=true"
    ]


def test_versions_restore_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "restored_version_id": "v1.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["restored_version_id"] == "v1.ipynb"


def test_versions_restore_command_reports_a_clean_error_for_a_missing_version(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook version not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore", "nb.ipynb", "does-not-exist.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook version not found")


def test_versions_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "list", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_versions_delete_command_reports_success_with_yes_flag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "deleted_version_id": "v1.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "delete", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Deleted version 'v1.ipynb' of 'nb.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/v1.ipynb"]


def test_versions_delete_command_dry_run_skips_the_prompt_and_prints_would_delete(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True, "filename": "nb.ipynb",
            "deleted_version_id": "v1.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "delete", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would delete version 'v1.ipynb' of 'nb.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/v1.ipynb?dry_run=true"]


def test_versions_delete_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "deleted_version_id": "v1.ipynb",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "delete", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["deleted_version_id"] == "v1.ipynb"


def test_versions_delete_command_aborts_without_yes_when_declined(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "versions", "delete", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        input="n\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_versions_delete_command_reports_a_clean_error_for_a_missing_version(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook version not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "delete", "nb.ipynb", "does-not-exist.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook version not found")


def test_versions_delete_batch_command_is_registered():

    proc = _run_cli(["versions", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "delete-batch" in proc.stdout


def test_versions_delete_batch_command_reports_success_with_yes_flag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "results": [
                {"version_id": "v1.ipynb", "status": "success"},
                {"version_id": "v2.ipynb", "status": "success"},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "delete-batch", "nb.ipynb", "v1.ipynb", "v2.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Deleted version 'v1.ipynb'" in proc.stdout
    assert "Deleted version 'v2.ipynb'" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/delete-batch"]
    assert json.loads(handler.bodies[0]) == {"version_ids": ["v1.ipynb", "v2.ipynb"]}


def test_versions_delete_batch_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "results": [
                {"version_id": "v1.ipynb", "status": "success"},
                {
                    "version_id": "missing.ipynb", "status": "error",
                    "detail": "Notebook version not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "delete-batch", "nb.ipynb", "v1.ipynb", "missing.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Deleted version 'v1.ipynb'" in proc.stdout
    assert "Failed to delete version 'missing.ipynb': Notebook version not found" in proc.stdout
    assert "1 succeeded, 1 failed" in proc.stdout


def test_versions_delete_batch_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "filename": "nb.ipynb",
        "results": [{"version_id": "v1.ipynb", "status": "success"}],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "delete-batch", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_versions_delete_batch_command_aborts_without_yes_when_declined(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "versions", "delete-batch", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        input="n\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_versions_delete_batch_command_dry_run_skips_confirmation_and_sends_dry_run_field(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "filename": "nb.ipynb",
            "results": [
                {"version_id": "v1.ipynb", "status": "success"},
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        [
            "versions", "delete-batch", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would delete version 'v1.ipynb'" in proc.stdout
    assert "1 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/delete-batch"]
    assert json.loads(handler.bodies[0]) == {
        "version_ids": ["v1.ipynb"], "dry_run": True
    }


def test_versions_delete_batch_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "delete-batch", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_versions_restore_batch_command_is_registered():

    proc = _run_cli(["versions", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "restore-batch" in proc.stdout


def test_versions_restore_batch_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": False,
            "results": [
                {"filename": "a.ipynb", "version_id": "v1.ipynb", "status": "success", "restored_version_id": "v1.ipynb"},
                {"filename": "b.ipynb", "version_id": "v2.ipynb", "status": "success", "restored_version_id": "v2.ipynb"},
            ],
            "succeeded_count": 2,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore-batch", "a.ipynb:v1.ipynb", "b.ipynb:v2.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Restored 'a.ipynb' to version 'v1.ipynb'" in proc.stdout
    assert "Restored 'b.ipynb' to version 'v2.ipynb'" in proc.stdout
    assert "2 succeeded, 0 failed" in proc.stdout
    assert handler.requests == ["/api/notebooks/versions/restore-batch"]
    assert json.loads(handler.bodies[0]) == {
        "entries": [
            {"filename": "a.ipynb", "version_id": "v1.ipynb"},
            {"filename": "b.ipynb", "version_id": "v2.ipynb"},
        ]
    }


def test_versions_restore_batch_command_reports_a_partial_failure(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "results": [
                {"filename": "a.ipynb", "version_id": "v1.ipynb", "status": "success", "restored_version_id": "v1.ipynb"},
                {
                    "filename": "missing.ipynb", "version_id": "v9.ipynb", "status": "error",
                    "detail": "Notebook file not found",
                },
            ],
            "succeeded_count": 1,
            "failed_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore-batch", "a.ipynb:v1.ipynb", "missing.ipynb:v9.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Restored 'a.ipynb' to version 'v1.ipynb'" in proc.stdout
    assert (
        "Failed to restore 'missing.ipynb' to version 'v9.ipynb': "
        "Notebook file not found" in proc.stdout
    )
    assert "1 succeeded, 1 failed" in proc.stdout


def test_versions_restore_batch_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success",
        "dry_run": False,
        "results": [
            {"filename": "a.ipynb", "version_id": "v1.ipynb", "status": "success", "restored_version_id": "v1.ipynb"},
        ],
        "succeeded_count": 1,
        "failed_count": 0,
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore-batch", "a.ipynb:v1.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_versions_restore_batch_command_dry_run_sends_dry_run_field(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "dry_run": True,
            "results": [
                {"filename": "a.ipynb", "version_id": "v1.ipynb", "status": "success", "restored_version_id": "v1.ipynb"},
            ],
            "succeeded_count": 1,
            "failed_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore-batch", "a.ipynb:v1.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would restore 'a.ipynb' to version 'v1.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/versions/restore-batch"]
    assert json.loads(handler.bodies[0]) == {
        "entries": [{"filename": "a.ipynb", "version_id": "v1.ipynb"}],
        "dry_run": True,
    }


def test_versions_restore_batch_command_rejects_a_malformed_entry(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["versions", "restore-batch", "no-colon-here"],
        cwd=workdir,
    )

    assert proc.returncode != 0
    assert "filename:version_id" in proc.stderr


def test_versions_restore_batch_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "restore-batch", "a.ipynb:v1.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_versions_clear_command_reports_success_with_yes_flag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "deleted_version_ids": ["v1.ipynb", "v2.ipynb"],
            "deleted_count": 2,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "clear", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Deleted 2 version(s) of 'nb.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions"]


def test_versions_clear_command_older_than_days_passes_the_param_through_and_prompts(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "older_than_days": 30,
            "deleted_version_ids": ["v1.ipynb"],
            "deleted_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli", "versions", "clear", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--older-than-days", "30",
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        input="y\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "older than 30 day(s)" in proc.stdout
    assert "Deleted 1 version(s) of 'nb.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions?older_than_days=30"]


def test_versions_clear_command_omits_older_than_days_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "older_than_days": None,
            "deleted_version_ids": ["v1.ipynb"],
            "deleted_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "clear", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions"]


def test_versions_clear_command_reports_no_history(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "deleted_version_ids": [], "deleted_count": 0,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "clear", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no version history" in proc.stdout


def test_versions_clear_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "nb.ipynb",
            "deleted_version_ids": ["v1.ipynb"], "deleted_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "clear", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["deleted_count"] == 1


def test_versions_clear_command_aborts_without_yes_when_declined(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "versions", "clear", "nb.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        input="n\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_versions_clear_command_dry_run_skips_confirmation_and_sends_dry_run_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True, "filename": "nb.ipynb",
            "deleted_version_ids": ["v1.ipynb", "v2.ipynb"],
            "deleted_count": 2,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        [
            "versions", "clear", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would delete 2 version(s) of 'nb.ipynb'" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions?dry_run=true"]


def test_versions_clear_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "clear", "does-not-exist.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def _versions_diff_notebook_bytes(function_source):
    return json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": function_source,
                }
            ],
        }
    ).encode("utf-8")


def test_versions_diff_command_is_registered():

    proc = _run_cli(["versions", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "diff" in proc.stdout


def test_versions_diff_command_compares_a_version_against_the_current_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        # First request: the "old" side (the version itself).
        _raw_response(
            200,
            _versions_diff_notebook_bytes(
                "def add(a: int, b: int) -> int:\n    return a + b\n\n"
                "def subtract(a: int, b: int) -> int:\n    return a - b\n"
            ),
        ),
        # Second request: the "new" side -- no --against, so the
        # notebook's current live content via GET /api/notebooks/{filename}.
        _raw_response(
            200,
            _versions_diff_notebook_bytes(
                "def add(a: int, b: int, c: int = 0) -> int:\n    return a + b + c\n\n"
                "def multiply(a: int, b: int) -> int:\n    return a * b\n"
            ),
        ),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "diff", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Comparing version 'v1.ipynb'" in proc.stdout
    assert "current live content" in proc.stdout
    assert "Added 1 endpoint(s):" in proc.stdout
    assert "POST /multiply" in proc.stdout
    assert "Removed 1 endpoint(s):" in proc.stdout
    assert "POST /subtract" in proc.stdout
    assert "Changed 1 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert handler.requests == [
        "/api/notebooks/nb.ipynb/versions/v1.ipynb",
        "/api/notebooks/nb.ipynb",
    ]


def test_versions_diff_command_content_flag_prints_a_line_level_diff(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _versions_diff_notebook_bytes(
                "def add(a: int, b: int) -> int:\n    return a + b\n"
            ),
        ),
        _raw_response(
            200,
            _versions_diff_notebook_bytes(
                "def add(a: int, b: int) -> int:\n    return a + b + 1\n"
            ),
        ),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "diff", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--content",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No changes to the compiled API surface." in proc.stdout
    assert "-    return a + b" in proc.stdout
    assert "+    return a + b + 1" in proc.stdout
    assert "version 'v1.ipynb'" in proc.stdout
    assert "the current live content of 'nb.ipynb'" in proc.stdout


def test_versions_diff_command_compares_two_versions_via_against(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    function_source = "def add(a: int, b: int) -> int:\n    return a + b\n"
    handler.responses = [
        _raw_response(200, _versions_diff_notebook_bytes(function_source)),
        _raw_response(200, _versions_diff_notebook_bytes(function_source)),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "diff", "nb.ipynb", "v1.ipynb",
            "--against", "v2.ipynb", "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Comparing version 'v1.ipynb'" in proc.stdout
    assert "against version 'v2.ipynb'" in proc.stdout
    assert "No changes to the compiled API surface." in proc.stdout
    assert handler.requests == [
        "/api/notebooks/nb.ipynb/versions/v1.ipynb",
        "/api/notebooks/nb.ipynb/versions/v2.ipynb",
    ]


def test_versions_diff_command_json_flag_emits_machine_readable_output(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    function_source = "def add(a: int, b: int) -> int:\n    return a + b\n"
    handler.responses = [
        _raw_response(200, _versions_diff_notebook_bytes(function_source)),
        _raw_response(200, _versions_diff_notebook_bytes(function_source)),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "diff", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data == {
        "added": [], "removed": [], "changed": [], "unchanged": ["add"],
        "compatible": True, "breaking_changes": [],
    }


def test_versions_diff_command_reports_a_clean_error_for_a_missing_version(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook version not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "diff", "nb.ipynb", "does-not-exist.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook version not found")


def test_versions_compare_command_is_registered():

    proc = _run_cli(["versions", "--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "compare" in proc.stdout


def test_versions_compare_command_compares_a_version_against_the_current_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "version_id": "v1.ipynb",
            "against": None,
            "added": [{"name": "multiply"}],
            "removed": [{"name": "subtract"}],
            "changed": [{"name": "add", "old": {}, "new": {}}],
            "unchanged": ["noop"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "compare", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Comparing version 'v1.ipynb'" in proc.stdout
    assert "current live content" in proc.stdout
    assert "Added 1 endpoint(s):" in proc.stdout
    assert "POST /multiply" in proc.stdout
    assert "Removed 1 endpoint(s):" in proc.stdout
    assert "POST /subtract" in proc.stdout
    assert "Changed 1 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb/versions/v1.ipynb/diff"]


def test_versions_compare_command_compares_two_versions_via_against(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "version_id": "v1.ipynb",
            "against": "v2.ipynb",
            "added": [], "removed": [], "changed": [], "unchanged": ["add"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "compare", "nb.ipynb", "v1.ipynb",
            "--against", "v2.ipynb", "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Comparing version 'v1.ipynb'" in proc.stdout
    assert "against version 'v2.ipynb'" in proc.stdout
    assert "No changes to the compiled API surface." in proc.stdout
    assert handler.requests == [
        "/api/notebooks/nb.ipynb/versions/v1.ipynb/diff?against=v2.ipynb"
    ]


def test_versions_compare_command_content_flag_passes_the_query_param_and_prints_the_diff(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "filename": "nb.ipynb",
            "version_id": "v1.ipynb",
            "against": None,
            "added": [], "removed": [], "changed": [], "unchanged": ["add"],
            "content_diff": [
                "--- version 'v1.ipynb'", "+++ current",
                '+    """docstring only"""',
            ],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "compare", "nb.ipynb", "v1.ipynb",
            "--content", "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "docstring only" in proc.stdout
    assert handler.requests == [
        "/api/notebooks/nb.ipynb/versions/v1.ipynb/diff?content=true"
    ]


def test_versions_compare_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "filename": "nb.ipynb", "version_id": "v1.ipynb",
        "against": None, "added": [], "removed": [], "changed": [], "unchanged": [],
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "compare", "nb.ipynb", "v1.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_versions_compare_command_reports_a_clean_error_for_a_missing_version(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook version not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "versions", "compare", "nb.ipynb", "does-not-exist.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook version not found")


def test_remote_files_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "remote-files" in proc.stdout


def test_remote_files_list_command_prints_the_compiled_files(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "generated_files": ["app.py", "requirements.txt"],
            "file_details": [
                {"filename": "app.py", "size_bytes": 1024, "modified_at": "2026-01-01T00:00:00+00:00"},
                {"filename": "requirements.txt", "size_bytes": 12, "modified_at": "2026-01-01T00:00:00+00:00"},
            ],
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "source_notebook_filename": "nb.ipynb",
            "source_notebook_exists": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "list", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "app.py  (1024 bytes, modified 2026-01-01T00:00:00+00:00)" in proc.stdout
    assert "Compiled from: nb.ipynb" in proc.stdout
    assert handler.requests == ["/api/generated"]


def test_remote_files_list_command_warns_when_generated_files_were_modified_since_compile(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "generated_files": ["app.py", "requirements.txt"],
            "file_details": [
                {"filename": "app.py", "size_bytes": 1024, "modified_at": "2026-01-01T00:00:00+00:00"},
            ],
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "source_notebook_filename": "nb.ipynb",
            "source_notebook_exists": True,
            "generated_files_modified_since_compile": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "list", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "modified since the last compile" in proc.stdout


def test_remote_files_list_command_omits_the_warning_when_nothing_was_modified(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "generated_files": ["app.py"],
            "file_details": [
                {"filename": "app.py", "size_bytes": 1024, "modified_at": "2026-01-01T00:00:00+00:00"},
            ],
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "source_notebook_filename": "nb.ipynb",
            "source_notebook_exists": True,
            "generated_files_modified_since_compile": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "list", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "modified since the last compile" not in proc.stdout


def test_remote_files_list_command_shows_the_compiled_version_id_when_present(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "generated_files": ["app.py"],
            "file_details": [
                {"filename": "app.py", "size_bytes": 1024, "modified_at": "2026-01-01T00:00:00+00:00"},
            ],
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "compiled_version_id": "20260101T000000000000_abcd.ipynb",
            "source_notebook_filename": "nb.ipynb",
            "source_notebook_exists": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "list", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "Compiled from: nb.ipynb (version '20260101T000000000000_abcd.ipynb')"
        in proc.stdout
    )


def test_remote_files_list_command_checksums_flag_passes_the_query_param_and_prints_hashes(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "generated_files": ["app.py"],
            "file_details": [
                {
                    "filename": "app.py", "size_bytes": 1024,
                    "modified_at": "2026-01-01T00:00:00+00:00",
                    "sha256": "a" * 64,
                },
            ],
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "source_notebook_filename": "nb.ipynb",
            "source_notebook_exists": True,
            "bundle_sha256": "b" * 64,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "list", "--checksums", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"sha256:{'a' * 64}" in proc.stdout
    assert f"Bundle sha256: {'b' * 64}" in proc.stdout
    assert handler.requests == ["/api/generated?checksums=true"]


def test_remote_files_list_command_flags_a_missing_source_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "generated_files": ["app.py"],
            "file_details": [
                {"filename": "app.py", "size_bytes": 1024, "modified_at": "2026-01-01T00:00:00+00:00"},
            ],
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "source_notebook_filename": "nb.ipynb",
            "source_notebook_exists": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "list", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Compiled from: nb.ipynb  [no longer uploaded]" in proc.stdout


def test_remote_files_list_command_reports_no_compiled_app(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "generated_files": [], "file_details": [],
            "compiled_at": None, "source_notebook_filename": None,
            "source_notebook_exists": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "list", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No compiled app found on the dashboard." in proc.stdout


def test_remote_files_list_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "generated_files": ["app.py"],
            "file_details": [
                {"filename": "app.py", "size_bytes": 1, "modified_at": "2026-01-01T00:00:00+00:00"}
            ],
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "source_notebook_filename": None, "source_notebook_exists": False,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "list", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["generated_files"] == ["app.py"]


def test_remote_files_get_command_prints_content_to_stdout_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "app.py",
            "content": "from fastapi import FastAPI\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "get", "app.py", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == "from fastapi import FastAPI\n"
    assert handler.requests == ["/api/generated/app.py"]


def test_remote_files_get_command_saves_to_output_when_given(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "requirements.txt",
            "content": "fastapi\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-files", "get", "requirements.txt",
            "--dashboard-url", dashboard_url, "--output", "reqs.txt",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "reqs.txt").read_text() == "fastapi\n"
    assert "Saved 'requirements.txt'" in proc.stdout


def test_remote_files_get_command_expected_sha256_succeeds_on_a_match(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "app.py",
            "content": "x = 1\n", "sha256": "abc123",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-files", "get", "app.py", "--dashboard-url", dashboard_url,
            "--output", "app.py", "--expected-sha256", "abc123",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "app.py").read_text() == "x = 1\n"
    assert "sha256: abc123" in proc.stdout


def test_remote_files_get_command_expected_sha256_fails_on_a_mismatch_and_writes_nothing(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "app.py",
            "content": "x = 1\n", "sha256": "abc123",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-files", "get", "app.py", "--dashboard-url", dashboard_url,
            "--output", "app.py", "--expected-sha256", "does-not-match",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "does not match the expected value")
    assert not (workdir / "app.py").exists()


def test_remote_files_get_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "filename": "app.py", "content": "x = 1\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-files", "get", "app.py",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["content"] == "x = 1\n"


def test_remote_files_get_command_reports_a_clean_error_for_a_missing_file(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Generated file not found. Run /api/compile first."})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "get", "missing.py", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Generated file not found")


def test_remote_files_delete_command_reports_success_with_yes_flag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "generated_dir": "/srv/generated"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "delete", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Deleted the compiled app" in proc.stdout
    assert handler.requests == ["/api/generated"]


def test_remote_files_delete_command_aborts_without_yes_when_declined(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "remote-files", "delete", "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        input="n\n",
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_remote_files_delete_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "generated_dir": "/srv/generated"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-files", "delete",
            "--dashboard-url", dashboard_url, "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["generated_dir"] == "/srv/generated"


def test_remote_files_delete_command_reports_a_clean_error_when_nothing_is_compiled(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "No compiled app found. Run /api/compile first."})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-files", "delete", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No compiled app found")


def test_remote_files_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-files", "list",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def _notebook_bytes_with_function(function_source):
    """The exact bytes _write_notebook_with_function writes to disk,
    without writing to disk -- for queuing as a fake dashboard's own GET
    /api/notebooks/{filename} response body in the remote-diff tests
    below, which need "the dashboard's copy" to exist only as response
    bytes, never as a local file.
    """
    return json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": function_source,
                }
            ],
        }
    ).encode("utf-8")


def test_remote_diff_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "remote-diff" in proc.stdout


def test_remote_diff_command_reports_added_removed_and_changed_functions(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n\n"
                "def subtract(a: int, b: int) -> int:\n    return a - b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    local_path = workdir / "local.ipynb"
    _write_notebook_with_function(
        local_path,
        "def add(a: int, b: int, c: int = 0) -> int:\n    return a + b + c\n\n"
        "def multiply(a: int, b: int) -> int:\n    return a * b\n",
    )

    proc = _run_cli(
        [
            "remote-diff", "nb.ipynb", str(local_path),
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Comparing local" in proc.stdout
    assert "Added 1 endpoint(s):" in proc.stdout
    assert "POST /multiply" in proc.stdout
    assert "Removed 1 endpoint(s):" in proc.stdout
    assert "POST /subtract" in proc.stdout
    assert "Changed 1 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb"]


def test_remote_diff_command_content_flag_prints_a_line_level_diff(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    local_path = workdir / "local.ipynb"
    _write_notebook_with_function(
        local_path, "def add(a: int, b: int) -> int:\n    return a + b + 1\n"
    )

    proc = _run_cli(
        [
            "remote-diff", "nb.ipynb", str(local_path),
            "--dashboard-url", dashboard_url, "--content",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No changes to the compiled API surface." in proc.stdout
    assert "-    return a + b" in proc.stdout
    assert "+    return a + b + 1" in proc.stdout
    assert f"'nb.ipynb' on {dashboard_url}" in proc.stdout


def test_remote_diff_command_reports_no_changes_for_identical_notebooks(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    function_source = "def add(a: int, b: int) -> int:\n    return a + b\n"
    handler.responses = [
        _raw_response(200, _notebook_bytes_with_function(function_source))
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    local_path = workdir / "local.ipynb"
    _write_notebook_with_function(local_path, function_source)

    proc = _run_cli(
        [
            "remote-diff", "nb.ipynb", str(local_path),
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No changes to the compiled API surface." in proc.stdout


def test_remote_diff_command_defaults_the_local_path_to_the_filename(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    function_source = "def add(a: int, b: int) -> int:\n    return a + b\n"
    handler.responses = [
        _raw_response(200, _notebook_bytes_with_function(function_source))
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    # No explicit local path passed -- must default to "nb.ipynb" in cwd,
    # matching the dashboard-side filename.
    _write_notebook_with_function(workdir / "nb.ipynb", function_source)

    proc = _run_cli(
        ["remote-diff", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No changes to the compiled API surface." in proc.stdout


def test_remote_diff_command_json_flag_emits_machine_readable_output(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    local_path = workdir / "local.ipynb"
    _write_notebook_with_function(
        local_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n    return a * b\n",
    )

    proc = _run_cli(
        [
            "remote-diff", "nb.ipynb", str(local_path),
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert [f["name"] for f in data["added"]] == ["multiply"]
    assert data["removed"] == []


def test_remote_diff_command_reports_a_clean_error_for_a_missing_local_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-diff", "nb.ipynb", str(workdir / "does-not-exist.ipynb"),
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No such file or directory")


def test_remote_diff_command_reports_a_clean_error_for_a_missing_remote_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    local_path = workdir / "local.ipynb"
    _write_notebook(local_path)

    proc = _run_cli(
        [
            "remote-diff", "nb.ipynb", str(local_path),
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_remote_diff_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    local_path = workdir / "local.ipynb"
    _write_notebook(local_path)

    proc = _run_cli(
        [
            "remote-diff", "nb.ipynb", str(local_path),
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_diff_notebooks_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "diff-notebooks" in proc.stdout


def test_diff_notebooks_command_reports_added_removed_and_changed_functions(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "old": "a.ipynb",
            "new": "b.ipynb",
            "added": [{"name": "multiply"}],
            "removed": [{"name": "subtract"}],
            "changed": [{"name": "add", "old": {}, "new": {}}],
            "unchanged": ["noop"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["diff-notebooks", "a.ipynb", "b.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Comparing 'a.ipynb' against 'b.ipynb'" in proc.stdout
    assert "Added 1 endpoint(s):" in proc.stdout
    assert "POST /multiply" in proc.stdout
    assert "Removed 1 endpoint(s):" in proc.stdout
    assert "POST /subtract" in proc.stdout
    assert "Changed 1 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert handler.requests == ["/api/notebooks/diff?old=a.ipynb&new=b.ipynb"]


def test_diff_notebooks_command_fail_on_breaking_exits_nonzero_when_incompatible(
    tmp_path, fake_dashboard
):
    """diff-notebooks is server-backed -- unlike `diff`, it never calls
    classify_notebook_diff itself, it just trusts GET /api/notebooks/diff's
    own "compatible" field (merged in by classify_notebook_diff there).
    """

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "old": "a.ipynb", "new": "b.ipynb",
            "added": [], "removed": [{"name": "subtract"}], "changed": [], "unchanged": [],
            "compatible": False,
            "breaking_changes": [{
                "type": "removed_endpoint",
                "name": "subtract",
                "detail": "Endpoint 'subtract' was removed.",
            }],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "diff-notebooks", "a.ipynb", "b.ipynb",
            "--fail-on-breaking", "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 1
    assert "Endpoint 'subtract' was removed." in proc.stdout


def test_diff_notebooks_command_passes_old_version_and_new_version_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "old": "a.ipynb", "new": "b.ipynb",
            "old_version": "20260101T000000000000_abcd.ipynb",
            "new_version": "20260102T000000000000_efgh.ipynb",
            "added": [], "removed": [], "changed": [], "unchanged": ["add"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "diff-notebooks", "a.ipynb", "b.ipynb",
            "--old-version", "20260101T000000000000_abcd.ipynb",
            "--new-version", "20260102T000000000000_efgh.ipynb",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "Comparing 'a.ipynb' version '20260101T000000000000_abcd.ipynb' "
        "against 'b.ipynb' version '20260102T000000000000_efgh.ipynb'"
    ) in proc.stdout
    assert handler.requests == [
        "/api/notebooks/diff?old=a.ipynb&new=b.ipynb"
        "&old_version=20260101T000000000000_abcd.ipynb"
        "&new_version=20260102T000000000000_efgh.ipynb"
    ]


def test_diff_notebooks_command_content_flag_passes_the_query_param_and_prints_the_diff(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "old": "a.ipynb", "new": "b.ipynb",
            "old_version": None, "new_version": None,
            "added": [], "removed": [], "changed": [], "unchanged": ["add"],
            "content_diff": [
                "--- a.ipynb", "+++ b.ipynb", '+    """docstring only"""',
            ],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "diff-notebooks", "a.ipynb", "b.ipynb",
            "--content", "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'docstring only' in proc.stdout
    assert handler.requests == [
        "/api/notebooks/diff?old=a.ipynb&new=b.ipynb&content=true"
    ]


def test_diff_notebooks_command_reports_no_changes_for_identical_notebooks(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "old": "a.ipynb", "new": "b.ipynb",
            "added": [], "removed": [], "changed": [], "unchanged": ["add"],
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["diff-notebooks", "a.ipynb", "b.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No changes to the compiled API surface." in proc.stdout


def test_diff_notebooks_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {
        "status": "success", "old": "a.ipynb", "new": "b.ipynb",
        "added": [], "removed": [], "changed": [], "unchanged": [],
    }
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "diff-notebooks", "a.ipynb", "b.ipynb",
            "--dashboard-url", dashboard_url, "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_diff_notebooks_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found: b.ipynb"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["diff-notebooks", "a.ipynb", "b.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found: b.ipynb")


def test_diff_notebooks_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "diff-notebooks", "a.ipynb", "b.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_remote_curl_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "remote-curl" in proc.stdout


def test_remote_curl_command_writes_a_script_with_a_command_per_function(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n\n"
                "def subtract(a: int, b: int) -> int:\n    return a - b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-curl", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cURL script for 'nb.ipynb'" in proc.stdout
    assert "written to: requests.sh (2 request(s))" in proc.stdout
    assert handler.requests == ["/api/notebooks/nb.ipynb"]

    script = (workdir / "requests.sh").read_text(encoding="utf-8")
    assert "curl -X POST http://localhost:8000/add" in script
    assert "curl -X POST http://localhost:8000/subtract" in script
    assert "X-API-Key: notebook-to-api-dev-key" in script


def test_remote_curl_command_version_id_fetches_that_version_instead_of_current_content(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-curl", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--version-id", "20260101T000000000000_abcd1234.ipynb",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "cURL script for 'nb.ipynb' version "
        "'20260101T000000000000_abcd1234.ipynb'"
    ) in proc.stdout
    assert handler.requests == [
        "/api/notebooks/nb.ipynb/versions/20260101T000000000000_abcd1234.ipynb"
    ]

    script = (workdir / "requests.sh").read_text(encoding="utf-8")
    assert "curl -X POST http://localhost:8000/add" in script
    assert "nb.ipynb' version '20260101T000000000000_abcd1234.ipynb'" in script


def test_remote_curl_command_omits_version_id_from_the_url_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-curl", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/notebooks/nb.ipynb"]


def test_remote_curl_command_respects_host_port_api_key_and_output(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-curl", "nb.ipynb", "--dashboard-url", dashboard_url,
            "--host", "api.example.com", "--port", "9000",
            "--api-key", "mykey123", "--output", "custom.sh",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    script = (workdir / "custom.sh").read_text(encoding="utf-8")
    assert "curl -X POST http://api.example.com:9000/add" in script
    assert "X-API-Key: mykey123" in script
    assert not (workdir / "requests.sh").exists()


def test_remote_curl_command_json_flag_emits_machine_readable_output(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _raw_response(
            200,
            _notebook_bytes_with_function(
                "def add(a: int, b: int) -> int:\n    return a + b\n"
            ),
        )
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-curl", "nb.ipynb", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)

    assert data["status"] == "success"
    assert data["path"] == "requests.sh"
    assert len(data["commands"]) == 1
    assert "curl -X POST http://localhost:8000/add" in data["commands"][0]


def test_remote_curl_command_reports_a_clean_error_for_a_missing_notebook(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "Notebook file not found"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-curl", "nb.ipynb", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Notebook file not found")


def test_remote_curl_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-curl", "nb.ipynb",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_remote_export_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "remote-export" in proc.stdout


def test_remote_export_openapi_command_prints_the_schema_to_stdout_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "format": "json",
            "path": "/srv/generated/openapi.json",
            "schema": {"openapi": "3.1.0", "info": {"title": "x"}},
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-export", "openapi", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    printed = json.loads(proc.stdout)
    assert printed == {"openapi": "3.1.0", "info": {"title": "x"}}
    assert handler.requests == ["/api/export-openapi"]
    assert json.loads(handler.bodies[0]) == {"format": "json"}


def test_remote_export_openapi_command_saves_to_output_when_given(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "format": "yaml",
            "path": "/srv/generated/openapi.yaml",
            "content": "openapi: 3.1.0\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-export", "openapi", "--format", "yaml",
            "--dashboard-url", dashboard_url, "--output", "schema.yaml",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "schema.yaml").read_text() == "openapi: 3.1.0\n"
    assert "Saved the OpenAPI yaml export" in proc.stdout
    assert json.loads(handler.bodies[0]) == {"format": "yaml"}


def test_remote_export_openapi_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "format": "json",
            "path": "/srv/generated/openapi.json",
            "schema": {"openapi": "3.1.0"},
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-export", "openapi", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["format"] == "json"
    assert data["schema"] == {"openapi": "3.1.0"}


def test_remote_export_openapi_command_reports_a_clean_error_when_nothing_is_compiled(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "No compiled app found. Run /api/compile first."})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-export", "openapi", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No compiled app found")


def test_remote_export_sdk_command_prints_the_code_to_stdout_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "language": "python",
            "path": "/srv/generated/sdk/python_client.py",
            "code": "class Client:\n    pass\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-export", "sdk", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == "class Client:\n    pass\n\n"
    assert handler.requests == ["/api/export-sdk"]
    assert json.loads(handler.bodies[0]) == {"language": "python"}


def test_remote_export_sdk_command_saves_to_output_when_given(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "language": "typescript",
            "path": "/srv/generated/sdk/typescript_client.ts",
            "code": "export class Client {}\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-export", "sdk", "--language", "typescript",
            "--dashboard-url", dashboard_url, "--output", "client.ts",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "client.ts").read_text() == "export class Client {}\n"
    assert "Saved the typescript SDK client" in proc.stdout
    assert json.loads(handler.bodies[0]) == {"language": "typescript"}


def test_remote_export_sdk_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "language": "python",
            "path": "/srv/generated/sdk/python_client.py", "code": "x = 1\n",
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-export", "sdk", "--dashboard-url", dashboard_url, "--json"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["code"] == "x = 1\n"


def test_remote_export_sdk_command_reports_a_clean_error_when_no_schema_exported(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "No exported OpenAPI schema found. Run /api/export-openapi first."})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-export", "sdk", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No exported OpenAPI schema found")


def test_remote_export_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-export", "openapi",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_remote_deploy_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "remote-deploy" in proc.stdout


def test_remote_deploy_command_reports_success(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "tag": "generated:latest", "pushed": False})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-deploy", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Built image 'generated:latest'" in proc.stdout
    assert "Pushed to the registry." not in proc.stdout
    assert handler.requests == ["/api/deploy"]
    assert json.loads(handler.bodies[0]) == {"push": False, "force": False}


def test_remote_deploy_command_passes_tag_push_platform_and_force_through(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "tag": "myapp:v1", "pushed": True})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-deploy", "--dashboard-url", dashboard_url,
            "--tag", "myapp:v1", "--push", "--platform", "linux/amd64", "--force",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Built image 'myapp:v1'" in proc.stdout
    assert "Pushed to the registry." in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "push": True, "force": True, "tag": "myapp:v1", "platform": "linux/amd64",
    }


def test_remote_deploy_command_passes_no_cache_through(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "tag": "generated:latest", "pushed": False})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-deploy", "--dashboard-url", dashboard_url, "--no-cache"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(handler.bodies[0]) == {
        "push": False, "force": False, "no_cache": True,
    }


def test_remote_deploy_command_omits_no_cache_by_default(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "tag": "generated:latest", "pushed": False})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-deploy", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no_cache" not in json.loads(handler.bodies[0])


def test_remote_deploy_command_dry_run_passes_the_flag_through_and_prints_would_build(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success", "dry_run": True,
            "tag": "myapp:v1", "pushed": True,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-deploy", "--dashboard-url", dashboard_url,
            "--tag", "myapp:v1", "--push", "--dry-run",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would build image 'myapp:v1'" in proc.stdout
    assert "Would push to the registry." in proc.stdout
    assert json.loads(handler.bodies[0]) == {
        "push": True, "force": False, "tag": "myapp:v1", "dry_run": True,
    }


def test_remote_deploy_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "tag": "generated:latest", "pushed": False})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-deploy", "--dashboard-url", dashboard_url, "--json"], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["tag"] == "generated:latest"


def test_remote_deploy_command_reports_a_clean_error_when_stale_without_force(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(409, {
            "detail": (
                "The currently-compiled app no longer matches its source "
                "notebook's current content -- it was edited since the "
                'last compile. Run /api/compile again first, or pass '
                '"force": true to deploy the stale build anyway.'
            )
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-deploy", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    _assert_clean_cli_error(proc, "no longer matches its source notebook")


def test_remote_deploy_command_reports_a_clean_error_when_nothing_is_compiled(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(404, {"detail": "No compiled app found. Run /api/compile first."})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-deploy", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    _assert_clean_cli_error(proc, "No compiled app found")


def test_remote_deploy_command_reports_a_clean_error_for_a_failed_build(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(500, {"detail": "Docker build failed: some error"})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["remote-deploy", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Docker build failed")


def test_remote_deploy_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "remote-deploy",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_deploy_history_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "deploy-history" in proc.stdout


def test_deploy_history_command_prints_past_deploys(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "entries": [
                {
                    "deployed_at": "2024-06-01T12:00:00+00:00",
                    "tag": "myapp:v2",
                    "platform": "linux/amd64",
                    "pushed": True,
                    "source_notebook_filename": "nb.ipynb",
                    "source_notebook_sha256": "abc123",
                },
                {
                    "deployed_at": "2024-05-01T12:00:00+00:00",
                    "tag": "myapp:v1",
                    "platform": None,
                    "pushed": False,
                    "source_notebook_filename": "nb.ipynb",
                    "source_notebook_sha256": "def456",
                },
            ],
            "entry_count": 2,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy-history", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "myapp:v2" in proc.stdout
    assert "pushed, from nb.ipynb" in proc.stdout
    assert "myapp:v1" in proc.stdout
    assert "not pushed, from nb.ipynb" in proc.stdout
    assert "2 deploy(s)" in proc.stdout
    assert handler.requests == ["/api/deploy/history"]


def test_deploy_history_command_reports_no_deploys(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy-history", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No deploys recorded" in proc.stdout


def test_deploy_history_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "entries": [], "entry_count": 0}
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy-history", "--dashboard-url", dashboard_url, "--json"], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_deploy_history_command_format_csv_prints_the_dashboards_raw_csv_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    csv_body = (
        "deployed_at,tag,platform,pushed,source_notebook_filename,"
        "source_notebook_sha256\r\n"
        "2024-01-01T00:00:00+00:00,myapp:latest,,False,nb.ipynb,abc123\r\n"
    )
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy-history", "--dashboard-url", dashboard_url, "--format", "csv"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    # subprocess's own text-mode universal newline translation turns the
    # CSV's "\r\n" line endings into plain "\n" by the time they reach
    # proc.stdout -- immaterial to what was actually printed.
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == ["/api/deploy/history?format=csv"]


def test_deploy_history_command_omits_format_query_param_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy-history", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/deploy/history"]


def test_deploy_history_command_sends_filter_and_limit_query_params(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "deploy-history",
            "--source-notebook", "nb.ipynb",
            "--platform", "linux/amd64",
            "--pushed-only",
            "--limit", "5",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(handler.requests) == 1
    request_path = handler.requests[0]
    assert request_path.startswith("/api/deploy/history?")

    query = urllib.parse.parse_qs(request_path.split("?", 1)[1])
    assert query == {
        "source_notebook_filename": ["nb.ipynb"],
        "platform": ["linux/amd64"],
        "pushed": ["true"],
        "limit": ["5"],
    }


def test_deploy_history_command_sends_deployed_after_and_before_query_params(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "deploy-history",
            "--deployed-after", "2026-01-01T00:00:00+00:00",
            "--deployed-before", "2026-06-01T00:00:00+00:00",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query == {
        "deployed_after": ["2026-01-01T00:00:00+00:00"],
        "deployed_before": ["2026-06-01T00:00:00+00:00"],
    }


def test_deploy_history_command_sends_source_sha256_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "deploy-history", "--source-sha256", "abc123",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query == {"source_notebook_sha256": ["abc123"]}


def test_deploy_history_command_sends_tag_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy-history", "--tag", "myapp:v2", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query == {"tag": ["myapp:v2"]}


def test_deploy_history_command_not_pushed_flag_sends_pushed_false(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy-history", "--not-pushed", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["pushed"] == ["false"]


def test_deploy_history_command_sends_offset_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy-history", "--offset", "5", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["offset"] == ["5"]


def test_deploy_history_command_omits_offset_query_param_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy-history", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests[0] == "/api/deploy/history"


def test_deploy_history_command_rejects_pushed_only_and_not_pushed_together(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "deploy-history", "--pushed-only", "--not-pushed",
            "--dashboard-url", "http://127.0.0.1:1",
        ],
        cwd=workdir,
    )

    assert proc.returncode != 0
    assert "not allowed with argument" in proc.stderr


def test_deploy_history_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "deploy-history",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_clear_deploy_history_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "clear-deploy-history" in proc.stdout


def test_clear_deploy_history_command_reports_success_with_yes_flag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "deleted_count": 3})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["clear-deploy-history", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Discarded 3 deploy history entr(y/ies)" in proc.stdout
    assert handler.requests == ["/api/deploy/history"]


def test_clear_deploy_history_command_sends_source_notebook_query_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "deleted_count": 1})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-deploy-history", "--source-notebook", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/deploy/history?source_notebook_filename=nb.ipynb"
    ]


def test_clear_deploy_history_command_sends_sha256_query_param(
    tmp_path, fake_dashboard
):
    """Mirrors test_clear_deploy_history_command_sends_source_notebook_query_param:
    DELETE /api/deploy/history's own "source_notebook_sha256" filter had
    no --sha256 flag of its own here at all, even though `clear-deploy-
    history --source-notebook` already threads the sibling filename
    filter through.
    """

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "deleted_count": 1})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-deploy-history", "--sha256", "abc123",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/deploy/history?source_notebook_sha256=abc123"
    ]


def test_clear_deploy_history_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "deleted_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-deploy-history", "--dashboard-url", dashboard_url,
            "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == {"status": "success", "deleted_count": 0}


def test_clear_deploy_history_command_aborts_without_yes_when_declined(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "clear-deploy-history", "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        input="n\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_clear_deploy_history_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-deploy-history",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_clear_deploy_history_command_sends_older_than_days_query_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "dry_run": False, "deleted_count": 2})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-deploy-history", "--older-than-days", "30",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/deploy/history?older_than_days=30"]


def test_clear_deploy_history_command_dry_run_skips_confirmation_and_sends_dry_run_field(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "dry_run": True, "deleted_count": 1})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        [
            "clear-deploy-history", "--dry-run",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would discard 1 deploy history entr(y/ies)" in proc.stdout
    assert handler.requests == ["/api/deploy/history?dry_run=true"]


def test_compile_history_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "compile-history" in proc.stdout


def test_compile_history_command_prints_past_compiles(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "success",
            "entries": [
                {
                    "compiled_at": "2024-06-01T12:00:00+00:00",
                    "notebook_filename": "nb.ipynb",
                    "source_notebook_sha256": "abc123",
                    "only": None,
                    "exclude": None,
                    "endpoint_count": 2,
                    "dependency_count": 1,
                    "skipped_function_count": 0,
                },
            ],
            "entry_count": 1,
        })
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["compile-history", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nb.ipynb" in proc.stdout
    assert "2 endpoint(s)" in proc.stdout
    assert "1 compile(s)" in proc.stdout
    assert handler.requests == ["/api/compile/history"]


def test_compile_history_command_reports_no_compiles(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["compile-history", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "No compiles recorded" in proc.stdout


def test_compile_history_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    body = {"status": "success", "entries": [], "entry_count": 0}
    handler.responses = [_json_response(200, body)]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["compile-history", "--dashboard-url", dashboard_url, "--json"], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == body


def test_compile_history_command_format_csv_prints_the_dashboards_raw_csv_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    csv_body = (
        "compiled_at,notebook_filename,source_notebook_sha256,only,exclude,"
        "endpoint_count,dependency_count,skipped_function_count\r\n"
        "2024-01-01T00:00:00+00:00,nb.ipynb,abc123,,,2,0,0\r\n"
    )
    handler.responses = [_raw_response(200, csv_body.encode("utf-8"), "text/csv")]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["compile-history", "--dashboard-url", dashboard_url, "--format", "csv"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == csv_body.replace("\r\n", "\n")
    assert handler.requests == ["/api/compile/history?format=csv"]


def test_compile_history_command_omits_format_query_param_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["compile-history", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/compile/history"]


def test_compile_history_command_sends_notebook_and_limit_query_params(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "compile-history",
            "--notebook", "nb.ipynb",
            "--limit", "5",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(handler.requests) == 1
    request_path = handler.requests[0]
    assert request_path.startswith("/api/compile/history?")

    query = urllib.parse.parse_qs(request_path.split("?", 1)[1])
    assert query == {"notebook_filename": ["nb.ipynb"], "limit": ["5"]}


def test_compile_history_command_sends_compiled_after_and_before_query_params(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "compile-history",
            "--compiled-after", "2026-01-01T00:00:00+00:00",
            "--compiled-before", "2026-06-01T00:00:00+00:00",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query == {
        "compiled_after": ["2026-01-01T00:00:00+00:00"],
        "compiled_before": ["2026-06-01T00:00:00+00:00"],
    }


def test_compile_history_command_sends_source_sha256_query_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "compile-history", "--source-sha256", "abc123",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query == {"source_notebook_sha256": ["abc123"]}


def test_compile_history_command_sends_offset_query_param(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["compile-history", "--offset", "5", "--dashboard-url", dashboard_url],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    query = urllib.parse.parse_qs(handler.requests[0].split("?", 1)[1])
    assert query["offset"] == ["5"]


def test_compile_history_command_omits_offset_query_param_by_default(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "entries": [], "entry_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["compile-history", "--dashboard-url", dashboard_url], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests[0] == "/api/compile/history"


def test_compile_history_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "compile-history",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_clear_compile_history_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "clear-compile-history" in proc.stdout


def test_clear_compile_history_command_reports_success_with_yes_flag(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "deleted_count": 3})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["clear-compile-history", "--dashboard-url", dashboard_url, "--yes"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Discarded 3 compile history entr(y/ies)" in proc.stdout
    assert handler.requests == ["/api/compile/history"]


def test_clear_compile_history_command_sends_notebook_filename_query_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "deleted_count": 1})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-compile-history", "--notebook", "nb.ipynb",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/compile/history?notebook_filename=nb.ipynb"]


def test_clear_compile_history_command_sends_sha256_query_param(
    tmp_path, fake_dashboard
):
    """Mirrors test_clear_deploy_history_command_sends_sha256_query_param
    for compile history: DELETE /api/compile/history's own
    "source_notebook_sha256" filter had no --sha256 flag of its own here
    either.
    """

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "deleted_count": 1})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-compile-history", "--sha256", "abc123",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == [
        "/api/compile/history?source_notebook_sha256=abc123"
    ]


def test_clear_compile_history_command_json_flag_emits_the_dashboards_own_response(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "deleted_count": 0})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-compile-history", "--dashboard-url", dashboard_url,
            "--yes", "--json",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == {"status": "success", "deleted_count": 0}


def test_clear_compile_history_command_aborts_without_yes_when_declined(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = []

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli",
            "clear-compile-history", "--dashboard-url", dashboard_url,
        ],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        input="n\n",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted." in proc.stdout
    assert handler.requests == []


def test_clear_compile_history_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-compile-history",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5", "--yes",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")


def test_clear_compile_history_command_sends_older_than_days_query_param(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "dry_run": False, "deleted_count": 2})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "clear-compile-history", "--older-than-days", "30",
            "--dashboard-url", dashboard_url, "--yes",
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert handler.requests == ["/api/compile/history?older_than_days=30"]


def test_clear_compile_history_command_dry_run_skips_confirmation_and_sends_dry_run_field(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {"status": "success", "dry_run": True, "deleted_count": 1})
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # No --yes, and no stdin input provided -- would hang/fail on a
    # confirmation prompt if --dry-run didn't skip it.
    proc = _run_cli(
        [
            "clear-compile-history", "--dry-run",
            "--dashboard-url", dashboard_url,
        ],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would discard 1 compile history entr(y/ies)" in proc.stdout
    assert handler.requests == ["/api/compile/history?dry_run=true"]


def test_status_command_is_registered():

    proc = _run_cli(["--help"], cwd=Path.cwd())

    assert proc.returncode == 0
    assert "status" in proc.stdout


def test_status_command_prints_health_and_config(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": True, "compiled_at": "2026-01-01T00:00:00+00:00",
        }),
        _json_response(200, {
            "status": "success",
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_source_url_length": 2048,
            "max_search_regex_length": 200,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"Dashboard at {dashboard_url}: healthy" in proc.stdout
    assert "compiled app present, last compiled at 2026-01-01T00:00:00+00:00" in proc.stdout
    assert "max upload size: 10485760 bytes" in proc.stdout
    assert "max batch upload files: 50" in proc.stdout
    assert "max description length: 500" in proc.stdout
    assert "max source url length: 2048" in proc.stdout
    assert "max search regex length: 200" in proc.stdout
    assert "max deploy history entries: 50" in proc.stdout
    assert "max compile history entries: 50" in proc.stdout
    assert "notebook sort keys: name, size, uploaded_at" in proc.stdout
    assert "Compiling Python version: 3.12" in proc.stdout
    assert handler.requests == ["/api/health", "/api/config"]


def test_status_command_prints_the_compiled_version_id_when_present(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": True,
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "compiled_version_id": "20260101T000000000000_abcd1234.ipynb",
        }),
        _json_response(200, {
            "status": "success",
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "compiled app present, last compiled at 2026-01-01T00:00:00+00:00 "
        "(version '20260101T000000000000_abcd1234.ipynb')"
    ) in proc.stdout


def test_status_command_warns_when_generated_files_were_modified_since_compile(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": True,
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "compiled_version_id": None,
            "generated_files_modified_since_compile": True,
        }),
        _json_response(200, {
            "status": "success",
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "modified since the last compile" in proc.stdout


def test_status_command_omits_the_warning_when_nothing_was_modified(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": True,
            "compiled_at": "2026-01-01T00:00:00+00:00",
            "compiled_version_id": None,
            "generated_files_modified_since_compile": False,
        }),
        _json_response(200, {
            "status": "success",
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "modified since the last compile" not in proc.stdout


def test_status_command_prints_url_import_timeout(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False,
        }),
        _json_response(200, {
            "status": "success",
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "url_import_timeout_seconds": 30,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "allowed_origins": [],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "URL import timeout: 30s" in proc.stdout


def test_status_command_prints_stale_upload_temp_file_threshold(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False,
        }),
        _json_response(200, {
            "status": "success",
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "url_import_timeout_seconds": 30,
            "stale_upload_temp_file_seconds": 3600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "allowed_origins": [],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "stale upload temp file threshold: 3600s" in proc.stdout


def test_status_command_prints_a_configured_max_notebooks(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False,
        }),
        _json_response(200, {
            "status": "success",
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebooks": 500,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "url_import_timeout_seconds": 30,
            "stale_upload_temp_file_seconds": 3600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "allowed_origins": [],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "max notebooks: 500" in proc.stdout


def test_status_command_prints_unlimited_for_a_disabled_max_notebooks(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False,
        }),
        _json_response(200, {
            "status": "success",
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebooks": 0,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "url_import_timeout_seconds": 30,
            "stale_upload_temp_file_seconds": 3600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "allowed_origins": [],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "max notebooks: unlimited" in proc.stdout


def test_status_command_prints_a_configured_dashboard_rate_limit(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False,
        }),
        _json_response(200, {
            "status": "success",
            "allowed_origins": [],
            "dashboard_rate_limit_per_minute": 120,
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebooks": 0,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "url_import_timeout_seconds": 30,
            "stale_upload_temp_file_seconds": 3600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dashboard rate limit: 120 requests/minute per client" in proc.stdout


def test_status_command_prints_disabled_for_no_dashboard_rate_limit(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False,
        }),
        _json_response(200, {
            "status": "success",
            "allowed_origins": [],
            "dashboard_rate_limit_per_minute": None,
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebooks": 0,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "url_import_timeout_seconds": 30,
            "stale_upload_temp_file_seconds": 3600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dashboard rate limit: disabled" in proc.stdout


def test_status_command_prints_allowed_origins(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False,
        }),
        _json_response(200, {
            "status": "success",
            "max_upload_bytes": 10485760,
            "max_batch_upload_files": 50,
            "max_notebook_versions": 20,
            "max_tag_length": 40,
            "max_tags_per_notebook": 20,
            "max_description_length": 500,
            "max_deploy_history_entries": 50,
            "max_compile_history_entries": 50,
            "deploy_subprocess_timeout_seconds": 600,
            "notebook_sort_keys": ["name", "size", "uploaded_at"],
            "notebook_sort_orders": ["asc", "desc"],
            "allowed_origins": ["http://localhost:5173", "http://localhost:3000"],
            "compiling_python_version": "3.12",
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "allowed origins: http://localhost:5173, http://localhost:3000"
        in proc.stdout
    )


def test_status_command_reports_a_matching_dashboard_version(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "version": NOTEBOOK_TO_API_VERSION,
            "compiled_app_present": False, "compiled_at": None,
        }),
        _json_response(200, {
            "status": "success", "max_upload_bytes": 1, "max_batch_upload_files": 1,
            "max_notebook_versions": 1, "max_tag_length": 1, "max_tags_per_notebook": 1,
            "deploy_subprocess_timeout_seconds": 1,
            "notebook_sort_keys": [], "notebook_sort_orders": [],
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"version: {NOTEBOOK_TO_API_VERSION} (matches this CLI)" in proc.stdout


def test_status_command_flags_a_mismatched_dashboard_version(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "version": "9.9.9",
            "compiled_app_present": False, "compiled_at": None,
        }),
        _json_response(200, {
            "status": "success", "max_upload_bytes": 1, "max_batch_upload_files": 1,
            "max_notebook_versions": 1, "max_tag_length": 1, "max_tags_per_notebook": 1,
            "deploy_subprocess_timeout_seconds": 1,
            "notebook_sort_keys": [], "notebook_sort_orders": [],
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        f"version: 9.9.9 (this CLI is {NOTEBOOK_TO_API_VERSION} -- mismatched)"
        in proc.stdout
    )


def test_status_command_omits_version_line_when_the_dashboard_does_not_report_one(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False, "compiled_at": None,
        }),
        _json_response(200, {
            "status": "success", "max_upload_bytes": 1, "max_batch_upload_files": 1,
            "max_notebook_versions": 1, "max_tag_length": 1, "max_tags_per_notebook": 1,
            "deploy_subprocess_timeout_seconds": 1,
            "notebook_sort_keys": [], "notebook_sort_orders": [],
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "\n  version:" not in proc.stdout


def test_status_command_check_writable_flag_passes_the_query_param_and_prints_results(
    tmp_path, fake_dashboard
):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False, "compiled_at": None,
            "upload_dir_writable": True, "generated_dir_writable": False,
        }),
        _json_response(200, {
            "status": "success", "max_upload_bytes": 1, "max_batch_upload_files": 1,
            "max_notebook_versions": 1, "max_tag_length": 1, "max_tags_per_notebook": 1,
            "deploy_subprocess_timeout_seconds": 1,
            "notebook_sort_keys": [], "notebook_sort_orders": [],
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["status", "--dashboard-url", dashboard_url, "--check-writable"],
        cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "upload directory writable: yes" in proc.stdout
    assert "generated directory writable: NO" in proc.stdout
    assert handler.requests == [
        "/api/health?check_writable=true", "/api/config",
    ]


def test_status_command_reports_no_compiled_app(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False, "compiled_at": None,
        }),
        _json_response(200, {
            "status": "success", "max_upload_bytes": 1, "max_batch_upload_files": 1,
            "max_notebook_versions": 1, "max_tag_length": 1, "max_tags_per_notebook": 1,
            "deploy_subprocess_timeout_seconds": 1,
            "notebook_sort_keys": [], "notebook_sort_orders": [],
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(["status", "--dashboard-url", dashboard_url], cwd=workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no compiled app yet" in proc.stdout


def test_status_command_json_flag_emits_a_combined_result(tmp_path, fake_dashboard):

    dashboard_url, handler = fake_dashboard
    handler.responses = [
        _json_response(200, {
            "status": "healthy", "service": "notebook-to-api",
            "compiled_app_present": False, "compiled_at": None,
        }),
        _json_response(200, {
            "status": "success", "max_upload_bytes": 1, "max_batch_upload_files": 1,
            "max_notebook_versions": 1, "max_tag_length": 1, "max_tags_per_notebook": 1,
            "deploy_subprocess_timeout_seconds": 1,
            "notebook_sort_keys": [], "notebook_sort_orders": [],
        }),
    ]

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["status", "--dashboard-url", dashboard_url, "--json"], cwd=workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["health"]["status"] == "healthy"
    assert data["config"]["max_upload_bytes"] == 1


def test_status_command_reports_a_clean_error_when_the_dashboard_is_unreachable(
    tmp_path,
):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        [
            "status",
            "--dashboard-url", "http://127.0.0.1:1", "--timeout", "5",
        ],
        cwd=workdir,
    )

    _assert_clean_cli_error(proc, "Is it running?")
