import http.server
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

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


def _write_add_subtract_notebook(path):
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
                            "\n"
                            "def subtract(a: int, b: int) -> int:\n"
                            "    return a - b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _install_fake_docker(bin_dir, log_path):
    """A fake `docker` executable that records how it was invoked instead of
    actually building an image, so these tests don't need a real Docker
    daemon (mirrors the fake `requests` module used to test the generated
    SDK client in test_sdk_generator.py).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        f'printf \'%s\\n\' "$@" > "{log_path}"\n'
        f'pwd >> "{log_path}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    docker_stub.chmod(docker_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _install_fake_docker_verbose_on_stdout(bin_dir, log_path):
    """Like _install_fake_docker, but -- unlike every other fake `docker`
    stub in this file -- also echoes realistic build-log lines to its own
    stdout before exiting, the way a *real* `docker build`/`docker push`
    always does ("Step 1/5 : FROM ...", "Successfully built ...", ...).

    Every other stub here writes only to a log file, never to stdout, so
    none of them can expose a --json run's stdout getting real build-log
    text mixed into it -- confirmed: this was the one gap that let
    `deploy --json`'s "stdout is valid JSON, full stop" guarantee go
    unverified against anything resembling real Docker's own verbosity.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        f'printf \'%s\\n\' "$@" > "{log_path}"\n'
        f'pwd >> "{log_path}"\n'
        "echo 'Step 1/5 : FROM python:3.12-slim'\n"
        "echo 'Successfully built abc123'\n"
        "echo 'Successfully tagged built_api:latest'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    docker_stub.chmod(docker_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _install_fake_docker_recording_all_calls(bin_dir, log_path):
    """Like _install_fake_docker, but appends a record per invocation
    instead of overwriting, separated by a marker line -- so a test
    exercising multiple docker calls in one run (build followed by push)
    can inspect each call independently, in the order they happened.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        f'{{ printf \'%s\\n\' "$@"; pwd; printf \'%s\\n\' "==CALL=="; }} >> "{log_path}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    docker_stub.chmod(docker_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _install_fake_docker_with_smoke_test_support(bin_dir, log_path):
    """Like _install_fake_docker_recording_all_calls, but also supports
    `run`/`port`/`stop`/`logs` (for `deploy --smoke-test`'s own
    _run_local_deploy_smoke_test) -- each subcommand's own behavior is
    driven by environment variables a test sets before invoking `deploy`
    (inherited by this subprocess, and by the further `docker ...`
    subprocesses it launches in turn):

      FAKE_DOCKER_RUN_EXIT_CODE (default 0), FAKE_DOCKER_RUN_STDOUT
      (default "fake-container-id"), FAKE_DOCKER_RUN_STDERR (default "")
      FAKE_DOCKER_PORT_EXIT_CODE (default 0), FAKE_DOCKER_PORT_STDOUT
      (default "") -- a real test points this at a real local HTTP
      server's own "127.0.0.1:<port>" (see fake_container_health_server
      below) so the smoke test's own GET /health polling has something
      real to actually reach.
      FAKE_DOCKER_LOGS_STDOUT (default "") -- returned by `docker logs`,
      surfaced in a failed smoke test's own "detail".
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        f'{{ printf \'%s\\n\' "$@"; pwd; printf \'%s\\n\' "==CALL=="; }} >> "{log_path}"\n'
        'case "$1" in\n'
        "  run)\n"
        '    printf \'%s\' "${FAKE_DOCKER_RUN_STDOUT:-fake-container-id}"\n'
        '    printf \'%s\' "${FAKE_DOCKER_RUN_STDERR:-}" >&2\n'
        '    exit "${FAKE_DOCKER_RUN_EXIT_CODE:-0}"\n'
        "    ;;\n"
        "  port)\n"
        '    printf \'%s\' "${FAKE_DOCKER_PORT_STDOUT:-}"\n'
        '    exit "${FAKE_DOCKER_PORT_EXIT_CODE:-0}"\n'
        "    ;;\n"
        "  stop)\n"
        "    exit 0\n"
        "    ;;\n"
        "  logs)\n"
        '    printf \'%s\' "${FAKE_DOCKER_LOGS_STDOUT:-}"\n'
        "    exit 0\n"
        "    ;;\n"
        "  *)\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker_stub.chmod(docker_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


class _SmokeTestHealthHandler(http.server.BaseHTTPRequestHandler):
    """Stands in for the compiled app's own GET /health inside the
    "container" `docker run` would otherwise have started for real --
    fake_container_health_server below points the fake docker's own
    "docker port" output at this real local server instead.
    """

    status_code = 200

    def do_GET(self):

        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(type(self).status_code)
        body = b'{"status": "healthy"}'
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_container_health_server():
    _SmokeTestHealthHandler.status_code = 200

    server = http.server.HTTPServer(("127.0.0.1", 0), _SmokeTestHealthHandler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        yield port, _SmokeTestHealthHandler
    finally:
        server.shutdown()
        server_thread.join(timeout=5)


def _run_cli(args, cwd, path_dirs=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    if path_dirs:
        env["PATH"] = os.pathsep.join([*path_dirs, env.get("PATH", "")])
    return subprocess.run(
        [sys.executable, "-m", "backend.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_deploy_command_is_registered(tmp_path):
    """Confirmed exploitable before this fix: `deploy` had a real dispatch
    branch in main() but no matching subparsers.add_parser("deploy", ...),
    so argparse rejected it outright with "invalid choice: 'deploy'"
    before the branch was ever reached.
    """

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    proc = subprocess.run(
        [sys.executable, "-m", "backend.cli", "deploy", "--help"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "invalid choice" not in proc.stderr
    assert "notebook" in proc.stdout


def test_deploy_compiles_and_builds_with_default_tag(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])

    proc = subprocess.run(
        [
            sys.executable, "-m", "backend.cli", "deploy",
            str(notebook_path), "--output", "built_api",
        ],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    # The notebook must actually have been compiled before the image build.
    assert (workdir / "built_api" / "app.py").exists()
    assert (workdir / "built_api" / "Dockerfile").exists()

    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines[:-1] == ["build", "-t", "built_api:latest", "."]
    # docker build must run with the compiled output dir as its context.
    assert log_lines[-1] == str((workdir / "built_api").resolve())

    assert "Docker image 'built_api:latest' built successfully." in proc.stdout


def test_deploy_prints_a_compile_summary_before_building(tmp_path):
    """`compile` and `serve` both print a summary of what actually got
    generated (endpoint list, background/task_id markers, dependencies)
    right after compiling -- see print_compile_summary in
    backend/inspector.py. `deploy` also compiles the notebook as its
    first step, but never called it, so a `deploy` run gave no visibility
    at all into what had been compiled before jumping straight to the
    Docker build -- only "Building Docker image ... built successfully."
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
                            "import pandas as pd\n\n"
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

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "built_api"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert "Generated 2 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert "POST /train_model  [background]" in proc.stdout
    add_line = next(
        line for line in proc.stdout.splitlines() if line.strip() == "POST /add"
    )
    assert "[background]" not in add_line
    assert "Dependencies: pandas" in proc.stdout

    # The summary must appear before the build starts, not after.
    summary_index = proc.stdout.index("Generated 2 endpoint(s):")
    build_index = proc.stdout.index("Docker image 'built_api:latest' built successfully.")
    assert summary_index < build_index


def test_deploy_respects_custom_tag(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "generated", "--tag", "myapp:v2"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines[:-1] == ["build", "-t", "myapp:v2", "."]


def test_deploy_respects_custom_platform(tmp_path):
    """`docker build`'s own default target platform is the local Docker
    daemon's host architecture -- not necessarily the deploy target's
    (almost every cloud PaaS runs linux/amd64). Before --platform
    existed, there was no way to override it short of bypassing this
    tool's own `deploy` command and running `docker build` by hand.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "generated",
            "--tag", "myapp:v2", "--platform", "linux/amd64",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines[:-1] == [
        "build", "-t", "myapp:v2", "--platform", "linux/amd64", ".",
    ]


def test_deploy_only_builds_an_image_with_just_the_named_function_compiled(tmp_path):
    """`compile`'s own --only/--exclude (see tests/test_cli.py) must work
    identically through `deploy`, since deploy compiles the notebook as
    its own first step before building the Docker image from whatever
    that compile produced -- a helper function a notebook author doesn't
    want as part of a *deployed* app's public surface needed a way to be
    left out of the image `deploy` actually builds and (optionally)
    pushes, not just out of a local `compile` a developer runs by hand.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_add_subtract_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "built_api",
            "--only", "add",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    generated_app = (workdir / "built_api" / "app.py").read_text(encoding="utf-8")
    assert '"/add"' in generated_app
    assert '"/subtract"' not in generated_app

    assert "Generated 1 endpoint(s):" in proc.stdout
    assert "POST /add" in proc.stdout
    assert "POST /subtract" not in proc.stdout


def test_deploy_omits_platform_flag_by_default(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "generated"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert "--platform" not in log_lines


def test_deploy_respects_no_cache(tmp_path):
    """Docker's own layer cache can silently reuse a stale `pip install`
    layer even after requirements.txt changed -- before --no-cache
    existed, there was no way to force a clean rebuild through this
    command at all short of running `docker build --no-cache` by hand.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "generated",
            "--tag", "myapp:v2", "--no-cache",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines[:-1] == [
        "build", "-t", "myapp:v2", "--no-cache", ".",
    ]


def test_deploy_omits_no_cache_flag_by_default(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "generated"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert "--no-cache" not in log_lines


def test_deploy_reports_a_clear_error_when_docker_is_missing(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    # An empty PATH-equivalent directory so `docker` genuinely can't be found,
    # instead of falling back to whatever happens to be installed on the
    # machine running the tests.
    empty_bin_dir = tmp_path / "empty_bin"
    empty_bin_dir.mkdir()

    env = dict(os.environ)
    env["PATH"] = str(empty_bin_dir)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    proc = subprocess.run(
        [sys.executable, "-m", "backend.cli", "deploy", str(notebook_path)],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 1
    assert "Traceback (most recent call last)" not in proc.stderr
    assert "Error: Docker CLI not found" in proc.stderr
    # The compile step must still have run before the docker lookup failed.
    assert (workdir / "generated" / "app.py").exists()


def _install_fake_docker_that_fails_build(bin_dir):
    """A fake `docker` whose `build` subcommand always exits non-zero, to
    exercise the subprocess.CalledProcessError path from `docker build`
    itself failing (as opposed to Docker not being installed at all).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        'echo "docker: build failed: no space left on device" >&2\n'
        "exit 1\n",
        encoding="utf-8",
    )
    docker_stub.chmod(docker_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def test_deploy_reports_a_clean_error_when_docker_build_fails(tmp_path):
    """Before CLI_USER_FACING_ERRORS existed, only "Docker CLI not found"
    (a FileNotFoundError converted to RuntimeError) was ever caught -- a
    `docker build` that ran but exited non-zero raised an uncaught
    subprocess.CalledProcessError, dumping a raw traceback instead of a
    clean error.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    _install_fake_docker_that_fails_build(bin_dir)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "generated"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 1
    assert "Traceback (most recent call last)" not in proc.stderr
    assert "Error: Command" in proc.stderr
    assert "returned non-zero exit status 1" in proc.stderr
    # The compile step must still have run before the failed docker build.
    assert (workdir / "generated" / "app.py").exists()


def _install_fake_docker_that_hangs(bin_dir, seconds):
    """A fake `docker` that sleeps instead of returning, to exercise the
    subprocess timeout -- before it existed, `deploy`'s docker build/push
    calls had no timeout at all, so a hung `docker build` (e.g. a stuck
    base-image pull) blocked the CLI forever with no way to configure a
    limit, unlike POST /api/deploy.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        f"#!/bin/sh\nsleep {seconds}\nexit 0\n",
        encoding="utf-8",
    )
    docker_stub.chmod(docker_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def test_deploy_reports_a_clean_error_when_docker_build_times_out(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    _install_fake_docker_that_hangs(bin_dir, seconds=5)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
    env["NOTEBOOK_API_DEPLOY_TIMEOUT_SECONDS"] = "1"

    proc = subprocess.run(
        [sys.executable, "-m", "backend.cli", "deploy", str(notebook_path), "--output", "generated"],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 1
    assert "Traceback (most recent call last)" not in proc.stderr
    assert "Error: " in proc.stderr
    # subprocess.TimeoutExpired reports the actual elapsed wall-clock time
    # (e.g. "0.9999847 seconds"), not the configured value verbatim, so
    # this checks the message shape rather than an exact "1 seconds" match
    # -- which is inherently timing-dependent and would be flaky.
    assert "docker" in proc.stderr
    assert "timed out after" in proc.stderr
    assert "seconds" in proc.stderr
    # The compile step must still have run before the docker build hung.
    assert (workdir / "generated" / "app.py").exists()


def test_deploy_docker_timeout_is_configurable_via_env_var(tmp_path):
    """Same NOTEBOOK_API_DEPLOY_TIMEOUT_SECONDS env var POST /api/deploy
    already reads (see DEPLOY_SUBPROCESS_TIMEOUT_SECONDS in
    routes/upload.py) -- a docker call that would exceed the old, fixed
    600s default but comfortably finishes within a longer configured
    timeout must still succeed, not be arbitrarily cut off.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    _install_fake_docker_that_hangs(bin_dir, seconds=1)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
    env["NOTEBOOK_API_DEPLOY_TIMEOUT_SECONDS"] = "30"

    proc = subprocess.run(
        [sys.executable, "-m", "backend.cli", "deploy", str(notebook_path), "--output", "generated"],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "built successfully" in proc.stdout


def test_deploy_reports_a_clean_error_for_a_missing_notebook(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    proc = _run_cli(
        ["deploy", str(workdir / "does-not-exist.ipynb")], cwd=workdir
    )

    assert proc.returncode == 1
    assert "Traceback (most recent call last)" not in proc.stderr
    assert "Error: " in proc.stderr
    assert "No such file or directory" in proc.stderr


def test_deploy_does_not_push_by_default(tmp_path):
    """Without --push, only `docker build` should run -- no `docker push`
    call at all.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_recording_all_calls(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "generated"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    assert len(calls) == 1
    assert "Pushing Docker image" not in proc.stdout
    assert "pushed successfully" not in proc.stdout


def test_deploy_push_runs_docker_push_after_a_successful_build(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_recording_all_calls(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "generated",
            "--tag", "registry.example.com/myapp:v1", "--push",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    assert len(calls) == 2

    build_call = calls[0].splitlines()
    assert build_call[:-1] == ["build", "-t", "registry.example.com/myapp:v1", "."]
    assert build_call[-1] == str((workdir / "generated").resolve())

    push_call = calls[1].splitlines()
    assert push_call[:-1] == ["push", "registry.example.com/myapp:v1"]
    assert push_call[-1] == str((workdir / "generated").resolve())

    assert "Docker image 'registry.example.com/myapp:v1' pushed successfully." in proc.stdout


def test_deploy_push_help_documents_the_flag(tmp_path):

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    proc = subprocess.run(
        [sys.executable, "-m", "backend.cli", "deploy", "--help"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--push" in proc.stdout


def test_deploy_json_flag_emits_machine_readable_output(tmp_path):
    """Before --json existed on `deploy`, a script driving it (CI, another
    tool shelling out to it) had no way to get its outcome (the tag that
    was actually built, whether it was pushed) as structured data -- only
    free-form progress text -- even though POST /api/deploy's REST
    response (routes/upload.py) already returns exactly this
    {"status", "tag", "pushed"} shape for the same operation. Matches
    that shape rather than inventing a different one, and none of
    compile_notebook's/print_compile_summary's/this command's own
    progress prints may leak onto stdout, or a script doing
    json.loads(stdout) would choke on it -- the same guarantee
    `compile --json` already makes.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "built_api", "--json"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (workdir / "built_api" / "app.py").exists()

    data = json.loads(proc.stdout)
    assert data == {
        "status": "success",
        "tag": "built_api:latest",
        "pushed": False,
    }


def test_deploy_json_flag_stdout_is_still_valid_json_against_a_verbose_docker(tmp_path):
    """`deploy --json`'s own contextlib.redirect_stdout only patches this
    process's Python-level sys.stdout -- it has no effect on a subprocess's
    inherited OS-level stdout file descriptor. Confirmed exploitable
    before this fix: `docker build`/`docker push` are always verbose on
    stdout in real life ("Step 1/5 : FROM ...", "Successfully built ...",
    ...), and without capture_output=True on the subprocess.run call
    itself, that text was written straight through to this process's real
    stdout -- immediately followed by the JSON blob -- so
    json.loads(stdout) failed outright ("Expecting value: line 1 column
    1") the moment Docker actually printed anything, which every other
    fake `docker` stub in this file (silent on stdout, only ever writing
    to a log file) could never expose. Uses
    _install_fake_docker_verbose_on_stdout specifically because it does
    what real Docker does: write build-log lines to its own stdout before
    exiting 0.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_verbose_on_stdout(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "built_api", "--json"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    # The whole point: stdout must be nothing but the JSON blob, with none
    # of the fake docker's build-log lines mixed in.
    data = json.loads(proc.stdout)
    assert data == {
        "status": "success",
        "tag": "built_api:latest",
        "pushed": False,
    }
    assert "Step 1/5" not in proc.stdout
    assert "Successfully built" not in proc.stdout


def test_deploy_without_json_still_streams_dockers_own_output_live(tmp_path):
    """The fix above must be scoped to --json only -- the human-readable
    `deploy` path (no --json) is expected to show `docker build`/`docker
    push`'s own live progress output on the real terminal, the same as
    always, not capture and hide it.
    """

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_verbose_on_stdout(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "built_api"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Step 1/5" in proc.stdout
    assert "Successfully built" in proc.stdout


def test_deploy_json_flag_reports_pushed_true_after_a_successful_push(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_recording_all_calls(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "generated",
            "--tag", "registry.example.com/myapp:v1", "--push", "--json",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    data = json.loads(proc.stdout)
    assert data == {
        "status": "success",
        "tag": "registry.example.com/myapp:v1",
        "pushed": True,
    }

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    assert len(calls) == 2


def test_deploy_dry_run_does_not_invoke_docker(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_recording_all_calls(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "built_api", "--dry-run",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Compiling still happens -- that's what --dry-run actually validates.
    assert (workdir / "built_api" / "app.py").exists()
    assert (workdir / "built_api" / "Dockerfile").exists()

    assert "Would build Docker image 'built_api:latest'" in proc.stdout
    # Docker must never have been invoked at all.
    assert not log_path.exists()


def test_deploy_dry_run_reports_would_push_when_requested(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_recording_all_calls(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "built_api",
            "--push", "--dry-run",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Would push Docker image 'built_api:latest'" in proc.stdout
    assert not log_path.exists()


def test_deploy_dry_run_json_flag_emits_dry_run_true(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_recording_all_calls(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "built_api",
            "--dry-run", "--json",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    data = json.loads(proc.stdout)
    assert data == {
        "status": "success",
        "tag": "built_api:latest",
        "pushed": False,
        "dry_run": True,
    }
    assert not log_path.exists()


def test_deploy_dry_run_json_flag_reports_pushed_true_when_push_requested(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_recording_all_calls(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "built_api",
            "--push", "--dry-run", "--json",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    data = json.loads(proc.stdout)
    assert data == {
        "status": "success",
        "tag": "built_api:latest",
        "pushed": True,
        "dry_run": True,
    }
    assert not log_path.exists()


def test_deploy_smoke_test_flag_documented_in_help(tmp_path):

    proc = _run_cli(["deploy", "--help"], cwd=tmp_path)

    assert proc.returncode == 0
    assert "--smoke-test" in proc.stdout


def test_deploy_smoke_test_passes_when_health_responds_200(
    tmp_path, fake_container_health_server, monkeypatch
):

    port, _handler = fake_container_health_server
    monkeypatch.setenv("FAKE_DOCKER_PORT_STDOUT", f"127.0.0.1:{port}")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "built_api", "--smoke-test"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Smoke test: passed" in proc.stdout

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    subcommands = [call.splitlines()[0] for call in calls]
    assert "run" in subcommands
    assert "stop" in subcommands


def test_deploy_smoke_test_json_flag_includes_smoke_test_field(
    tmp_path, fake_container_health_server, monkeypatch
):

    port, _handler = fake_container_health_server
    monkeypatch.setenv("FAKE_DOCKER_PORT_STDOUT", f"127.0.0.1:{port}")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "built_api",
            "--smoke-test", "--json",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    data = json.loads(proc.stdout)
    assert data["smoke_test"] == {"passed": True, "status_code": 200, "detail": None}


def test_deploy_smoke_test_exits_1_when_docker_run_fails(tmp_path, monkeypatch):

    monkeypatch.setenv("FAKE_DOCKER_RUN_EXIT_CODE", "1")
    monkeypatch.setenv("FAKE_DOCKER_RUN_STDERR", "no such image")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "built_api", "--smoke-test"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Smoke test: FAILED" in proc.stdout
    assert "no such image" in proc.stdout


def test_deploy_smoke_test_json_flag_exits_1_on_failure(tmp_path, monkeypatch):

    monkeypatch.setenv("FAKE_DOCKER_RUN_EXIT_CODE", "1")
    monkeypatch.setenv("FAKE_DOCKER_RUN_STDERR", "no such image")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "built_api",
            "--smoke-test", "--json",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["smoke_test"]["passed"] is False


def test_deploy_smoke_test_never_blocks_a_requested_push(tmp_path, monkeypatch):

    monkeypatch.setenv("FAKE_DOCKER_RUN_EXIT_CODE", "1")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "built_api",
            "--smoke-test", "--push",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "pushed successfully" in proc.stdout

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    subcommands = [call.splitlines()[0] for call in calls]
    assert "push" in subcommands


def test_deploy_smoke_test_omitted_by_default(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)

    proc = _run_cli(
        ["deploy", str(notebook_path), "--output", "built_api"],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Smoke test" not in proc.stdout

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    subcommands = [call.splitlines()[0] for call in calls]
    assert "run" not in subcommands


def test_deploy_smoke_test_ignored_under_dry_run(tmp_path):

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    notebook_path = workdir / "nb.ipynb"
    _write_notebook(notebook_path)

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)

    proc = _run_cli(
        [
            "deploy", str(notebook_path), "--output", "built_api",
            "--smoke-test", "--dry-run",
        ],
        cwd=workdir,
        path_dirs=[str(bin_dir)],
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not log_path.exists()
