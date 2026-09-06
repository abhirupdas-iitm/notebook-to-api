import ast
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import nbformat
import pytest

from backend.compiler import (
    _drop_private_functions,
    _explicit_requirement_package_name,
    _extract_excluded_imports,
    _extract_explicit_apt_packages,
    _extract_explicit_requirements,
    _extract_private_function_names,
    _filter_functions_by_name,
    clear_stale_export_artifacts,
    COMPILE_LOCK,
    compile_notebook,
    compile_notebook_to_api,
    compiling_python_version,
    extract_third_party_imports,
    package_name_for_output_dir,
    resolve_requirements,
    STANDARD_LIBS,
    THIS_TOOLS_OWN_PACKAGE_NAME,
)
from backend.generator.docker_generator import (
    apt_install_content,
    docker_compose_content,
    dockerfile_content,
    env_example_content,
    generate_dockerfile,
    generate_dockerignore,
    generate_docker_compose,
    generate_env_example,
)
from backend.generator.kubernetes_generator import (
    generate_kubernetes_manifest,
    kubernetes_manifest_content,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compiler_pipeline():

    output_dir = "test_generated"

    compile_notebook(
        "notebooks/sample.ipynb",
        output_dir
    )

    assert Path(
        f"{output_dir}/app.py"
    ).exists()

    assert Path(
        f"{output_dir}/requirements.txt"
    ).exists()

    assert Path(
        f"{output_dir}/Dockerfile"
    ).exists()


def test_compiler_pipeline_dockerfile_runs_as_non_root_with_a_healthcheck(tmp_path):
    """Confirmed exploitable before this fix: the generated Dockerfile had
    no USER directive (the container ran as root, needlessly widening the
    blast radius of any RCE-class bug) and no HEALTHCHECK, even though
    the generated app already exposes GET /health for exactly that
    purpose -- so orchestrators (Compose, Swarm, a bare `docker run`) had
    no way to tell a hung/crashed process apart from a healthy one.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    dockerfile = (output_dir / "Dockerfile").read_text(encoding="utf-8")

    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile
    # USER must come after the app's files are owned by that user, and
    # before the CMD that actually runs the app as it.
    assert dockerfile.index("chown") < dockerfile.index("USER appuser") < dockerfile.index("CMD [")


def test_compiler_pipeline_dockerfile_sets_unbuffered_and_no_bytecode_env_vars(
    tmp_path,
):
    """Confirmed missing before this fix: without PYTHONUNBUFFERED=1, a
    container's stdout is block-buffered (never a real terminal), so
    uvicorn's own request logs and any print() the notebook's own code
    does can sit unflushed for a long time or be lost entirely if the
    container is killed -- exactly the real-time output `docker logs` and
    any log-aggregation pipeline are expected to see. Without
    PYTHONDONTWRITEBYTECODE=1, the container writes a .pyc cache into its
    writable layer on every cold start -- the exact kind of artifact this
    project already treats as noise to exclude everywhere else it can
    appear (see EXCLUDED_GENERATED_DIR_NAMES in backend/inspector.py) --
    and would outright fail on a container run with a read-only root
    filesystem.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    dockerfile = (output_dir / "Dockerfile").read_text(encoding="utf-8")

    assert "ENV PYTHONUNBUFFERED=1" in dockerfile
    assert "ENV PYTHONDONTWRITEBYTECODE=1" in dockerfile
    # Set immediately after FROM, before anything that runs Python (pip,
    # then the app itself), so both apply universally rather than only
    # to some later step.
    assert (
        dockerfile.index("FROM python")
        < dockerfile.index("ENV PYTHONUNBUFFERED=1")
        < dockerfile.index("RUN pip install")
    )


def test_generate_dockerfile_sets_unbuffered_and_no_bytecode_env_vars(tmp_path):

    output_path = tmp_path / "Dockerfile"

    generate_dockerfile(str(output_path), "generated")

    dockerfile = output_path.read_text(encoding="utf-8")

    assert "ENV PYTHONUNBUFFERED=1" in dockerfile
    assert "ENV PYTHONDONTWRITEBYTECODE=1" in dockerfile


def test_generate_dockerfile_cmd_actually_honors_port_env_var_at_runtime(tmp_path):
    """Most real PaaS deploy targets (Cloud Run, Render, Heroku, ...)
    assign the container's listening port via a $PORT environment variable
    at start time and require the process to actually bind to it -- there
    is no fixed port they'll forward to instead. Before this, CMD was a
    plain exec-form array with "--port", "8000" hardcoded, which -- with
    no shell involved in exec form -- couldn't read $PORT at all no matter
    what a deploy target set it to.

    Runs the Dockerfile's actual CMD shell command (with `uvicorn` swapped
    for `echo` so no real server needs to start) to prove $PORT is
    genuinely substituted by the shell at container start, not just
    present as literal text somewhere in the Dockerfile.
    """

    output_path = tmp_path / "Dockerfile"

    generate_dockerfile(str(output_path), "generated")

    dockerfile = output_path.read_text(encoding="utf-8")

    cmd_line = next(
        line for line in dockerfile.splitlines()
        if line.startswith('CMD ["sh", "-c",')
    )
    shell_command = json.loads(cmd_line[len("CMD "):])[2]
    shell_command = shell_command.replace("uvicorn", "echo", 1)

    default_result = subprocess.run(
        ["sh", "-c", shell_command], capture_output=True, text=True
    )
    assert "--port 8000" in default_result.stdout

    custom_env = dict(os.environ)
    custom_env["PORT"] = "8080"

    custom_result = subprocess.run(
        ["sh", "-c", shell_command],
        capture_output=True,
        text=True,
        env=custom_env,
    )
    assert "--port 8080" in custom_result.stdout


def test_generate_dockerfile_healthcheck_actually_honors_port_env_var_at_runtime(
    tmp_path,
):
    """The HEALTHCHECK must probe whatever port uvicorn actually bound to
    (see the CMD test above), not a stale hardcoded 8000 -- otherwise a
    deploy target assigning a non-default $PORT would leave Docker
    reporting the container "unhealthy" forever, regardless of how healthy
    the app inside it actually is.

    Runs the Dockerfile's actual HEALTHCHECK python snippet against a real
    local HTTP server bound to a non-default port, with $PORT set to match
    -- confirming it resolves and reaches that exact port rather than only
    checking the Dockerfile's text for the right substring.
    """

    output_path = tmp_path / "Dockerfile"

    generate_dockerfile(str(output_path), "generated")

    dockerfile = output_path.read_text(encoding="utf-8")

    healthcheck_line = next(
        line for line in dockerfile.splitlines()
        if line.strip().startswith("CMD python -c")
    )
    snippet = healthcheck_line.split('CMD python -c "', 1)[1].rsplit('"', 1)[0]

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        custom_env = dict(os.environ)
        custom_env["PORT"] = str(port)

        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env=custom_env,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    finally:
        server.shutdown()
        server_thread.join(timeout=5)


def test_compiler_pipeline_example_payload_is_a_list_for_an_optional_list_parameter(
    tmp_path,
):
    """Confirmed exploitable before this fix: normalize_type_annotation
    (backend/parser/ast_parser.py) peeled "Optional[" off
    "Optional[List[float]]" with a blind ".replace(']', '')" that
    stripped *every* closing bracket in the string, corrupting the
    surviving "List[float]]" into the mismatched "List[float" instead of
    "List[float]" -- which matched none of the type_defaults lookups, so
    an extremely common real-world signature like
    `scores: Optional[List[float]] = None` baked a `None` example into
    the generated app's own OpenAPI schema for a field that's actually a
    list.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "from typing import List, Optional\n\n"
            "def summarize(scores: Optional[List[float]] = None) -> int:\n"
            "    return len(scores) if scores else 0\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    app_source = (output_dir / "app.py").read_text(encoding="utf-8")

    assert "'example': {'scores': []}" in app_source
    assert "'example': {'scores': None}" not in app_source


def test_compiling_python_version_matches_the_running_interpreter():

    version = compiling_python_version()

    assert version == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_generate_dockerfile_defaults_to_python_3_11_when_not_specified(tmp_path):
    """Preserves generate_dockerfile's previous behavior for a direct
    caller that doesn't pass python_version -- only compile_notebook_to_api
    (via compiling_python_version()) is expected to override it.
    """

    output_path = tmp_path / "Dockerfile"

    generate_dockerfile(str(output_path), "generated")

    assert "FROM python:3.11-slim" in output_path.read_text(encoding="utf-8")


def test_generate_dockerfile_uses_the_given_python_version(tmp_path):

    output_path = tmp_path / "Dockerfile"

    generate_dockerfile(str(output_path), "generated", python_version="3.12")

    assert "FROM python:3.12-slim" in output_path.read_text(encoding="utf-8")


def test_apt_install_content_is_empty_for_no_packages():

    assert apt_install_content(None) == ""
    assert apt_install_content([]) == ""


def test_apt_install_content_lists_every_package_on_one_line():

    content = apt_install_content(["libpq-dev", "gcc"])

    assert "RUN apt-get update && apt-get install -y --no-install-recommends" in content
    assert "libpq-dev gcc" in content
    assert "rm -rf /var/lib/apt/lists/*" in content


def test_dockerfile_content_omits_apt_block_by_default():
    """The overwhelming majority of notebooks use no "apt-requires"
    directive at all -- the generated Dockerfile for one of them must be
    byte-for-byte identical to what dockerfile_content already produced
    before this parameter existed.
    """

    without_param = dockerfile_content("generated", "3.12")
    with_empty_list = dockerfile_content("generated", "3.12", apt_packages=[])
    with_none = dockerfile_content("generated", "3.12", apt_packages=None)

    assert without_param == with_empty_list == with_none
    assert "apt-get" not in without_param


def test_dockerfile_content_includes_apt_block_before_pip_install():
    """A system library needed to *build* a pip package (not merely to
    run it) must already be present before `pip install` runs, or the
    build itself fails with nothing installed yet to fix it.
    """

    content = dockerfile_content("generated", "3.12", apt_packages=["libpq-dev"])

    assert "libpq-dev" in content
    assert content.index("apt-get install") < content.index("pip install")


def test_dockerfile_content_apt_block_matches_apt_install_content():

    apt_packages = ["libpq-dev", "gcc"]

    full_dockerfile = dockerfile_content("generated", "3.12", apt_packages)

    assert apt_install_content(apt_packages) in full_dockerfile


def test_generate_dockerfile_writes_the_apt_block_when_given_packages(tmp_path):

    output_path = tmp_path / "Dockerfile"

    generate_dockerfile(
        str(output_path), "generated", python_version="3.12",
        apt_packages=["libpq-dev"],
    )

    content = output_path.read_text(encoding="utf-8")
    assert "apt-get install -y --no-install-recommends" in content
    assert "libpq-dev" in content


def test_compiler_pipeline_dockerfile_base_image_matches_the_compiling_interpreter(
    tmp_path, monkeypatch
):
    """Confirmed broken before this fix: the Dockerfile always hardcoded
    "FROM python:3.11-slim" regardless of what interpreter actually ran
    the compile -- while requirements.txt's versions (_pinned_requirement)
    are pinned against exactly that interpreter's installed packages. A
    pinned package whose wheels don't cover 3.11 (or that needs a newer
    Python) would silently break `docker build`'s
    `pip install -r requirements.txt` for anyone compiling on a different
    Python version, which this repository's own environment already is.
    """

    monkeypatch.setattr(
        "backend.compiler.compiling_python_version", lambda: "3.99"
    )

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    dockerfile = (output_dir / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.99-slim" in dockerfile
    assert "FROM python:3.11-slim" not in dockerfile


def test_compiler_pipeline_generates_a_dockerignore_excluding_git_and_caches(tmp_path):
    """Confirmed exploitable before this fix: nothing wrote a
    .dockerignore alongside the Dockerfile, so `COPY . {package_name}/`
    picked up .git, __pycache__, local venvs, and notebooks from the
    build context into the image -- bloating it and, for .git, risking
    shipping history that was never meant to be in the image.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    dockerignore_path = output_dir / ".dockerignore"
    assert dockerignore_path.exists()

    dockerignore = dockerignore_path.read_text(encoding="utf-8")
    assert ".git/" in dockerignore
    assert "__pycache__/" in dockerignore
    assert ".venv/" in dockerignore


def test_generate_dockerignore_excludes_openapi_and_sdk_export_artifacts(tmp_path):
    """Confirmed exploitable before this fix: POST /api/export-openapi,
    POST /api/export-sdk, and the CLI's export-openapi/export-sdk
    commands all write openapi.json/openapi.yaml/sdk/ straight into the
    same output directory as the compiled app -- but the running app
    never reads any of them (it builds its OpenAPI schema live via
    custom_openapi(), not from a file on disk). Building/deploying any
    time after such an export baked these purely client-facing artifacts
    into the served image with no runtime benefit, the exact kind of
    build-context noise this .dockerignore already exists to keep out for
    .git/__pycache__/venvs/notebooks.
    """

    output_path = tmp_path / ".dockerignore"

    generate_dockerignore(str(output_path))

    dockerignore = output_path.read_text(encoding="utf-8")
    assert "openapi.json" in dockerignore
    assert "openapi.yaml" in dockerignore
    assert "sdk/" in dockerignore


def test_generate_dockerignore_excludes_compile_metadata(tmp_path):
    """.compile_metadata.json (write_compile_metadata, backend/compiler.py)
    is dashboard-internal bookkeeping written into the same output
    directory as the compiled app on every compile -- never read by the
    running app itself -- and its "source_notebook" field is the source
    notebook's absolute filesystem path on the compiling server. Before
    this fix, every `deploy`/`docker build` baked that server-side path
    straight into the shipped image, the exact class of build-context leak
    this .dockerignore already exists to prevent for openapi.json/
    openapi.yaml/sdk/.
    """

    output_path = tmp_path / ".dockerignore"

    generate_dockerignore(str(output_path))

    dockerignore = output_path.read_text(encoding="utf-8")
    assert ".compile_metadata.json" in dockerignore


def test_generate_dockerignore_excludes_docker_compose(tmp_path):
    """docker-compose.yml (generate_docker_compose, backend/generator/
    docker_generator.py) is now written into the same output directory
    as the compiled app on every compile too -- a purely local-dev/
    deploy-tooling convenience file the running app never reads at
    runtime, the identical "never read by the app, so it shouldn't ship
    in the image" reasoning this .dockerignore already applies to
    openapi.json/openapi.yaml/sdk/.
    """

    output_path = tmp_path / ".dockerignore"

    generate_dockerignore(str(output_path))

    dockerignore = output_path.read_text(encoding="utf-8")
    assert "docker-compose.yml" in dockerignore


def test_generate_dockerignore_excludes_readme(tmp_path):
    """README.md (generate_readme, backend/generator/docker_generator.py)
    is purely documentation for a human looking at the compiled output
    directory or a downloaded bundle -- never read by the running app
    itself, the identical reasoning this .dockerignore already applies to
    .env.example/docker-compose.yml.
    """

    output_path = tmp_path / ".dockerignore"

    generate_dockerignore(str(output_path))

    dockerignore = output_path.read_text(encoding="utf-8")
    assert "README.md" in dockerignore


def test_docker_compose_content_matches_generate_docker_composes_own_output(tmp_path):
    """docker_compose_content is the pure string generate_docker_compose
    itself writes to disk -- see dockerfile_content's own docstring for
    why this split exists. Confirms the two can't drift apart, the same
    "preview matches the real write" guarantee already covered for
    dockerfile_content/generate_dockerfile above.
    """

    env_vars = [
        {"name": "NOTEBOOK_API_KEY", "default": "dev-key", "description": "..."},
    ]

    output_path = tmp_path / "docker-compose.yml"

    generate_docker_compose(str(output_path), "myapp", env_vars)

    assert (
        output_path.read_text(encoding="utf-8")
        == docker_compose_content("myapp", env_vars)
    )


def test_docker_compose_content_uses_the_given_package_name_as_the_service_name():

    content = docker_compose_content("myapp", [])

    assert "services:" in content
    assert "  myapp:" in content
    assert "    build: ." in content


def test_docker_compose_content_sets_an_unless_stopped_restart_policy():
    """Compose's own default restart policy is "no" -- without this, a
    container that crashed or was OOM-killed just stayed down until an
    operator noticed and re-ran `docker compose up` by hand, defeating
    the whole point of a `docker compose up -d`-style unattended
    deployment.
    """

    content = docker_compose_content("generated", [])

    assert "    restart: unless-stopped\n" in content


def test_docker_compose_content_maps_port_on_both_sides_via_the_port_env_var():
    """The Dockerfile's own CMD/HEALTHCHECK bind/probe whatever $PORT is
    set to at container start (see dockerfile_content above) -- the
    compose file's own port mapping must track the exact same variable
    on *both* sides (host and container), or a caller overriding $PORT
    would map traffic to a container port the app never actually bound
    to.
    """

    content = docker_compose_content("generated", [])

    assert '"${PORT:-8000}:${PORT:-8000}"' in content
    assert "PORT=${PORT:-8000}" in content


def test_docker_compose_content_lists_every_env_var_with_its_own_default():

    env_vars = [
        {"name": "NOTEBOOK_API_KEY", "default": "dev-key", "description": "..."},
        {"name": "NOTEBOOK_API_MAX_TASKS", "default": "10000", "description": "..."},
    ]

    content = docker_compose_content("generated", env_vars)

    assert "NOTEBOOK_API_KEY=${NOTEBOOK_API_KEY:-dev-key}" in content
    assert "NOTEBOOK_API_MAX_TASKS=${NOTEBOOK_API_MAX_TASKS:-10000}" in content


def test_docker_compose_content_with_no_env_vars_still_maps_port():
    """An empty env_vars list (or None) must still produce a valid,
    usable compose file -- just with nothing beyond PORT in its own
    "environment:" section -- rather than a malformed file missing the
    "environment:" key's own required list entirely.
    """

    content = docker_compose_content("generated", [])

    assert "environment:\n      - PORT=${PORT:-8000}\n" in content

    content_none = docker_compose_content("generated", None)

    assert content_none == content


def test_generate_dockerignore_excludes_env_example(tmp_path):
    """.env.example (generate_env_example, backend/generator/
    docker_generator.py) is now written into the same output directory
    as the compiled app on every compile too -- a template for an
    operator to copy to their own .env, never read by the running app
    itself, the identical "never read by the app, so it shouldn't ship
    in the image" reasoning this .dockerignore already applies to
    Dockerfile/.dockerignore/docker-compose.yml.
    """

    output_path = tmp_path / ".dockerignore"

    generate_dockerignore(str(output_path))

    dockerignore = output_path.read_text(encoding="utf-8")
    assert ".env.example" in dockerignore


def test_env_example_content_matches_generate_env_examples_own_output(tmp_path):
    """env_example_content is the pure string generate_env_example itself
    writes to disk -- see dockerfile_content's own docstring for why this
    split exists. Confirms the two can't drift apart, the same "preview
    matches the real write" guarantee already covered for
    dockerfile_content/generate_dockerfile above.
    """

    env_vars = [
        {"name": "NOTEBOOK_API_KEY", "default": "dev-key", "description": "A key."},
    ]

    output_path = tmp_path / ".env.example"

    generate_env_example(str(output_path), env_vars)

    assert (
        output_path.read_text(encoding="utf-8")
        == env_example_content(env_vars)
    )


def test_env_example_content_lists_every_env_var_with_its_own_default_and_description():

    env_vars = [
        {
            "name": "NOTEBOOK_API_KEY", "default": "dev-key",
            "description": "The API key clients must present.",
        },
        {
            "name": "NOTEBOOK_API_MAX_TASKS", "default": "10000",
            "description": "Maximum pending background tasks.",
        },
    ]

    content = env_example_content(env_vars)

    assert "NOTEBOOK_API_KEY=dev-key" in content
    assert "# The API key clients must present." in content
    assert "NOTEBOOK_API_MAX_TASKS=10000" in content
    assert "# Maximum pending background tasks." in content


def test_env_example_content_always_includes_port():
    """PORT is deliberately excluded from GENERATED_APP_ENV_VARS itself
    (it's read by the Dockerfile's own CMD/HEALTHCHECK and docker-
    compose.yml's own "ports" mapping, never by the compiled app) --
    but docker_compose_content already includes it unconditionally in
    its own "environment:" section, and this must too, for the same
    reason: an operator commonly wants to override the host-side port
    without touching the generated docker-compose.yml itself.
    """

    content = env_example_content([])

    assert "PORT=8000" in content


def test_env_example_content_with_no_env_vars_is_still_a_valid_file():

    content = env_example_content(None)

    assert "PORT=8000" in content
    assert content == env_example_content([])


def test_env_example_content_produces_a_value_that_can_actually_be_parsed_as_env_assignments():
    """Every non-comment, non-blank line must be a real NAME=value
    assignment -- the whole point is that `cp .env.example .env` alone
    already reproduces the compiled app's own unconfigured behavior.
    """

    env_vars = [
        {"name": "NOTEBOOK_API_KEY", "default": "dev-key", "description": "A key."},
        {"name": "NOTEBOOK_API_MAX_TASKS", "default": "10000", "description": "Cap."},
    ]

    content = env_example_content(env_vars)

    assignment_lines = [
        line for line in content.splitlines()
        if line and not line.startswith("#")
    ]

    assert assignment_lines == [
        "PORT=8000",
        "NOTEBOOK_API_KEY=dev-key",
        "NOTEBOOK_API_MAX_TASKS=10000",
    ]


def test_compiler_pipeline_writes_apt_requires_directive_into_the_dockerfile(
    tmp_path
):
    """Confirmed missing before this feature: a notebook whose own
    dependency needs a system package present inside the image (e.g.
    `psycopg2` needing libpq-dev to build, or `opencv-python` needing
    libgl1 present at runtime) had exactly one path to a working image --
    hand-editing the generated Dockerfile after every single compile,
    since compile_notebook_to_api's own generate_dockerfile call never
    knew about anything beyond `pip install`.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "# notebook-to-api: apt-requires libpq-dev\n"
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    dockerfile = (output_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "apt-get install -y --no-install-recommends" in dockerfile
    assert "libpq-dev" in dockerfile
    assert dockerfile.index("apt-get install") < dockerfile.index("pip install")


def test_compiler_pipeline_omits_apt_block_for_a_notebook_with_no_directive(
    tmp_path
):

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    dockerfile = (output_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "apt-get" not in dockerfile


def test_compiler_pipeline_generates_a_docker_compose_file(tmp_path):
    """Confirmed missing before this feature: a compiled app had a
    Dockerfile but nothing to actually run it with beyond a hand-typed
    `docker run` -- POST /api/compile (and the CLI's own `compile`) now
    also writes a ready-to-use docker-compose.yml alongside it, on every
    compile, the same way the Dockerfile/.dockerignore already are.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    compose_path = output_dir / "docker-compose.yml"
    assert compose_path.is_file()

    compose = compose_path.read_text(encoding="utf-8")
    assert "services:\n  generated:\n    build: .\n" in compose
    assert "    restart: unless-stopped\n" in compose
    assert "NOTEBOOK_API_KEY=${NOTEBOOK_API_KEY:-notebook-to-api-dev-key}" in compose
    assert "NOTEBOOK_API_RATE_LIMIT_PER_MINUTE=${NOTEBOOK_API_RATE_LIMIT_PER_MINUTE:-0}" in compose


def test_compiler_pipeline_generates_an_env_example_file(tmp_path):
    """Confirmed missing before this feature: GET /api/env-vars-preview
    already answered "what env vars does a compiled app recognize" as
    structured JSON, but nothing ever actually wrote a ready-to-use
    .env.example an operator could `cp .env.example .env` from, unlike
    every other deployment artifact (Dockerfile, .dockerignore,
    docker-compose.yml) a compile already writes alongside app.py.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    env_example_path = output_dir / ".env.example"
    assert env_example_path.is_file()

    env_example = env_example_path.read_text(encoding="utf-8")
    assert "PORT=8000" in env_example
    assert "NOTEBOOK_API_KEY=notebook-to-api-dev-key" in env_example
    assert "NOTEBOOK_API_RATE_LIMIT_PER_MINUTE=0" in env_example


def test_generate_dockerignore_excludes_kubernetes_manifest(tmp_path):
    """kubernetes.yaml (generate_kubernetes_manifest, backend/generator/
    kubernetes_generator.py) is now written into the same output directory
    as the compiled app on every compile too -- a purely deploy-tooling
    convenience file the running app never reads at runtime, the identical
    "never read by the app, so it shouldn't ship in the image" reasoning
    this .dockerignore already applies to docker-compose.yml/.env.example.
    """

    output_path = tmp_path / ".dockerignore"

    generate_dockerignore(str(output_path))

    dockerignore = output_path.read_text(encoding="utf-8")
    assert "kubernetes.yaml" in dockerignore


def test_kubernetes_manifest_content_matches_generate_kubernetes_manifests_own_output(
    tmp_path
):
    """kubernetes_manifest_content is the pure string
    generate_kubernetes_manifest itself writes to disk -- see
    dockerfile_content's own docstring for why this split exists. Confirms
    the two can't drift apart, the same "preview matches the real write"
    guarantee already covered for dockerfile_content/generate_dockerfile
    above.
    """

    env_vars = [
        {"name": "NOTEBOOK_API_KEY", "default": "dev-key", "description": "..."},
    ]

    output_path = tmp_path / "kubernetes.yaml"

    generate_kubernetes_manifest(str(output_path), "myapp", env_vars)

    assert (
        output_path.read_text(encoding="utf-8")
        == kubernetes_manifest_content("myapp", env_vars)
    )


def test_kubernetes_manifest_content_uses_the_given_package_name_throughout():

    content = kubernetes_manifest_content("myapp", [])

    assert "  name: myapp\n" in content
    assert "    app: myapp\n" in content
    assert "image: myapp:latest\n" in content


def test_kubernetes_manifest_content_renders_both_a_deployment_and_a_service():

    content = kubernetes_manifest_content("generated", [])

    documents = content.split("\n---\n")
    assert len(documents) == 2
    assert "kind: Deployment" in documents[0]
    assert "kind: Service" in documents[1]


def test_kubernetes_manifest_content_wires_up_health_and_readiness_probes():
    """GET /health and GET /ready are the compiled app's own two built-in
    routes with no Depends(verify_api_key) (see RESERVED_INFRASTRUCTURE_NAMES,
    backend/generator/api_generator.py) -- the same unauthenticated routes
    the Dockerfile's own HEALTHCHECK already curls, so a probe here needs
    no credential this manifest would otherwise have to embed.
    """

    content = kubernetes_manifest_content("generated", [])

    assert "livenessProbe:" in content
    assert "readinessProbe:" in content
    assert content.count("path: /health") == 1
    assert content.count("path: /ready") == 1


def test_kubernetes_manifest_content_maps_container_port_to_the_port_env_var():

    content = kubernetes_manifest_content("generated", [])

    assert "containerPort: 8000" in content
    assert '- name: PORT\n              value: "8000"' in content


def test_kubernetes_manifest_content_lists_every_env_var_with_its_own_default():

    env_vars = [
        {"name": "NOTEBOOK_API_KEY", "default": "dev-key", "description": "..."},
        {"name": "NOTEBOOK_API_MAX_TASKS", "default": "10000", "description": "..."},
    ]

    content = kubernetes_manifest_content("generated", env_vars)

    assert '- name: NOTEBOOK_API_KEY\n              value: "dev-key"' in content
    assert (
        '- name: NOTEBOOK_API_MAX_TASKS\n              value: "10000"' in content
    )


def test_kubernetes_manifest_content_with_no_env_vars_still_maps_port():
    """An empty env_vars list (or None) must still produce a valid,
    usable manifest -- just with nothing beyond PORT in its own "env:"
    list -- rather than a malformed file missing the "env:" key's own
    required list entirely.
    """

    content = kubernetes_manifest_content("generated", [])

    assert '- name: PORT\n              value: "8000"' in content

    content_none = kubernetes_manifest_content("generated", None)

    assert content_none == content


def test_compiler_pipeline_generates_a_kubernetes_manifest_file(tmp_path):
    """Confirmed missing before this feature: a compiled app already got a
    Dockerfile, a docker-compose.yml for a single-host `docker compose up`,
    and a .env.example -- but nothing for a Kubernetes cluster, the
    deployment target GET /api/env-vars-preview's own docstring already
    names alongside docker-compose.yml/.env.example in passing. POST
    /api/compile (and the CLI's own `compile`) now also writes a
    ready-to-`kubectl apply` kubernetes.yaml alongside them, on every
    compile.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    manifest_path = output_dir / "kubernetes.yaml"
    assert manifest_path.is_file()

    manifest = manifest_path.read_text(encoding="utf-8")
    assert "  name: generated\n" in manifest
    assert 'value: "notebook-to-api-dev-key"' in manifest
    assert "NOTEBOOK_API_KEY" in manifest


def test_compiler_pipeline_generates_a_readme_file(tmp_path):
    """Confirmed missing before this feature: a compiled app shipped
    app.py, requirements.txt, a Dockerfile/.dockerignore/docker-
    compose.yml/.env.example, and optionally an OpenAPI export and SDK
    clients -- but nothing telling a human what any of it actually was.
    An operator who downloads GET /api/download's zip, or clones a deploy
    target's repo, had no single file saying which endpoints this
    specific compile exposes, that every one needs an X-API-Key header,
    or even the one command that actually runs the thing they just got.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def train_model(data: list) -> dict:\n    return {}\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    readme_path = output_dir / "README.md"
    assert readme_path.is_file()

    readme = readme_path.read_text(encoding="utf-8")
    assert readme.startswith("# generated")
    assert "`POST /add`" in readme
    assert (
        "`POST /train_model` -- enqueues a background task" in readme
    )
    assert "X-API-Key" in readme
    assert "docker compose up --build" in readme
    assert "NOTEBOOK_API_KEY" in readme


def test_compiler_pipeline_readme_reflects_only_and_exclude_filtering(tmp_path):
    """The README's own endpoint list must reflect what this compile
    actually exposes -- the same functions/only/exclude-filtered list
    generate_fastapi_code itself compiles into endpoints -- not every
    function the notebook happens to define.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def subtract(a: int, b: int) -> int:\n    return a - b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir), only=["add"])

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "`POST /add`" in readme
    assert "`POST /subtract`" not in readme


def test_compiler_pipeline_bakes_source_notebook_sha256_into_info_endpoint(tmp_path):
    """A running deployed container had no way to self-report which exact
    notebook content actually produced it, short of cross-referencing
    this dashboard's own deploy/compile history externally -- GET /info
    now reports it directly, baked in at compile time.
    """
    import hashlib

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    expected_sha256 = hashlib.sha256(notebook_path.read_bytes()).hexdigest()

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
info = client.get("/info").json()
assert info["source_notebook_sha256"] == {expected_sha256!r}, info

print("SOURCE_NOTEBOOK_SHA256_INFO_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SOURCE_NOTEBOOK_SHA256_INFO_E2E_OK" in proc.stdout


def test_compiler_pipeline_bakes_the_real_tool_version_into_generated_endpoints(tmp_path):
    """GET / and GET /info both previously reported a hardcoded "1.0.0"
    literal completely unrelated to which actual version of this tool
    compiled the app -- the same "two independent, inevitably-drifting
    hardcoded version literals" bug NOTEBOOK_TO_API_VERSION
    (backend/compiler.py) was already introduced to deduplicate for this
    dashboard's own GET /api/health and GET /, just never threaded
    through to the *generated* app's own identical two literals.
    """

    from backend.compiler import NOTEBOOK_TO_API_VERSION

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
root = client.get("/").json()
info = client.get("/info").json()
assert root["generator_version"] == {NOTEBOOK_TO_API_VERSION!r}, root
assert info["version"] == {NOTEBOOK_TO_API_VERSION!r}, info

print("NOTEBOOK_TO_API_VERSION_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NOTEBOOK_TO_API_VERSION_E2E_OK" in proc.stdout


def test_compiler_pipeline_bakes_the_real_tool_version_into_the_openapi_schema(tmp_path):
    """A third hardcoded "1.0.0" literal missed the first time this was
    fixed: the FastAPI(...) app object's own `version=` kwarg, which
    feeds directly into this app's own OpenAPI "info.version" --
    user-visible in every compiled app's own /docs (Swagger UI), and
    baked directly into whatever POST /api/export-openapi writes out
    (export_openapi_schema serializes app.openapi() unchanged).
    """

    from backend.compiler import NOTEBOOK_TO_API_VERSION

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from backend.exporters.openapi_exporter import export_openapi_schema
export_openapi_schema("generated/openapi.json", "generated")

import json
with open("generated/openapi.json") as f:
    schema = json.load(f)

assert schema["info"]["version"] == {NOTEBOOK_TO_API_VERSION!r}, schema["info"]

print("OPENAPI_INFO_VERSION_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OPENAPI_INFO_VERSION_E2E_OK" in proc.stdout


def test_compiler_pipeline_openapi_schema_reports_a_configured_public_url(tmp_path):
    """Confirmed dead code before this fix: the FastAPI(...) constructor's
    own servers=[...] kwarg was silently discarded by custom_openapi,
    which never itself passed servers= to get_openapi(...) --
    app.openapi()["servers"] was never even a key in the resulting
    schema, no matter what NOTEBOOK_API_PUBLIC_URL was set to. Also
    confirms the env var itself is actually read at compiled-app import
    time (when a real deployment's own environment is in effect), not
    baked in at compile time on this dashboard.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    script = f"""
import os
import sys

os.environ["NOTEBOOK_API_PUBLIC_URL"] = "https://api.example.com"

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app

schema = app.openapi()
assert schema["servers"] == [
    {{"url": "https://api.example.com", "description": "This deployment"}}
], schema.get("servers")

print("OPENAPI_SERVERS_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OPENAPI_SERVERS_E2E_OK" in proc.stdout


def test_compiler_pipeline_docs_are_reachable_by_default(tmp_path):

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
assert client.get("/docs").status_code == 200
assert client.get("/redoc").status_code == 200
assert client.get("/openapi.json").status_code == 200

print("DOCS_ENABLED_BY_DEFAULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DOCS_ENABLED_BY_DEFAULT_E2E_OK" in proc.stdout


def test_compiler_pipeline_disable_docs_hides_docs_but_not_the_rest_of_the_app(tmp_path):
    """NOTEBOOK_API_DISABLE_DOCS=true must 404 /docs, /redoc, and
    /openapi.json -- every request this app accepts is already
    authenticated via X-API-Key, but the schema and docs UI themselves
    were always served with no such requirement, exposing every
    endpoint's own name, parameters, and example payloads to anyone who
    could merely reach the deployment. Every other route (health, the
    real notebook-derived endpoints) must keep working, and this
    dashboard's own POST /api/export-openapi/export-sdk (which call
    app.openapi() directly, in-process, never through the disabled HTTP
    routes) must too.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    script = f"""
import os
import sys

os.environ["NOTEBOOK_API_DISABLE_DOCS"] = "true"

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
assert client.get("/docs").status_code == 404
assert client.get("/redoc").status_code == 404
assert client.get("/openapi.json").status_code == 404
assert client.get("/health").status_code == 200

from backend.exporters.openapi_exporter import export_openapi_schema
export_openapi_schema("generated/openapi.json", "generated")

import json
with open("generated/openapi.json") as f:
    schema = json.load(f)
assert "paths" in schema and schema["paths"]

print("DISABLE_DOCS_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DISABLE_DOCS_E2E_OK" in proc.stdout


def test_compiler_pipeline_disable_docs_accepts_common_truthy_spellings(tmp_path):

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    for truthy_value in ("true", "TRUE", "1", "yes", "on"):

        workdir = tmp_path / f"workdir_{truthy_value}"
        workdir.mkdir()

        script = f"""
import os
import sys

os.environ["NOTEBOOK_API_DISABLE_DOCS"] = {truthy_value!r}

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
assert client.get("/docs").status_code == 404, {truthy_value!r}

print("TRUTHY_OK")
"""

        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert proc.returncode == 0, f"{truthy_value}: " + proc.stdout + proc.stderr
        assert "TRUTHY_OK" in proc.stdout


def test_compiler_pipeline_dockerignore_excludes_a_real_exported_openapi_and_sdk(
    tmp_path,
):
    """End-to-end: compile a notebook, actually export its OpenAPI schema
    and SDK into the same output_dir (mirroring what POST
    /api/export-openapi + POST /api/export-sdk, or a real `deploy` run
    after them, would do), and confirm the generated .dockerignore's
    patterns actually match the real files that landed on disk -- not
    just that the right literal substrings appear somewhere in its text.
    """
    import fnmatch

    from backend.exporters.openapi_exporter import export_openapi_schema
    from backend.exporters.sdk_generator import generate_python_sdk

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(tmp_path)!r})

from backend.exporters.openapi_exporter import export_openapi_schema
from backend.exporters.sdk_generator import generate_python_sdk

export_openapi_schema({str(output_dir / "openapi.json")!r}, "generated")
generate_python_sdk(
    {str(output_dir / "openapi.json")!r},
    {str(output_dir / "sdk" / "python_client.py")!r},
)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    dockerignore_patterns = (
        (output_dir / ".dockerignore").read_text(encoding="utf-8").splitlines()
    )

    def is_ignored(relative_path):
        return any(
            fnmatch.fnmatch(relative_path, pattern)
            or relative_path.startswith(pattern)
            for pattern in dockerignore_patterns
        )

    assert is_ignored("openapi.json")
    assert is_ignored("sdk/python_client.py")
    # write_compile_metadata (backend/compiler.py) already wrote this
    # alongside app.py as part of compile_notebook above -- it must be
    # ignored too, since it's never read by the running app and its
    # "source_notebook" field is the compiling server's own filesystem
    # path.
    assert is_ignored(".compile_metadata.json")
    # docker-compose.yml (generate_docker_compose) is a real file
    # compile_notebook above already wrote alongside the Dockerfile --
    # a local-dev/deploy-tooling convenience file the running app never
    # reads, so it must be ignored the same way.
    assert (output_dir / "docker-compose.yml").is_file()
    assert is_ignored("docker-compose.yml")
    # kubernetes.yaml (generate_kubernetes_manifest) is the same kind of
    # deploy-tooling convenience file docker-compose.yml already is, just
    # for a Kubernetes cluster instead of a single host -- it must be
    # ignored for the identical reason.
    assert (output_dir / "kubernetes.yaml").is_file()
    assert is_ignored("kubernetes.yaml")
    # The actually-deployable artifacts must NOT be swept up by the same
    # patterns.
    assert not is_ignored("app.py")
    assert not is_ignored("requirements.txt")
    assert not is_ignored("runtime/notebook_module.py")


def test_compile_notebook_to_api_holds_compile_lock_for_its_whole_write_phase(
    tmp_path, monkeypatch
):
    """POST /api/compile runs as a plain `def` route, scheduled onto
    FastAPI's worker threadpool (see routes/upload.py) specifically so a
    slow compile doesn't block other requests -- which also means two
    overlapping compiles can now genuinely run in two different threads
    at once. Without COMPILE_LOCK serializing compile_notebook_to_api's
    multi-file write sequence, their writes could interleave into a
    corrupted, mismatched output directory.

    Verified directly: a second thread's non-blocking attempt to acquire
    COMPILE_LOCK must fail while the first compile is mid-write, and must
    succeed again once it finishes.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )
    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    entered_write = threading.Event()
    release_write = threading.Event()

    import backend.compiler as compiler_module

    original_write_runtime_module = compiler_module.write_runtime_module

    def blocking_write_runtime_module(code_cells, out_dir):
        entered_write.set()
        assert release_write.wait(timeout=5)
        return original_write_runtime_module(code_cells, out_dir)

    monkeypatch.setattr(
        compiler_module, "write_runtime_module", blocking_write_runtime_module
    )

    compile_thread = threading.Thread(
        target=compile_notebook_to_api,
        args=(str(notebook_path), str(output_dir / "app.py")),
    )
    compile_thread.start()

    assert entered_write.wait(timeout=5), "compile never reached its write phase"

    # COMPILE_LOCK must still be held by the compile in progress.
    assert COMPILE_LOCK.acquire(blocking=False) is False

    release_write.set()
    compile_thread.join(timeout=5)
    assert not compile_thread.is_alive()

    # Free again once the compile has finished.
    assert COMPILE_LOCK.acquire(blocking=False) is True
    COMPILE_LOCK.release()


def test_concurrent_compiles_to_the_same_output_dir_never_produce_a_mixed_result(
    tmp_path,
):
    """Confirmed exploitable before COMPILE_LOCK existed: compiling two
    different notebooks into the same output_dir from two threads at once
    (now possible -- see the test above) could leave app.py describing one
    notebook's function(s) while the runtime module actually holds a
    different notebook's code, since compile_notebook_to_api writes them
    as separate, non-atomic steps. With the lock in place, one compile
    always fully finishes before the other starts, so the final output
    must always match exactly one notebook end to end -- never a mix.
    """

    def _notebook(source):
        notebook = nbformat.v4.new_notebook()
        notebook.cells.append(nbformat.v4.new_code_cell(source))
        return notebook

    notebook_a_path = tmp_path / "a.ipynb"
    with open(notebook_a_path, "w", encoding="utf-8") as f:
        nbformat.write(
            _notebook("def add(a: int, b: int) -> int:\n    return a + b\n"), f
        )

    notebook_b_path = tmp_path / "b.ipynb"
    with open(notebook_b_path, "w", encoding="utf-8") as f:
        nbformat.write(
            _notebook("def multiply(a: int, b: int) -> int:\n    return a * b\n"), f
        )

    output_dir = tmp_path / "generated"
    output_path = str(output_dir / "app.py")

    threads = [
        threading.Thread(
            target=compile_notebook_to_api, args=(str(notebook_a_path), output_path)
        ),
        threading.Thread(
            target=compile_notebook_to_api, args=(str(notebook_b_path), output_path)
        ),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    app_source = (output_dir / "app.py").read_text(encoding="utf-8")
    runtime_source = (
        output_dir / "runtime" / "notebook_module.py"
    ).read_text(encoding="utf-8")

    if "/add" in app_source:
        assert "def add(" in runtime_source
        assert "def multiply(" not in runtime_source
        assert "/multiply" not in app_source
    else:
        assert "/multiply" in app_source
        assert "def multiply(" in runtime_source
        assert "def add(" not in runtime_source
        assert "/add" not in app_source


def test_compiler_pipeline_handles_magics_and_broken_cells(tmp_path):
    """A notebook with Jupyter magics/shell escapes, and a cell that is
    still unparseable after stripping them, must compile end-to-end
    instead of crashing, and must not lose imports detected in other,
    valid cells.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "%matplotlib inline\n"
            "!pip install pandas\n"
            "import pandas as pd\n\n"
            "def summarize(count: int) -> int:\n"
            "    return count * 2\n"
        )
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "%%bash\necho this cell is not python"
        )
    )

    notebook_path = tmp_path / "magics.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    runtime_module = (
        output_dir / "runtime" / "notebook_module.py"
    ).read_text(encoding="utf-8")

    # The generated runtime module must itself be valid, importable Python.
    ast.parse(runtime_module)

    requirements = (output_dir / "requirements.txt").read_text(
        encoding="utf-8"
    )

    assert "pandas" in requirements


def test_compiler_pipeline_does_not_expose_a_writefile_cells_own_function(tmp_path):
    """Confirmed exploitable before this fix: %%writefile writes its own
    cell body to a file instead of executing it in the notebook's own
    namespace -- a real Jupyter kernel never defines a function written
    this way at all, so a later cell calling it raises NameError. Before
    this fix, only the "%%writefile ..." line itself was commented out,
    leaving a syntactically-valid-Python body (the common real-world
    case) untouched and compiled straight into a real, working endpoint
    -- a fidelity gap between what the source notebook actually does and
    what got served, with no warning anywhere.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "%%writefile helper_module.py\n"
            "def greet(name: str) -> str:\n"
            "    return f'hello {name}'\n"
        )
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "writefile.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    app_source = (output_dir / "app.py").read_text(encoding="utf-8")

    assert '"/add"' in app_source
    assert "greet" not in app_source

    runtime_module = (
        output_dir / "runtime" / "notebook_module.py"
    ).read_text(encoding="utf-8")

    # The %%writefile cell's own body survives as inert, commented-out
    # text (the same "keep line numbers stable" treatment strip_magic_
    # commands already gives an ordinary magic line) -- what matters is
    # that it defines no live, callable top-level function.
    tree = ast.parse(runtime_module)
    top_level_function_names = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "greet" not in top_level_function_names
    assert "add" in top_level_function_names


def test_compiler_pipeline_handles_a_leftover_introspection_query(tmp_path):
    """A cell left over from interactive exploration with a trailing
    ``func?``/``?func`` IPython introspection query (inline docstring/
    source lookup) must not lose the function(s) defined in that same
    cell -- before strip_magic_commands covered this syntax, `ast.parse`
    failed on the whole cell (it parses a cell as a single unit), so
    is_parseable_python dropped the entire cell, silently taking a
    perfectly good `train_model` down with it.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def train_model(epochs: int) -> str:\n"
            "    return f'trained for {epochs} epochs'\n\n"
            "train_model?\n"
        )
    )

    notebook_path = tmp_path / "introspection.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    runtime_module = (
        output_dir / "runtime" / "notebook_module.py"
    ).read_text(encoding="utf-8")

    ast.parse(runtime_module)
    assert "def train_model(epochs: int) -> str:" in runtime_module

    app_source = (output_dir / "app.py").read_text(encoding="utf-8")
    assert '@app.post("/train_model"' in app_source


def test_compiler_pipeline_does_not_expose_class_methods_or_nested_functions(
    tmp_path
):
    """A class method or a closure nested inside another function is not
    callable as a standalone module-level function, so it must not be
    turned into its own generated API endpoint.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "class Model:\n"
            "    def predict(self, x: int) -> int:\n"
            "        return x * 2\n\n"
            "def run(x: int) -> int:\n"
            "    def helper(y: int) -> int:\n"
            "        return y + 1\n"
            "    return helper(x)\n"
        )
    )

    notebook_path = tmp_path / "methods.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    assert '"/run"' in generated_app or "'/run'" in generated_app
    assert '"/predict"' not in generated_app
    assert '"/helper"' not in generated_app


def test_compiler_pipeline_deduplicates_functions_redefined_across_cells(
    tmp_path
):
    """Iteratively re-running a cell with a fixed version of the same
    function is a normal notebook workflow. The compiler must not
    register two conflicting routes for the same path -- FastAPI/Starlette
    would route every request to the *first*-registered one while the
    OpenAPI schema (dict-keyed by path) would document the *last*, so the
    served and documented behaviour would silently diverge.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n"
            "    # fixed version\n"
            "    return a + b + 1\n"
        )
    )

    notebook_path = tmp_path / "redefined.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    assert generated_app.count('"/add"') == 1
    assert generated_app.count("def add(") == 1


def _add_and_subtract_notebook(tmp_path, filename="add_subtract.ipynb"):
    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n"
            "def subtract(a: int, b: int) -> int:\n"
            "    return a - b\n"
        )
    )

    notebook_path = tmp_path / filename

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    return notebook_path


def test_filter_functions_by_name_only_keeps_just_the_named_functions():

    functions = [{"name": "add"}, {"name": "subtract"}]

    filtered = _filter_functions_by_name(functions, only=["add"], exclude=None)

    assert [f["name"] for f in filtered] == ["add"]


def test_filter_functions_by_name_exclude_drops_just_the_named_functions():

    functions = [{"name": "add"}, {"name": "subtract"}]

    filtered = _filter_functions_by_name(functions, only=None, exclude=["subtract"])

    assert [f["name"] for f in filtered] == ["add"]


def test_filter_functions_by_name_with_neither_returns_everything_unchanged():

    functions = [{"name": "add"}, {"name": "subtract"}]

    filtered = _filter_functions_by_name(functions, only=None, exclude=None)

    assert filtered is functions


def test_filter_functions_by_name_rejects_only_and_exclude_together():

    with pytest.raises(ValueError, match="can't both be given"):
        _filter_functions_by_name(
            [{"name": "add"}], only=["add"], exclude=["add"]
        )


def test_filter_functions_by_name_rejects_an_unknown_only_name():

    with pytest.raises(ValueError, match="not defined in this notebook"):
        _filter_functions_by_name(
            [{"name": "add"}], only=["nope"], exclude=None
        )


def test_filter_functions_by_name_rejects_an_unknown_exclude_name():

    with pytest.raises(ValueError, match="not defined in this notebook"):
        _filter_functions_by_name(
            [{"name": "add"}], only=None, exclude=["nope"]
        )


def test_extract_private_function_names_matches_a_directive_immediately_above_a_def():

    code_cells = [
        "# notebook-to-api: private\ndef helper(x):\n    return x\n"
    ]

    assert _extract_private_function_names(code_cells) == {"helper"}


def test_extract_private_function_names_tolerates_blank_lines_between_directive_and_def():

    code_cells = [
        "# notebook-to-api: private\n\n\ndef helper(x):\n    return x\n"
    ]

    assert _extract_private_function_names(code_cells) == {"helper"}


def test_extract_private_function_names_matches_an_async_def():

    code_cells = [
        "# notebook-to-api: private\nasync def helper(x):\n    return x\n"
    ]

    assert _extract_private_function_names(code_cells) == {"helper"}


def test_extract_private_function_names_ignores_a_directive_with_no_following_def():

    code_cells = [
        "# notebook-to-api: private\nx = 1\n"
    ]

    assert _extract_private_function_names(code_cells) == set()


def test_extract_private_function_names_ignores_an_unrelated_comment():

    code_cells = [
        "# just a regular comment\ndef add(a, b):\n    return a + b\n"
    ]

    assert _extract_private_function_names(code_cells) == set()


def test_extract_private_function_names_only_marks_the_function_directly_below():

    code_cells = [
        "def add(a, b):\n    return a + b\n\n"
        "# notebook-to-api: private\n"
        "def helper(x):\n    return x\n"
    ]

    assert _extract_private_function_names(code_cells) == {"helper"}


def test_drop_private_functions_removes_the_marked_function():

    functions = [{"name": "add"}, {"name": "helper"}]
    code_cells = ["# notebook-to-api: private\ndef helper(x):\n    return x\n"]

    filtered, exclude = _drop_private_functions(functions, code_cells)

    assert [f["name"] for f in filtered] == ["add"]
    assert exclude is None


def test_drop_private_functions_with_no_directive_returns_functions_unchanged():

    functions = [{"name": "add"}]
    code_cells = ["def add(a, b):\n    return a + b\n"]

    filtered, exclude = _drop_private_functions(functions, code_cells, exclude=["add"])

    assert filtered is functions
    assert exclude == ["add"]


def test_drop_private_functions_rejects_only_naming_a_private_function():

    functions = [{"name": "add"}, {"name": "helper"}]
    code_cells = ["# notebook-to-api: private\ndef helper(x):\n    return x\n"]

    with pytest.raises(ValueError, match='"# notebook-to-api: private"'):
        _drop_private_functions(functions, code_cells, only=["helper"])


def test_drop_private_functions_treats_exclude_naming_a_private_function_as_a_no_op():
    """Naming an already-private function via `exclude` is redundant, not
    an error -- _filter_functions_by_name's own "not defined in this
    notebook" check would otherwise misfire once the function has
    already been dropped from `functions` by the time it runs.
    """

    functions = [{"name": "add"}, {"name": "helper"}]
    code_cells = ["# notebook-to-api: private\ndef helper(x):\n    return x\n"]

    filtered, exclude = _drop_private_functions(
        functions, code_cells, exclude=["helper"]
    )

    assert [f["name"] for f in filtered] == ["add"]
    assert exclude == []

    # The adjusted exclude must still compose cleanly with
    # _filter_functions_by_name -- no "not defined" error for a name
    # that's already gone.
    assert _filter_functions_by_name(filtered, only=None, exclude=exclude) == filtered


def test_compile_notebook_with_only_generates_an_endpoint_for_just_that_function(
    tmp_path
):
    """The generated app.py must expose exactly the requested function as
    an endpoint -- and no others -- while the excluded function must still
    be present and callable in the runtime module, since a compiled-out
    function may still be called internally by one that *is* exposed.
    """

    notebook_path = _add_and_subtract_notebook(tmp_path)
    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir), only=["add"])

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")
    runtime_module = (
        output_dir / "runtime" / "notebook_module.py"
    ).read_text(encoding="utf-8")

    assert '"/add"' in generated_app
    assert '"/subtract"' not in generated_app
    assert "def add(" in runtime_module
    assert "def subtract(" in runtime_module


def test_compile_notebook_with_exclude_omits_just_that_functions_endpoint(
    tmp_path
):

    notebook_path = _add_and_subtract_notebook(tmp_path)
    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir), exclude=["subtract"])

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    assert '"/add"' in generated_app
    assert '"/subtract"' not in generated_app


def test_compile_notebook_with_neither_only_nor_exclude_compiles_every_function(
    tmp_path
):
    """Preserves the previous, still-default behavior -- every top-level
    function becomes an endpoint when --only/--exclude aren't given.
    """

    notebook_path = _add_and_subtract_notebook(tmp_path)
    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    assert '"/add"' in generated_app
    assert '"/subtract"' in generated_app


def test_compile_notebook_never_exposes_a_private_directive_marked_function(tmp_path):
    """A function marked "# notebook-to-api: private" must never get its
    own endpoint -- but must still be present and callable in the
    runtime module, since a caller-exposed function may still call it
    internally, the same "still present, just not its own endpoint"
    contract --exclude already provides.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "# notebook-to-api: private\n"
            "def helper(x: int) -> int:\n"
            "    return x * 2\n\n"
            "def add(a: int, b: int) -> int:\n"
            "    return helper(a) + helper(b)\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")
    runtime_module = (
        output_dir / "runtime" / "notebook_module.py"
    ).read_text(encoding="utf-8")

    assert '"/add"' in generated_app
    assert '"/helper"' not in generated_app
    assert "def helper(" in runtime_module


def test_compile_notebook_only_naming_a_private_function_is_a_clean_error(tmp_path):

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "# notebook-to-api: private\n"
            "def helper(x: int) -> int:\n"
            "    return x\n\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    with pytest.raises(ValueError, match='"# notebook-to-api: private"'):
        compile_notebook(str(notebook_path), str(output_dir), only=["helper"])


def test_compile_notebook_only_and_exclude_together_is_a_clean_error(tmp_path):

    notebook_path = _add_and_subtract_notebook(tmp_path)
    output_dir = tmp_path / "generated"

    with pytest.raises(ValueError, match="can't both be given"):
        compile_notebook(
            str(notebook_path), str(output_dir), only=["add"], exclude=["subtract"]
        )


def test_compiler_pipeline_generates_awaitable_endpoint_for_async_function(
    tmp_path
):
    """`async def` functions are common in notebooks that call external
    APIs (httpx/aiohttp). Compiling one must produce a valid, importable
    generated app whose endpoint actually awaits the coroutine instead of
    returning it unresolved.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "async def fetch_data(url: str) -> dict:\n"
            "    return {'url': url}\n"
        )
    )

    notebook_path = tmp_path / "async_func.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    ast.parse(generated_app)

    assert "async def fetch_data(" in generated_app
    assert "await notebook_module.fetch_data(" in generated_app


def test_compiler_pipeline_calls_keyword_only_args_by_keyword(tmp_path):
    """`def train(data, *, epochs=10)` is a common ML-notebook signature.
    Keyword-only params must be forwarded as `epochs=req.epochs`, not
    positionally, or the generated endpoint raises a TypeError on every
    call.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def score(data: list, *, epochs: int = 10) -> dict:\n"
            "    return {'data': data, 'epochs': epochs}\n"
        )
    )

    notebook_path = tmp_path / "kwonly.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    ast.parse(generated_app)

    assert "notebook_module.score(req.data, epochs=req.epochs)" in generated_app


def test_compiler_pipeline_positional_only_args_work_end_to_end(tmp_path):
    """Confirmed exploitable before this fix: positional-only params (those
    before a bare `/`) were dropped during extraction, so the generated
    endpoint called notebook_module.f(...) without them and every request
    raised a TypeError for missing required arguments.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def combine(a: int, b: int, /, c: int) -> int:\n"
            "    return a + b + c\n"
        )
    )

    notebook_path = tmp_path / "posonly.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    ast.parse(generated_app)

    assert "notebook_module.combine(req.a, req.b, req.c)" in generated_app


def test_compiler_pipeline_zero_argument_function_compiles_end_to_end(tmp_path):
    """Confirmed exploitable before this fix: a zero-parameter notebook
    function produced an empty Pydantic model class body (no fields, no
    model_config), which is a SyntaxError -- app.py failed to even
    `compile()`, breaking every endpoint in the generated API, not just
    the zero-arg one.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def get_status() -> dict:\n"
            "    return {'ok': True}\n"
        )
    )

    notebook_path = tmp_path / "zeroarg.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    ast.parse(generated_app)
    compile(generated_app, "app.py", "exec")


def test_compiler_pipeline_parameter_type_containing_a_quote_compiles_end_to_end(
    tmp_path,
):
    """Confirmed exploitable before this fix: a parameter's type
    annotation (ast.unparse'd from the notebook's own source -- e.g.
    Literal["a\\"quoted\\"value"] unparses to Literal['a"quoted"value'])
    is arbitrary, notebook-author-controlled text that can itself
    legitimately contain a double quote. That quote was embedded as a
    raw f-string inside a hand-written description="..." literal for the
    generated Pydantic Field, closing the string early and corrupting
    the rest of the line into a SyntaxError that failed to compile the
    *entire* generated app.py, not just this one parameter.
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
                            "from typing import Literal\n\n"
                            'def classify(label: Literal["a\\"quoted\\"value"]) -> str:\n'
                            "    return label\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

resp = client.post("/classify", json={{"label": 'a"quoted"value'}}, headers=headers)
assert resp.status_code == 200, resp.text
assert resp.json() == {{"result": 'a"quoted"value'}}, resp.json()

schema = app.openapi()
field_description = schema["components"]["schemas"]["ClassifyRequest"]["properties"]["label"]["description"]
assert field_description == "Parameter 'label' of type Literal['a\\"quoted\\"value']", field_description

print("QUOTED_PARAMETER_TYPE_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "QUOTED_PARAMETER_TYPE_E2E_OK" in proc.stdout


def test_compiler_pipeline_return_type_containing_a_quote_compiles_end_to_end(
    tmp_path,
):
    """Mirrors
    test_compiler_pipeline_parameter_type_containing_a_quote_compiles_end_to_end
    for a *return* type annotation, which flows into an endpoint's own
    responses={{200: {{"description": ...}}}} entry (response_description
    for a synchronous endpoint, task_response_description for a
    background one) through the exact same unescaped-embedding hazard.
    Covers both code paths in one notebook, since each builds this
    description differently.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "from typing import Literal\n\n"
            "def classify_sync(x: int) -> "
            'Literal["a\\"quoted\\"value"]:\n'
            '    return "a\\"quoted\\"value"\n\n'
            "def process_classify(x: int) -> "
            'Literal["a\\"quoted\\"value"]:\n'
            '    return "a\\"quoted\\"value"\n'
        )
    )

    notebook_path = tmp_path / "quoted_return_type.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    ast.parse(generated_app)
    compile(generated_app, "app.py", "exec")


def test_compiler_pipeline_function_docstring_becomes_the_endpoint_description(
    tmp_path,
):
    """Before this fix, extract_functions_from_code (parser/ast_parser.py)
    never even extracted a function's own docstring, so it was always
    discarded no matter what a notebook author wrote -- every endpoint's
    OpenAPI description was the same generic templated sentence
    ("Auto-generated endpoint for <name>. Operation ID: <name>.
    Parameters: <names>."), regardless of how much real documentation the
    author had already written directly on the function.
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

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app

schema = app.openapi()
description = schema["paths"]["/add"]["post"]["description"]
assert description == "Add two numbers and return their sum.", description

print("DOCSTRING_DESCRIPTION_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DOCSTRING_DESCRIPTION_E2E_OK" in proc.stdout


def test_compiler_pipeline_function_without_docstring_keeps_the_auto_generated_description(
    tmp_path,
):
    """A function with no docstring must keep getting the previous
    behavior's auto-generated description -- this feature is additive,
    not a replacement for every endpoint's docs.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app

schema = app.openapi()
description = schema["paths"]["/add"]["post"]["description"]
assert description == (
    "Auto-generated endpoint for add. Operation ID: add. "
    "Parameters: a, b."
), description

print("AUTO_DESCRIPTION_FALLBACK_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AUTO_DESCRIPTION_FALLBACK_E2E_OK" in proc.stdout


def test_compiler_pipeline_docstring_containing_quotes_and_newlines_compiles_end_to_end(
    tmp_path,
):
    """Mirrors
    test_compiler_pipeline_parameter_type_containing_a_quote_compiles_end_to_end
    for the same unescaped-embedding hazard, now on a function's own
    docstring: it's arbitrary, notebook-author-controlled text that can
    legitimately contain a double quote, a backslash, or span multiple
    lines. Before description was repr()'d rather than embedded as a raw
    f-string, any of those would close the description="..." literal
    early and corrupt the whole @app.post(...) call into a SyntaxError,
    failing the entire compile over a single endpoint's docs. Covers both
    the synchronous and background/task_id code paths, since each builds
    this @app.post(...) call separately.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def classify_sync(x: int) -> str:\n"
            '    """Classify x as "high" or "low".\n\n'
            "    Uses a backslash \\\\ in this line too.\n"
            '    """\n'
            '    return "high"\n\n'
            "def process_classify(x: int) -> str:\n"
            '    """Classify x as "high" or "low" (background version)."""\n'
            '    return "high"\n'
        )
    )

    notebook_path = tmp_path / "quoted_docstring.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    ast.parse(generated_app)
    compile(generated_app, "app.py", "exec")


def test_compiler_pipeline_sync_endpoint_reports_a_raised_exception_as_a_clean_500(
    tmp_path,
):
    """Confirmed exploitable before this fix: a synchronous notebook
    function raising an exception (a ZeroDivisionError here, but any bug
    or a legitimately bad input Pydantic's own type validation can't
    catch -- a KeyError, a bad file path, ...) propagated straight out of
    the endpoint unhandled, crashing with a bare, detail-free "Internal
    Server Error" -- exactly the gap _run_background_task already closed
    for the background/task_id path (it reports the task "failed" with
    str(e) instead of leaving it stuck forever), but with no equivalent
    on the synchronous path at all.
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
                            "def divide(a: int, b: int) -> float:\n"
                            "    return a / b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

# raise_server_exceptions=False so this behaves like a real deployment
# (Starlette's ServerErrorMiddleware catching the unhandled exception)
# instead of TestClient re-raising it into this test process.
client = TestClient(app, raise_server_exceptions=False)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

resp = client.post("/divide", json={{"a": 5, "b": 0}}, headers=headers)
assert resp.status_code == 500, resp.text
assert "divide" in resp.text
assert "ZeroDivisionError" in resp.text

print("SYNC_RAISED_EXCEPTION_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SYNC_RAISED_EXCEPTION_E2E_OK" in proc.stdout


def test_compiler_pipeline_sync_endpoint_lets_a_deliberate_httpexception_through_unwrapped(
    tmp_path,
):
    """A notebook function that imports fastapi itself and deliberately
    raises an HTTPException (e.g. to signal its own 404/403/409) is
    choosing that status code and message on purpose -- it must reach the
    caller as-is, not get swallowed into a generic 500 by the same
    except-Exception block that now catches everything else.
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
                            "from fastapi import HTTPException\n\n"
                            "def lookup(item_id: int) -> int:\n"
                            "    raise HTTPException(status_code=404, detail='item not found')\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

resp = client.post("/lookup", json={{"item_id": 1}}, headers=headers)
assert resp.status_code == 404, resp.text
assert resp.json() == {{"detail": "item not found"}}, resp.json()

print("SYNC_DELIBERATE_HTTPEXCEPTION_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SYNC_DELIBERATE_HTTPEXCEPTION_E2E_OK" in proc.stdout


def test_compiler_pipeline_sync_endpoint_with_unserializable_result_returns_a_clean_500(
    tmp_path,
):
    """Mirrors
    test_compiler_pipeline_background_task_with_unserializable_result_is_reported_as_failed
    for a synchronous (non-background) endpoint. Confirmed exploitable
    before this fix: a synchronous function returning something FastAPI's
    response serialization can't encode (e.g. a complex number -- Python
    builtin, no extra dependency needed to demonstrate this; a raw numpy
    array or pandas DataFrame is the more common real-world case for
    "compute_stats" but requires numpy as a test dependency this project
    doesn't otherwise have) crashed with an unhandled ValueError deep
    inside FastAPI's routing internals -- which a real (non-test-client)
    deployment surfaces to the caller as a bare "Internal Server Error"
    with no detail at all, unlike every other failure mode this
    generated app already reports clearly (auth, reserved names,
    oversized bodies, ...).
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
                            "def compute_stats(x: int) -> complex:\n"
                            "    return complex(x, x * 2)\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

# raise_server_exceptions=False so this behaves like a real deployment
# (Starlette's ServerErrorMiddleware catching an unhandled exception and
# turning it into a plain 500) instead of TestClient re-raising it into
# this test process.
client = TestClient(app, raise_server_exceptions=False)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

resp = client.post("/compute_stats", json={{"x": 5}}, headers=headers)
assert resp.status_code == 500, resp.text
assert "compute_stats" in resp.text
assert "not JSON-serializable" in resp.text

print("SYNC_UNSERIALIZABLE_RESULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SYNC_UNSERIALIZABLE_RESULT_E2E_OK" in proc.stdout


def test_compiler_pipeline_sync_endpoint_result_is_run_through_jsonable_encoder(
    tmp_path,
):
    """Mirrors
    test_compiler_pipeline_background_task_result_is_run_through_jsonable_encoder
    for a synchronous endpoint: a type json.dumps alone can't handle (a
    datetime) must still be delivered correctly, converted into JSON-safe
    data rather than merely happening not to break on it.
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
                            "import datetime\n\n"
                            "def report(year: int) -> dict:\n"
                            "    return {\n"
                            "        'year': year,\n"
                            "        'generated_on': datetime.date(2024, 1, 1),\n"
                            "    }\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

resp = client.post("/report", json={{"year": 2024}}, headers=headers)
assert resp.status_code == 200, resp.text
assert resp.json() == {{"result": {{"year": 2024, "generated_on": "2024-01-01"}}}}, resp.json()

print("SYNC_JSONABLE_ENCODER_RESULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SYNC_JSONABLE_ENCODER_RESULT_E2E_OK" in proc.stdout


def test_compiler_pipeline_rejects_notebook_function_named_verify_api_key(tmp_path):
    """Confirmed exploitable before this fix: a notebook function named
    verify_api_key rebinds the generated app's own auth-check function at
    module load time, silently disabling API-key authentication for every
    endpoint defined after it. compile_notebook must fail loudly instead
    of producing that app.
    """
    from backend.generator.api_generator import ReservedFunctionNameError

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def verify_api_key() -> dict:\n"
            "    return {'ok': True}\n"
        )
    )

    notebook_path = tmp_path / "reserved.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    with pytest.raises(ReservedFunctionNameError):
        compile_notebook(str(notebook_path), str(output_dir))


def test_compiler_pipeline_rejects_notebook_function_named_evict_expired_tasks(
    tmp_path,
):
    """Confirmed exploitable before this fix: a notebook function named
    _evict_expired_tasks silently overwrote the generated app's own
    module-level helper of that name at import time -- and broke every
    *other* background endpoint's own submission, not just this one's,
    since each one calls this exact now-shadowed name before enqueuing a
    new task. Reproduced directly against a real compiled+running app: a
    completely unrelated `train_model` background endpoint crashed with
    "TypeError: _evict_expired_tasks() missing 1 required positional
    argument" the moment it tried to submit a task, nothing to do with
    train_model's own logic at all. compile_notebook must fail loudly
    instead of producing that app.
    """
    from backend.generator.api_generator import ReservedFunctionNameError

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def _evict_expired_tasks(x: int) -> int:\n"
            "    return x\n\n"
            "def train_model(epochs: int) -> str:\n"
            "    return 'done'\n"
        )
    )

    notebook_path = tmp_path / "reserved_helper.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    with pytest.raises(ReservedFunctionNameError):
        compile_notebook(str(notebook_path), str(output_dir))


def test_compiler_pipeline_leaves_a_previous_successful_compile_untouched_on_failure(
    tmp_path
):
    """Confirmed exploitable before this fix: generate_fastapi_code (and
    the ReservedFunctionNameError it can raise) previously ran *after*
    write_runtime_module and write_requirements had already overwritten
    the runtime module and requirements.txt from a working previous
    compile with content generated from the *failing* notebook -- while
    app.py, the Dockerfile, and .compile_metadata.json were left
    untouched from that previous compile. A failed recompile (e.g. a typo
    that introduces a reserved-name collision) left output_dir in an
    inconsistent state matching neither the old nor the new notebook,
    with app.py expecting functions the runtime module no longer defined.
    """
    from backend.generator.api_generator import ReservedFunctionNameError

    good_notebook = nbformat.v4.new_notebook()
    good_notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(good_notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    runtime_path = output_dir / "runtime" / "notebook_module.py"
    requirements_path = output_dir / "requirements.txt"
    metadata_path = output_dir / ".compile_metadata.json"
    app_path = output_dir / "app.py"

    runtime_before = runtime_path.read_text(encoding="utf-8")
    requirements_before = requirements_path.read_text(encoding="utf-8")
    metadata_before = metadata_path.read_text(encoding="utf-8")
    app_before = app_path.read_text(encoding="utf-8")

    bad_notebook = nbformat.v4.new_notebook()
    bad_notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import json\n\n"
            "def health_check() -> dict:\n    return {}\n"
        )
    )
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(bad_notebook, f)

    with pytest.raises(ReservedFunctionNameError):
        compile_notebook(str(notebook_path), str(output_dir))

    # Every artifact from the previous successful compile must be
    # completely untouched -- not just individually valid, but an exact
    # match for the working app that was there before the failed attempt.
    assert runtime_path.read_text(encoding="utf-8") == runtime_before
    assert requirements_path.read_text(encoding="utf-8") == requirements_before
    assert metadata_path.read_text(encoding="utf-8") == metadata_before
    assert app_path.read_text(encoding="utf-8") == app_before


def test_clear_stale_export_artifacts_removes_openapi_and_sdk_files(tmp_path):

    (tmp_path / "openapi.json").write_text("{}", encoding="utf-8")
    (tmp_path / "openapi.yaml").write_text("{}", encoding="utf-8")
    sdk_dir = tmp_path / "sdk"
    sdk_dir.mkdir()
    (sdk_dir / "python_client.py").write_text("# client", encoding="utf-8")

    clear_stale_export_artifacts(str(tmp_path))

    assert not (tmp_path / "openapi.json").exists()
    assert not (tmp_path / "openapi.yaml").exists()
    assert not sdk_dir.exists()


def test_clear_stale_export_artifacts_is_a_no_op_when_nothing_was_ever_exported(
    tmp_path,
):
    # Must not raise just because there was never a prior export to clear.
    clear_stale_export_artifacts(str(tmp_path))


def test_compiler_pipeline_recompile_clears_a_stale_exported_openapi_and_sdk(
    tmp_path,
):
    """Confirmed exploitable before this fix: POST /api/export-openapi and
    POST /api/export-sdk (and the CLI's export-openapi/export-sdk
    commands) write openapi.json/openapi.yaml/sdk/ straight into
    output_dir, alongside the compiled app -- but recompiling the
    notebook only ever overwrote app.py, the runtime module,
    requirements.txt, and the Dockerfile, leaving any previously exported
    openapi.json/openapi.yaml/sdk/ completely untouched. A caller
    downloading the "compiled app" afterwards (GET /api/download, or GET
    /api/generated/openapi.json) got a schema/SDK describing the
    *previous* compile's endpoints, silently mismatched against the
    app.py sitting right next to it in the same directory.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    # Simulate a prior POST /api/export-openapi + POST /api/export-sdk
    # against this compile, without actually dynamically importing the
    # compiled app module (which export_openapi_schema does, and which
    # this project's own SDK tests deliberately run out-of-process to
    # avoid caching across tests in the same pytest process -- irrelevant
    # to what's being tested here, which is only whether a recompile
    # clears these files out, not what they contain).
    (output_dir / "openapi.json").write_text(
        json.dumps({"paths": {"/add": {}}}), encoding="utf-8"
    )
    (output_dir / "openapi.yaml").write_text("paths:\n  /add: {}\n", encoding="utf-8")
    sdk_dir = output_dir / "sdk"
    sdk_dir.mkdir()
    (sdk_dir / "python_client.py").write_text(
        "def add(self, payload): ...\n", encoding="utf-8"
    )

    other_notebook = nbformat.v4.new_notebook()
    other_notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def multiply(a: int, b: int) -> int:\n    return a * b\n"
        )
    )
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(other_notebook, f)

    compile_notebook(str(notebook_path), str(output_dir))

    app_source = (output_dir / "app.py").read_text(encoding="utf-8")
    assert "def multiply(" in app_source
    assert "def add(" not in app_source

    assert not (output_dir / "openapi.json").exists()
    assert not (output_dir / "openapi.yaml").exists()
    assert not sdk_dir.exists()


def test_compiler_pipeline_failed_recompile_leaves_stale_exports_untouched(tmp_path):
    """Mirrors
    test_compiler_pipeline_leaves_a_previous_successful_compile_untouched_on_failure
    for export artifacts specifically: a compile that fails (e.g. a
    reserved-name collision) must leave a previous compile's exported
    openapi.json/sdk/ untouched too, the same as it already leaves
    app.py/requirements.txt/the runtime module untouched -- clearing
    stale exports only makes sense once a new, actually-successful
    compile exists to replace what they described.
    """
    from backend.generator.api_generator import ReservedFunctionNameError

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    openapi_path = output_dir / "openapi.json"
    openapi_path.write_text(json.dumps({"paths": {"/add": {}}}), encoding="utf-8")
    sdk_dir = output_dir / "sdk"
    sdk_dir.mkdir()
    (sdk_dir / "python_client.py").write_text("# client", encoding="utf-8")

    bad_notebook = nbformat.v4.new_notebook()
    bad_notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def health_check() -> dict:\n    return {}\n"
        )
    )
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(bad_notebook, f)

    with pytest.raises(ReservedFunctionNameError):
        compile_notebook(str(notebook_path), str(output_dir))

    assert openapi_path.exists()
    assert sdk_dir.exists()


def test_compiler_pipeline_case_colliding_function_names_get_distinct_models(tmp_path):
    """Confirmed exploitable before this fix: two notebook functions
    differing only by the case of their first letter (e.g. "get_data" and
    "Get_data") produced identically-named Pydantic request model classes,
    so the second class definition silently shadowed the first -- one
    endpoint ended up validating requests against the *other* function's
    fields.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def get_data(query: str) -> dict:\n"
            "    return {'query': query}\n"
        )
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def Get_data(id: int) -> dict:\n"
            "    return {'id': id}\n"
        )
    )

    notebook_path = tmp_path / "collide.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")

    ast.parse(generated_app)
    compile(generated_app, "app.py", "exec")

    assert generated_app.count("class Get_dataRequest(BaseModel):") == 1
    assert generated_app.count("class Get_dataRequest_2(BaseModel):") == 1


def test_package_name_for_output_dir_uses_basename():

    assert package_name_for_output_dir("generated") == "generated"
    assert package_name_for_output_dir("my_output") == "my_output"
    assert package_name_for_output_dir("build/my_output") == "my_output"


def test_package_name_for_output_dir_rejects_invalid_identifier():

    with pytest.raises(ValueError):
        package_name_for_output_dir("my-output")


def test_package_name_for_output_dir_rejects_python_keyword():

    with pytest.raises(ValueError):
        package_name_for_output_dir("import")


@pytest.mark.parametrize("stdlib_name", ["json", "os", "sys", "time", "re"])
def test_package_name_for_output_dir_rejects_a_standard_library_module_name(
    stdlib_name,
):
    """Confirmed exploitable before this fix: `--output json` (or any
    real standard-library module name) passed this function's own
    isidentifier()/keyword checks fine and compiled without error, but
    the generated app.py's `import json.runtime.notebook_module as
    notebook_module` statement then resolved to the real, already-
    imported stdlib `json` module instead of the locally compiled
    package -- Python's import system finds a standard-library module
    ahead of a same-named package under the working directory. The
    generated app was entirely unusable (`python -m uvicorn json.app:app`
    -- what `serve`, the generated Dockerfile's CMD, and any real
    deployment all run -- failed with "No module named 'json.app'"),
    with the failure only ever surfacing later, disconnected from the
    --output choice that actually caused it.
    """

    with pytest.raises(ValueError, match="standard library module"):
        package_name_for_output_dir(stdlib_name)


def test_package_name_for_output_dir_allows_a_name_that_merely_looks_like_a_builtin():
    """Only real *importable modules* collide this way -- "list"/"dict"/
    "str" are builtin *types*, not modules, so there's no standard-
    library module for `import list.runtime.notebook_module` to
    incorrectly resolve to instead.
    """

    assert package_name_for_output_dir("list") == "list"


@pytest.mark.parametrize("installed_package_name", ["fastapi", "pytest", "httpx"])
def test_package_name_for_output_dir_rejects_an_installed_third_party_package_name(
    installed_package_name,
):
    """Confirmed exploitable before this fix: `--output fastapi` (or any
    other package genuinely `pip install`ed in the compiling environment
    -- fastapi/pytest/httpx are all in this project's own requirements.txt,
    so they're guaranteed present here) passed the isidentifier()/keyword/
    STANDARD_LIBS checks fine and compiled without error, but the
    generated app.py's `import fastapi.runtime.notebook_module` statement
    then resolved to the real, already-installed `fastapi` package
    instead of the locally compiled one -- reproduced against a real
    `python -m uvicorn fastapi.app:app`, which fails outright with
    "Could not import module 'fastapi.app'" since the real package has no
    such submodule. This is the exact same import-shadowing hazard the
    standard-library check above already guards against, just for a
    third-party package instead of one built into the interpreter.
    """

    with pytest.raises(ValueError, match="already-installed"):
        package_name_for_output_dir(installed_package_name)


def test_package_name_for_output_dir_allows_a_name_with_no_installed_package():
    """A name that isn't a standard-library module and isn't an installed
    third-party package either has nothing real for
    `import <name>.runtime.notebook_module` to incorrectly resolve to, so
    it's allowed through exactly as before.
    """

    assert (
        package_name_for_output_dir("definitely_not_an_installed_package_xyz")
        == "definitely_not_an_installed_package_xyz"
    )


def test_package_name_for_output_dir_does_not_flag_this_tools_own_prior_output_dirs():
    """A directory this tool itself already compiled into (e.g.
    "generated", the documented default) is a real, on-disk Python
    package the moment it exists -- but it was never `pip install`ed, so
    it must not trip the installed-third-party-package check above. Using
    importlib.util.find_spec (which also matches local, non-installed
    directories) instead of importlib.metadata.packages_distributions()
    would have made this tool's own default --output start failing the
    very first time it was reused for a second compile.
    """

    assert package_name_for_output_dir("generated") == "generated"
    assert package_name_for_output_dir("test_generated") == "test_generated"


def test_this_tools_own_package_name_is_backend():
    """Sanity check on the constant itself: derived from __name__
    ("backend.compiler") rather than hardcoded, so it can't drift if this
    package is ever renamed -- but the collision check below is only
    meaningful if it actually resolves to the real package name.
    """

    assert THIS_TOOLS_OWN_PACKAGE_NAME == "backend"


def test_package_name_for_output_dir_rejects_this_tools_own_package_name():
    """Confirmed exploitable before this fix: `--output backend` passed
    the isidentifier()/keyword/STANDARD_LIBS/installed-third-party-package
    checks fine (this project was never `pip install`ed, so
    importlib.metadata.packages_distributions() has no metadata for
    "backend" at all -- the identical reason this tool's own prior
    --output dirs like "generated" are deliberately allowed through) and
    compiled without error -- but the generated app.py's `import
    backend.runtime.notebook_module` statement would then resolve to this
    tool's own real "backend" package instead of the locally compiled
    one. Unlike an ordinary already-installed third-party package, this
    collision isn't merely possible: backend/compiler.py (the module
    performing this very check) is part of the "backend" package, so it
    is unconditionally already imported in every single invocation of
    this tool.
    """

    with pytest.raises(ValueError, match="this tool's own top-level package"):
        package_name_for_output_dir("backend")


def test_compiler_pipeline_rejects_output_dir_colliding_with_this_tools_own_package(
    tmp_path,
):
    """End-to-end confirmation that compile_notebook itself -- not just
    the package_name_for_output_dir helper in isolation -- refuses to
    compile into an --output directory named "backend", and does so
    before writing anything.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "backend"

    with pytest.raises(ValueError, match="this tool's own top-level package"):
        compile_notebook(str(notebook_path), str(output_dir))

    assert not (output_dir / "app.py").exists()


def test_compiler_pipeline_rejects_output_dir_colliding_with_an_installed_package(
    tmp_path,
):
    """End-to-end confirmation that compile_notebook itself -- not just
    the package_name_for_output_dir helper in isolation -- refuses to
    compile into an --output directory whose basename shadows an
    installed package, and does so before writing anything.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "fastapi"

    with pytest.raises(ValueError, match="already-installed"):
        compile_notebook(str(notebook_path), str(output_dir))

    # The collision is rejected before anything is actually written --
    # only the (empty) output directory itself may exist, from
    # compile_notebook_to_api's own os.makedirs call that precedes the
    # package-name check.
    assert not (output_dir / "app.py").exists()


def test_compiler_pipeline_respects_custom_output_dir(tmp_path):
    """The --output flag is documented as configurable (it has a CLI flag
    with a default), but write_runtime_module used to hardcode
    "generated/runtime/..." regardless of output_dir while the generated
    app.py always imported the fixed name "generated" -- so any non-
    default --output directory produced files in the wrong place with an
    import that could never resolve them.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "my_custom_output"

    compile_notebook(str(notebook_path), str(output_dir))

    # The runtime module must live under the actual output directory, not
    # the old hardcoded "generated/runtime/" path.
    assert (output_dir / "runtime" / "notebook_module.py").exists()

    generated_app = (output_dir / "app.py").read_text(encoding="utf-8")
    ast.parse(generated_app)
    assert "import my_custom_output.runtime.notebook_module" in generated_app

    dockerfile = (output_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY . my_custom_output/" in dockerfile
    assert "uvicorn my_custom_output.app:app" in dockerfile


def test_compiler_pipeline_custom_output_dir_actually_runs(tmp_path):
    """Static checks confirm the generated files are consistent with each
    other; this drives a real request through the compiled app with a
    custom --output directory to confirm it actually imports and runs,
    not just that the generated source text looks right. Run in a fresh
    subprocess/cwd since the generated package name and its import must
    be resolved by a real Python import machinery run from the directory
    compilation happened in.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "my_custom_output")

from my_custom_output.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
resp = client.post(
    "/add",
    json={{"a": 2, "b": 3}},
    headers={{"X-API-Key": "notebook-to-api-dev-key"}},
)
assert resp.status_code == 200, resp.text
assert resp.json() == {{"result": 5}}, resp.json()
print("CUSTOM_OUTPUT_DIR_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CUSTOM_OUTPUT_DIR_E2E_OK" in proc.stdout


def test_standard_libs_covers_common_stdlib_modules_beyond_the_old_hardcoded_list():
    """The old STANDARD_LIBS was a hand-picked set of 12 names, missing
    the vast majority of the standard library. Any notebook using one of
    the missed modules got it written into requirements.txt as if it were
    a third-party PyPI package -- and for some names (e.g. "asyncio"),
    PyPI has an unrelated real package that pip actually installs,
    shadowing the built-in module.
    """

    commonly_missed = {
        "asyncio", "random", "logging", "subprocess", "csv", "sqlite3",
        "uuid", "hashlib", "threading", "shutil", "glob", "base64",
        "enum", "dataclasses", "copy", "pickle", "warnings", "traceback",
        "inspect", "urllib", "string", "decimal", "tempfile", "io",
    }

    assert commonly_missed <= STANDARD_LIBS


def test_standard_libs_does_not_exclude_third_party_packages():

    third_party = {"pandas", "numpy", "requests", "sklearn", "fastapi"}

    assert not (third_party & STANDARD_LIBS)


def test_compiler_pipeline_excludes_stdlib_modules_from_requirements(tmp_path):

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import asyncio\n"
            "import random\n"
            "import pandas as pd\n\n"
            "def compute(x: int) -> int:\n"
            "    return x\n"
        )
    )

    notebook_path = tmp_path / "stdlib_imports.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(
        encoding="utf-8"
    )
    # Dependencies are pinned to "name==version" when the compiling
    # environment has the package installed (see
    # test_requirements_pins_installed_dependency_versions), so a
    # dependency line no longer necessarily equals the bare package name.
    dep_names = {line.split("==")[0] for line in requirements.split()}

    assert "asyncio" not in dep_names
    assert "random" not in dep_names
    assert "pandas" in dep_names


def test_compiler_pipeline_maps_a_dangerously_ambiguous_import_to_its_real_pypi_name(
    tmp_path,
):
    """Confirmed missing before this fix: PyPI hosts a real, unrelated,
    unofficial package under the bare import name "dotenv" -- unmapped,
    requirements.txt listed "dotenv" itself, and `pip install -r
    requirements.txt` in the generated Dockerfile's build would silently
    install the *wrong* package instead of python-dotenv, the one the
    notebook actually needs.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "from dotenv import load_dotenv\n\n"
            "def get_config() -> dict:\n"
            "    return {}\n"
        )
    )

    notebook_path = tmp_path / "dotenv_import.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(
        encoding="utf-8"
    )
    dep_names = {line.split("==")[0] for line in requirements.split()}

    assert "python-dotenv" in dep_names
    assert "dotenv" not in dep_names


def test_requirements_omits_watchdog(tmp_path):
    """watchdog is a dependency of this tool's own `serve` command (hot
    recompilation while developing locally) -- the generated app itself
    never imports it (see generator/api_generator.py). Before this fix,
    every compiled app shipped it as an unused line in requirements.txt
    and, from there, an unused package baked into its Docker image.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")

    assert "watchdog" not in requirements


def test_requirements_pins_core_dependencies_to_installed_versions(tmp_path):

    import importlib.metadata

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    lines = set(requirements.split())

    for package in ("fastapi", "uvicorn", "pydantic"):
        installed_version = importlib.metadata.version(package)
        assert f"{package}=={installed_version}" in lines


def test_requirements_pins_a_notebook_dependency_installed_in_this_environment(
    tmp_path
):
    """Without pinning, requirements.txt just listed the bare package
    name -- `pip install -r requirements.txt` at deploy time would then
    resolve whatever the latest release happens to be, not the version
    the notebook was actually compiled and tested against.

    Uses nbformat as the notebook's import: it's a hard dependency of
    this very test file (imported at the top), so it's guaranteed
    installed in any environment capable of running this suite at all --
    unlike pandas, which this test used before and which isn't listed in
    the project's requirements.txt, so it's absent in a clean CI install
    and importlib.metadata.version() raised PackageNotFoundError there
    (confirmed: this test only ever passed locally by accident, because
    pandas happened to already be installed in that environment for
    unrelated reasons).
    """

    import importlib.metadata

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import nbformat\n\n"
            "def summarize(count: int) -> int:\n    return count * 2\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")

    installed_nbformat_version = importlib.metadata.version("nbformat")
    assert f"nbformat=={installed_nbformat_version}" in requirements.split()


def test_requirements_falls_back_to_a_bare_name_for_an_uninstalled_dependency(
    tmp_path
):
    """A notebook can import a third-party library the machine compiling
    it doesn't happen to have installed -- this tool has no way to look up
    a version it can't introspect, so the previous, unpinned behavior
    (just the bare name) must still be used instead of failing the
    compile outright.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import definitely_not_installed_pkg_hopefully\n\n"
            "def noop() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    lines = set(requirements.split())

    assert "definitely_not_installed_pkg_hopefully" in lines


def test_pinned_requirement_helper_pins_an_installed_package():

    import importlib.metadata

    from backend.compiler import _pinned_requirement

    assert _pinned_requirement("pytest") == f"pytest=={importlib.metadata.version('pytest')}"


def test_pinned_requirement_helper_falls_back_for_an_unknown_package():

    from backend.compiler import _pinned_requirement

    assert _pinned_requirement("definitely_not_a_real_package_xyz") == (
        "definitely_not_a_real_package_xyz"
    )


def test_pinned_requirement_helper_strips_a_pep440_local_version_segment(monkeypatch):
    """PyPI rejects any upload whose version contains a "+" (PEP 440's
    local version segment) -- it exists specifically to distinguish a
    locally-modified build (a CUDA-specific wheel, a setuptools-scm/git
    "+dirty" build, an editable install, ...) from the public release it's
    based on, never to be redistributed itself. Pinning the exact local
    version importlib.metadata.version() reports bakes an unresolvable
    "package==version+local" line into requirements.txt: confirmed
    reproduced against a real `pip install` of exactly this kind of pin,
    which fails with "No matching distribution found for ...".
    """

    import backend.compiler as compiler_module

    monkeypatch.setattr(
        compiler_module.importlib.metadata,
        "version",
        lambda package_name: "2.1.0+cu121",
    )

    assert compiler_module._pinned_requirement("torch_local_version_test") == (
        "torch_local_version_test==2.1.0"
    )


def test_pinned_requirement_helper_leaves_an_ordinary_version_untouched(monkeypatch):
    """The common case -- no "+" in the reported version at all -- must
    behave exactly as before: pinned to the full version string, unchanged.
    """

    import backend.compiler as compiler_module

    monkeypatch.setattr(
        compiler_module.importlib.metadata,
        "version",
        lambda package_name: "1.2.3",
    )

    assert compiler_module._pinned_requirement("ordinary_version_test") == (
        "ordinary_version_test==1.2.3"
    )


def test_requirements_strips_a_local_version_segment_from_a_pinned_dependency(
    tmp_path, monkeypatch
):
    """End-to-end: a notebook importing a package whose installed version
    happens to carry a PEP 440 local version segment must still get a
    requirements.txt pin that `pip install` can actually resolve, not the
    unresolvable exact local build.
    """

    import backend.compiler as compiler_module

    real_version = compiler_module.importlib.metadata.version

    def fake_version(package_name):
        if package_name == "local_version_dependency_test":
            return "0.1.0+dirty"
        return real_version(package_name)

    monkeypatch.setattr(compiler_module.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(
        compiler_module,
        "distribution_name_for_import",
        lambda import_name: (
            "local_version_dependency_test"
            if import_name == "local_version_dependency_test"
            else import_name
        ),
    )

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import local_version_dependency_test\n\n"
            "def noop() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    lines = set(requirements.split())

    assert "local_version_dependency_test==0.1.0" in lines
    assert "local_version_dependency_test==0.1.0+dirty" not in lines
    assert not any("+dirty" in line for line in lines)


def test_resolve_requirements_drops_the_auto_detected_line_that_conflicts_with_an_explicit_one():
    """A notebook importing a package directly while also declaring
    "# notebook-to-api: requires <same-package>==<version>" (to pin a
    specific version this tool's own auto-resolution wouldn't otherwise
    choose) previously got *both* lines written to requirements.txt --
    confirmed exploitable: two version-pinned lines for the same
    distribution is a requirement pip refuses outright ("Double
    requirement given"), breaking `deploy`'s own Docker build over
    exactly the kind of explicit override this directive exists to let a
    notebook author make. The explicit line must win.
    """

    requirements = resolve_requirements(
        ["numpy"], explicit_requirements=["numpy==1.24.0"]
    )

    numpy_lines = [line for line in requirements if line.split("==")[0] == "numpy"]
    assert numpy_lines == ["numpy==1.24.0"]


def test_resolve_requirements_conflict_detection_is_case_insensitive():
    """PyPI distribution names are themselves case-insensitive -- pip
    normalizes "NumPy"/"numpy"/"nUmPy" to the identical project -- so a
    directive spelled differently than distribution_name_for_import's
    own resolved name must still be recognized as the same package.
    """

    requirements = resolve_requirements(
        ["numpy"], explicit_requirements=["NumPy==1.24.0"]
    )

    numpy_lines = [
        line for line in requirements if line.lower().split("==")[0] == "numpy"
    ]
    assert numpy_lines == ["NumPy==1.24.0"]


def test_resolve_requirements_keeps_auto_detected_lines_with_no_explicit_conflict():

    requirements = resolve_requirements(
        ["requests"], explicit_requirements=["a-private-pkg==1.0.0"]
    )

    assert any(line.startswith("requests") for line in requirements)
    assert "a-private-pkg==1.0.0" in requirements


def test_compile_drops_the_auto_detected_dependency_that_conflicts_with_an_explicit_pin(
    tmp_path
):
    """Uses python-multipart the same way
    test_requirements_resolves_an_import_name_to_its_actual_distribution_name
    (below) does: its import name ("multipart") differs from its
    distribution name ("python-multipart") -- so the explicit directive
    here, naming the *distribution*, must still suppress the
    auto-detected import's own resolved "python-multipart==<installed>"
    line, not just an exact-text match against the raw import name.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "# notebook-to-api: requires python-multipart==999.0.0\n"
            "import multipart\n\n"
            "def noop() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    lines = requirements.split()

    multipart_lines = [
        line for line in lines if line.split("==")[0] == "python-multipart"
    ]
    assert multipart_lines == ["python-multipart==999.0.0"]


def test_compile_raises_for_conflicting_explicit_requirement_directives(tmp_path):

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "# notebook-to-api: requires numpy==1.24.0\n"
            "# notebook-to-api: requires numpy==1.26.0\n"
            "def noop() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    with pytest.raises(ValueError, match="numpy"):
        compile_notebook(str(notebook_path), str(output_dir))


def test_compile_with_conflicting_requirements_does_not_corrupt_a_previous_good_compile(
    tmp_path
):
    """Mirrors the identical "output_dir left in an inconsistent state"
    regression compile_notebook_to_api's own comment above (right before
    generate_fastapi_code is called) already documents fixing for a
    ReservedFunctionNameError -- this closes the same class of bug for a
    conflicting-requirements notebook: recompiling a working app with one
    must leave every file from the last working compile completely
    untouched, not a torn mix of the old app.py/Dockerfile alongside a
    requirements.txt already rewritten from the failing notebook.
    """

    good_notebook = nbformat.v4.new_notebook()
    good_notebook.cells.append(
        nbformat.v4.new_code_cell("def add(a: int, b: int) -> int:\n    return a + b\n")
    )
    good_notebook_path = tmp_path / "good.ipynb"
    with open(good_notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(good_notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(good_notebook_path), str(output_dir))

    app_py_before = (output_dir / "app.py").read_text(encoding="utf-8")
    requirements_before = (output_dir / "requirements.txt").read_text(encoding="utf-8")

    bad_notebook = nbformat.v4.new_notebook()
    bad_notebook.cells.append(
        nbformat.v4.new_code_cell(
            "# notebook-to-api: requires numpy==1.24.0\n"
            "# notebook-to-api: requires numpy==1.26.0\n"
            "def multiply(a: int, b: int) -> int:\n    return a * b\n"
        )
    )
    bad_notebook_path = tmp_path / "bad.ipynb"
    with open(bad_notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(bad_notebook, f)

    with pytest.raises(ValueError, match="numpy"):
        compile_notebook(str(bad_notebook_path), str(output_dir))

    assert (output_dir / "app.py").read_text(encoding="utf-8") == app_py_before
    assert (
        (output_dir / "requirements.txt").read_text(encoding="utf-8")
        == requirements_before
    )


def test_requirements_resolves_an_import_name_to_its_actual_distribution_name(
    tmp_path
):
    """A notebook's `import` statement names a *module*, not necessarily
    the PyPI *distribution* that provides it -- `pip install <name>` only
    works for the latter, and the two frequently differ. Before this,
    write_requirements wrote the raw import name straight into
    requirements.txt unchanged, so `pip install -r requirements.txt` --
    and from there every `deploy`/`docker build` -- failed outright for
    any notebook using one of these.

    Uses python-multipart as the notebook's import: its import name
    ("multipart") differs from its distribution name ("python-multipart"),
    and it's a direct, guaranteed dependency of this very project (see
    requirements.txt) -- reliably installed in any environment capable of
    running this suite at all, the same reliability rationale
    test_requirements_pins_a_notebook_dependency_installed_in_this_environment
    (just above) already documents for its own choice of nbformat.
    """

    import importlib.metadata

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import multipart\n\n"
            "def noop() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    lines = set(requirements.split())

    installed_version = importlib.metadata.version("python-multipart")
    assert f"python-multipart=={installed_version}" in lines
    # Must not also (or instead) list the raw, uninstallable import name.
    assert "multipart" not in lines
    assert not any(line.startswith("multipart==") for line in lines)


def test_requirements_deduplicates_distinct_imports_resolving_to_the_same_distribution(
    tmp_path, monkeypatch
):
    """Two distinct import names occasionally resolve to the same PyPI
    distribution (e.g. "attr" and "attrs" are both provided by the
    "attrs" distribution) -- this must not write a duplicate
    requirements.txt line for it.
    """

    import backend.compiler as compiler_module

    monkeypatch.setattr(
        compiler_module,
        "distribution_name_for_import",
        lambda import_name: "shared_distribution_test_pkg",
    )

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import alias_one\n"
            "import alias_two\n\n"
            "def noop() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    lines = requirements.split()

    assert lines.count("shared_distribution_test_pkg") == 1


def test_distribution_name_for_import_helper_resolves_a_known_alias():

    from backend.compiler import distribution_name_for_import

    assert distribution_name_for_import("multipart") == "python-multipart"


def test_distribution_name_for_import_helper_falls_back_for_an_unknown_import():

    from backend.compiler import distribution_name_for_import

    assert distribution_name_for_import("definitely_not_a_real_import_xyz") == (
        "definitely_not_a_real_import_xyz"
    )


def test_compile_writes_metadata_recording_the_source_notebook(tmp_path):
    """Nothing on disk (or via the API) previously recorded which
    notebook produced a given `generated/` output -- GET /api/notebooks
    had no way to say "this is the one currently compiled" as a result
    (see test_list_notebooks_marks_the_currently_compiled_notebook in
    test_upload_routes.py).
    """

    import json

    from backend.compiler import COMPILE_METADATA_FILENAME

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    metadata_path = output_dir / COMPILE_METADATA_FILENAME
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["source_notebook"] == str(notebook_path.resolve())
    assert "compiled_at" in metadata

    # Must be a real, parseable ISO 8601 UTC timestamp, not just any string.
    from datetime import datetime
    parsed = datetime.fromisoformat(metadata["compiled_at"])
    assert parsed.tzinfo is not None


def test_compile_metadata_records_an_absolute_path_even_for_a_relative_input(
    tmp_path, monkeypatch
):

    import json

    from backend.compiler import COMPILE_METADATA_FILENAME

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    monkeypatch.chdir(tmp_path)

    compile_notebook("nb.ipynb", "built")

    metadata_path = Path("built") / COMPILE_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert Path(metadata["source_notebook"]).is_absolute()
    assert Path(metadata["source_notebook"]) == notebook_path.resolve()


def test_compile_notebook_with_a_source_notebook_path_records_it_but_hashes_the_compiled_content(
    tmp_path,
):
    """compile_notebook's own "source_notebook_path" -- used by POST
    /api/compile's "version_id" (backend/routes/upload.py) to compile one
    of a notebook's own previously snapshotted versions without restoring
    it over the notebook's current content -- must record the *real*
    notebook as "source_notebook" while hashing the content that actually
    got compiled (the version snapshot), not the real notebook's own
    current (different) content.
    """

    import json

    from backend.compiler import COMPILE_METADATA_FILENAME, hash_notebook_file

    real_notebook = nbformat.v4.new_notebook()
    real_notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def multiply(a: int, b: int) -> int:\n    return a * b\n"
        )
    )
    real_notebook_path = tmp_path / "nb.ipynb"
    with open(real_notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(real_notebook, f)

    old_version = nbformat.v4.new_notebook()
    old_version.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )
    old_version_path = tmp_path / "nb_v1.ipynb"
    with open(old_version_path, "w", encoding="utf-8") as f:
        nbformat.write(old_version, f)

    output_dir = tmp_path / "generated"
    compile_notebook(
        str(old_version_path),
        str(output_dir),
        source_notebook_path=str(real_notebook_path),
    )

    app_code = (output_dir / "app.py").read_text(encoding="utf-8")
    assert "add" in app_code
    assert "multiply" not in app_code

    metadata_path = output_dir / COMPILE_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["source_notebook"] == str(real_notebook_path.resolve())
    assert metadata["source_notebook_sha256"] == hash_notebook_file(str(old_version_path))
    assert metadata["source_notebook_sha256"] != hash_notebook_file(str(real_notebook_path))


def test_compile_notebook_records_the_given_version_id(tmp_path):

    import json

    from backend.compiler import COMPILE_METADATA_FILENAME

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell("def add(a: int, b: int) -> int:\n    return a + b\n")
    )
    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(
        str(notebook_path), str(output_dir), version_id="20260101T000000000000_abcd.ipynb",
    )

    metadata_path = output_dir / COMPILE_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["compiled_version_id"] == "20260101T000000000000_abcd.ipynb"


def test_compile_notebook_records_a_null_version_id_by_default(tmp_path):

    import json

    from backend.compiler import COMPILE_METADATA_FILENAME

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell("def add(a: int, b: int) -> int:\n    return a + b\n")
    )
    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    metadata_path = output_dir / COMPILE_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["compiled_version_id"] is None


def test_recompiling_overwrites_the_previous_compile_metadata(tmp_path):

    import json

    from backend.compiler import COMPILE_METADATA_FILENAME

    def _make_notebook(path, source):
        notebook = nbformat.v4.new_notebook()
        notebook.cells.append(nbformat.v4.new_code_cell(source))
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

    notebook_a = tmp_path / "a.ipynb"
    notebook_b = tmp_path / "b.ipynb"
    _make_notebook(notebook_a, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _make_notebook(notebook_b, "def sub(a: int, b: int) -> int:\n    return a - b\n")

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_a), str(output_dir))
    compile_notebook(str(notebook_b), str(output_dir))

    metadata_path = output_dir / COMPILE_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["source_notebook"] == str(notebook_b.resolve())


def test_recompile_removes_stale_compile_metadata_when_a_later_write_step_fails(
    tmp_path, monkeypatch
):
    """app.py (and its runtime module) are written before Dockerfile/
    .dockerignore generation and write_compile_metadata -- so a failure in
    any of those three later steps (a real disk/permission failure; here
    simulated via generate_dockerfile) previously left app.py already
    reflecting the *new* notebook while .compile_metadata.json, untouched,
    still described whichever notebook the *previous* successful compile
    actually produced it for. That's not merely stale, it's silently
    wrong: every metadata-driven consumer (GET /api/notebooks'
    "currently_compiled"/"compiled_at"/"notebook_changed_since_compile",
    GET /api/generated's "source_notebook_filename"/
    "source_notebook_exists") would confidently report the wrong notebook
    as the one actually being served, with nothing to indicate the
    mismatch. Confirmed reproduced before this fix: recompiling a working
    "add" app with a notebook exposing "multiply" while generate_dockerfile
    was made to raise left the runtime module already containing only
    "multiply", with .compile_metadata.json's "source_notebook" still
    naming the "add" notebook.
    """

    import backend.compiler as compiler_module

    def _make_notebook(path, source):
        notebook = nbformat.v4.new_notebook()
        notebook.cells.append(nbformat.v4.new_code_cell(source))
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

    notebook_a = tmp_path / "add.ipynb"
    notebook_b = tmp_path / "multiply.ipynb"
    _make_notebook(notebook_a, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _make_notebook(
        notebook_b, "def multiply(a: int, b: int) -> int:\n    return a * b\n"
    )

    output_dir = tmp_path / "generated"

    # First, a genuinely successful compile.
    compile_notebook(str(notebook_a), str(output_dir))

    metadata_path = output_dir / compiler_module.COMPILE_METADATA_FILENAME
    assert metadata_path.is_file()

    # Now recompile with a different notebook, but make the post-app.py
    # Dockerfile generation step fail -- standing in for a real
    # disk/permission failure at that point.
    monkeypatch.setattr(
        compiler_module,
        "generate_dockerfile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("simulated disk failure")
        ),
    )

    with pytest.raises(PermissionError):
        compile_notebook(str(notebook_b), str(output_dir))

    # app.py's runtime module already moved on to the new notebook...
    runtime_module_source = (
        output_dir / "runtime" / "notebook_module.py"
    ).read_text(encoding="utf-8")
    assert "def multiply(" in runtime_module_source
    assert "def add(" not in runtime_module_source

    # ...so the now-stale metadata (still describing the *old* notebook)
    # must be gone, not left silently pointing at the wrong one.
    assert not metadata_path.exists()


def test_first_ever_compile_failing_at_a_post_app_py_step_leaves_no_metadata_file(
    tmp_path, monkeypatch
):
    """The no-previous-compile case: there's no stale metadata file to
    remove (none was ever written), so this must be a clean no-op rather
    than crashing trying to remove a file that was never there.
    """

    import backend.compiler as compiler_module

    notebook = tmp_path / "nb.ipynb"
    nb = nbformat.v4.new_notebook()
    nb.cells.append(
        nbformat.v4.new_code_cell("def add(a: int, b: int) -> int:\n    return a + b\n")
    )
    with open(notebook, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    output_dir = tmp_path / "generated"

    monkeypatch.setattr(
        compiler_module,
        "generate_dockerfile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("simulated disk failure")
        ),
    )

    with pytest.raises(PermissionError):
        compile_notebook(str(notebook), str(output_dir))

    assert not (output_dir / compiler_module.COMPILE_METADATA_FILENAME).exists()


def test_hash_notebook_file_is_deterministic_and_content_sensitive(tmp_path):

    from backend.compiler import hash_notebook_file

    notebook_a = tmp_path / "a.ipynb"
    notebook_a.write_text('{"cells": []}', encoding="utf-8")

    notebook_a_copy = tmp_path / "a_copy.ipynb"
    notebook_a_copy.write_text('{"cells": []}', encoding="utf-8")

    notebook_b = tmp_path / "b.ipynb"
    notebook_b.write_text('{"cells": [1]}', encoding="utf-8")

    # Same content (even in a different file) -> same hash.
    assert hash_notebook_file(notebook_a) == hash_notebook_file(notebook_a_copy)
    # Different content -> different hash.
    assert hash_notebook_file(notebook_a) != hash_notebook_file(notebook_b)
    # A plain hex sha256 digest, not some other encoding.
    digest = hash_notebook_file(notebook_a)
    assert len(digest) == 64
    int(digest, 16)  # must not raise


def test_compile_metadata_records_the_source_notebooks_content_hash(tmp_path):
    """Closes the gap left by write_compile_metadata recording only
    *which* notebook was compiled: even knowing that, there was no way to
    tell whether the notebook had since been edited and re-uploaded,
    leaving the currently-served app silently stale relative to it (see
    test_list_notebooks_flags_a_notebook_changed_since_its_last_compile
    in test_upload_routes.py for the end-to-end behavior this enables).
    """

    import json

    from backend.compiler import COMPILE_METADATA_FILENAME, hash_notebook_file

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    metadata_path = output_dir / COMPILE_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["source_notebook_sha256"] == hash_notebook_file(notebook_path)


def test_compile_metadata_records_generated_files_sha256(tmp_path):
    """The output-side counterpart to source_notebook_sha256 above: a
    baseline hash over the compile-produced files themselves (app.py,
    requirements.txt, Dockerfile, ...), recorded at the very end of a
    successful compile so a later caller can tell whether the *compiled
    output* has since been hand-edited on the server, not just whether
    the source notebook has.
    """

    import json

    from backend.compiler import COMPILE_METADATA_FILENAME, _generated_files_sha256

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    metadata_path = output_dir / COMPILE_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["generated_files_sha256"] == _generated_files_sha256(str(output_dir))


def test_generated_files_sha256_changes_when_a_generated_file_is_hand_edited(tmp_path):

    from backend.compiler import _generated_files_sha256

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    baseline = _generated_files_sha256(str(output_dir))

    (output_dir / "requirements.txt").write_text(
        "fastapi==0.0.0\n", encoding="utf-8"
    )

    assert _generated_files_sha256(str(output_dir)) != baseline


def test_generated_files_sha256_changes_when_env_example_is_hand_edited(tmp_path):

    from backend.compiler import _generated_files_sha256

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    baseline = _generated_files_sha256(str(output_dir))

    (output_dir / ".env.example").write_text(
        "PORT=9999\n", encoding="utf-8"
    )

    assert _generated_files_sha256(str(output_dir)) != baseline


def test_generated_files_sha256_changes_when_kubernetes_manifest_is_hand_edited(
    tmp_path
):
    """kubernetes.yaml (generate_kubernetes_manifest) is now a real
    compile-produced artifact, like docker-compose.yml/.env.example/
    README.md -- it must participate in the same hand-edit detection
    _generated_files_sha256 already gives those, or GET /api/notebooks'
    own "generated_files_modified_since_compile" would silently miss a
    hand-edit to it.
    """

    from backend.compiler import _generated_files_sha256

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    baseline = _generated_files_sha256(str(output_dir))

    (output_dir / "kubernetes.yaml").write_text(
        "kind: Deployment\n", encoding="utf-8"
    )

    assert _generated_files_sha256(str(output_dir)) != baseline


def test_generated_files_sha256_changes_when_readme_is_hand_edited(tmp_path):
    """README.md is a real compile-produced artifact (generate_readme,
    backend/generator/docker_generator.py) just like Dockerfile/
    docker-compose.yml/.env.example -- a hand-edit to it must be detected
    the identical way theirs already are.
    """

    from backend.compiler import _generated_files_sha256

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    baseline = _generated_files_sha256(str(output_dir))

    (output_dir / "README.md").write_text(
        "hand-edited\n", encoding="utf-8"
    )

    assert _generated_files_sha256(str(output_dir)) != baseline


def test_generated_files_sha256_ignores_files_outside_the_known_set(tmp_path):
    """A generic directory walk would pick up an unrelated file an
    operator (or a later POST /api/export-openapi/export-sdk) dropped
    into output_dir -- this hash must only ever reflect the specific
    files a compile itself actually produces.
    """

    from backend.compiler import _generated_files_sha256

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    baseline = _generated_files_sha256(str(output_dir))

    (output_dir / "openapi.json").write_text("{}", encoding="utf-8")

    assert _generated_files_sha256(str(output_dir)) == baseline


def test_generated_files_sha256_skips_a_missing_file_without_raising(tmp_path):

    from backend.compiler import _generated_files_sha256

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    baseline = _generated_files_sha256(str(output_dir))

    (output_dir / ".dockerignore").unlink()

    changed = _generated_files_sha256(str(output_dir))

    assert changed != baseline  # a missing file still changes the hash
    # ...but doesn't raise, confirmed simply by reaching this assertion.


def test_compiler_pipeline_optional_none_default_param_is_actually_optional(
    tmp_path
):
    """`def greet(name, title=None)` is an extremely common Python idiom
    for an optional parameter. Confirmed live before this fix: the
    generated endpoint 422'd on a request that omitted `title`, because
    the generated Pydantic field was marked required -- default=None was
    indistinguishable from "no default" once extracted.
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
                            "def greet(name: str, title: str = None) -> str:\n"
                            "    return ((title or '') + ' ' + name).strip()\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
resp = client.post(
    "/greet",
    json={{"name": "Ada"}},
    headers={{"X-API-Key": "notebook-to-api-dev-key"}},
)
assert resp.status_code == 200, resp.text
assert resp.json() == {{"result": "Ada"}}, resp.json()
print("OPTIONAL_NONE_DEFAULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OPTIONAL_NONE_DEFAULT_E2E_OK" in proc.stdout


def test_compiler_pipeline_api_key_auth_still_works_end_to_end(tmp_path):
    """Behavioral check that switching the API key comparison to
    hmac.compare_digest didn't change any of the three real outcomes:
    missing header and wrong key both 401, correct key succeeds.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
payload = {{"a": 1, "b": 2}}

no_header = client.post("/add", json=payload)
assert no_header.status_code == 401, no_header.text

wrong_key = client.post("/add", json=payload, headers={{"X-API-Key": "wrong"}})
assert wrong_key.status_code == 401, wrong_key.text

correct_key = client.post(
    "/add", json=payload, headers={{"X-API-Key": "notebook-to-api-dev-key"}}
)
assert correct_key.status_code == 200, correct_key.text
assert correct_key.json() == {{"result": 3}}, correct_key.json()

print("API_KEY_AUTH_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "API_KEY_AUTH_E2E_OK" in proc.stdout


def test_compiler_pipeline_generated_app_allows_cross_origin_requests_by_default(
    tmp_path,
):
    """Before this, the generated app had no CORS configuration at all --
    a browser-based frontend calling a deployed generated API (the whole
    point of generating one) was blocked by CORS with no way to fix it
    short of hand-editing the generated file.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

preflight = client.options(
    "/add",
    headers={{
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "POST",
    }},
)
assert preflight.status_code == 200, preflight.text
# CORSMiddleware emits a literal "*" (not a reflected Origin) when "*" is
# in allow_origins and allow_credentials is False.
assert preflight.headers["access-control-allow-origin"] == "*"

resp = client.post(
    "/add",
    json={{"a": 1, "b": 2}},
    headers={{
        "X-API-Key": "notebook-to-api-dev-key",
        "Origin": "https://example.com",
    }},
)
assert resp.status_code == 200, resp.text
assert resp.headers["access-control-allow-origin"] == "*"
assert "access-control-allow-credentials" not in resp.headers

print("CORS_DEFAULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CORS_DEFAULT_E2E_OK" in proc.stdout


def test_compiler_pipeline_generated_app_respects_configured_allowed_origins(
    tmp_path,
):
    """NOTEBOOK_API_ALLOWED_ORIGINS lets a real deployment lock the
    permissive "*" default down to a known frontend origin, matching the
    dashboard API's own NOTEBOOK_API_ALLOWED_ORIGINS convention (see
    allowed_origins() in backend/dashboard.py).
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import os
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

os.environ["NOTEBOOK_API_ALLOWED_ORIGINS"] = "https://allowed.example.com"

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

allowed = client.post(
    "/add",
    json={{"a": 1, "b": 2}},
    headers={{
        "X-API-Key": "notebook-to-api-dev-key",
        "Origin": "https://allowed.example.com",
    }},
)
assert allowed.status_code == 200, allowed.text
assert allowed.headers["access-control-allow-origin"] == "https://allowed.example.com"

disallowed = client.post(
    "/add",
    json={{"a": 1, "b": 2}},
    headers={{
        "X-API-Key": "notebook-to-api-dev-key",
        "Origin": "https://not-allowed.example.com",
    }},
)
assert "access-control-allow-origin" not in disallowed.headers

print("CORS_CONFIGURED_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CORS_CONFIGURED_E2E_OK" in proc.stdout


def test_compiler_pipeline_generated_app_stamps_security_headers_on_every_response(
    tmp_path,
):
    """Confirmed exploitable before this fix: a real compiled app set
    none of X-Content-Type-Options/X-Frame-Options/Referrer-Policy on any
    response -- not the ones a client actually wants (2xx), and not the
    ones where a browser's own default MIME-sniffing/framing behavior
    matters just as much: a 401 (invalid key), a 413 (oversized body,
    from MaxRequestBodySizeMiddleware -- registered *before* this new
    middleware, so it must still see a response this middleware wraps),
    and even /docs itself.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import os
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

os.environ["NOTEBOOK_API_MAX_REQUEST_BYTES"] = "50"

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}


def assert_hardened(resp):
    assert resp.headers["X-Content-Type-Options"] == "nosniff", resp.headers
    assert resp.headers["X-Frame-Options"] == "DENY", resp.headers
    assert resp.headers["Referrer-Policy"] == "no-referrer", resp.headers


success = client.post("/add", json={{"a": 1, "b": 2}}, headers=headers)
assert success.status_code == 200, success.text
assert_hardened(success)

unauthorized = client.post("/add", json={{"a": 1, "b": 2}}, headers={{"X-API-Key": "wrong"}})
assert unauthorized.status_code == 401, unauthorized.text
assert_hardened(unauthorized)

oversized = client.post(
    "/add", json={{"a": 1, "b": 2, "padding": "x" * 200}}, headers=headers,
)
assert oversized.status_code == 413, oversized.text
assert_hardened(oversized)

docs = client.get("/docs")
assert docs.status_code == 200, docs.text
assert_hardened(docs)

print("SECURITY_HEADERS_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SECURITY_HEADERS_E2E_OK" in proc.stdout


def test_compiler_pipeline_generated_app_stamps_x_process_time_ms(tmp_path):
    """Confirmed exploitable before this fix: a real compiled app gave an
    operator no way to see per-request latency at all -- no header, no
    endpoint -- short of instrumenting it externally.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
resp = client.get("/health")
assert resp.status_code == 200, resp.text
elapsed_ms = float(resp.headers["X-Process-Time-Ms"])
assert elapsed_ms >= 0, resp.headers

auto_id_resp = client.get("/health")
assert auto_id_resp.headers["X-Request-ID"], auto_id_resp.headers

echoed_resp = client.get("/health", headers={{"X-Request-ID": "caller-supplied-xyz"}})
assert echoed_resp.headers["X-Request-ID"] == "caller-supplied-xyz", echoed_resp.headers

print("PROCESS_TIME_HEADER_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PROCESS_TIME_HEADER_E2E_OK" in proc.stdout


def test_compiler_pipeline_generated_app_gzip_compresses_a_large_response(tmp_path):
    """Confirmed exploitable before this fix: a real compiled app never
    compressed any response, no matter how large -- a caller sending
    Accept-Encoding: gzip against an endpoint returning a large payload
    still always got the full uncompressed body back, a real bandwidth
    cost this app had no way to avoid on its own. Only kicks in when the
    caller's own Accept-Encoding actually asks for it (Accept-Encoding:
    identity must still get an uncompressed response), and the
    decompressed content must still be exactly right either way.
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
                            "def get_large_payload() -> str:\n"
                            "    return 'x' * 2000\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import os
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

compressed = client.post(
    "/get_large_payload", json={{}}, headers={{**headers, "Accept-Encoding": "gzip"}},
)
assert compressed.status_code == 200, compressed.text
assert compressed.headers.get("content-encoding") == "gzip", compressed.headers
assert compressed.json() == {{"result": "x" * 2000}}, compressed.json()

uncompressed = client.post(
    "/get_large_payload", json={{}}, headers={{**headers, "Accept-Encoding": "identity"}},
)
assert uncompressed.status_code == 200, uncompressed.text
assert "content-encoding" not in uncompressed.headers, uncompressed.headers
assert uncompressed.json() == {{"result": "x" * 2000}}, uncompressed.json()

print("GZIP_COMPRESSION_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GZIP_COMPRESSION_E2E_OK" in proc.stdout


def test_compiler_pipeline_generated_app_rejects_an_oversized_request_body(tmp_path):
    """Before this, every endpoint on the generated app accepted a JSON
    request body of any size -- unlike this tool's own dashboard
    /api/upload, which has always capped uploads at MAX_UPLOAD_BYTES (see
    routes/upload.py) for exactly this reason. Configurable via
    NOTEBOOK_API_MAX_REQUEST_BYTES, matching the NOTEBOOK_API_* env-var
    convention this generated app's other limits already follow.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import os
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

os.environ["NOTEBOOK_API_MAX_REQUEST_BYTES"] = "50"

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

small = client.post("/add", json={{"a": 1, "b": 2}}, headers=headers)
assert small.status_code == 200, small.text
assert small.json() == {{"result": 3}}, small.json()

oversized = client.post(
    "/add",
    json={{"a": 1, "b": 2, "padding": "x" * 200}},
    headers=headers,
)
assert oversized.status_code == 413, oversized.text
assert "50 bytes" in oversized.text, oversized.text

print("MAX_REQUEST_BODY_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MAX_REQUEST_BODY_E2E_OK" in proc.stdout


def test_compiler_pipeline_generated_app_default_request_body_limit_allows_normal_requests(
    tmp_path,
):
    """The default (10MB, matching MAX_UPLOAD_BYTES's own default) must
    not reject an ordinary, unconfigured request -- this middleware is
    meant to catch genuinely oversized bodies, not interfere with normal
    usage when NOTEBOOK_API_MAX_REQUEST_BYTES is left unset.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
resp = client.post(
    "/add",
    json={{"a": 1, "b": 2}},
    headers={{"X-API-Key": "notebook-to-api-dev-key"}},
)
assert resp.status_code == 200, resp.text
assert resp.json() == {{"result": 3}}, resp.json()

print("MAX_REQUEST_BODY_DEFAULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MAX_REQUEST_BODY_DEFAULT_E2E_OK" in proc.stdout


def test_compiler_pipeline_supports_zero_downtime_api_key_rotation(tmp_path):
    """Confirmed exploitable before this fix: /auth/info advertised
    'key_rotation': True, but the generated app only ever read a single
    key from NOTEBOOK_API_KEY -- there was no way to accept an old and a
    new key at once, so "rotating" the key meant a hard cutover where
    every client using the old key started getting 401s the moment the
    env var changed. NOTEBOOK_API_KEY is now a comma-separated list, so
    both an old and a new key can be valid at once during a rotation
    window.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import os
import sys

os.environ["NOTEBOOK_API_KEY"] = "old-key, new-key"

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
payload = {{"a": 1, "b": 2}}

old_key = client.post("/add", json=payload, headers={{"X-API-Key": "old-key"}})
assert old_key.status_code == 200, old_key.text

new_key = client.post("/add", json=payload, headers={{"X-API-Key": "new-key"}})
assert new_key.status_code == 200, new_key.text

unrelated_key = client.post("/add", json=payload, headers={{"X-API-Key": "someone-elses-key"}})
assert unrelated_key.status_code == 401, unrelated_key.text

info = client.get("/auth/info")
assert info.json()["configured_keys"] == 2, info.json()

print("API_KEY_ROTATION_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "API_KEY_ROTATION_E2E_OK" in proc.stdout


def test_compiler_pipeline_rate_limiting_disabled_by_default(tmp_path):
    """NOTEBOOK_API_RATE_LIMIT_PER_MINUTE defaults to "0", which must
    mean unlimited (the previous, pre-rate-limiting behavior) rather than
    "zero requests allowed" -- a request volume well past any real
    per-minute limit still succeeds every time, and /auth/info reports
    the feature as off.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}
payload = {{"a": 1, "b": 2}}

for _ in range(25):
    resp = client.post("/add", json=payload, headers=headers)
    assert resp.status_code == 200, resp.text

info = client.get("/auth/info").json()
assert info["rate_limiting"] is False, info
assert info["rate_limit_per_minute"] is None, info

print("RATE_LIMIT_DISABLED_BY_DEFAULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RATE_LIMIT_DISABLED_BY_DEFAULT_E2E_OK" in proc.stdout


def test_compiler_pipeline_rate_limit_returns_429_once_exceeded(tmp_path):
    """NOTEBOOK_API_RATE_LIMIT_PER_MINUTE, once set, caps how many
    requests a single API key may make per rolling 60s window -- the
    (N+1)th request within the window must be rejected with 429 and a
    Retry-After header, while a *different* key's own window is
    unaffected (tracked independently per key, not globally).
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import os
import sys

os.environ["NOTEBOOK_API_KEY"] = "key-a, key-b"
os.environ["NOTEBOOK_API_RATE_LIMIT_PER_MINUTE"] = "2"

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
payload = {{"a": 1, "b": 2}}
headers_a = {{"X-API-Key": "key-a"}}
headers_b = {{"X-API-Key": "key-b"}}

first = client.post("/add", json=payload, headers=headers_a)
assert first.status_code == 200, first.text

second = client.post("/add", json=payload, headers=headers_a)
assert second.status_code == 200, second.text

third = client.post("/add", json=payload, headers=headers_a)
assert third.status_code == 429, third.text
assert "Retry-After" in third.headers, third.headers
assert int(third.headers["Retry-After"]) >= 1, third.headers

other_key = client.post("/add", json=payload, headers=headers_b)
assert other_key.status_code == 200, other_key.text

info = client.get("/auth/info").json()
assert info["rate_limiting"] is True, info
assert info["rate_limit_per_minute"] == 2, info

print("RATE_LIMIT_429_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RATE_LIMIT_429_E2E_OK" in proc.stdout


def test_compiler_pipeline_rate_limit_sends_x_ratelimit_headers(tmp_path):
    """Confirmed exploitable before this fix: a rate-limited request only
    ever got a Retry-After header, and only once it had already been
    rejected with 429 -- there was no X-RateLimit-Limit/-Remaining/-Reset
    on a *successful* response (the standard GitHub/Stripe-style
    contract), so a well-behaved caller had no way to see it was about to
    be throttled and back off on its own; the only signal was a 429 it
    had already triggered. Each header must also appear on the 429
    response itself, with Remaining pinned to 0 there.
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
                            "    return a + b\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import os
import sys

os.environ["NOTEBOOK_API_KEY"] = "key-a"
os.environ["NOTEBOOK_API_RATE_LIMIT_PER_MINUTE"] = "2"

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
payload = {{"a": 1, "b": 2}}
headers = {{"X-API-Key": "key-a"}}

first = client.post("/add", json=payload, headers=headers)
assert first.status_code == 200, first.text
assert first.headers["X-RateLimit-Limit"] == "2", first.headers
assert first.headers["X-RateLimit-Remaining"] == "1", first.headers
assert int(first.headers["X-RateLimit-Reset"]) > 0, first.headers

second = client.post("/add", json=payload, headers=headers)
assert second.status_code == 200, second.text
assert second.headers["X-RateLimit-Remaining"] == "0", second.headers

third = client.post("/add", json=payload, headers=headers)
assert third.status_code == 429, third.text
assert third.headers["X-RateLimit-Limit"] == "2", third.headers
assert third.headers["X-RateLimit-Remaining"] == "0", third.headers
assert int(third.headers["X-RateLimit-Reset"]) > 0, third.headers

# An unauthenticated/unlimited endpoint (rate limiting only ever applies
# once a request has already authenticated via verify_api_key) gets none
# of these -- confirms they're not stamped globally on every response.
health = client.get("/health")
assert "X-RateLimit-Limit" not in health.headers, health.headers

print("RATE_LIMIT_HEADERS_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RATE_LIMIT_HEADERS_E2E_OK" in proc.stdout


def test_compiler_pipeline_typing_generic_and_enum_params_work_end_to_end(tmp_path):
    """Confirmed exploitable before this fix: a parameter typed with a
    typing-module generic (List[float], Optional[str], Dict[str, Any]) or
    a notebook-defined Enum produced a generated Pydantic field
    referencing a name nothing in the generated app imports. The class
    definition itself didn't fail (deferred annotation evaluation), but
    the very first real use -- building the schema for /docs, /openapi.json,
    or the first request -- raised PydanticUserError/NameError.
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
                            "from typing import List, Optional, Dict, Any\n"
                            "from enum import Enum\n\n"
                            "class Priority(Enum):\n"
                            "    LOW = 'low'\n"
                            "    HIGH = 'high'\n\n"
                            "def summarize(\n"
                            "    scores: List[float],\n"
                            "    label: Optional[str] = None,\n"
                            "    meta: Dict[str, Any] = None,\n"
                            "    priority: Optional[Priority] = None,\n"
                            ") -> str:\n"
                            "    return label or 'none'\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app, SummarizeRequest
from fastapi.testclient import TestClient

# Building the schema is exactly what raised PydanticUserError before this fix.
schema = SummarizeRequest.model_json_schema()
assert schema["properties"]["scores"]["type"] == "array", schema

client = TestClient(app)
resp = client.post(
    "/summarize",
    json={{"scores": [1.0, 2.0], "label": "x", "meta": {{"a": 1}}}},
    headers={{"X-API-Key": "notebook-to-api-dev-key"}},
)
assert resp.status_code == 200, resp.text
assert resp.json() == {{"result": "x"}}, resp.json()
print("TYPING_GENERIC_ENUM_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TYPING_GENERIC_ENUM_E2E_OK" in proc.stdout


def test_compiler_pipeline_enum_default_param_is_usable_end_to_end(tmp_path):
    """Confirmed exploitable before this fix: a parameter defaulting to a
    notebook-defined Enum member (e.g. `priority: Priority = Priority.HIGH`)
    got repr()'d into the generated Pydantic model exactly like a literal
    default, silently turning it into the *string* "Priority.HIGH" instead
    of the actual enum member. A caller omitting that field to take its
    default then passed the raw string straight into the notebook's own
    function, which crashed with an AttributeError the moment it tried to
    use it as an actual Priority (e.g. `.value`).
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
                            "from enum import Enum\n\n"
                            "class Priority(Enum):\n"
                            "    LOW = 'low'\n"
                            "    HIGH = 'high'\n\n"
                            "def set_priority(priority: Priority = Priority.HIGH) -> str:\n"
                            "    return priority.value\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

# Omitting "priority" entirely must fall back to the real Priority.HIGH
# enum member, not the string "Priority.HIGH".
default_resp = client.post("/set_priority", json={{}}, headers=headers)
assert default_resp.status_code == 200, default_resp.text
assert default_resp.json() == {{"result": "high"}}, default_resp.json()

explicit_resp = client.post(
    "/set_priority", json={{"priority": "low"}}, headers=headers
)
assert explicit_resp.status_code == 200, explicit_resp.text
assert explicit_resp.json() == {{"result": "low"}}, explicit_resp.json()

print("ENUM_DEFAULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ENUM_DEFAULT_E2E_OK" in proc.stdout


def test_compiler_pipeline_background_tasks_are_evicted_after_ttl_expires(tmp_path):
    """Confirmed exploitable before this fix: the generated TASKS registry
    never evicted anything on its own -- a long-running deployment
    handling steady background-task traffic accumulated one entry per
    call forever. With TASK_TTL_SECONDS forced to 0 (via the
    NOTEBOOK_API_TASK_TTL_SECONDS env var the generated app already
    reads), a task created before a second task must be gone by the time
    the second one is created, since eviction runs opportunistically on
    every new task's creation.
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
                            "def process_data(x: int) -> int:\n"
                            "    return x\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import os
import sys
import time

os.environ["NOTEBOOK_API_TASK_TTL_SECONDS"] = "0"

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

first = client.post("/process_data", json={{"x": 1}}, headers=headers)
assert first.status_code == 200, first.text
first_task_id = first.json()["task_id"]

# Any nonzero elapsed time exceeds a TTL of 0, so the first task is
# eligible for eviction by the time the second one is created.
time.sleep(0.01)

second = client.post("/process_data", json={{"x": 2}}, headers=headers)
assert second.status_code == 200, second.text

lookup = client.get(f"/tasks/{{first_task_id}}", headers=headers)
assert lookup.status_code == 404, lookup.text
assert first_task_id in lookup.json()["detail"], lookup.json()

print("TASK_TTL_EVICTION_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TASK_TTL_EVICTION_E2E_OK" in proc.stdout


def test_compiler_pipeline_unknown_task_id_returns_404(tmp_path):
    """Confirmed exploitable before this fix: GET/DELETE /tasks/{task_id}
    for a task_id that was never created (or has since been evicted --
    see test_compiler_pipeline_background_tasks_are_evicted_after_ttl_expires
    above, or simply deleted by another caller) returned HTTP 200 with a
    body of {"error": "Task not found"}, instead of a 404. That's not
    just a wrong status code for its own sake: this is the exact endpoint
    the generated Python/TypeScript SDK's get_task/wait_for_task poll
    (backend/exporters/sdk_generator.py), and both rely on
    response.raise_for_status() to signal failure -- which never fires
    for a 200. wait_for_task additionally only checks whether
    task.get('status') != 'processing' to decide a task is "finished",
    so a task_id that no longer exists reads as status=None, which is
    trivially != 'processing' -- meaning wait_for_task returned
    {"error": "Task not found"} straight to the caller as if it were the
    task's actual, successful result, with no exception raised at all.
    Now a real 404 lets raise_for_status() do its job.
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
                            "def process_data(x: int) -> int:\n"
                            "    return x\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

lookup = client.get("/tasks/does-not-exist", headers=headers)
assert lookup.status_code == 404, lookup.text
assert "does-not-exist" in lookup.json()["detail"], lookup.json()

deletion = client.delete("/tasks/does-not-exist", headers=headers)
assert deletion.status_code == 404, deletion.text
assert "does-not-exist" in deletion.json()["detail"], deletion.json()

print("UNKNOWN_TASK_ID_404_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "UNKNOWN_TASK_ID_404_E2E_OK" in proc.stdout


def test_compiler_pipeline_background_task_with_unserializable_result_is_reported_as_failed(
    tmp_path,
):
    """Confirmed exploitable before this fix: a background function
    returning something FastAPI's own response serialization can't
    encode (e.g. a complex number -- Python builtin, no extra dependency
    needed to demonstrate this; a raw numpy array or pandas DataFrame is
    the more common real-world case for "process_data", an entirely
    ordinary thing for the 'process'/'train'/'generate'/'embed' keywords
    that route a function to a background task in the first place, but
    requires numpy as a test dependency this project doesn't otherwise
    have) marked the task "completed" with that unserializable result
    stored as-is. GET /tasks/{task_id} then crashed with an unhandled
    500 the moment FastAPI tried to serialize the response -- and so did
    GET /tasks entirely, for *every* task in the registry, not just the
    offending one, since it returns them all in a single response.
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
                            "def process_data(x: int) -> complex:\n"
                            "    return complex(x, x * 2)\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
import time

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

submitted = client.post("/process_data", json={{"x": 5}}, headers=headers)
assert submitted.status_code == 200, submitted.text
task_id = submitted.json()["task_id"]

deadline = time.time() + 5
while True:
    lookup = client.get(f"/tasks/{{task_id}}", headers=headers)
    assert lookup.status_code == 200, lookup.text
    if lookup.json().get("status") != "processing":
        break
    assert time.time() < deadline, "task never left processing"
    time.sleep(0.01)

task = lookup.json()
assert task["status"] == "failed", task
assert "error" in task, task
assert "result" not in task, task

# GET /tasks must still succeed too -- not crash for *every* task in the
# registry just because one of them has an unserializable result.
listing = client.get("/tasks", headers=headers)
assert listing.status_code == 200, listing.text
assert listing.json()["tasks"][task_id]["status"] == "failed"

print("UNSERIALIZABLE_RESULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "UNSERIALIZABLE_RESULT_E2E_OK" in proc.stdout


def test_compiler_pipeline_background_task_result_is_run_through_jsonable_encoder(
    tmp_path,
):
    """A JSON-safe result (unlike the numpy case above) must still be
    delivered correctly -- and jsonable_encoder should actually convert a
    type json.dumps alone can't handle natively (a datetime) into a
    JSON-safe value, rather than merely happening not to break on it.
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
                            "import datetime\n\n"
                            "def generate_report(year: int) -> dict:\n"
                            "    return {\n"
                            "        'year': year,\n"
                            "        'generated_on': datetime.date(2024, 1, 1),\n"
                            "    }\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import sys
import time

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

submitted = client.post("/generate_report", json={{"year": 2024}}, headers=headers)
assert submitted.status_code == 200, submitted.text
task_id = submitted.json()["task_id"]

deadline = time.time() + 5
while True:
    lookup = client.get(f"/tasks/{{task_id}}", headers=headers)
    assert lookup.status_code == 200, lookup.text
    if lookup.json().get("status") != "processing":
        break
    assert time.time() < deadline, "task never left processing"
    time.sleep(0.01)

task = lookup.json()
assert task["status"] == "completed", task
assert task["result"] == {{"year": 2024, "generated_on": "2024-01-01"}}, task

print("JSONABLE_ENCODER_RESULT_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "JSONABLE_ENCODER_RESULT_E2E_OK" in proc.stdout


def _free_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_serving(base_url, deadline):
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1):
                return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.05)
    raise TimeoutError(f"Server at {base_url} never became ready")


def test_compiler_pipeline_background_task_does_not_block_the_event_loop(tmp_path):
    """Confirmed exploitable before this fix: _run_background_task called
    a synchronous notebook function directly, which ran its entire body
    inline on this app's single asyncio event loop -- the exact same loop
    every other request, including a completely unrelated GET /health,
    is served from. Confirmed against a real (non-TestClient) uvicorn
    server: a background task doing nothing but time.sleep(1) froze a
    concurrent GET /health for the full second, the opposite of what
    "background" is supposed to mean -- and especially damaging since the
    "train"/"process"/"generate"/"embed"/"scrape" keywords that route a
    function to a background task in the first place are routinely slow,
    CPU-bound work, not quick one-liners.

    Runs a real uvicorn subprocess (TestClient's in-process request
    handling doesn't reliably reproduce this class of event-loop-blocking
    bug) and measures how long a concurrent GET /health actually takes
    while a slow background task is in flight.
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
                            "import time\n\n"
                            "def train_slow(x: int) -> int:\n"
                            "    time.sleep(1)\n"
                            "    return x\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from backend.compiler import compile_notebook

    compile_notebook(str(notebook_path), str(workdir / "generated"))

    port = _free_tcp_port()
    base_url = f"http://127.0.0.1:{port}"

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{workdir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "generated.app:app",
            "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=str(workdir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_until_serving(base_url, time.time() + 20)

        headers = {"X-API-Key": "notebook-to-api-dev-key", "Content-Type": "application/json"}
        submit_req = urllib.request.Request(
            f"{base_url}/train_slow", data=b'{"x": 5}', headers=headers, method="POST"
        )
        with urllib.request.urlopen(submit_req, timeout=5) as resp:
            task_id = json.loads(resp.read())["task_id"]

        # The background task is now "processing" (it sleeps for a full
        # second). A concurrent, completely unrelated request must not be
        # stuck waiting behind it -- generously bounded well under the
        # task's own 1s sleep to leave room for scheduling jitter, while
        # still being a strong signal against the ~1s this took before
        # this fix.
        start = time.monotonic()
        with urllib.request.urlopen(f"{base_url}/health", timeout=5):
            pass
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"GET /health took {elapsed:.3f}s while a background task was "
            "running -- the event loop is still being blocked"
        )

        # The task itself must still actually complete with the right
        # result, not just "not block anything else".
        deadline = time.time() + 10
        while True:
            with urllib.request.urlopen(
                urllib.request.Request(f"{base_url}/tasks/{task_id}", headers=headers),
                timeout=5,
            ) as resp:
                task = json.loads(resp.read())
            if task["status"] != "processing":
                break
            assert time.time() < deadline, "task never left processing"
            time.sleep(0.05)

        assert task == {"status": "completed", "result": 5, "created_at": task["created_at"]}

    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def test_compiler_pipeline_background_endpoint_rejects_tasks_past_the_configured_limit(
    tmp_path,
):
    """Confirmed exploitable before this fix: _evict_expired_tasks only
    bounds TASKS' *long-term* growth (nothing older than
    TASK_TTL_SECONDS survives) -- a burst of requests arriving faster
    than that TTL still grew TASKS without any limit in the meantime.
    With NOTEBOOK_API_MAX_TASKS forced to 2, a third concurrent
    background request must be refused with 503 instead of silently
    accepted.
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
                            "def train_model(epochs: int) -> str:\n"
                            "    return 'done'\n"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
import os
import sys

os.environ["NOTEBOOK_API_MAX_TASKS"] = "2"

sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(workdir)!r})

from backend.compiler import compile_notebook

compile_notebook({str(notebook_path)!r}, "generated")

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
headers = {{"X-API-Key": "notebook-to-api-dev-key"}}

first = client.post("/train_model", json={{"epochs": 1}}, headers=headers)
assert first.status_code == 200, first.text

second = client.post("/train_model", json={{"epochs": 1}}, headers=headers)
assert second.status_code == 200, second.text

third = client.post("/train_model", json={{"epochs": 1}}, headers=headers)
assert third.status_code == 503, third.text
assert "Too many pending background tasks" in third.text, third.text

print("MAX_PENDING_TASKS_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MAX_PENDING_TASKS_E2E_OK" in proc.stdout


def test_compiler_pipeline_tasks_endpoints_reject_unauthenticated_requests(tmp_path):
    """Confirmed exploitable before this fix: GET /tasks and GET
    /tasks/{task_id} returned stored function call inputs/outputs with no
    API key at all, and the DELETE/POST tasks endpoints let anyone wipe
    task state -- every other endpoint in the generated app (including
    /auth/validate) required Depends(verify_api_key), but the entire
    /tasks family was left open.
    """

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def process_data(x: int) -> int:\n"
            "    return x\n"
        )
    )

    notebook_path = tmp_path / "tasksauth.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"

    compile_notebook(str(notebook_path), str(output_dir))

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(output_dir.parent)!r})

from generated.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

assert client.get("/tasks").status_code == 401
assert client.get("/tasks/whatever").status_code == 401
assert client.delete("/tasks/completed").status_code == 401
assert client.delete("/tasks/failed").status_code == 401
assert client.post("/tasks/cleanup").status_code == 401
assert client.post("/tasks/reset").status_code == 401
assert client.delete("/tasks/whatever").status_code == 401

headers = {{"X-API-Key": "notebook-to-api-dev-key"}}
assert client.get("/tasks", headers=headers).status_code == 200
assert client.post("/tasks/reset", headers=headers).status_code == 200

print("TASKS_AUTH_E2E_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(output_dir.parent),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TASKS_AUTH_E2E_OK" in proc.stdout

def test_extract_explicit_requirements_finds_a_directive_in_a_cell():

    code_cells = [
        "# notebook-to-api: requires opencv-python-headless==4.9.0.80\n"
        "import pandas\n"
    ]

    assert _extract_explicit_requirements(code_cells) == [
        "opencv-python-headless==4.9.0.80"
    ]


def test_extract_explicit_requirements_finds_a_directive_indented_inside_a_function():

    code_cells = [
        "def f() -> int:\n"
        "    # notebook-to-api: requires extra-runtime-only-pkg==2.0\n"
        "    return 1\n"
    ]

    assert _extract_explicit_requirements(code_cells) == [
        "extra-runtime-only-pkg==2.0"
    ]


def test_extract_explicit_requirements_supports_a_vcs_or_extras_spec():

    code_cells = [
        "# notebook-to-api: requires my-private-pkg @ git+https://example.com/pkg.git\n"
        "# notebook-to-api: requires somepkg[extra]==1.2.3\n"
    ]

    specs = _extract_explicit_requirements(code_cells)

    assert "my-private-pkg @ git+https://example.com/pkg.git" in specs
    assert "somepkg[extra]==1.2.3" in specs


def test_extract_explicit_requirements_deduplicates_exact_matches_across_cells():

    code_cells = [
        "# notebook-to-api: requires somepkg==1.0\n",
        "# notebook-to-api: requires somepkg==1.0\n",
    ]

    assert _extract_explicit_requirements(code_cells) == ["somepkg==1.0"]


def test_extract_explicit_requirements_ignores_an_unrelated_comment():

    code_cells = [
        "# this notebook-to-api project requires careful review\n"
        "import pandas\n"
    ]

    assert _extract_explicit_requirements(code_cells) == []


def test_extract_explicit_requirements_returns_an_empty_list_with_no_directives():

    code_cells = ["import pandas\n\ndef f() -> int:\n    return 1\n"]

    assert _extract_explicit_requirements(code_cells) == []


def test_extract_explicit_apt_packages_finds_a_directive_in_a_cell():

    code_cells = [
        "# notebook-to-api: apt-requires libpq-dev\n"
        "import psycopg2\n"
    ]

    assert _extract_explicit_apt_packages(code_cells) == ["libpq-dev"]


def test_extract_explicit_apt_packages_finds_a_directive_indented_inside_a_function():

    code_cells = [
        "def f() -> int:\n"
        "    # notebook-to-api: apt-requires libgl1\n"
        "    return 1\n"
    ]

    assert _extract_explicit_apt_packages(code_cells) == ["libgl1"]


def test_extract_explicit_apt_packages_supports_a_version_pin():

    code_cells = [
        "# notebook-to-api: apt-requires libpq-dev=13.11-0+deb12u1\n"
    ]

    assert _extract_explicit_apt_packages(code_cells) == [
        "libpq-dev=13.11-0+deb12u1"
    ]


def test_extract_explicit_apt_packages_deduplicates_exact_matches_across_cells():

    code_cells = [
        "# notebook-to-api: apt-requires libpq-dev\n",
        "# notebook-to-api: apt-requires libpq-dev\n",
    ]

    assert _extract_explicit_apt_packages(code_cells) == ["libpq-dev"]


def test_extract_explicit_apt_packages_preserves_first_seen_order_across_cells():

    code_cells = [
        "# notebook-to-api: apt-requires libpq-dev\n",
        "# notebook-to-api: apt-requires libgl1\n",
    ]

    assert _extract_explicit_apt_packages(code_cells) == ["libpq-dev", "libgl1"]


def test_extract_explicit_apt_packages_allows_two_different_pins_for_the_same_package():
    """Unlike _extract_explicit_requirements' own "requires" directive,
    apt-get itself doesn't hard-fail on two mentions of the same package
    -- it simply installs whichever comes last -- so there is no
    equivalent conflict to raise for here.
    """

    code_cells = [
        "# notebook-to-api: apt-requires libpq-dev=1.0\n"
        "# notebook-to-api: apt-requires libpq-dev=2.0\n"
    ]

    assert _extract_explicit_apt_packages(code_cells) == [
        "libpq-dev=1.0", "libpq-dev=2.0",
    ]


def test_extract_explicit_apt_packages_ignores_an_unrelated_comment():

    code_cells = [
        "# this notebook-to-api project requires careful review\n"
        "import pandas\n"
    ]

    assert _extract_explicit_apt_packages(code_cells) == []


def test_extract_explicit_apt_packages_does_not_match_the_requires_directive():

    code_cells = [
        "# notebook-to-api: requires psycopg2==2.9.9\n"
    ]

    assert _extract_explicit_apt_packages(code_cells) == []


def test_extract_explicit_apt_packages_returns_an_empty_list_with_no_directives():

    code_cells = ["import pandas\n\ndef f() -> int:\n    return 1\n"]

    assert _extract_explicit_apt_packages(code_cells) == []


def test_extract_explicit_requirements_raises_for_conflicting_specs_in_the_same_cell():
    """Two different specs for the same package both surviving into
    requirements.txt is the identical "Double requirement given" pip
    failure resolve_requirements' own docstring already describes fixing
    for the auto-detected-vs-explicit case (see
    test_resolve_requirements_drops_the_auto_detected_line_that_conflicts_with_an_explicit_one
    below) -- just for two explicit directives naming the same package
    instead, e.g. left behind after pinning a different version while
    iterating on a notebook.
    """

    code_cells = [
        "# notebook-to-api: requires numpy==1.24.0\n"
        "# notebook-to-api: requires numpy==1.26.0\n"
    ]

    with pytest.raises(ValueError, match="numpy"):
        _extract_explicit_requirements(code_cells)


def test_extract_explicit_requirements_raises_for_conflicting_specs_across_cells():

    code_cells = [
        "# notebook-to-api: requires numpy==1.24.0\nimport pandas\n",
        "def f() -> int:\n    return 1\n",
        "# notebook-to-api: requires numpy==1.26.0\n",
    ]

    with pytest.raises(ValueError, match="numpy"):
        _extract_explicit_requirements(code_cells)


def test_extract_explicit_requirements_conflict_detection_is_case_insensitive():
    """PyPI distribution names are themselves case-insensitive -- pip
    normalizes "NumPy"/"numpy"/"nUmPy" to the identical project -- so a
    directive spelled differently than another must still be recognized
    as naming the same package.
    """

    code_cells = [
        "# notebook-to-api: requires NumPy==1.24.0\n"
        "# notebook-to-api: requires numpy==1.26.0\n"
    ]

    with pytest.raises(ValueError, match="(?i)numpy"):
        _extract_explicit_requirements(code_cells)


def test_extract_explicit_requirements_error_names_both_conflicting_specs():

    code_cells = [
        "# notebook-to-api: requires numpy==1.24.0\n"
        "# notebook-to-api: requires numpy==1.26.0\n"
    ]

    with pytest.raises(ValueError) as exc_info:
        _extract_explicit_requirements(code_cells)

    message = str(exc_info.value)
    assert "numpy==1.24.0" in message
    assert "numpy==1.26.0" in message


def test_extract_explicit_requirements_allows_identical_specs_repeated():
    """An exact-duplicate line is not a conflict -- exact-duplicate
    removal (test_extract_explicit_requirements_deduplicates_exact_matches_across_cells
    above) already handles this case cleanly, and must keep doing so
    with conflict detection layered on top of it.
    """

    code_cells = [
        "# notebook-to-api: requires somepkg==1.0\n",
        "# notebook-to-api: requires somepkg==1.0\n",
    ]

    assert _extract_explicit_requirements(code_cells) == ["somepkg==1.0"]


def test_extract_explicit_requirements_allows_unrelated_packages():

    code_cells = [
        "# notebook-to-api: requires numpy==1.24.0\n"
        "# notebook-to-api: requires pandas==2.0.0\n"
    ]

    assert _extract_explicit_requirements(code_cells) == [
        "numpy==1.24.0", "pandas==2.0.0",
    ]


def test_extract_explicit_requirements_does_not_false_flag_two_unrelated_vcs_specs():
    """_explicit_requirement_package_name has no reliable way to name a
    bare VCS/URL spec with no "name @ " prefix (see its own docstring) --
    two such specs must never be treated as conflicting just because
    neither can be named, the same "no reliable name, so no comparison"
    reasoning that function's own None return already establishes for
    resolve_requirements' identical auto-detected-conflict check.
    """

    code_cells = [
        "# notebook-to-api: requires git+https://example.com/one.git\n"
        "# notebook-to-api: requires git+https://example.com/two.git\n"
    ]

    assert _extract_explicit_requirements(code_cells) == [
        "git+https://example.com/one.git",
        "git+https://example.com/two.git",
    ]


def test_explicit_requirement_package_name_returns_none_for_a_bare_vcs_url():
    """Confirmed exploitable before this: "[A-Za-z0-9._-]*" alone had no
    reason to stop before "git+https://...write.git"'s own "+", so it
    silently captured "git" as though that were the actual package name
    -- indistinguishable from a real, unrelated package literally named
    "git".
    """

    assert _explicit_requirement_package_name(
        "git+https://example.com/pkg.git"
    ) is None


def test_explicit_requirement_package_name_handles_every_pep_508_continuation():

    assert _explicit_requirement_package_name("requests") == "requests"
    assert _explicit_requirement_package_name("somepkg[extra]==1.2.3") == "somepkg"
    assert _explicit_requirement_package_name("somepkg>=1.0") == "somepkg"
    assert _explicit_requirement_package_name("somepkg~=1.0") == "somepkg"
    assert _explicit_requirement_package_name("somepkg!=1.0") == "somepkg"
    assert _explicit_requirement_package_name(
        'somepkg;python_version<"3.11"'
    ) == "somepkg"
    assert _explicit_requirement_package_name(
        "my-private-pkg @ git+https://example.com/pkg.git"
    ) == "my-private-pkg"


def test_extract_excluded_imports_finds_a_directive_in_a_cell():

    code_cells = [
        "# notebook-to-api: exclude pytest\n"
        "import pandas\n"
    ]

    assert _extract_excluded_imports(code_cells) == {"pytest"}


def test_extract_excluded_imports_finds_a_directive_indented_inside_a_function():

    code_cells = [
        "def f() -> int:\n"
        "    # notebook-to-api: exclude debug_only_pkg\n"
        "    return 1\n"
    ]

    assert _extract_excluded_imports(code_cells) == {"debug_only_pkg"}


def test_extract_excluded_imports_collects_several_across_cells():

    code_cells = [
        "# notebook-to-api: exclude pytest\n",
        "# notebook-to-api: exclude ipdb\n",
    ]

    assert _extract_excluded_imports(code_cells) == {"pytest", "ipdb"}


def test_extract_excluded_imports_ignores_an_unrelated_comment():

    code_cells = [
        "# this notebook-to-api project should exclude nothing\n"
        "import pandas\n"
    ]

    assert _extract_excluded_imports(code_cells) == set()


def test_extract_excluded_imports_returns_an_empty_set_with_no_directives():

    code_cells = ["import pandas\n\ndef f() -> int:\n    return 1\n"]

    assert _extract_excluded_imports(code_cells) == set()


def test_extract_third_party_imports_omits_an_excluded_import():

    code_cells = [
        "# notebook-to-api: exclude pytest\n"
        "import pytest\n"
        "import pandas\n"
    ]

    imports = extract_third_party_imports(code_cells)

    assert "pandas" in imports
    assert "pytest" not in imports


def test_extract_third_party_imports_keeps_a_non_excluded_import():

    code_cells = ["import pandas\n"]

    assert extract_third_party_imports(code_cells) == ["pandas"]


def test_compile_notebook_excludes_a_directive_named_import_from_requirements_txt(
    tmp_path
):

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "# notebook-to-api: exclude nbformat\n"
            "import nbformat\n\n"
            "def f() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")

    assert "nbformat" not in requirements
    assert any(
        line.startswith("fastapi==") for line in requirements.splitlines()
    )


def test_compile_notebook_writes_an_explicit_requirement_directive_to_requirements_txt(
    tmp_path
):

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "# notebook-to-api: requires opencv-python-headless==4.9.0.80\n"
            "\n"
            "def f() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    lines = set(requirements.split())

    assert "opencv-python-headless==4.9.0.80" in lines


def test_compile_notebook_explicit_requirement_coexists_with_auto_detected_ones(
    tmp_path
):

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "# notebook-to-api: requires my-private-pkg @ git+https://example.com/pkg.git\n"
            "import nbformat\n\n"
            "def f() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    lines = requirements.splitlines()

    assert "my-private-pkg @ git+https://example.com/pkg.git" in lines
    assert any(line.startswith("nbformat==") for line in lines)
    assert any(line.startswith("fastapi==") for line in lines)


def test_compile_notebook_without_any_directive_behaves_as_before(tmp_path):
    """Preserves the previous, still-default behavior -- a notebook with
    no "# notebook-to-api: requires ..." comment compiles exactly as it
    always has.

    Uses nbformat as the auto-detected import -- see
    test_requirements_pins_a_notebook_dependency_installed_in_this_environment's
    own docstring above for why this can't be pandas: it isn't listed in
    the project's requirements.txt, so it's absent in a clean CI install.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import nbformat\n\ndef f() -> int:\n    return 1\n"
        )
    )

    notebook_path = tmp_path / "nb.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    output_dir = tmp_path / "generated"
    compile_notebook(str(notebook_path), str(output_dir))

    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    lines = requirements.split()

    assert any(line.startswith("nbformat==") for line in lines)
    assert any(line.startswith("fastapi==") for line in lines)
