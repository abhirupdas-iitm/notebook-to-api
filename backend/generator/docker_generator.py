import textwrap

from backend.generator.api_generator import LONG_RUNNING_KEYWORDS


def dockerfile_content(
    package_name="generated",
    python_version="3.11",
):
    """The exact Dockerfile text generate_dockerfile (below) writes to
    disk, as a pure string -- no filesystem access at all.

    Split out from generate_dockerfile so a caller that only wants to
    know what the Dockerfile *would* contain (see POST
    /api/dockerfile-preview, backend/routes/upload.py) can get that
    without a real output_path to write into, or a real compiled output
    directory backing "package_name" -- the same "answer the question
    without touching disk" split extract_third_party_imports/
    resolve_requirements already give write_requirements, and
    generate_fastapi_code already gives write_generated_api.
    """

    return f"""\
FROM python:{python_version}-slim

# Python buffers stdout/stderr in blocks (not line-by-line) whenever it
# isn't attached to a real terminal -- which a container's stdout never
# is. Left unset, uvicorn's own request logs and any print() the
# notebook's own code does (it runs for real inside every request this
# app handles) can sit in that buffer for a long time, or be lost
# entirely if the container is killed before it fills -- exactly the
# output `docker logs` and any log-aggregation pipeline watching it are
# expected to see in real time.
#
# PYTHONDONTWRITEBYTECODE avoids writing a .pyc bytecode cache into the
# container's writable layer on every cold start: this project already
# treats __pycache__ as noise to actively exclude, not ship, everywhere
# else it could appear (see EXCLUDED_GENERATED_DIR_NAMES in
# backend/inspector.py, which keeps it out of `inspect`'s generated-files
# listing and the downloadable app bundle for the same reason) -- letting
# the deployed container itself accumulate one anyway, silently, was the
# one place that exclusion never reached. Also matters for a container
# run with a read-only root filesystem (a common hardening practice),
# where writing it at all would fail.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy generated output into /app/{package_name}/ to preserve module paths
COPY . {package_name}/

# Running as root inside the container is privilege the app never needs,
# and widens the blast radius of any RCE-class bug in this generated code
# or a transitive dependency.
RUN useradd --create-home --uid 1000 appuser \\
    && chown -R appuser:appuser /app
USER appuser

# 8000 is this image's default listening port for a plain `docker run`
# with no further configuration, but most real PaaS deploy targets
# (Cloud Run, Render, Heroku, ...) assign the container a port at
# deploy/start time via a $PORT environment variable and require the
# process to actually bind to it -- there's no fixed port they'll agree to
# forward to instead. EXPOSE is just image metadata (it can't reference an
# env var and doesn't affect what the process actually binds to), so it
# stays a literal 8000 documenting the default; the CMD and HEALTHCHECK
# below are what must actually track $PORT at container start.
EXPOSE 8000

# The generated app already exposes GET /health for exactly this purpose
# (see api_generator.py); without a HEALTHCHECK, Docker/orchestrators
# (Compose, Swarm, a bare `docker run`) have no way to distinguish a
# hung/crashed process from a healthy one. Reads $PORT the same way the
# CMD below does, so the healthcheck still hits the port uvicorn actually
# bound to instead of a stale hardcoded 8000.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health', timeout=2)" || exit 1

# A plain exec-form `CMD ["uvicorn", ..., "--port", "8000"]` can't read an
# environment variable at all -- there's no shell involved to expand one --
# so the port would stay hardcoded to 8000 no matter what $PORT a deploy
# target set. Routing through `sh -c` lets `${{PORT:-8000}}` actually
# expand at container start, falling back to 8000 when $PORT isn't set
# (e.g. a local `docker run` with no other configuration).
CMD ["sh", "-c", "uvicorn {package_name}.app:app --host 0.0.0.0 --port ${{PORT:-8000}}"]
"""


def generate_dockerfile(
    output_path="generated/Dockerfile",
    package_name="generated",
    python_version="3.11",
):
    """Write a Dockerfile for the compiled app at `output_path`.

    python_version selects the base image's Python ("<major>.<minor>",
    e.g. "3.12") and should be the interpreter that actually ran the
    compile -- see compiler.compiling_python_version(), the caller
    compile_notebook_to_api always passes. requirements.txt's versions are
    pinned by _pinned_requirement against whatever's installed in *that*
    interpreter's environment; a fixed base image Python unrelated to it
    (this previously always hardcoded "3.11" regardless of what compiled
    the notebook) can silently break `docker build`'s
    `pip install -r requirements.txt` the moment a pinned package's wheels
    don't cover that Python version, or fall back to a source build that
    behaves differently from what was actually resolved and tested
    locally.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dockerfile_content(package_name, python_version))

    print(f"Dockerfile generated at: {output_path}")


def docker_compose_content(package_name="generated", env_vars=None):
    """The exact docker-compose.yml text generate_docker_compose (below)
    writes to disk, as a pure string -- no filesystem access at all. See
    dockerfile_content's own docstring above for why this split exists.

    Before this, a compiled app had a Dockerfile but nothing to actually
    run it with beyond a hand-typed `docker run` -- an operator who
    wanted a one-command `docker compose up` (e.g. for local smoke-
    testing before a real deploy, or as a starting point for a real
    compose-based deployment) had to write every one of these lines
    themselves: the host<->container $PORT mapping the Dockerfile's own
    CMD/HEALTHCHECK already require (see dockerfile_content above), and
    every NOTEBOOK_API_* variable the compiled app itself reads (see
    api_generator.py) -- discoverable via GET /api/env-vars-preview, but
    still meant transcribing by hand into a compose file, one entry at a
    time, with nothing to catch a typo'd name or a stale default the
    moment either drifted from what the app would actually read.

    `env_vars` is GENERATED_APP_ENV_VARS (backend/generator/
    api_generator.py) -- the exact same list generate_fastapi_code's own
    os.getenv(name, default) calls are themselves built from (see
    _generated_app_env_var_default there) and GET /api/env-vars-preview
    already reads back unmodified -- so the "environment:" section below
    can never list a variable the compiled app doesn't actually
    recognize, or a default that's drifted from what it would actually
    fall back to. Passed in rather than imported directly here to avoid
    a circular import: compiler.py (the caller of generate_docker_compose
    below) already imports both this module and api_generator.py, but
    api_generator.py has no reason of its own to import this module back.

    Each entry becomes "NAME=${NAME:-default}", a plain compose
    environment-list entry -- not a fixed value, but a shell-style
    default that still lets an operator override any one of them for a
    real deployment (a `.env` file alongside this one, or the calling
    shell's own environment) without editing this generated file, the
    same "configurable without editing generated code" precedent
    dockerfile_content's own $PORT interpolation already sets for the
    Dockerfile itself. "PORT" -- read by the Dockerfile's own CMD/
    HEALTHCHECK, not by the compiled app itself (see GET
    /api/env-vars-preview's own docstring for why it's deliberately
    excluded from GENERATED_APP_ENV_VARS) -- gets the identical
    treatment here, driving both sides of the host:container port
    mapping so the exposed port always matches whatever the container
    itself actually bound to.

    "restart: unless-stopped" was previously absent entirely -- Compose's
    own default restart policy is "no" (never restart), so a container
    that crashed or was OOM-killed (a real possibility: the compiled
    app's own NOTEBOOK_API_MAX_TASKS/NOTEBOOK_API_MAX_REQUEST_BYTES limit
    a single process' load, but nothing stops the host itself from
    running low on memory) just stayed down until an operator noticed
    and re-ran `docker compose up` by hand -- for a `docker compose up
    -d`-style deployment meant to keep running unattended (this file's
    own reason to exist, per this docstring's opening paragraph), that's
    a silent outage with no self-healing at all. "unless-stopped" (rather
    than the more aggressive "always") still respects an operator's own
    explicit `docker compose stop`: it restarts after a crash or a host
    reboot, but not after a deliberate stop, the same distinction Docker
    Swarm/Kubernetes' own default restart policies already draw between
    an unexpected exit and an intentional one.
    """

    env_vars = env_vars or []

    environment_lines = "\n".join(
        f"      - {entry['name']}=${{{entry['name']}:-{entry['default']}}}"
        for entry in env_vars
    )

    return (
        "services:\n"
        f"  {package_name}:\n"
        "    build: .\n"
        "    restart: unless-stopped\n"
        "    ports:\n"
        '      - "${PORT:-8000}:${PORT:-8000}"\n'
        "    environment:\n"
        "      - PORT=${PORT:-8000}\n"
        f"{environment_lines}\n"
    )


def generate_docker_compose(
    output_path="generated/docker-compose.yml",
    package_name="generated",
    env_vars=None,
):
    """Write a docker-compose.yml for the compiled app at `output_path`,
    alongside the Dockerfile/.dockerignore generate_dockerfile/
    generate_dockerignore already write there on every compile -- see
    docker_compose_content's own docstring above for why this exists and
    what it contains.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(docker_compose_content(package_name, env_vars))

    print(f"docker-compose.yml generated at: {output_path}")


def env_example_content(env_vars=None):
    """The exact .env.example text generate_env_example (below) writes to
    disk, as a pure string -- no filesystem access at all. See
    dockerfile_content's own docstring above for why this split exists.

    GET /api/env-vars-preview (backend/routes/upload.py) already answers
    "what environment variables does a compiled app read, and what do
    they default to" -- but only as structured JSON. An operator actually
    standing up a deployment (`docker compose up`, a Kubernetes Secret/
    ConfigMap, a plain systemd EnvironmentFile) still had to transcribe
    that JSON into a real env file by hand, one variable at a time, with
    nothing to catch a typo'd name or a stale default the moment either
    drifted from what the compiled app would actually read -- the exact
    "computed but never actually handed over as a ready-to-use artifact"
    gap docker_compose_content's own docstring already closed for
    docker-compose.yml itself.

    `env_vars` is GENERATED_APP_ENV_VARS (backend/generator/
    api_generator.py) -- the same list docker_compose_content already
    takes (see its own docstring for why it's passed in rather than
    imported directly here, to avoid a circular import) -- so a value
    here can never drift from what docker-compose.yml's own
    "environment:" section, or GET /api/env-vars-preview, already report
    for the same variable.

    Each entry becomes a wrapped "# <description>" comment followed by
    "NAME=default" -- a real, valid env file on its own (every value is
    already the same default the compiled app itself falls back to), not
    a placeholder that must be filled in before it works, so `cp
    .env.example .env` alone already reproduces the compiled app's own
    unconfigured behavior; only a value an operator actually wants to
    override needs editing.

    "PORT" (deliberately excluded from GENERATED_APP_ENV_VARS itself --
    see GET /api/env-vars-preview's own docstring: it's read by the
    Dockerfile's own CMD/HEALTHCHECK and docker-compose.yml's own "ports"
    mapping, never by the compiled app) gets the identical unconditional
    inclusion docker_compose_content's own "environment:" section already
    gives it, for the same reason: a `docker compose up` deployment
    commonly wants to override the host-side port without touching the
    generated docker-compose.yml itself.
    """
    env_vars = env_vars or []

    lines = [
        "# Copy this file to .env and override any value below -- every",
        "# one already matches the same default the compiled app (or, for",
        "# PORT, docker-compose.yml/the Dockerfile) itself falls back to,",
        "# so this file alone already reproduces the unconfigured behavior.",
        "",
        "# Host:container port docker-compose.yml's own \"ports\"/",
        "# \"environment\" sections map, and the Dockerfile's own CMD/",
        "# HEALTHCHECK read -- not read by the compiled app itself.",
        "PORT=8000",
        "",
    ]

    for entry in env_vars:

        for description_line in textwrap.wrap(entry["description"], width=76):
            lines.append(f"# {description_line}")

        lines.append(f"{entry['name']}={entry['default']}")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def generate_env_example(output_path="generated/.env.example", env_vars=None):
    """Write a .env.example for the compiled app at `output_path`,
    alongside the Dockerfile/.dockerignore/docker-compose.yml
    generate_dockerfile/generate_dockerignore/generate_docker_compose
    already write there on every compile -- see env_example_content's own
    docstring above for why this exists and what it contains.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(env_example_content(env_vars))

    print(f".env.example generated at: {output_path}")


def readme_content(package_name="generated", functions=None, env_vars=None):
    """The exact README.md text generate_readme (below) writes to disk,
    as a pure string -- no filesystem access at all. See
    dockerfile_content's own docstring above for why this split exists.

    Before this, a real compile wrote app.py, requirements.txt, a
    Dockerfile/.dockerignore/docker-compose.yml/.env.example, and
    optionally an OpenAPI export and SDK clients -- but nothing that told
    a human what any of it actually was. An operator who downloads GET
    /api/download's zip, or clones a deploy target's repo, had no single
    file saying which endpoints this specific compile exposes, that every
    one of them needs an X-API-Key header, or even the one command
    (`docker compose up --build`) that actually runs the thing they just
    got -- only /docs (which needs the app already running), the raw
    Dockerfile/docker-compose.yml (accurate, but not written to be read),
    and whatever the notebook author's own commit message happened to say
    elsewhere.

    `functions` is the same list generate_fastapi_code (api_generator.py)
    itself compiles into endpoints -- each already carrying "name" and,
    for a background one, matching LONG_RUNNING_KEYWORDS. Reusing that
    exact classification (rather than re-deriving "is this background"
    from scratch) means the "(background task)" markers below can never
    drift from what the compiled app.py this README ships alongside
    actually does -- the same "can't drift from the real thing" guarantee
    docker_compose_content's own "environment:" section already gives
    GENERATED_APP_ENV_VARS.

    `env_vars` is GENERATED_APP_ENV_VARS itself, passed in (not imported
    directly here) for the identical circular-import reason
    docker_compose_content/env_example_content's own docstrings already
    give: api_generator.py has no reason to import this module back.
    """
    functions = functions or []
    env_vars = env_vars or []

    endpoint_lines = []

    for func in sorted(functions, key=lambda f: f["name"]):

        is_background = any(
            kw in func["name"].lower() for kw in LONG_RUNNING_KEYWORDS
        )

        suffix = (
            " -- enqueues a background task; poll `GET /tasks/{task_id}` "
            "for the result"
            if is_background else ""
        )

        endpoint_lines.append(f"- `POST /{func['name']}`{suffix}")

    endpoints_section = (
        "\n".join(endpoint_lines)
        if endpoint_lines
        else "_This notebook doesn't expose any functions yet._"
    )

    env_var_lines = "\n".join(
        f"- `{entry['name']}` (default: `{entry['default']}`) -- "
        f"{entry['description']}"
        for entry in env_vars
    )

    return f"""\
# {package_name}

Generated by Notebook-to-API from a Jupyter notebook -- do not edit \
`app.py` or `runtime/notebook_module.py` directly, they're overwritten \
on every recompile. Edit the source notebook instead.

## Endpoints

Every endpoint below (and every built-in one -- `/health`, `/ready`, \
`/info`, `/config`, `/metrics`, `/uptime`, `/auth/status`, `/auth/info`, \
`/auth/validate`, `/tasks`, `/tasks/{{task_id}}`) requires an \
`X-API-Key` header. See Authentication below.

{endpoints_section}

Interactive docs are served at `/docs` (Swagger UI) and `/redoc`, unless \
`NOTEBOOK_API_DISABLE_DOCS=true`.

## Running locally

```bash
pip install -r requirements.txt
uvicorn {package_name}.app:app --reload
```

Or with Docker:

```bash
docker compose up --build
```

## Authentication

Every request must carry an `X-API-Key` header matching one of the keys \
in `NOTEBOOK_API_KEY` (comma-separated to support zero-downtime \
rotation). Defaults to `notebook-to-api-dev-key` for local development \
-- set a real value before deploying anywhere reachable.

## Configuration

Every setting below is a real default this app already falls back to; \
see `.env.example` for a ready-to-copy file.

{env_var_lines}
"""


def generate_readme(
    output_path="generated/README.md", package_name="generated",
    functions=None, env_vars=None,
):
    """Write a README.md for the compiled app at `output_path`, alongside
    the Dockerfile/.dockerignore/docker-compose.yml/.env.example
    generate_dockerfile/generate_dockerignore/generate_docker_compose/
    generate_env_example already write there on every compile -- see
    readme_content's own docstring above for why this exists and what it
    contains.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(readme_content(package_name, functions, env_vars))

    print(f"README.md generated at: {output_path}")


def dockerignore_content():
    """The exact .dockerignore text generate_dockerignore (below) writes
    to disk, as a pure string -- no filesystem access at all. See
    dockerfile_content's own docstring above for why this split exists.
    """

    return """\
.git/
.gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
env/
*.ipynb
.ipynb_checkpoints/
Dockerfile
.dockerignore
docker-compose.yml
kubernetes.yaml
.env.example
README.md
openapi.json
openapi.yaml
sdk/
.compile_metadata.json
"""


def generate_dockerignore(output_path="generated/.dockerignore"):
    """Without this, `COPY . {package_name}/` in the generated Dockerfile
    picks up .git, __pycache__, local venvs, notebooks, and other
    unrelated files from the build context into the image -- bloating it
    and, for .git in particular, potentially leaking history that was
    never meant to ship.

    openapi.json/openapi.yaml/sdk/ are also excluded: POST
    /api/export-openapi, POST /api/export-sdk, and the CLI's
    export-openapi/export-sdk commands (by default) all write these
    straight into this same output directory, alongside the compiled app
    -- but the running app never reads any of them at runtime (it builds
    its own OpenAPI schema live via custom_openapi() in api_generator.py,
    not from a file on disk). Left unexcluded, a `deploy`/`docker build`
    run any time after an export had happened baked these purely
    client-facing artifacts into the served image for no runtime benefit,
    the exact kind of build-context noise this .dockerignore already
    exists to keep out.

    .env.example (see env_example_content above) and kubernetes.yaml (see
    kubernetes_manifest_content, backend/generator/kubernetes_generator.py)
    are excluded for the same reason Dockerfile/.dockerignore/
    docker-compose.yml already are: each exists purely to hand an operator
    a ready-to-use deployment artifact -- a template to copy to their own
    `.env`, or a manifest to `kubectl apply` -- never read by the running
    app itself.

    README.md (see readme_content above) is excluded for the identical
    reason: purely documentation for a human looking at the compiled
    output directory or a downloaded bundle, never read by the running
    app itself.

    .compile_metadata.json (the literal filename write_compile_metadata
    uses -- see COMPILE_METADATA_FILENAME in backend/compiler.py, which
    this can't import without a circular import, since compiler.py already
    imports this module) is excluded for a sharper reason than build-context
    noise: it's dashboard-internal bookkeeping (read only by
    list_notebooks/_currently_compiled_notebook_metadata in
    routes/upload.py), never by the running app, and its "source_notebook"
    field is the source notebook's *absolute filesystem path on the
    compiling server*. Left unexcluded, every `deploy`/`docker build`
    baked that server-side path straight into the shipped image.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dockerignore_content())

    print(f".dockerignore generated at: {output_path}")
