import ast
import builtins
import typing
from pathlib import Path

# Top-level names the generated app itself defines. A notebook function
# sharing one of these names would be emitted as `def <name>(...)`,
# rebinding the real one at module-load time -- most dangerously
# "verify_api_key": every endpoint defined *after* such a collision gets
# `Depends(verify_api_key)` resolved (at def-statement time) against the
# notebook's own function instead of the real auth guard, silently
# disabling API-key authentication for the rest of the app with no error
# anywhere. Rejecting these outright avoids ever emitting that endpoint
# ordering trap.
RESERVED_INFRASTRUCTURE_NAMES = frozenset({
    "app", "TASKS", "API_KEYS", "API_KEY_HEADER_NAME", "START_TIME",
    "GENERATED_AT", "PYTHON_VERSION", "NOTEBOOK_TO_API_VERSION", "ALLOWED_ORIGINS",
    "PUBLIC_URL", "DISABLE_DOCS",
    "MAX_REQUEST_BODY_BYTES", "MaxRequestBodySizeMiddleware",
    "MAX_PENDING_TASKS", "WEBHOOK_TIMEOUT_SECONDS", "WEBHOOK_SECRET",
    "TASK_EXECUTION_TIMEOUT_SECONDS",
    "_deliver_task_webhook",
    "verify_api_key", "custom_openapi",
    "root", "health_check", "readiness_check", "auth_status", "auth_info",
    "validate_auth", "service_info", "service_config", "metrics", "uptime",
    "get_task", "list_tasks", "delete_task", "cleanup_tasks",
    "delete_completed_tasks", "delete_failed_tasks", "reset_tasks",
    "notebook_module",
    # Confirmed exploitable: these two private helpers (both defined at
    # module scope, like every other name above) were missing here, so a
    # notebook function literally named "_evict_expired_tasks" or
    # "_run_background_task" compiled fine and silently overwrote the
    # real one at module-execution time -- Python has no protection
    # against redefining a name, the later `def` always wins. Every
    # *other* background endpoint's own submission still calls the exact
    # same shadowed name (`_evict_expired_tasks()` before enqueuing a new
    # task, `background_tasks.add_task(_run_background_task, ...)` to run
    # one), so this didn't just break the notebook's own colliding
    # endpoint -- it broke background task submission or execution
    # *entirely*, app-wide. Reproduced: a notebook exposing both
    # `_evict_expired_tasks(x: int) -> int` and an unrelated `train_model`
    # crashed `POST /train_model` itself with "TypeError:
    # _evict_expired_tasks() missing 1 required positional argument:
    # 'req'", nothing to do with train_model's own logic at all.
    "_evict_expired_tasks", "_run_background_task",
})


class ReservedFunctionNameError(ValueError):
    """A notebook function's name collides with an identifier the
    generated app itself defines."""


# Keywords indicating a function should be run as a background task
LONG_RUNNING_KEYWORDS = [
    "train",
    "process",
    "generate",
    "embed",
    "scrape",
]

# Every environment variable the generated app itself reads to configure a
# runtime limit or credential -- one single source of truth codegen below
# builds its own os.getenv(name, default) calls from (see
# _generated_app_env_var_default), so GET /api/env-vars-preview
# (routes/upload.py) can never drift from what a real compile's own
# app.py would actually read, the same "can't drift from the real thing"
# guarantee dockerfile_content/dockerignore_content (backend/generator/
# docker_generator.py) already provide for their own artifact.
GENERATED_APP_ENV_VARS = [
    {
        "name": "NOTEBOOK_API_KEY",
        "default": "notebook-to-api-dev-key",
        "description": (
            "Comma-separated list of API keys accepted on the X-API-Key "
            "header every generated endpoint requires (see "
            "verify_api_key) -- a list, not a single value, so a key can "
            "be rotated with zero downtime: add the new key alongside "
            "the old one, restart, let clients switch over, then remove "
            "the old key and restart again."
        ),
    },
    {
        "name": "NOTEBOOK_API_ALLOWED_ORIGINS",
        "default": "*",
        "description": (
            "Comma-separated list of origins allowed to call this API "
            "from a browser (CORSMiddleware's own allow_origins). "
            "Unset, every origin is allowed -- safe here since every "
            "request is authenticated via X-API-Key, never a cookie."
        ),
    },
    {
        "name": "NOTEBOOK_API_MAX_REQUEST_BYTES",
        "default": str(10 * 1024 * 1024),
        "description": (
            "Maximum accepted request body size in bytes -- a request "
            "declaring a larger Content-Length is rejected with 413 "
            "before its body is even read."
        ),
    },
    {
        "name": "NOTEBOOK_API_TASK_TTL_SECONDS",
        "default": "3600",
        "description": (
            "How long a background task's own result stays in TASKS "
            "after completing/failing before it's evicted."
        ),
    },
    {
        "name": "NOTEBOOK_API_MAX_TASKS",
        "default": "10000",
        "description": (
            "Maximum number of background tasks pending at once -- a "
            "new one submitted while at this limit is rejected with 503 "
            "until some already-tracked tasks are evicted."
        ),
    },
    {
        "name": "NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS",
        "default": "0",
        "description": (
            "Maximum seconds a single background task's own execution "
            "may run before it's cancelled and marked 'failed' with a "
            "timeout error. Without this, a hung or runaway notebook "
            "function (an infinite loop, a network call with no timeout "
            "of its own) ties up one of this process' limited worker "
            "threads forever -- the same threadpool every *synchronous* "
            "endpoint (including GET /health) also runs on, so enough "
            "hung tasks eventually starve the entire app, not just "
            "background ones. 0 (the default) disables this entirely, "
            "preserving the previous unbounded-execution-time behavior."
        ),
    },
    {
        "name": "NOTEBOOK_API_RATE_LIMIT_PER_MINUTE",
        "default": "0",
        "description": (
            "Maximum requests a single API key may make per rolling "
            "60-second window before being rejected with 429 (and a "
            "Retry-After header). Every request against this key, "
            "successful or not, also gets X-RateLimit-Limit/"
            "-Remaining/-Reset response headers so a well-behaved "
            "caller can back off before actually being throttled, not "
            "just after. Tracked independently per configured key, so "
            "one key being throttled never affects another. 0 (the "
            "default) disables rate limiting entirely, preserving the "
            "previous unbounded behavior."
        ),
    },
    {
        "name": "NOTEBOOK_API_WEBHOOK_TIMEOUT_SECONDS",
        "default": "5",
        "description": (
            "How long a background task's own optional ?callback_url= "
            "webhook delivery may take before giving up. Purely "
            "best-effort: the task's own status/result recorded in "
            "TASKS is never affected by webhook delivery failing, "
            "timing out, or being misconfigured -- this only bounds "
            "how long that one delivery attempt can block the worker "
            "thread running it."
        ),
    },
    {
        "name": "NOTEBOOK_API_WEBHOOK_SECRET",
        "default": "",
        "description": (
            "Shared secret used to sign a background task's own optional "
            "?callback_url= webhook delivery with HMAC-SHA256, sent as "
            "X-Webhook-Signature: sha256=<hex>, so the receiving endpoint "
            "can verify a request actually came from this app and wasn't "
            "tampered with in transit -- the same X-Hub-Signature-256 "
            "contract GitHub/Stripe webhooks already use. Empty (the "
            "default) sends the webhook unsigned, exactly as before this "
            "existed."
        ),
    },
    {
        "name": "NOTEBOOK_API_PUBLIC_URL",
        "default": "http://localhost:8000",
        "description": (
            "The base URL this deployment is actually reachable at, "
            "reported as this app's own OpenAPI \"servers\" entry -- what "
            "/docs' own Swagger UI \"Try it out\" defaults its request "
            "URL to. Left at the default outside local development, "
            "Swagger UI keeps sending \"Try it out\" requests to "
            "http://localhost:8000 no matter where the app is actually "
            "deployed, failing every one of them from a browser that "
            "isn't itself on the same machine."
        ),
    },
    {
        "name": "NOTEBOOK_API_DISABLE_DOCS",
        "default": "false",
        "description": (
            "Set to \"true\" to disable this app's own interactive /docs "
            "(Swagger UI), /redoc, and /openapi.json entirely (each "
            "returns a plain 404) -- every request this app accepts is "
            "already authenticated via X-API-Key, but the schema and "
            "docs UI themselves were always served with no such "
            "requirement, exposing every endpoint's name, parameters, "
            "and example payloads to anyone who can merely reach the "
            "deployment, not just anyone who could actually call it. "
            "This dashboard's own POST /api/export-openapi/export-sdk "
            "are unaffected either way: they call this app's own "
            "openapi() method directly (in-process, at compile/export "
            "time), never through the HTTP routes this setting disables."
        ),
    },
]


def _generated_app_env_var_default(name):
    """The default value GENERATED_APP_ENV_VARS declares for `name`,
    embedded into the matching os.getenv(name, default) call generated
    below -- see GENERATED_APP_ENV_VARS' own docstring for why codegen
    reads it from there instead of repeating the literal a second time.
    """
    return next(
        entry["default"] for entry in GENERATED_APP_ENV_VARS
        if entry["name"] == name
    )


def _auth_and_rate_limit_error_responses():
    """The {401, 429} OpenAPI response entries every generated notebook-
    function endpoint can actually produce, regardless of what the
    notebook function itself does -- verify_api_key/_enforce_rate_limit
    (both above) run via Depends(verify_api_key) before the endpoint's
    own body ever executes, for every one of them, sync or background.

    Before this, a generated endpoint's own OpenAPI schema documented
    only its 200 response and FastAPI's own automatically-added 422
    (Pydantic validation error) -- FastAPI has no way to infer a plain
    dependency function's own `raise HTTPException(...)` calls the way
    it already does for 422, so 401 (an invalid/missing X-API-Key) and
    429 (NOTEBOOK_API_RATE_LIMIT_PER_MINUTE exceeded) were completely
    undocumented in the served schema, /docs, and any third-party tool
    generating a client from it -- even though every single endpoint
    already requires passing both checks before its own body ever runs.
    The exact numeric limit/window for 429 is a runtime NOTEBOOK_API_*
    env var this function has no way to know at compile time (see
    GENERATED_APP_ENV_VARS), so its own description points at the
    variable name instead of a number that could be wrong the moment an
    operator overrides the default.
    """
    return {
        401: {
            "description": "Missing or invalid X-API-Key header.",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid API key"}
                }
            },
        },
        429: {
            "description": (
                "Rate limit exceeded (see "
                "NOTEBOOK_API_RATE_LIMIT_PER_MINUTE)."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "detail": (
                            "Rate limit exceeded: 60 requests per 60s "
                            "per API key"
                        )
                    }
                }
            },
        },
    }


def _call_arg_expr(arg):
    """Render a single argument for the notebook_module.<fn>(...) call.

    Keyword-only parameters (those after a bare `*`, e.g.
    `def train(data, *, epochs=10)`) cannot be passed positionally, so
    they must be forwarded as `name=req.name` rather than plain `req.name`.
    """
    if arg.get("kind") == "keyword_only":
        return f"{arg['name']}=req.{arg['name']}"
    return f"req.{arg['name']}"


_TYPING_EXPORTS = frozenset(
    name for name in dir(typing) if not name.startswith("_")
)
_BUILTIN_NAMES = frozenset(dir(builtins))


class _AnnotationNameQualifier(ast.NodeTransformer):
    """Rewrites bare names in a type-annotation AST so every name the
    generated Pydantic model references is actually resolvable.

    A name that belongs to `typing` (List, Dict, Optional, Union, ...) is
    left alone but recorded so the caller can emit the matching
    `from typing import ...` line. Any other name that isn't a Python
    builtin is assumed to come from the notebook itself -- a class/Enum it
    defines, or something it imported at module level -- since the
    function using it as an annotation lives in that same module, the bare
    name must already resolve there. It's rewritten to
    `notebook_module.<name>` (the alias the generated app already imports
    the notebook's runtime module under) instead of failing with a
    NameError/PydanticUserError when the model class is built.
    """

    def __init__(self):
        self.typing_names = set()

    def visit_Name(self, node):
        if node.id in _TYPING_EXPORTS:
            self.typing_names.add(node.id)
            return node

        if node.id in _BUILTIN_NAMES:
            return node

        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(id="notebook_module", ctx=ast.Load()),
                attr=node.id,
                ctx=node.ctx,
            ),
            node,
        )


def _build_model_names(functions):
    """Map each function's name to a Pydantic request-model class name,
    guaranteed unique even when two function names collide once reduced
    to a class name (e.g. "get_data" and "Get_data" -- only the first
    character was ever uppercased, so both produced the identical class
    name "Get_dataRequest", and the second definition silently shadowed
    the first's fields: whichever endpoint referenced that name ended up
    validating requests against the *other* function's parameters).
    """
    used_names = set()
    model_names = {}

    for func in functions:
        func_name = func["name"]
        base_name = f"{func_name[0].upper()}{func_name[1:]}Request"
        candidate = base_name
        suffix = 2

        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1

        used_names.add(candidate)
        model_names[func_name] = candidate

    return model_names


def _resolve_annotation_source(type_str):
    """Turn a raw `ast.unparse`d annotation string (as stored in
    arg["type"] by the parser) into source the generated app can actually
    evaluate, plus the set of `typing` names it needs imported.

    Before this, arg["type"] was written into the generated Pydantic model
    verbatim: `List[float]`, `Optional[str]`, `Dict[str, Any]`, or a
    notebook-defined class/Enum name all produced a field annotation
    referencing a name nothing in the generated file ever imports, which
    breaks model construction (`PredictRequest.model_json_schema()` /
    `model_rebuild()`) the first time FastAPI actually needs the schema --
    i.e. on the very first request or /docs load, not at compile time.
    """
    if not type_str:
        return "str", set()

    try:
        tree = ast.parse(type_str, mode="eval")
    except SyntaxError:
        return type_str, set()

    qualifier = _AnnotationNameQualifier()
    rewritten = qualifier.visit(tree)
    ast.fix_missing_locations(rewritten)

    return ast.unparse(rewritten), qualifier.typing_names


def _annotation_has_own_field_description(type_str):
    """Whether `type_str` (a raw `ast.unparse`d annotation, as stored in
    arg["type"] by the parser -- the *original*, pre-
    _resolve_annotation_source string, not its notebook_module-qualified
    rewrite) is an `Annotated[T, ..., Field(..., description=...), ...]`
    whose own metadata already carries a description.

    generate_fastapi_code below always used to append its own
    `description=repr(field_description)` to every field's `Field(...)`
    call, unconditionally -- including a field whose annotation is
    itself `Annotated[int, Field(gt=0, description="must be positive")]`,
    a increasingly common way for a notebook author to document (and
    constrain) a parameter directly on modern Pydantic v2. Confirmed
    exploitable before this: Pydantic merges the `Annotated[...]`
    metadata's own FieldInfo with the one assigned as the field's default
    value, and the *assigned* one's "description" wins on conflict --
    so the notebook author's own carefully-written "must be positive"
    was silently replaced by the generic, auto-generated "Parameter 'x'
    of type Annotated[int, Field(gt=0, description=...)]" in the actual
    served OpenAPI schema and both generated SDK clients, with nothing
    to indicate the author's own description had been discarded.
    Skipping the generated description entirely whenever the annotation
    already supplies its own leaves every other `Field(...)` argument
    (gt, le, a default, ...) working exactly as before -- this only ever
    suppresses the one field this codebase has no business overriding.
    """
    if not type_str:
        return False

    try:
        tree = ast.parse(type_str, mode="eval").body
    except SyntaxError:
        return False

    if not isinstance(tree, ast.Subscript):
        return False

    base = tree.value
    base_name = (
        base.id if isinstance(base, ast.Name)
        else getattr(base, "attr", None)
    )

    if base_name != "Annotated":
        return False

    metadata_slice = tree.slice
    metadata = (
        metadata_slice.elts if isinstance(metadata_slice, ast.Tuple)
        else [metadata_slice]
    )

    for item in metadata[1:]:

        if not isinstance(item, ast.Call):
            continue

        func = item.func
        func_name = (
            func.id if isinstance(func, ast.Name)
            else getattr(func, "attr", None)
        )

        if func_name != "Field":
            continue

        if any(kw.arg == "description" for kw in item.keywords):
            return True

    return False


# Template for generating the FastAPI application source code
def generate_fastapi_code(
    functions, package_name="generated", source_notebook_sha256=None,
    notebook_to_api_version="1.0.0",
):
    """Generate FastAPI app code for the given functions.

    Each function is examined; if its name contains any of the
    LONG_RUNNING_KEYWORDS, an endpoint is created that enqueues the
    function as a BackgroundTask and returns a task_id. Otherwise a
    regular synchronous endpoint is generated.

    package_name is the top-level package the generated app imports its
    runtime module from (`<package_name>.runtime.notebook_module`). It
    must match the basename of wherever this generated code actually gets
    written -- see compiler.package_name_for_output_dir.

    source_notebook_sha256 (optional) is baked into the generated app
    itself as a fixed constant, returned by its own GET /info -- see that
    endpoint's own "source_notebook_sha256" field below for why this
    exists: a running deployed container had no way to self-report which
    exact notebook content actually produced it, short of cross-
    referencing this dashboard's own deploy/compile history externally
    (assuming that history is even still available, and the caller
    already knows which dashboard/tag to look under). None (the default,
    used by any caller not passing it) means "unknown" -- GET /info
    reports it as null, exactly as if this parameter didn't exist.

    notebook_to_api_version (optional) is baked into the generated app
    the identical way, as its own "generator_version" (GET /), "version"
    (GET /info), and -- unlike those first two, this one was missed the
    first time this parameter was added, and only caught afterward --
    the FastAPI(...) app object's own `version=` kwarg itself, which
    `custom_openapi` (below) passes straight through to
    get_openapi(..., version=app.version, ...) as this app's own
    OpenAPI "info.version". All three previously carried the identical
    hardcoded "1.0.0" literal, completely unrelated to which actual
    version of this tool compiled the app, the same "two independent,
    inevitably-drifting hardcoded version literals" bug NOTEBOOK_TO_API_
    VERSION (backend/compiler.py) was already introduced to deduplicate
    for this dashboard's own GET /api/health and GET / -- just never
    threaded through to the *generated* app's own three literals. Unlike
    "generator_version"/"version" (informational JSON fields only), a
    stale "info.version" is user-visible in every compiled app's own
    /docs (Swagger UI) and gets baked directly into whatever POST
    /api/export-openapi writes out (export_openapi_schema serializes
    app.openapi() unchanged) -- the exact schema any external tooling
    (an API catalog, a codegen tool other than this project's own
    generate_python_sdk/generate_typescript_sdk, which never read
    "info.version" themselves) would read "info.version" from.
    compile_notebook_to_api (backend/compiler.py) always passes its own
    NOTEBOOK_TO_API_VERSION here; the "1.0.0" default is only ever seen
    by a caller of this function that doesn't (a direct unit test, most
    commonly), preserving this function's previous literal exactly for
    it rather than silently changing behavior no caller asked for.
    """
    colliding_names = sorted(
        {func["name"] for func in functions} & RESERVED_INFRASTRUCTURE_NAMES
    )
    if colliding_names:
        raise ReservedFunctionNameError(
            "Notebook function name(s) "
            f"{', '.join(colliding_names)} collide with identifiers the "
            "generated app itself defines (auth, task management, or "
            "infrastructure routes). Rename the function(s) in the "
            "notebook and recompile."
        )

    # GET /tasks below always needs Optional[str] for its own `status`
    # query param, regardless of whether any notebook function's own
    # annotations need typing imports.
    needed_typing_names = {"Optional"}
    for func in functions:
        for arg in func.get("args", []):
            _, typing_names = _resolve_annotation_source(arg.get("type"))
            needed_typing_names |= typing_names

            if arg.get("has_default") and not arg.get(
                "default_is_literal", True
            ):
                _, default_typing_names = _resolve_annotation_source(
                    arg.get("default")
                )
                needed_typing_names |= default_typing_names

    model_names = _build_model_names(functions)

    lines = []
    # Imports for the generated FastAPI app
    lines.append(
        "from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, "
        "Depends, Query, Response"
    )
    lines.append("from fastapi.middleware.cors import CORSMiddleware")
    lines.append("from fastapi.middleware.gzip import GZipMiddleware")
    lines.append("from fastapi.responses import JSONResponse")
    lines.append("from fastapi.encoders import jsonable_encoder")
    lines.append("import anyio.to_thread")
    lines.append("import functools")
    lines.append("import uuid")
    lines.append("import os")
    lines.append("import sys")
    lines.append("import inspect")
    lines.append("import hmac")
    lines.append("import json")
    lines.append("import urllib.request")
    lines.append("import urllib.error")
    lines.append("from urllib.parse import urlparse")
    lines.append("from datetime import datetime")
    lines.append("import time")
    lines.append("from pydantic import BaseModel, Field")
    if needed_typing_names:
        lines.append(f"from typing import {', '.join(sorted(needed_typing_names))}")
    lines.append(f"import {package_name}.runtime.notebook_module as notebook_module")
    lines.append("")
    # Read before app = FastAPI(...) below (the servers= kwarg needs it
    # at construction time) -- see GET /api/env-vars-preview's own
    # NOTEBOOK_API_PUBLIC_URL entry for what this actually drives, and
    # why leaving it at the default outside local development silently
    # breaks Swagger UI's own "Try it out" for anyone not on the same
    # machine as the deployment.
    lines.append(
        'PUBLIC_URL = os.getenv('
        '"NOTEBOOK_API_PUBLIC_URL", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_PUBLIC_URL")}"'
        ')'
    )
    # Also read before app = FastAPI(...) below -- docs_url/redoc_url/
    # openapi_url are only ever honored at construction time; FastAPI has
    # no supported way to toggle them afterward. Membership in a truthy
    # set (not a bare bool(...) of the string, which -- confirmed --
    # would treat NOTEBOOK_API_DISABLE_DOCS=false as truthy, since a
    # non-empty string is always truthy in Python regardless of its own
    # text) mirrors dashboard_reload()'s own identical "tolerate
    # true/1/yes/on, case-insensitively" convention (backend/dashboard.py)
    # for a hand-typed env var, just inverted: that one's falsy set turns
    # a default-on behavior off, this truthy set turns a default-off
    # behavior on.
    lines.append(
        'DISABLE_DOCS = os.getenv('
        '"NOTEBOOK_API_DISABLE_DOCS", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_DISABLE_DOCS")}"'
        ').strip().lower() in ("true", "1", "yes", "on")'
    )
    lines.append("")
    lines.append(
        'app = FastAPI('
        'title="Notebook-to-API Generated Service", '
        'description="Automatically generated from notebook analysis.", '
        f'version={notebook_to_api_version!r}, '
        'contact={"name": "Notebook-to-API"}, '
        'license_info={"name": "MIT"}, '
        'servers=[{"url": PUBLIC_URL, '
        '"description": "This deployment"}], '
        'docs_url=None if DISABLE_DOCS else "/docs", '
        'redoc_url=None if DISABLE_DOCS else "/redoc", '
        'openapi_url=None if DISABLE_DOCS else "/openapi.json"'
        ')'
    )
    lines.append("")
    lines.append("app.openapi_schema = None")
    lines.append("")
    # Every request this app accepts is authenticated via the X-API-Key
    # header (see verify_api_key below), never a cookie, so -- unlike the
    # dashboard API's own CORS setup in backend/dashboard.py, which has to
    # restrict allow_origins to an explicit list precisely because it
    # accepts credentialed (cookie-based) cross-origin requests --
    # reflecting an arbitrary Origin here carries no cross-site credential
    # risk. allow_credentials is explicitly False, which is also what
    # makes a "*" default safe (browsers refuse "*" together with
    # allow_credentials=True). Without this, the single most common way to
    # actually consume a deployed generated API -- a browser-based
    # frontend calling it directly -- was blocked by CORS with no way to
    # fix it short of hand-editing this generated file. Configurable via
    # NOTEBOOK_API_ALLOWED_ORIGINS (comma-separated) to lock this down for
    # a real deployment instead.
    lines.append(
        'ALLOWED_ORIGINS = ['
        'o.strip() for o in os.getenv('
        '"NOTEBOOK_API_ALLOWED_ORIGINS", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_ALLOWED_ORIGINS")}"'
        ').split(",") if o.strip()'
        '] or ["*"]'
    )
    # Browsers only ever expose a small built-in safelist of *response*
    # headers to cross-origin JS (Cache-Control, Content-Language,
    # Content-Length, Content-Type, Expires, Last-Modified, Pragma) --
    # everything else, X-RateLimit-Limit/-Remaining/-Reset and
    # Retry-After (see _enforce_rate_limit above) included, is invisible
    # to `fetch(...).headers.get(...)` cross-origin no matter what
    # allow_origins/allow_headers above are set to, unless explicitly
    # listed in expose_headers. Confirmed: response.headers.get(...) for
    # any of these four returned null from cross-origin JS before this,
    # even though the server sent them every time -- a browser-based
    # frontend wanting to show "you're about to be rate limited" (the
    # whole point of Commit #3 adding these) had no way to read them at
    # all short of a same-origin request.
    lines.append(
        "app.add_middleware("
        "CORSMiddleware, "
        "allow_origins=ALLOWED_ORIGINS, "
        "allow_credentials=False, "
        "allow_methods=['*'], "
        "allow_headers=['*'], "
        "expose_headers=["
        "'X-RateLimit-Limit', 'X-RateLimit-Remaining', "
        "'X-RateLimit-Reset', 'Retry-After'"
        "]"
        ")"
    )
    lines.append("")
    # Every endpoint below accepts an arbitrary JSON request body with no
    # limit on its size -- unlike this very tool's own dashboard
    # /api/upload, which has always capped uploads at MAX_UPLOAD_BYTES
    # (see routes/upload.py) for exactly this reason. A deployed generated
    # app has no equivalent: one oversized request can consume unbounded
    # memory building the body before FastAPI/Pydantic ever gets a chance
    # to validate or reject it. Rejecting outright on a declared
    # Content-Length over the limit, before the body is read at all, is
    # the same "reject before reading" approach MAX_UPLOAD_BYTES already
    # uses. Matches the NOTEBOOK_API_* env-var convention this generated
    # app's other limits (API_KEYS, TASK_TTL_SECONDS, ALLOWED_ORIGINS)
    # already follow, defaulting to the same 10MB MAX_UPLOAD_BYTES already
    # defaults to.
    lines.append(
        'MAX_REQUEST_BODY_BYTES = int(os.getenv('
        '"NOTEBOOK_API_MAX_REQUEST_BYTES", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_MAX_REQUEST_BYTES")}"'
        '))'
    )
    lines.append("")
    lines.append("class MaxRequestBodySizeMiddleware:")
    lines.append("    def __init__(self, app):")
    lines.append("        self.app = app")
    lines.append("")
    lines.append("    async def __call__(self, scope, receive, send):")
    lines.append("        if scope['type'] == 'http':")
    lines.append("            for name, value in scope.get('headers') or []:")
    lines.append("                if name != b'content-length':")
    lines.append("                    continue")
    lines.append("                try:")
    lines.append("                    too_large = int(value) > MAX_REQUEST_BODY_BYTES")
    lines.append("                except ValueError:")
    lines.append("                    break")
    lines.append("                if too_large:")
    lines.append("                    response = JSONResponse(")
    lines.append("                        {")
    lines.append("                            'detail': (")
    lines.append("                                'Request body exceeds the maximum allowed '")
    lines.append("                                f'size of {MAX_REQUEST_BODY_BYTES} bytes'")
    lines.append("                            )")
    lines.append("                        },")
    lines.append("                        status_code=413,")
    lines.append("                    )")
    lines.append("                    await response(scope, receive, send)")
    lines.append("                    return")
    lines.append("                break")
    lines.append("        await self.app(scope, receive, send)")
    lines.append("")
    lines.append("app.add_middleware(MaxRequestBodySizeMiddleware)")
    lines.append("")
    # Registered last (see MaxRequestBodySizeMiddleware/CORSMiddleware
    # above -- the same "middleware added last ends up outermost, since
    # Starlette's own add_middleware inserts each new one at the *front*
    # of its internal list and then wraps outward-in over that list in
    # reverse" rule those already rely on) so these headers land on
    # *every* response this app ever sends, including a 413 from
    # MaxRequestBodySizeMiddleware or a 429/401 HTTPException -- not just
    # the successful ones a handler-level fix would only ever reach.
    # Baseline OWASP-recommended hardening with no functional downside
    # (unlike CORS/rate limiting, nothing here can reject a legitimate
    # request), so -- unlike NOTEBOOK_API_DISABLE_DOCS/ALLOWED_ORIGINS/
    # RATE_LIMIT_PER_MINUTE above -- these are unconditional, not gated
    # behind an env var an operator has to remember to set. Every
    # response from this app is JSON (or, unless NOTEBOOK_API_DISABLE_DOCS
    # is set, the /docs Swagger UI's own HTML), never content meant to be
    # framed or MIME-sniffed by a browser: X-Content-Type-Options blocks a
    # browser from ever guessing a response is something other than what
    # Content-Type already says it is (relevant here since notebook-
    # author-controlled strings -- docstrings, example payloads -- flow
    # straight into response bodies), X-Frame-Options blocks embedding any
    # response (including /docs itself) in a third-party <iframe>, and
    # Referrer-Policy stops this deployment's own URL (which can itself
    # carry sensitive path segments, e.g. a task_id) from leaking into the
    # Referer header of a request /docs' own "Try it out" -- or any link a
    # response body might contain -- makes to a different origin.
    lines.append("@app.middleware('http')")
    lines.append("async def _add_security_headers(request, call_next):")
    lines.append("    response = await call_next(request)")
    lines.append("    response.headers['X-Content-Type-Options'] = 'nosniff'")
    lines.append("    response.headers['X-Frame-Options'] = 'DENY'")
    lines.append("    response.headers['Referrer-Policy'] = 'no-referrer'")
    lines.append("    return response")
    lines.append("")
    # Registered last -- see _add_security_headers' own comment above for
    # why that makes this the outermost middleware -- so it compresses
    # the truly final response body (headers/status already finalized by
    # every layer above), rather than something an inner layer might
    # still rewrite. GZipMiddleware only compresses when the client's own
    # Accept-Encoding actually says it can decode gzip, so this changes
    # nothing for a caller that doesn't ask for it; for one that does, a
    # large JSON response (GET /tasks -- still up to 100 entries per page
    # even after pagination, GET /openapi.json, a notebook function
    # returning a large result) previously always went out uncompressed,
    # a real bandwidth/latency cost on any deployment reached over a slow
    # or metered link that this app had no way to avoid short of a
    # reverse proxy in front of it doing the compression itself.
    # Starlette's own default minimum_size (500 bytes) is left as-is --
    # below that, gzip's own framing overhead can make a compressed
    # response larger than the original.
    lines.append("app.add_middleware(GZipMiddleware)")
    lines.append("")
    # Registered last -- see _add_security_headers' own comment above for
    # why that makes this the outermost middleware -- so the timer spans
    # every other layer too (rate limiting, gzip compression, the
    # endpoint itself), reporting what a real client actually
    # experienced, not just handler time. Before this, this app gave an
    # operator no way to see per-request latency short of instrumenting
    # it externally (a reverse proxy's own access log, an APM agent) --
    # every other operational signal this app exposes (uptime, task
    # counts, rate-limit state) was already free via GET /metrics or
    # response headers, but response latency itself had no equivalent.
    lines.append("@app.middleware('http')")
    lines.append("async def _add_process_time_header(request, call_next):")
    lines.append("    start_time = time.perf_counter()")
    lines.append("    response = await call_next(request)")
    lines.append(
        "    response.headers['X-Process-Time-Ms'] = "
        "f'{(time.perf_counter() - start_time) * 1000:.2f}'"
    )
    lines.append("    return response")
    lines.append("")
    # Registered last -- outermost, wrapping X-Process-Time-Ms above --
    # so this request's own id is stamped even on a response one of the
    # earlier layers short-circuits (a 429 from rate limiting, a 413 from
    # MaxRequestBodySizeMiddleware). Honors a caller-supplied X-Request-ID
    # (e.g. from an upstream gateway that already assigns one to
    # correlate a single logical request across several downstream
    # services) instead of always minting a fresh one, so a trace started
    # upstream doesn't fork into two disconnected ids the moment it
    # reaches this app; only generates a new uuid4 when the caller didn't
    # send one at all. Before this, correlating "which request logged
    # this error" between a caller's own logs and this app's -- e.g. to
    # investigate a specific failed call reported after the fact, with no
    # other identifying information -- had no shared id to search by at
    # all.
    lines.append("@app.middleware('http')")
    lines.append("async def _add_request_id_header(request, call_next):")
    lines.append(
        "    request_id = request.headers.get('X-Request-ID') or "
        "str(uuid.uuid4())"
    )
    lines.append("    response = await call_next(request)")
    lines.append("    response.headers['X-Request-ID'] = request_id")
    lines.append("    return response")
    lines.append("")
    # Simple in‑memory task registry used by background endpoints
    lines.append("TASKS = {}")
    lines.append(
        'TASK_TTL_SECONDS = int(os.getenv('
        '"NOTEBOOK_API_TASK_TTL_SECONDS", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_TASK_TTL_SECONDS")}"'
        '))'
    )
    # _evict_expired_tasks bounds TASKS' *long-term* growth (nothing
    # older than TASK_TTL_SECONDS survives), but a burst of background
    # requests arriving faster than that TTL still grows TASKS without
    # limit in the meantime -- eviction alone doesn't stop a client
    # (malicious, or just a retry loop against a stuck deploy) from
    # submitting far more tasks than this process could ever actually
    # get to, exhausting memory well before any of them would expire.
    # Matches the same NOTEBOOK_API_* convention this generated app's
    # other limits (API_KEYS, TASK_TTL_SECONDS, MAX_REQUEST_BODY_BYTES)
    # already follow.
    lines.append(
        'MAX_PENDING_TASKS = int(os.getenv('
        '"NOTEBOOK_API_MAX_TASKS", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_MAX_TASKS")}"'
        '))'
    )
    # Bounds _run_background_task's own single execution below (see its
    # own docstring) -- 0 (the default) disables this entirely, the
    # identical "0 means off, preserving the previous unbounded behavior"
    # convention RATE_LIMIT_PER_MINUTE/MAX_PENDING_TASKS's own defaults
    # already follow.
    lines.append(
        'TASK_EXECUTION_TIMEOUT_SECONDS = int(os.getenv('
        '"NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS")}"'
        '))'
    )
    # Bounds _deliver_task_webhook's own single delivery attempt below --
    # a caller-supplied ?callback_url= pointing at a slow or unresponsive
    # endpoint must never be allowed to tie up a worker thread (and, by
    # extension, this process' limited thread pool) indefinitely.
    lines.append(
        'WEBHOOK_TIMEOUT_SECONDS = int(os.getenv('
        '"NOTEBOOK_API_WEBHOOK_TIMEOUT_SECONDS", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_WEBHOOK_TIMEOUT_SECONDS")}"'
        '))'
    )
    # Read by _deliver_task_webhook below to sign the webhook body with
    # HMAC-SHA256 (the same X-Hub-Signature-256 contract GitHub/Stripe
    # webhooks already use) -- empty (the default) sends the webhook
    # unsigned, exactly as before this existed. A caller-supplied
    # ?callback_url= is, by definition, a URL the caller themselves
    # chose to receive this app's own task results at, but it's still
    # commonly a public endpoint reachable by anyone who learns it (a
    # webhook.site-style debugging URL, or a real endpoint whose path
    # alone isn't a secret) -- without a signature, that endpoint's own
    # handler has no way to tell a request that actually came from this
    # app apart from one an attacker crafted by hand with a guessed or
    # leaked task_id/result.
    lines.append(
        'WEBHOOK_SECRET = os.getenv('
        '"NOTEBOOK_API_WEBHOOK_SECRET", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_WEBHOOK_SECRET")}"'
        ')'
    )
    lines.append(
        '# A comma-separated list, not a single value, so a key can be'
    )
    lines.append(
        '# rotated with zero downtime: add the new key alongside the old'
    )
    lines.append(
        '# one, restart, let clients switch over, then remove the old key'
    )
    lines.append(
        '# and restart again -- requests are never rejected mid-rotation.'
    )
    lines.append(
        'API_KEYS = tuple('
        'k.strip() for k in os.getenv('
        '"NOTEBOOK_API_KEY", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_KEY")}"'
        ').split(",") if k.strip()'
        ')'
    )
    lines.append("API_KEY_HEADER_NAME = 'X-API-Key'")
    lines.append("")
    # Tracked per API key (not globally, and not per-IP): a shared global
    # counter would let one heavy, legitimate key starve every other
    # key's own quota, and this app has no reliable notion of client
    # identity below the API key layer anyway (a proxy/load balancer in
    # front of it can make every request appear to come from the same
    # IP). API_KEYS is a small, fixed set fully known at startup, so
    # _RATE_LIMIT_WINDOWS below never grows past len(API_KEYS) entries --
    # unlike TASKS (which needs its own TTL-based eviction above), a
    # matching eviction scheme isn't needed here.
    lines.append(
        'RATE_LIMIT_PER_MINUTE = int(os.getenv('
        '"NOTEBOOK_API_RATE_LIMIT_PER_MINUTE", '
        f'"{_generated_app_env_var_default("NOTEBOOK_API_RATE_LIMIT_PER_MINUTE")}"'
        '))'
    )
    lines.append("RATE_LIMIT_WINDOW_SECONDS = 60")
    lines.append("_RATE_LIMIT_WINDOWS = {}")
    lines.append("")
    lines.append("def _enforce_rate_limit(api_key, response):")
    lines.append("    # RATE_LIMIT_PER_MINUTE <= 0 (the default) means rate")
    lines.append("    # limiting is disabled entirely -- no window is even")
    lines.append("    # tracked, so this is a no-op on the hot path for every")
    lines.append("    # deployment that never opts into it.")
    lines.append("    if RATE_LIMIT_PER_MINUTE <= 0:")
    lines.append("        return")
    lines.append("    now = time.time()")
    lines.append("    window_start, count = _RATE_LIMIT_WINDOWS.get(api_key, (now, 0))")
    lines.append("    # Fixed window, not sliding: once RATE_LIMIT_WINDOW_SECONDS")
    lines.append("    # has elapsed since this key's window opened, it resets to a")
    lines.append("    # fresh window rather than decaying the count gradually --")
    lines.append("    # the same lazy, no-background-thread eviction style")
    lines.append("    # _evict_expired_tasks above already uses for TASKS.")
    lines.append("    if now - window_start >= RATE_LIMIT_WINDOW_SECONDS:")
    lines.append("        window_start, count = now, 0")
    lines.append("    count += 1")
    lines.append("    _RATE_LIMIT_WINDOWS[api_key] = (window_start, count)")
    lines.append("    reset_at = int(window_start + RATE_LIMIT_WINDOW_SECONDS)")
    lines.append("    remaining = max(0, RATE_LIMIT_PER_MINUTE - count)")
    lines.append("    # Set on every rate-limited request, not just a 429 -- the")
    lines.append("    # standard client contract (GitHub/Stripe/...) these three")
    lines.append("    # headers follow lets a well-behaved caller see it's about to")
    lines.append("    # be throttled (a low/zero Remaining) and back off on its own,")
    lines.append("    # rather than the only previous signal being a 429 it's")
    lines.append("    # already received. `response` is the actual Response FastAPI")
    lines.append("    # is about to send back -- injecting it into this dependency")
    lines.append("    # (see verify_api_key below) rather than building a separate")
    lines.append("    # Response of its own is the documented way to mutate headers")
    lines.append("    # on a request that succeeds; it plays no part when this")
    lines.append("    # raises below; instead, the 429 branch attaches the same")
    lines.append("    # three headers directly to the HTTPException itself.")
    lines.append("    response.headers['X-RateLimit-Limit'] = str(RATE_LIMIT_PER_MINUTE)")
    lines.append("    response.headers['X-RateLimit-Remaining'] = str(remaining)")
    lines.append("    response.headers['X-RateLimit-Reset'] = str(reset_at)")
    lines.append("    if count > RATE_LIMIT_PER_MINUTE:")
    lines.append("        retry_after = max(")
    lines.append("            1, int(RATE_LIMIT_WINDOW_SECONDS - (now - window_start))")
    lines.append("        )")
    lines.append("        raise HTTPException(")
    lines.append("            status_code=429,")
    lines.append("            detail=(")
    lines.append("                f'Rate limit exceeded: {RATE_LIMIT_PER_MINUTE} '")
    lines.append("                f'requests per {RATE_LIMIT_WINDOW_SECONDS}s per API key'")
    lines.append("            ),")
    lines.append("            headers={")
    lines.append("                'Retry-After': str(retry_after),")
    lines.append("                'X-RateLimit-Limit': str(RATE_LIMIT_PER_MINUTE),")
    lines.append("                'X-RateLimit-Remaining': '0',")
    lines.append("                'X-RateLimit-Reset': str(reset_at),")
    lines.append("            },")
    lines.append("        )")
    lines.append("")
    lines.append("def _evict_expired_tasks():")
    lines.append("    # TASKS is an in-memory dict with no automatic eviction anywhere")
    lines.append("    # else in this app -- without this, a long-running deployment")
    lines.append("    # handling steady background-task traffic accumulates one entry")
    lines.append("    # per call forever, growing memory usage without bound. Called")
    lines.append("    # opportunistically on every new task's creation (lazy expiry)")
    lines.append("    # rather than a periodic background loop, so it needs no extra")
    lines.append("    # scheduler/thread and behaves the same whether or not anything")
    lines.append("    # ever polls /tasks.")
    lines.append("    #")
    lines.append("    # A task still 'processing' is never evicted here, no matter how")
    lines.append("    # old its created_at is. TASK_TTL_SECONDS bounds how long a")
    lines.append("    # *finished* task's own result lingers in memory -- it was never")
    lines.append("    # meant to be a deadline on how long the underlying notebook")
    lines.append("    # function itself is allowed to run. Confirmed exploitable before")
    lines.append("    # this: a background task (train/process/generate/embed/scrape --")
    lines.append("    # routinely slow, long-running work by design) that took longer")
    lines.append("    # than TASK_TTL_SECONDS to finish had its own TASKS entry evicted")
    lines.append("    # by this exact sweep while still running, out from under it --")
    lines.append("    # so _run_background_task's own eventual")
    lines.append("    # TASKS[task_id][\"status\"] = ... write (see below) raised a bare")
    lines.append("    # KeyError, an unhandled exception in a fire-and-forget asyncio")
    lines.append("    # task that's silently swallowed (logged, at best, as an opaque")
    lines.append("    # 'Task exception was never retrieved'). The task's real result")
    lines.append("    # (or error) was lost forever, and a caller polling GET")
    lines.append("    # /tasks/{task_id} for it saw a plain 404 instead -- indistinguishable")
    lines.append("    # from a task_id that never existed at all.")
    lines.append("    now = time.time()")
    lines.append("    expired_ids = [")
    lines.append("        task_id")
    lines.append("        for task_id, task in TASKS.items()")
    lines.append("        if task.get('status') != 'processing'")
    lines.append("        and now - task.get('created_at', now) > TASK_TTL_SECONDS")
    lines.append("    ]")
    lines.append("    for task_id in expired_ids:")
    lines.append("        TASKS.pop(task_id, None)")
    lines.append("")
    lines.append("def verify_api_key(response: Response, x_api_key: str = Header(None)):")
    lines.append("    # hmac.compare_digest instead of != : a plain string")
    lines.append("    # comparison short-circuits on the first differing byte, which")
    lines.append("    # makes response time leak how many leading characters of a")
    lines.append("    # guess were correct -- a classic timing side-channel for")
    lines.append("    # guessing the key byte by byte. Checked against every")
    lines.append("    # configured key (not just the first) so rotation doesn't")
    lines.append("    # reintroduce that leak by short-circuiting once a candidate")
    lines.append("    # key's own length/prefix happens to fail fast.")
    lines.append("    if x_api_key is None or not any(")
    lines.append("        hmac.compare_digest(x_api_key, key) for key in API_KEYS")
    lines.append("    ):")
    lines.append("        raise HTTPException(")
    lines.append("            status_code=401,")
    lines.append("            detail='Invalid API key'")
    lines.append("        )")
    lines.append("    # Rate limiting only ever applies once a request has already")
    lines.append("    # authenticated as a specific key -- an invalid/missing key")
    lines.append("    # already gets rejected with 401 above, before it could consume")
    lines.append("    # any key's own quota. x_api_key itself (not a separately")
    lines.append("    # re-matched entry from API_KEYS) is the right identity to key")
    lines.append("    # on: the any(...) check above already proved it's exactly")
    lines.append("    # equal to one of them.")
    lines.append("    _enforce_rate_limit(x_api_key, response)")
    lines.append("")
    lines.append("from fastapi.openapi.utils import get_openapi")
    lines.append("")
    lines.append("def custom_openapi():")
    lines.append("    if app.openapi_schema:")
    lines.append("        return app.openapi_schema")
    lines.append("")
    # servers=app.servers -- confirmed missing before this fix: FastAPI's
    # own app.openapi() would include the servers=[...] this app's own
    # constructor call above passes it automatically, but app.openapi is
    # overridden with this function entirely (see app.openapi =
    # custom_openapi below), and get_openapi() only ever returns what a
    # caller explicitly asks it to build. Without this line, the
    # PUBLIC_URL constructor kwarg above was silently discarded --
    # confirmed live: app.openapi()["servers"] was never even a key in
    # the resulting schema, let alone reflecting PUBLIC_URL -- so every
    # compiled app's own /docs (Swagger UI) had no configured servers
    # entry at all, no matter what NOTEBOOK_API_PUBLIC_URL was set to.
    lines.append("    openapi_schema = get_openapi(")
    lines.append("        title=app.title,")
    lines.append("        version=app.version,")
    lines.append("        description=app.description,")
    lines.append("        routes=app.routes,")
    lines.append("        servers=app.servers,")
    lines.append("    )")
    lines.append("")
    lines.append("    openapi_schema.setdefault('components', {})")
    lines.append("    openapi_schema['components'].setdefault('securitySchemes', {})")
    lines.append("")
    lines.append("    openapi_schema['components']['securitySchemes']['ApiKeyAuth'] = {")
    lines.append("        'type': 'apiKey',")
    lines.append("        'in': 'header',")
    lines.append("        'name': API_KEY_HEADER_NAME")
    lines.append("    }")
    lines.append("")
    lines.append("    app.openapi_schema = openapi_schema")
    lines.append("    return app.openapi_schema")
    lines.append("")
    lines.append("app.openapi = custom_openapi")
    lines.append("")
    lines.append("START_TIME = time.time()")
    lines.append(
        "GENERATED_AT = datetime.utcnow().isoformat() + 'Z'"
    )
    lines.append(
        "PYTHON_VERSION = sys.version.split()[0]"
    )
    lines.append(
        f"SOURCE_NOTEBOOK_SHA256 = {source_notebook_sha256!r}"
    )
    lines.append(
        f"NOTEBOOK_TO_API_VERSION = {notebook_to_api_version!r}"
    )
    lines.append("")
    protected_endpoint_count = len(functions)
    endpoint_list = [
        f"/{func['name']}"
        for func in functions
    ]
    total_generated_endpoint_count = len(endpoint_list)
    background_endpoint_count = sum(
        1
        for func in functions
        if any(
            kw in func["name"].lower()
            for kw in LONG_RUNNING_KEYWORDS
        )
    )
    lines.append("# Public infrastructure endpoints")
    lines.append("@app.get('/')")
    lines.append("def root():")
    lines.append("    return {")
    lines.append("        'service': 'Notebook-to-API Generated Service',")
    lines.append("        'generator': 'notebook-to-api',")
    lines.append("        'generator_version': NOTEBOOK_TO_API_VERSION,")
    lines.append(
        "        'generated_at': GENERATED_AT,"
    )
    lines.append(
        "        'python_version': PYTHON_VERSION,"
    )

    lines.append(
        "        'framework': 'FastAPI',"
    )
    lines.append(
        "        'background_task_support': True,"
    )

    lines.append(
        f"        'background_endpoint_count': {background_endpoint_count},"
    )
    lines.append(
        "        'available_features': ["
    )

    lines.append(
        "            'authentication',"
    )

    lines.append(
        "            'background_tasks',"
    )

    lines.append(
        "            'openapi_docs',"
    )

    lines.append(
        "            'metrics',"
    )

    lines.append(
        "            'task_monitoring',"
    )

    lines.append(
        "            'health_checks'"
    )

    lines.append(
        "        ],"
    )
    lines.append("        'documentation': {")
    lines.append("            'swagger': '/docs',")
    lines.append("            'openapi': '/openapi.json',")
    lines.append("            'redoc': '/redoc'")
    lines.append("        },")
    lines.append("        'operations': {")
    lines.append("            'health': '/health',")
    lines.append("            'ready': '/ready',")
    lines.append("            'info': '/info',")
    lines.append("            'config': '/config',")
    lines.append("            'metrics': '/metrics',")
    lines.append("            'uptime': '/uptime'")
    lines.append("        },")
    lines.append("        'task_management': {")
    lines.append("            'list': '/tasks',")
    lines.append("            'metrics': '/metrics',")
    lines.append("            'cleanup': '/tasks/cleanup',")
    lines.append("            'reset': '/tasks/reset'")
    lines.append("        },")
    lines.append("        'authentication': {")
    lines.append("            'status': '/auth/status',")
    lines.append("            'info': '/auth/info',")
    lines.append("            'validate': '/auth/validate'")
    lines.append("        },")
    lines.append(
        f"        'endpoint_count': {total_generated_endpoint_count},"
    )
    lines.append(
        f"        'protected_endpoints': {protected_endpoint_count},"
    )

    lines.append(
        f"        'sample_endpoints': {repr(endpoint_list[:10])}"
    )
    lines.append("    }")
    lines.append("")
    lines.append("@app.get('/health')")
    lines.append("def health_check():")
    lines.append("    return {'status': 'healthy'}")
    lines.append("")
    lines.append("@app.get('/ready')")
    lines.append("def readiness_check():")

    lines.append("    return {")
    lines.append("        'status': 'ready',")
    lines.append("        'tasks_registered': len(TASKS)")
    lines.append("    }")

    lines.append("")
    lines.append("@app.get('/auth/status')")
    lines.append("def auth_status():")

    lines.append("    return {")
    lines.append("        'authentication': 'enabled',")
    lines.append("        'api_key_configured': bool(API_KEYS)")
    lines.append("    }")

    lines.append("")
    lines.append("@app.get('/auth/info')")
    lines.append("def auth_info():")

    lines.append("    return {")
    lines.append("        'authentication': 'api_key',")
    lines.append("        'header': API_KEY_HEADER_NAME,")
    lines.append("        'environment_variable': 'NOTEBOOK_API_KEY',")
    lines.append("        'rate_limiting': RATE_LIMIT_PER_MINUTE > 0,")
    lines.append("        'rate_limit_per_minute': RATE_LIMIT_PER_MINUTE or None,")
    lines.append("        'key_rotation': True,")
    lines.append("        'configured_keys': len(API_KEYS),")
    lines.append(
        f"        'protected_endpoints': {protected_endpoint_count}"
    )
    lines.append("    }")

    lines.append("")
    lines.append("@app.get('/auth/validate')")
    lines.append("def validate_auth(_: None = Depends(verify_api_key)):")

    lines.append("    return {")
    lines.append("        'authenticated': True")
    lines.append("    }")

    lines.append("")
    lines.append("@app.get('/info')")
    lines.append("def service_info():")
    lines.append("    return {")
    lines.append('        "service": "Notebook-to-API Generated Service",')
    lines.append('        "version": NOTEBOOK_TO_API_VERSION,')
    lines.append('        "status": "running",')
    lines.append(f'        "endpoints": {repr(endpoint_list)},')
    lines.append(f'        "endpoint_count": {len(endpoint_list)},')
    lines.append(f'        "background_endpoint_count": {background_endpoint_count},')
    lines.append('        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,')
    lines.append('        "authentication": {')
    lines.append('            "enabled": True,')
    lines.append('            "type": "api_key"')
    lines.append('        }')
    lines.append("    }")
    lines.append("")
    # Every NOTEBOOK_API_* limit this app enforces (MAX_REQUEST_BODY_BYTES,
    # TASK_TTL_SECONDS, MAX_PENDING_TASKS, RATE_LIMIT_PER_MINUTE,
    # ALLOWED_ORIGINS, DISABLE_DOCS, PUBLIC_URL) was previously only
    # discoverable by reading the deployment's own environment directly --
    # shell access to the container, or knowledge of what was passed to
    # `docker run -e ...` -- with /auth/info's own "rate_limiting"/
    # "rate_limit_per_minute" the sole exception. An operator (or a
    # caller building a client that wants to size its own retries/backoff
    # against MAX_REQUEST_BODY_BYTES/RATE_LIMIT_PER_MINUTE without a
    # separate 413/429 round trip first) had no way to just ask the
    # running app what it's actually configured with -- the same gap GET
    # /api/config already closes for this dashboard's own configuration
    # (see routes/upload.py), just never given an equivalent here. No
    # secrets here (API_KEYS' own values are deliberately never
    # returned), so -- like /info/auth/status/auth/info above -- this
    # needs no authentication of its own.
    lines.append("@app.get('/config')")
    lines.append("def service_config():")
    lines.append("    return {")
    lines.append("        'max_request_body_bytes': MAX_REQUEST_BODY_BYTES,")
    lines.append("        'task_ttl_seconds': TASK_TTL_SECONDS,")
    lines.append("        'max_pending_tasks': MAX_PENDING_TASKS,")
    lines.append(
        "        'task_execution_timeout_seconds': "
        "TASK_EXECUTION_TIMEOUT_SECONDS or None,"
    )
    lines.append("        'webhook_timeout_seconds': WEBHOOK_TIMEOUT_SECONDS,")
    lines.append("        'webhook_signing_enabled': bool(WEBHOOK_SECRET),")
    lines.append("        'rate_limit_per_minute': RATE_LIMIT_PER_MINUTE or None,")
    lines.append("        'allowed_origins': ALLOWED_ORIGINS,")
    lines.append("        'disable_docs': DISABLE_DOCS,")
    lines.append("        'public_url': PUBLIC_URL,")
    lines.append("    }")
    lines.append("")
    lines.append("@app.get('/tasks')")
    lines.append("def list_tasks(")
    lines.append("    status: Optional[str] = None,")
    lines.append("    limit: int = Query(default=100, ge=1, le=1000),")
    lines.append("    offset: int = Query(default=0, ge=0),")
    lines.append("    _: None = Depends(verify_api_key),")
    lines.append("):")

    lines.append("    valid_statuses = ('processing', 'completed', 'failed')")
    lines.append("    if status is not None and status not in valid_statuses:")
    lines.append("        raise HTTPException(")
    lines.append("            status_code=400,")
    lines.append(
        "            detail=f\"Invalid status '{status}'; must be one of: "
        "{', '.join(valid_statuses)}\""
    )
    lines.append("        )")
    lines.append("")

    lines.append("    completed_tasks = sum(")
    lines.append("        1")
    lines.append("        for task in TASKS.values()")
    lines.append("        if task.get('status') == 'completed'")
    lines.append("    )")

    lines.append("    failed_tasks = sum(")
    lines.append("        1")
    lines.append("        for task in TASKS.values()")
    lines.append("        if task.get('status') == 'failed'")
    lines.append("    )")

    lines.append("    processing_tasks = sum(")
    lines.append("        1")
    lines.append("        for task in TASKS.values()")
    lines.append("        if task.get('status') == 'processing'")
    lines.append("    )")

    lines.append("    matching_items = [")
    lines.append("        (task_id, task)")
    lines.append("        for task_id, task in TASKS.items()")
    lines.append("        if status is None or task.get('status') == status")
    lines.append("    ]")
    lines.append(
        "    matching_items.sort("
        "key=lambda item: item[1].get('created_at', 0), reverse=True)"
    )
    lines.append("    page_items = matching_items[offset:offset + limit]")

    lines.append("    return {")
    lines.append("        'active_tasks': len(TASKS),")
    lines.append("        'processing_tasks': processing_tasks,")
    lines.append("        'completed_tasks': completed_tasks,")
    lines.append("        'failed_tasks': failed_tasks,")
    lines.append("        'matching_tasks': len(matching_items),")
    lines.append("        'limit': limit,")
    lines.append("        'offset': offset,")
    lines.append("        'tasks': dict(page_items)")
    lines.append("    }")
    lines.append("")
    lines.append("@app.get('/tasks/{task_id}')")
    lines.append("def get_task(task_id: str, _: None = Depends(verify_api_key)):")
    lines.append("    task = TASKS.get(task_id)")
    lines.append("")
    lines.append("    if not task:")
    lines.append("        raise HTTPException(")
    lines.append("            status_code=404,")
    lines.append("            detail=f'Task {task_id} not found'")
    lines.append("        )")
    lines.append("")
    lines.append("    return task")
    lines.append("")
    lines.append("@app.delete('/tasks/completed')")
    lines.append("def delete_completed_tasks(_: None = Depends(verify_api_key)):")

    lines.append("    completed_task_ids = [")
    lines.append("        task_id")
    lines.append("        for task_id, task in TASKS.items()")
    lines.append("        if task.get('status') == 'completed'")
    lines.append("    ]")

    lines.append("    for task_id in completed_task_ids:")
    lines.append("        TASKS.pop(task_id, None)")

    lines.append("    return {")
    lines.append("        'deleted': len(completed_task_ids),")
    lines.append("        'remaining_tasks': len(TASKS)")
    lines.append("    }")

    lines.append("")
    lines.append("@app.delete('/tasks/failed')")
    lines.append("def delete_failed_tasks(_: None = Depends(verify_api_key)):")

    lines.append("    failed_task_ids = [")
    lines.append("        task_id")
    lines.append("        for task_id, task in TASKS.items()")
    lines.append("        if task.get('status') == 'failed'")
    lines.append("    ]")

    lines.append("    for task_id in failed_task_ids:")
    lines.append("        TASKS.pop(task_id, None)")

    lines.append("    return {")
    lines.append("        'deleted': len(failed_task_ids),")
    lines.append("        'remaining_tasks': len(TASKS)")
    lines.append("    }")

    lines.append("")
    lines.append("@app.post('/tasks/cleanup')")
    lines.append("def cleanup_tasks(_: None = Depends(verify_api_key)):")

    lines.append("    completed_deleted = 0")
    lines.append("    failed_deleted = 0")

    lines.append("    task_ids = list(TASKS.keys())")

    lines.append("    for task_id in task_ids:")
    lines.append("        status = TASKS[task_id].get('status')")

    lines.append("        if status == 'completed':")
    lines.append("            TASKS.pop(task_id, None)")
    lines.append("            completed_deleted += 1")

    lines.append("        elif status == 'failed':")
    lines.append("            TASKS.pop(task_id, None)")
    lines.append("            failed_deleted += 1")

    lines.append("    return {")
    lines.append("        'completed_deleted': completed_deleted,")
    lines.append("        'failed_deleted': failed_deleted,")
    lines.append("        'remaining_tasks': len(TASKS)")
    lines.append("    }")

    lines.append("")
    lines.append("@app.get('/metrics')")
    lines.append("def metrics():")

    lines.append("    processing = sum(")
    lines.append("        1")
    lines.append("        for task in TASKS.values()")
    lines.append("        if task.get('status') == 'processing'")
    lines.append("    )")

    lines.append("    completed = sum(")
    lines.append("        1")
    lines.append("        for task in TASKS.values()")
    lines.append("        if task.get('status') == 'completed'")
    lines.append("    )")

    lines.append("    failed = sum(")
    lines.append("        1")
    lines.append("        for task in TASKS.values()")
    lines.append("        if task.get('status') == 'failed'")
    lines.append("    )")

    lines.append("    return {")
    lines.append("        'total_tasks': len(TASKS),")
    lines.append("        'processing': processing,")
    lines.append("        'completed': completed,")
    lines.append("        'failed': failed")
    lines.append("    }")

    lines.append("")
    lines.append("@app.get('/uptime')")
    lines.append("def uptime():")

    lines.append("    return {")
    lines.append("        'uptime_seconds': int(time.time() - START_TIME)")
    lines.append("    }")

    lines.append("")
    lines.append("@app.post('/tasks/reset')")
    lines.append("def reset_tasks(_: None = Depends(verify_api_key)):")

    lines.append("    deleted_tasks = len(TASKS)")

    lines.append("    TASKS.clear()")

    lines.append("    return {")
    lines.append("        'deleted_tasks': deleted_tasks")
    lines.append("    }")

    lines.append("")
    lines.append("@app.delete('/tasks/{task_id}')")
    lines.append("def delete_task(task_id: str, _: None = Depends(verify_api_key)):")

    lines.append("    task = TASKS.get(task_id)")
    lines.append("")
    lines.append("    if task is None:")
    lines.append("        raise HTTPException(")
    lines.append("            status_code=404,")
    lines.append("            detail=f'Task {task_id} not found'")
    lines.append("        )")
    lines.append("")
    # Confirmed exploitable before this: deleting a still-processing task
    # popped its TASKS entry immediately, but the background task itself
    # kept running -- there is no way to actually cancel work already
    # handed to anyio.to_thread.run_sync/asyncio. When it eventually
    # finished, _run_background_task's own TASKS[task_id][...] write (see
    # below) raised a bare KeyError against the now-missing entry, an
    # unhandled exception in a fire-and-forget asyncio task that's
    # silently swallowed rather than surfaced anywhere -- permanently
    # losing that task's real result or error with nothing to show for
    # it. Rejecting the delete outright while a task is still processing
    # (mirroring the 503 MAX_PENDING_TASKS already returns for "try again
    # once some have completed") gives a caller an actionable answer
    # instead of a delete that "succeeds" while quietly corrupting the
    # task it just claimed to remove.
    lines.append("    if task.get('status') == 'processing':")
    lines.append("        raise HTTPException(")
    lines.append("            status_code=409,")
    lines.append("            detail=(")
    lines.append(
        "                f'Task {task_id} is still processing and cannot be '"
    )
    lines.append(
        "                'deleted -- wait for it to complete or fail first'"
    )
    lines.append("            ),")
    lines.append("        )")
    lines.append("")
    lines.append("    deleted_task = TASKS.pop(task_id)")

    lines.append("    return {")
    lines.append("        'message': 'Task deleted',")
    lines.append("        'task_id': task_id,")
    lines.append("        'status': deleted_task.get('status')")
    lines.append("    }")

    lines.append("")
    # Delivers a single best-effort POST of `payload` (the finished task's
    # own TASKS record: status/result, or status/error) to `callback_url`,
    # so a caller can opt out of polling get_task/wait_for_task for a
    # background task's completion. Deliberately synchronous (urllib, not
    # an async HTTP client) -- called only from inside
    # anyio.to_thread.run_sync below, the same worker-thread pattern a
    # plain (non-async) notebook function itself already runs under, for
    # the identical reason: it must never block this app's single event
    # loop for up to WEBHOOK_TIMEOUT_SECONDS waiting on a caller-controlled
    # endpoint that might be slow or unresponsive. Any failure (a DNS
    # failure, connection refused, a non-2xx response, a timeout) is
    # swallowed here, not raised -- delivery is purely a convenience on
    # top of the task's own real result, which is already durably recorded
    # in TASKS by the time this is ever called; a caller who needs a
    # guarantee should poll get_task/wait_for_task instead.
    #
    # When WEBHOOK_SECRET is configured, the request also carries an
    # X-Webhook-Signature: sha256=<hex hmac> header -- computed over the
    # exact same `body` bytes being sent, using hmac.compare_digest's own
    # module (already imported above for API-key comparison) rather than
    # a second, separate crypto dependency. digestmod is passed as the
    # plain string "sha256" specifically so this needs no `import
    # hashlib` of its own: hmac.new resolves a string digestmod via
    # hashlib.new internally. A receiving endpoint recomputes the same
    # HMAC over the raw body it received (using the identical shared
    # secret) and compares it against this header -- via
    # hmac.compare_digest, never `==`, for the same timing-attack reason
    # verify_api_key below already uses it -- to confirm both that the
    # request actually came from this app and that the body wasn't
    # altered in transit, the same X-Hub-Signature-256 contract GitHub/
    # Stripe webhooks already use. Omitted entirely when WEBHOOK_SECRET
    # is empty (the default), so an existing receiver that predates this
    # feature keeps working unmodified.
    lines.append("def _deliver_task_webhook(callback_url, payload):")
    lines.append("    try:")
    lines.append("        body = json.dumps(payload).encode('utf-8')")
    lines.append("        headers = {'Content-Type': 'application/json'}")
    lines.append("        if WEBHOOK_SECRET:")
    lines.append("            signature = hmac.new(")
    lines.append(
        "                WEBHOOK_SECRET.encode('utf-8'), body, 'sha256'"
    )
    lines.append("            ).hexdigest()")
    lines.append(
        "            headers['X-Webhook-Signature'] = f'sha256={signature}'"
    )
    lines.append("        request = urllib.request.Request(")
    lines.append("            callback_url,")
    lines.append("            data=body,")
    lines.append("            headers=headers,")
    lines.append("            method='POST',")
    lines.append("        )")
    lines.append(
        "        urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT_SECONDS).close()"
    )
    lines.append("    except (urllib.error.URLError, ValueError, OSError):")
    lines.append("        pass")
    lines.append("")
    lines.append(
        "async def _run_background_task(func, task_id, *args, "
        "callback_url=None, **kwargs):"
    )
    lines.append("    try:")
    lines.append("        # Calling a plain (non-async) notebook function directly")
    lines.append("        # here would run its entire body synchronously, inline, on")
    lines.append("        # this coroutine -- which is this app's single asyncio")
    lines.append("        # event loop, shared by every other request the process is")
    lines.append("        # handling right now, including completely unrelated ones")
    lines.append("        # like GET /health. Confirmed against a real (non-")
    lines.append("        # TestClient) uvicorn server: a background task doing")
    lines.append("        # nothing but time.sleep(2) froze a concurrent GET /health")
    lines.append("        # for the full 2 seconds -- the exact opposite of what a")
    lines.append("        # 'background' task is supposed to mean, and especially")
    lines.append("        # damaging here since the 'train'/'process'/'generate'/")
    lines.append("        # 'embed'/'scrape' keywords that route a function to a")
    lines.append("        # background task in the first place are routinely slow,")
    lines.append("        # CPU-bound work. anyio.to_thread.run_sync (anyio is")
    lines.append("        # already a transitive dependency of fastapi/starlette --")
    lines.append("        # BackgroundTasks itself is built on it -- not a new one)")
    lines.append("        # runs a synchronous function in a worker thread instead,")
    lines.append("        # so it no longer blocks every other request this server")
    lines.append("        # is handling for its entire duration. An `async def`")
    lines.append("        # notebook function is awaited directly instead, exactly")
    lines.append("        # as before -- it already cooperates with the event loop")
    lines.append("        # on its own and has no need for a worker thread.")
    lines.append("        # anyio.fail_after(None) (TASK_EXECUTION_TIMEOUT_SECONDS'")
    lines.append("        # own default, 0, is falsy -- `0 or None` is None) never")
    lines.append("        # times out at all, preserving the previous unbounded-")
    lines.append("        # execution-time behavior exactly. Given a real timeout,")
    lines.append("        # it raises a plain TimeoutError (caught below) the moment")
    lines.append("        # it elapses. abandon_on_cancel=True on the sync branch is")
    lines.append("        # what actually makes that timeout meaningful there: a real")
    lines.append("        # OS thread already running arbitrary notebook code can't be")
    lines.append("        # forcibly killed, but without this, anyio's own default")
    lines.append("        # (abandon_on_cancel=False) still *blocks this coroutine*")
    lines.append("        # until that thread finishes on its own -- silently")
    lines.append("        # defeating the timeout for the one case (a hung sync call)")
    lines.append("        # it exists to catch. Abandoning it instead frees this")
    lines.append("        # coroutine (and the capacity-limiter slot backing every")
    lines.append("        # other endpoint's own threadpool use) immediately; the")
    lines.append("        # orphaned thread itself still runs to completion")
    lines.append("        # afterward, an unavoidable limit of cooperatively")
    lines.append("        # cancelling arbitrary synchronous code at all.")
    lines.append("        with anyio.fail_after(TASK_EXECUTION_TIMEOUT_SECONDS or None):")
    lines.append("            if inspect.iscoroutinefunction(func):")
    lines.append("                result = await func(*args, **kwargs)")
    lines.append("            else:")
    lines.append("                result = await anyio.to_thread.run_sync(")
    lines.append("                    functools.partial(func, *args, **kwargs),")
    lines.append("                    abandon_on_cancel=True,")
    lines.append("                )")
    lines.append("        # jsonable_encoder both validates that `result` is")
    lines.append("        # actually something GET /tasks/{task_id} -- and GET")
    lines.append("        # /tasks, which returns *every* task in one response --")
    lines.append("        # can serialize, and converts common library-native")
    lines.append("        # return types (numpy arrays, pandas DataFrames, ...)")
    lines.append("        # into plain JSON-safe data first. Without this, a")
    lines.append("        # background function returning one of those -- an")
    lines.append("        # entirely ordinary thing for the 'train'/'process'/")
    lines.append("        # 'generate'/'embed'/'scrape' keywords that route a")
    lines.append("        # function here in the first place -- marked the task")
    lines.append("        # 'completed' with an unserializable result, which then")
    lines.append("        # crashed every subsequent GET /tasks/{task_id} for it,")
    lines.append("        # and GET /tasks entirely (for every task, not just this")
    lines.append("        # one), the moment FastAPI's own response serialization")
    lines.append("        # ran into it.")
    lines.append("        result = jsonable_encoder(result)")
    # `task_id in TASKS` rather than an unconditional TASKS[task_id][...]
    # write: _evict_expired_tasks above never removes a 'processing' task,
    # but POST /tasks/reset still unconditionally clears every entry,
    # in-flight or not (an admin nuke, by design). Without this guard, a
    # task still running when /tasks/reset fires raised a bare KeyError
    # here -- an unhandled exception in a fire-and-forget asyncio task,
    # silently logged (at best) as an opaque "Task exception was never
    # retrieved" rather than surfaced anywhere. This task's own result is
    # honestly gone either way (the caller asked to forget it); the fix
    # is just to let that happen quietly instead of crashing on it.
    lines.append("        if task_id in TASKS:")
    lines.append("            TASKS[task_id][\"status\"] = \"completed\"")
    lines.append("            TASKS[task_id][\"result\"] = result")
    # Built from the local outcome directly, not read back from TASKS --
    # this must still deliver the real result even when the entry is
    # already gone by the time we get here (POST /tasks/reset firing
    # mid-run; see the "if task_id in TASKS" guard above), rather than
    # silently sending a webhook with nothing in it.
    lines.append("        if callback_url:")
    lines.append("            await anyio.to_thread.run_sync(")
    lines.append(
        "                _deliver_task_webhook, callback_url, "
        "{\"task_id\": task_id, \"status\": \"completed\", \"result\": result}"
    )
    lines.append("            )")
    # A separate except clause from the generic Exception one below,
    # rather than letting it fall through to that one's own str(e) --
    # anyio.fail_after's own TimeoutError carries no message at all
    # (str(e) is just ""), which would otherwise record a completely
    # uninformative empty "error", indistinguishable from any other
    # unlabeled failure.
    lines.append("    except TimeoutError:")
    lines.append(
        "        timeout_error = ("
        "f'Task exceeded its {TASK_EXECUTION_TIMEOUT_SECONDS}s '"
        "'execution timeout')"
    )
    lines.append("        if task_id in TASKS:")
    lines.append("            TASKS[task_id][\"status\"] = \"failed\"")
    lines.append("            TASKS[task_id][\"error\"] = timeout_error")
    lines.append("        if callback_url:")
    lines.append("            await anyio.to_thread.run_sync(")
    lines.append(
        "                _deliver_task_webhook, callback_url, "
        "{\"task_id\": task_id, \"status\": \"failed\", \"error\": timeout_error}"
    )
    lines.append("            )")
    lines.append("    except Exception as e:")
    lines.append("        if task_id in TASKS:")
    lines.append("            TASKS[task_id][\"status\"] = \"failed\"")
    lines.append("            TASKS[task_id][\"error\"] = str(e)")
    lines.append("        if callback_url:")
    lines.append("            await anyio.to_thread.run_sync(")
    lines.append(
        "                _deliver_task_webhook, callback_url, "
        "{\"task_id\": task_id, \"status\": \"failed\", \"error\": str(e)}"
    )
    lines.append("            )")
    lines.append("")
    # Generate Pydantic models for request bodies
    for func in functions:
        func_name = func["name"]
        model_name = model_names[func_name]
        example_payload = func.get(
            "example_payload",
            {}
        )
        lines.append(f"class {model_name}(BaseModel):")
        if not func.get("args"):
            # A zero-parameter notebook function (e.g. `def health(): ...`)
            # produces a class body with no fields and, since
            # example_payload is empty too, no model_config block either --
            # an empty class body is a SyntaxError, which would fail to
            # compile the *entire* generated app, not just this endpoint.
            lines.append("    pass")
        for arg in func.get("args", []):
            arg_name = arg.get("name", "param")
            raw_arg_type = arg.get("type")
            arg_type, _ = _resolve_annotation_source(raw_arg_type)

            # repr()'d below (see description=repr(field_description)),
            # not embedded as a raw f-string inside a hand-written
            # description="..." literal -- raw_arg_type is arbitrary,
            # notebook-author-controlled text (ast.unparse of the
            # parameter's own annotation), and can itself legitimately
            # contain a double quote (e.g. a real, valid Python
            # `Literal["a\"b"]` parameter unparses to `Literal['a"b']`).
            # Confirmed exploitable before this fix: that embedded `"`
            # closed the description="..." string literal early,
            # corrupting the rest of the line into a SyntaxError that
            # failed to compile the *entire* generated app.py, not just
            # this one field.
            #
            # None (added to a `Field(...)` call below only when not
            # None) whenever the annotation already supplies its own
            # description via Annotated[T, Field(..., description=...)]
            # -- see _annotation_has_own_field_description's own
            # docstring for the silent-override bug this avoids.
            # Otherwise prefers the notebook author's own per-parameter
            # docstring description (extract_functions_from_code's
            # "description", from a Google-style "Args:" section -- see
            # _parse_docstring_arg_descriptions, backend/parser/
            # ast_parser.py) over the generic fallback every field used
            # to get regardless of how the function was actually
            # documented.
            if _annotation_has_own_field_description(raw_arg_type):
                field_description = None
            else:
                field_description = arg.get("description") or (
                    f"Parameter '{arg_name}' "
                    f"of type {raw_arg_type or 'str'}"
                )

            default_value = arg.get("default")

            if arg.get("has_default"):
                if arg.get("default_is_literal", True):
                    default_expr = repr(default_value)
                else:
                    # A non-literal default (e.g. a notebook-defined Enum
                    # member like `Priority.HIGH` -- see
                    # extract_functions_from_code's default_is_literal in
                    # ast_parser.py). `default_value` here is raw notebook
                    # source, not a Python value, so repr()-ing it like a
                    # literal default would embed it as the *string*
                    # "Priority.HIGH" instead of the actual enum member --
                    # confirmed broken before this fix: a caller omitting
                    # this field to take its default raised an
                    # AttributeError inside the notebook's own function the
                    # moment it tried to use the (wrongly stringified)
                    # value. Reusing _resolve_annotation_source qualifies
                    # any bare notebook-defined name it references (e.g.
                    # "Priority") to notebook_module.Priority, exactly as
                    # it already does for type annotations, then the
                    # qualified source is embedded directly as a code
                    # expression rather than a string literal.
                    default_expr, _ = _resolve_annotation_source(default_value)
                if field_description is not None:
                    lines.append(
                        f'    {arg_name}: {arg_type} = Field('
                        f'default={default_expr}, '
                        f'description={repr(field_description)}'
                        f')'
                    )
                else:
                    lines.append(
                        f'    {arg_name}: {arg_type} = Field('
                        f'default={default_expr}'
                        f')'
                    )
            elif field_description is not None:
                lines.append(
                    f'    {arg_name}: {arg_type} = Field('
                    f'description={repr(field_description)}'
                    f')'
                )
            else:
                # No default to assign, and the annotation already
                # carries its own description (see
                # _annotation_has_own_field_description above) -- a bare
                # annotated field, with no assignment at all, is already
                # a valid required Pydantic field declaration, exactly
                # like a hand-written `Annotated[...]`-only field would
                # be; an empty `= Field()` here would add nothing.
                lines.append(f'    {arg_name}: {arg_type}')
        if example_payload:
            lines.append("")
            lines.append("    model_config = {")
            lines.append(
                f"        'json_schema_extra': {{'example': {repr(example_payload)}}}"
            )
            lines.append("    }")
        lines.append("")
    # Generate endpoints
    for func in functions:
        func_name = func["name"]
        operation_id = func_name
        tag = "General"
        if "train" in func_name.lower():
            tag = "Training"
        elif "predict" in func_name.lower():
            tag = "Inference"
        elif any(
            kw in func_name.lower()
            for kw in ["scrape", "extract", "process"]
        ):
            tag = "Data Processing"
        elif any(
            kw in func_name.lower()
            for kw in ["embed", "vector"]
        ):
            tag = "Embeddings"
        category = tag
        args = func.get("args", [])
        example_response = func.get(
            "example_response",
            {"result": None}
        )
        return_type = func.get(
            "return_type",
            "unknown"
        )
        response_description = (
            f"Returns {return_type}"
        )
        model_name = model_names[func_name]
        call_args = ", ".join(_call_arg_expr(arg) for arg in args)
        is_background = any(kw in func_name.lower() for kw in LONG_RUNNING_KEYWORDS)
        summary = (
            func_name
            .replace("_", " ")
            .title()
        )
        # A notebook function's own docstring is exactly the description
        # its author already wrote, on purpose, for this exact function --
        # strictly more useful than a generic templated sentence that
        # doesn't even manage to say what the endpoint *does*. Before this,
        # extract_functions_from_code (parser/ast_parser.py) didn't even
        # extract it, so it was always discarded no matter what a notebook
        # author wrote; only ever falls back to the auto-generated summary
        # below for a function with no docstring at all (or one that's
        # empty/all-whitespace, which ast.get_docstring(clean=True) already
        # normalizes down to a falsy value).
        docstring = func.get("docstring")
        description = (
            docstring
            if docstring
            else (
                f"Auto-generated endpoint for {func_name}. "
                f"Operation ID: {operation_id}. "
                f"Parameters: {', '.join(arg['name'] for arg in args) if args else 'None'}."
            )
        )
        if is_background:
            # A background endpoint doesn't return `example_response`/
            # `response_description` (the notebook function's own return
            # value) at all -- it returns {"task_id": ..., "status":
            # "processing"} (see the `return` statement below) and the
            # real result only becomes available later via GET
            # /tasks/{task_id}. Documenting the function's own return
            # shape here instead was actively misleading: /docs, and any
            # third-party tool generating a client from openapi.json,
            # would expect a response this endpoint never actually sends.
            # repr()'d below for the same reason field_description is:
            # return_type is arbitrary, notebook-author-controlled text
            # (ast.unparse of the function's own return annotation) that
            # can itself legitimately contain a double quote -- embedding
            # it as a raw f-string inside a hand-written
            # "description": "..." literal let that quote close the
            # string early, corrupting the whole responses={...} dict
            # literal into a SyntaxError that failed to compile the
            # entire generated app.py.
            task_response_description = (
                f"Task enqueued. Poll GET /tasks/{{task_id}} for the "
                f"completed {return_type} result."
            )
            task_example_response = {"task_id": "<uuid>", "status": "processing"}
            # A background endpoint's own two extra failure modes, on top
            # of the {401, 429} every endpoint can already produce (see
            # _auth_and_rate_limit_error_responses): 503 when
            # NOTEBOOK_API_MAX_TASKS is already at capacity, and 400 for
            # a caller-supplied ?callback_url= that isn't http(s) (see
            # this same function's own body below) -- neither was
            # documented anywhere in the served schema before this,
            # despite both being real, reachable responses. The whole
            # dict is repr()'d as one native Python object below, not
            # hand-assembled via string concatenation the way this used
            # to be -- repr() can never produce invalid Python source no
            # matter what a notebook author's own docstring/return-type
            # text contains, closing the exact class of quote-escaping
            # bug e91b1fa already had to fix here once by hand.
            task_responses = {
                200: {
                    "description": task_response_description,
                    "content": {
                        "application/json": {"example": task_example_response}
                    },
                },
                400: {
                    "description": (
                        "callback_url was given but isn't an http:// or "
                        "https:// URL."
                    ),
                    "content": {
                        "application/json": {
                            "example": {
                                "detail": (
                                    "callback_url must be an http:// or "
                                    "https:// URL"
                                )
                            }
                        }
                    },
                },
                503: {
                    "description": (
                        "Too many pending background tasks (see "
                        "NOTEBOOK_API_MAX_TASKS)."
                    ),
                    "content": {
                        "application/json": {
                            "example": {
                                "detail": (
                                    "Too many pending background tasks "
                                    "(limit 10000); try again once some "
                                    "have finished."
                                )
                            }
                        }
                    },
                },
                **_auth_and_rate_limit_error_responses(),
            }
            lines.append(
                f'@app.post("/{func_name}", '
                f'summary="{summary}", '
                # repr()'d, not embedded as a raw f-string like the fixed,
                # server-generated boilerplate this replaces when there's
                # no docstring: description can now be a notebook author's
                # own docstring, arbitrary content that can legitimately
                # contain a double quote, a newline, or a backslash --
                # embedding it as a raw "description="..."" literal would
                # let any of those close the string early, corrupting the
                # whole @app.post(...) call into a SyntaxError and failing
                # the entire compile, not just this one endpoint's docs
                # (the exact bug class e91b1fa already fixed for the
                # Pydantic Field description and the responses={} dict's
                # own "description" entry).
                f'description={repr(description)}, '
                f'tags=["{tag}"], '
                f'operation_id="{operation_id}", '
                # "x-notebook-to-api-return-type" (the notebook function's
                # own raw return annotation, exactly as extracted by
                # extract_functions_from_code -- not resolved/qualified
                # the way _resolve_annotation_source does for the actual
                # Python model, since a downstream consumer of this
                # schema has no notebook_module of its own to qualify
                # against) is read by generate_typescript_sdk
                # (backend/exporters/sdk_generator.py) the same way it
                # already reads "x-notebook-to-api-async"/
                # "x-notebook-to-api-category" -- an out-of-band channel
                # for information the OpenAPI spec itself has no field
                # for, since this endpoint's own declared 200 response
                # schema is deliberately {} (see task_responses above):
                # nothing about the *eventual* result a real
                # GET /tasks/{{task_id}} will carry is otherwise
                # discoverable from this schema at all.
                f'openapi_extra={{"x-notebook-to-api-category": "{category}", "x-notebook-to-api-async": True, "x-notebook-to-api-return-type": {repr(return_type)}, "security": [{{"ApiKeyAuth": []}}]}}, '
                f'responses={repr(task_responses)})'
            )
            lines.append(
                f"def {func_name}(req: {model_name}, background_tasks: "
                "BackgroundTasks, callback_url: Optional[str] = None, "
                "_: None = Depends(verify_api_key)):"
            )
            lines.append("    _evict_expired_tasks()")
            lines.append("    if len(TASKS) >= MAX_PENDING_TASKS:")
            lines.append("        raise HTTPException(")
            lines.append("            status_code=503,")
            lines.append("            detail=(")
            lines.append("                f'Too many pending background tasks (limit '")
            lines.append("                f'{MAX_PENDING_TASKS}); try again once some have '")
            lines.append("                'finished.'")
            lines.append("            ),")
            lines.append("        )")
            # A caller opting into webhook delivery (rather than polling
            # get_task/wait_for_task) supplies this per-request, not via a
            # server-side operator setting -- so, unlike every other limit
            # this app enforces, this is the one input here that's fully
            # attacker/caller-controlled. Restricted to http(s) so a typo'd
            # or malicious "file:///etc/passwd"-style scheme can't reach
            # urllib.request.urlopen inside _deliver_task_webhook at all --
            # validated here, before a task is even created, so the error
            # is immediate and actionable rather than a silent delivery
            # failure discovered only much later.
            lines.append("    if callback_url is not None:")
            lines.append("        if urlparse(callback_url).scheme not in ('http', 'https'):")
            lines.append("            raise HTTPException(")
            lines.append("                status_code=400,")
            lines.append("                detail=(")
            lines.append(
                "                    'callback_url must be an http:// or '"
            )
            lines.append(
                "                    'https:// URL'"
            )
            lines.append("                ),")
            lines.append("            )")
            lines.append("    task_id = uuid.uuid4().hex")
            lines.append("    TASKS[task_id] = {\"status\": \"processing\", \"created_at\": time.time()}")
            # Pass positional arguments to the background function
            call_parts = (
                [f"notebook_module.{func_name}", "task_id"]
                + [_call_arg_expr(arg) for arg in args]
                + ["callback_url=callback_url"]
            )
            lines.append(
                "    background_tasks.add_task(_run_background_task, "
                f"{', '.join(call_parts)})"
            )
            lines.append("    return {\"task_id\": task_id, \"status\": \"processing\"}")
        else:
            # A synchronous endpoint's own extra failure mode, on top of
            # the {401, 429} every endpoint can already produce (see
            # _auth_and_rate_limit_error_responses): 500, wrapping either
            # the notebook function's own exception or a non-JSON-
            # serializable return value (see this same function's own
            # body below) -- undocumented anywhere in the served schema
            # before this. repr()'d as one native Python object, not
            # hand-assembled via string concatenation, the same
            # quote-escaping-proof technique the background branch above
            # now uses too.
            sync_responses = {
                200: {
                    "description": response_description,
                    "content": {
                        "application/json": {"example": example_response}
                    },
                },
                500: {
                    "description": (
                        f"'{func_name}' raised an exception, or returned "
                        "a value that isn't JSON-serializable."
                    ),
                    "content": {
                        "application/json": {
                            "example": {
                                "detail": (
                                    f"'{func_name}' raised ValueError: "
                                    "<message>"
                                )
                            }
                        }
                    },
                },
                **_auth_and_rate_limit_error_responses(),
            }
            lines.append(
                f'@app.post("/{func_name}", '
                f'summary="{summary}", '
                # repr()'d, not embedded as a raw f-string like the fixed,
                # server-generated boilerplate this replaces when there's
                # no docstring: description can now be a notebook author's
                # own docstring, arbitrary content that can legitimately
                # contain a double quote, a newline, or a backslash --
                # embedding it as a raw "description="..."" literal would
                # let any of those close the string early, corrupting the
                # whole @app.post(...) call into a SyntaxError and failing
                # the entire compile, not just this one endpoint's docs
                # (the exact bug class e91b1fa already fixed for the
                # Pydantic Field description and the responses={} dict's
                # own "description" entry).
                f'description={repr(description)}, '
                f'tags=["{tag}"], '
                f'operation_id="{operation_id}", '
                # See the background branch's own identical
                # "x-notebook-to-api-return-type" comment above --
                # this endpoint's own declared 200 response schema is
                # deliberately {} too (see sync_responses above), so
                # generate_typescript_sdk has no other way to learn what
                # "result" actually contains.
                f'openapi_extra={{"x-notebook-to-api-category": "{category}", "x-notebook-to-api-return-type": {repr(return_type)}, "security": [{{"ApiKeyAuth": []}}]}}, '
                f'responses={repr(sync_responses)})'
            )
            is_async = func.get("is_async", False)
            def_keyword = "async def" if is_async else "def"
            call_prefix = "await " if is_async else ""
            lines.append(f"{def_keyword} {func_name}(req: {model_name}, _: None = Depends(verify_api_key)):")
            # _run_background_task already wraps a background function's own
            # call the same way (reporting the task "failed" with str(e)
            # instead of leaving it stuck "processing" forever), but a
            # synchronous endpoint had no equivalent at all: the notebook
            # function's own exception -- a ZeroDivisionError, a KeyError, a
            # bad file path, anything -- propagated straight out unhandled,
            # crashing with the exact same bare, detail-free "Internal
            # Server Error" the jsonable_encoder gap below already had.
            # HTTPException is re-raised as-is (not wrapped into a generic
            # 500) since a notebook function that imports fastapi itself and
            # deliberately raises one (e.g. HTTPException(404, ...)) is
            # already choosing its own status code and message on purpose.
            lines.append("    try:")
            lines.append(f"        result = {call_prefix}notebook_module.{func_name}({call_args})")
            lines.append("    except HTTPException:")
            lines.append("        raise")
            lines.append("    except Exception as e:")
            lines.append("        raise HTTPException(")
            lines.append("            status_code=500,")
            lines.append(
                f"            detail=f\"'{func_name}' raised "
                "{type(e).__name__}: {e}\","
            )
            lines.append("        )")
            # Same reasoning as _run_background_task's own jsonable_encoder
            # call: without pre-encoding here, a result FastAPI's response
            # serialization can't handle on its own (a raw numpy array, a
            # pandas DataFrame, ...) doesn't fail with anything resembling
            # this app's other error responses -- it crashes deep inside
            # FastAPI's routing internals as an unhandled ValueError, which
            # a real (non-test-client) deployment surfaces to the caller as
            # a bare "Internal Server Error" with no detail at all, unlike
            # every other failure mode this generated app already reports
            # clearly (auth, reserved names, oversized bodies, ...).
            lines.append("    try:")
            lines.append("        result = jsonable_encoder(result)")
            lines.append("    except Exception as e:")
            lines.append("        raise HTTPException(")
            lines.append("            status_code=500,")
            lines.append(
                f"            detail=f\"'{func_name}' returned a value that "
                "is not JSON-serializable: {e}\","
            )
            lines.append("        )")
            lines.append("    return {\"result\": result}")
        lines.append("")
    return "\n".join(lines)

# Helper to write the generated FastAPI source file
def write_generated_api(code, output_path="generated/app.py"):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Generated API written to: {output_path}")


def endpoint_openapi_metadata(
    schema_generator,
    endpoint
):

    description = (
        schema_generator
        .generate_openapi_description(
            endpoint
        )
    )

    return {

        "summary":
            description.summary,

        "description":
            description.description,

        "tags":
            description.tags
    }


def endpoint_examples(
    schema_generator,
    endpoint
):

    example = (
        schema_generator
        .generate_api_examples(
            endpoint
        )
    )

    return {

        "request":
            example.request_example,

        "response":
            example.response_example
    }


def endpoint_errors(
    schema_generator
):

    return (
        schema_generator
        .generate_api_error_docs()
    )

# Simple demo when run directly
if __name__ == "__main__":
    sample_functions = [
        {
            "name": "add",
            "args": [
                {"name": "a", "type": "int"},
                {"name": "b", "type": "int"}
            ],
            "return_type": "int"
        },
        {
            "name": "train_model",
            "args": [
                {"name": "epochs", "type": "int"}
            ],
            "return_type": "str"
        }
    ]
    generated_code = generate_fastapi_code(sample_functions)
    write_generated_api(generated_code)