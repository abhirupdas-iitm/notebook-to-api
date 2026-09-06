import json
import sys
import types
import urllib.error

import pytest

from backend.generator.api_generator import (
    GENERATED_APP_ENV_VARS,
    generate_fastapi_code,
    RESERVED_INFRASTRUCTURE_NAMES,
    ReservedFunctionNameError,
)


def test_generated_app_env_vars_default_matches_the_actual_generated_code():
    """GENERATED_APP_ENV_VARS (read back by GET /api/env-vars-preview,
    backend/routes/upload.py) must be the single source of truth
    generate_fastapi_code's own os.getenv(...) calls are built from --
    not a second, independently-maintained copy of the same five
    defaults that could silently drift out of sync with what a compiled
    app.py actually falls back to.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]
    code = generate_fastapi_code(functions)

    assert {entry["name"] for entry in GENERATED_APP_ENV_VARS} == {
        "NOTEBOOK_API_KEY",
        "NOTEBOOK_API_ALLOWED_ORIGINS",
        "NOTEBOOK_API_MAX_REQUEST_BYTES",
        "NOTEBOOK_API_TASK_TTL_SECONDS",
        "NOTEBOOK_API_MAX_TASKS",
        "NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS",
        "NOTEBOOK_API_RATE_LIMIT_PER_MINUTE",
        "NOTEBOOK_API_WEBHOOK_TIMEOUT_SECONDS",
        "NOTEBOOK_API_WEBHOOK_SECRET",
        "NOTEBOOK_API_PUBLIC_URL",
        "NOTEBOOK_API_DISABLE_DOCS",
    }

    for entry in GENERATED_APP_ENV_VARS:
        assert f'os.getenv("{entry["name"]}", "{entry["default"]}")' in code
        assert entry["description"]


def test_generate_fastapi_code_bakes_in_the_given_source_notebook_sha256():

    functions = [{"name": "add", "args": [], "return_type": "int"}]
    sha = "a" * 64

    code = generate_fastapi_code(functions, source_notebook_sha256=sha)

    assert f"SOURCE_NOTEBOOK_SHA256 = '{sha}'" in code
    assert '"source_notebook_sha256": SOURCE_NOTEBOOK_SHA256' in code


def test_generate_fastapi_code_defaults_source_notebook_sha256_to_none():

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "SOURCE_NOTEBOOK_SHA256 = None" in code


def test_generate_fastapi_code_bakes_in_the_given_notebook_to_api_version():
    """GET / and GET /info both previously reported a hardcoded "1.0.0"
    literal completely unrelated to which actual version of this tool
    compiled the app -- the same "two independent, inevitably-drifting
    hardcoded version literals" bug NOTEBOOK_TO_API_VERSION
    (backend/compiler.py) was already introduced to deduplicate for this
    dashboard's own GET /api/health and GET /, just never threaded
    through to the *generated* app's own identical two literals.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions, notebook_to_api_version="0.4.2")

    assert "NOTEBOOK_TO_API_VERSION = '0.4.2'" in code
    assert "'generator_version': NOTEBOOK_TO_API_VERSION," in code
    assert '"version": NOTEBOOK_TO_API_VERSION,' in code


def test_generate_fastapi_code_bakes_the_given_version_into_the_fastapi_app_itself():
    """A third hardcoded "1.0.0" literal missed the first time this
    parameter was added: the FastAPI(...) app object's own `version=`
    kwarg, which custom_openapi passes straight through as this app's
    own OpenAPI "info.version" -- user-visible in every compiled app's
    own /docs (Swagger UI), and baked directly into whatever POST
    /api/export-openapi writes out (export_openapi_schema serializes
    app.openapi() unchanged), unlike "generator_version"/"version" above
    (informational JSON fields only).
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions, notebook_to_api_version="0.4.2")

    assert "version='0.4.2'" in code
    assert 'version="1.0.0"' not in code


def test_generate_fastapi_code_defaults_notebook_to_api_version_to_one_point_zero_point_zero():
    """A caller not passing "notebook_to_api_version" at all (a direct
    unit test, most commonly -- every real compile always passes
    compiler.py's own NOTEBOOK_TO_API_VERSION) must see this function's
    own previous literal exactly, not a silently different default no
    caller asked for.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "NOTEBOOK_TO_API_VERSION = '1.0.0'" in code
    assert "version='1.0.0'" in code


def test_notebook_to_api_version_is_a_reserved_infrastructure_name():
    """A notebook function (or module-level assignment) literally named
    NOTEBOOK_TO_API_VERSION would silently rebind the real one at
    module-load time -- the exact endpoint-ordering trap this reserved-
    names set already exists to reject outright, the same protection
    GENERATED_AT/PYTHON_VERSION (the other baked-in metadata constants)
    already have.
    """

    assert "NOTEBOOK_TO_API_VERSION" in RESERVED_INFRASTRUCTURE_NAMES


def test_generate_fastapi_code_passes_servers_to_get_openapi():
    """Confirmed dead code before this fix: the FastAPI(...) constructor's
    own servers=[...] kwarg was silently discarded, since custom_openapi
    completely overrides app.openapi and never itself passed servers= to
    get_openapi(...) -- app.openapi()["servers"] was never even a key in
    the resulting schema, no matter what the constructor was given.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "servers=app.servers," in code


def test_generate_fastapi_code_bakes_the_given_public_url_into_the_servers_entry():
    """NOTEBOOK_API_PUBLIC_URL (read into PUBLIC_URL before app =
    FastAPI(...), since the servers= kwarg needs it at construction time)
    drives the same "servers" entry GET /docs' own Swagger UI "Try it
    out" defaults its request URL to -- left at the previous hardcoded
    "http://localhost:8000" outside local development, every "Try it
    out" request failed from a browser that wasn't itself on the same
    machine as the deployment.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert (
        'PUBLIC_URL = os.getenv("NOTEBOOK_API_PUBLIC_URL", "http://localhost:8000")'
        in code
    )
    assert 'servers=[{"url": PUBLIC_URL, "description": "This deployment"}]' in code


def test_public_url_is_a_reserved_infrastructure_name():

    assert "PUBLIC_URL" in RESERVED_INFRASTRUCTURE_NAMES


def test_generate_fastapi_code_defaults_to_docs_enabled():
    """docs_url/redoc_url/openapi_url must default to their own normal
    FastAPI paths -- NOTEBOOK_API_DISABLE_DOCS defaults to "false", so an
    existing deployment that never opts in sees no behavior change.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert 'DISABLE_DOCS = os.getenv("NOTEBOOK_API_DISABLE_DOCS", "false")' in code
    assert 'docs_url=None if DISABLE_DOCS else "/docs"' in code
    assert 'redoc_url=None if DISABLE_DOCS else "/redoc"' in code
    assert 'openapi_url=None if DISABLE_DOCS else "/openapi.json"' in code


def test_disable_docs_is_a_reserved_infrastructure_name():

    assert "DISABLE_DOCS" in RESERVED_INFRASTRUCTURE_NAMES


def _register_fake_notebook_module(monkeypatch, package_name="generated"):
    """Generated code always contains a real
    `import <package_name>.runtime.notebook_module as notebook_module`
    statement (see api_generator.py). A plain `namespace = {"notebook_module":
    ...}` dict passed to exec() does NOT satisfy that -- `import X as Y`
    always performs a real import of X via sys.modules/sys.path and
    ignores whatever's already bound to the name Y, so exec()ing generated
    code without actually registering these modules only "works" by
    accident if a real `<package_name>/runtime/notebook_module.py`
    happens to already exist somewhere importable (e.g. a stray leftover
    `generated/` directory from a previous local run) -- which silently
    passes locally but fails with ModuleNotFoundError in a clean checkout.
    """
    parent = types.ModuleType(package_name)
    runtime_pkg = types.ModuleType(f"{package_name}.runtime")
    notebook_module = types.ModuleType(f"{package_name}.runtime.notebook_module")

    monkeypatch.setitem(sys.modules, package_name, parent)
    monkeypatch.setitem(sys.modules, f"{package_name}.runtime", runtime_pkg)
    monkeypatch.setitem(
        sys.modules, f"{package_name}.runtime.notebook_module", notebook_module
    )

    return notebook_module


def test_api_generation():

    functions = [
        {
            "name": "add",
            "args": [
                {
                    "name": "a",
                    "type": "int"
                },
                {
                    "name": "b",
                    "type": "int"
                }
            ],
            "return_type": "int"
        }
    ]

    code = generate_fastapi_code(functions)

    assert "@app.post" in code


def test_api_key_check_uses_constant_time_comparison():
    """A plain `x_api_key != API_KEY` short-circuits on the first
    differing byte, leaking via response timing how many leading
    characters of a guess were correct -- a classic timing side-channel
    for guessing the key byte by byte. Must use hmac.compare_digest.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "import hmac" in code
    assert "hmac.compare_digest(x_api_key, key) for key in API_KEYS" in code
    assert "x_api_key != " not in code
    assert "x_api_key in API_KEYS" not in code


def test_api_key_check_still_rejects_missing_header():
    """hmac.compare_digest raises TypeError on None, so the missing-header
    case (x_api_key defaults to None) must be checked before calling it,
    not delegated to it.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "if x_api_key is None or not any(" in code


def test_rate_limit_dependency_injects_response_to_set_headers():
    """verify_api_key must accept and forward a `response: Response`
    parameter to _enforce_rate_limit -- confirmed exploitable before this
    fix: without it, _enforce_rate_limit had no way to set headers on a
    request that *succeeds*, so X-RateLimit-Limit/-Remaining/-Reset could
    only ever be attached to the 429 it raises, never to the requests
    leading up to it.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "from fastapi import" in code.splitlines()[0]
    assert "Response" in code.splitlines()[0]
    assert (
        "def verify_api_key(response: Response, x_api_key: str = Header(None)):"
        in code
    )
    assert "def _enforce_rate_limit(api_key, response):" in code
    assert "_enforce_rate_limit(x_api_key, response)" in code
    assert "response.headers['X-RateLimit-Limit'] = str(RATE_LIMIT_PER_MINUTE)" in code
    assert "response.headers['X-RateLimit-Remaining'] = str(remaining)" in code
    assert "response.headers['X-RateLimit-Reset'] = str(reset_at)" in code
    assert "'X-RateLimit-Limit': str(RATE_LIMIT_PER_MINUTE)," in code
    assert "'X-RateLimit-Remaining': '0'," in code
    assert "'X-RateLimit-Reset': str(reset_at)," in code


def test_generated_app_configures_cors_middleware_with_a_permissive_default():
    """Before this, the generated app had no CORS configuration at all --
    a browser-based frontend, the single most common way to actually
    consume a deployed generated API, was blocked by CORS with no way to
    fix it short of hand-editing the generated file. Default is
    permissive ("*") because every endpoint is authenticated via the
    X-API-Key header, not a cookie, so allow_credentials stays False and a
    wildcard origin carries no cross-site credential risk.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "from fastapi.middleware.cors import CORSMiddleware" in code
    assert 'os.getenv("NOTEBOOK_API_ALLOWED_ORIGINS", "*")' in code
    assert "allow_credentials=False" in code
    assert "app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS" in code


def test_generated_app_cors_exposes_the_rate_limit_headers_to_cross_origin_js():
    """Confirmed exploitable before this fix: a browser only ever exposes
    a small built-in safelist of response headers to cross-origin JS
    (Cache-Control, Content-Language, Content-Length, Content-Type,
    Expires, Last-Modified, Pragma) -- X-RateLimit-Limit/-Remaining/-Reset
    and Retry-After (see _enforce_rate_limit) are not on it, so
    `fetch(...).headers.get('X-RateLimit-Remaining')` from cross-origin JS
    always returned null, even though the server sent the header every
    time, unless CORSMiddleware's own expose_headers explicitly lists it.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert (
        "expose_headers=['X-RateLimit-Limit', 'X-RateLimit-Remaining', "
        "'X-RateLimit-Reset', 'Retry-After']"
        in code
    )


def test_notebook_function_named_allowed_origins_is_rejected():
    """ALLOWED_ORIGINS is a module-level name the generated app itself
    defines (see RESERVED_INFRASTRUCTURE_NAMES) -- same collision hazard
    class as API_KEYS or TASKS.
    """

    functions = [{"name": "ALLOWED_ORIGINS", "args": [], "return_type": "dict"}]

    with pytest.raises(ReservedFunctionNameError, match="ALLOWED_ORIGINS"):
        generate_fastapi_code(functions)


def test_generated_app_configures_a_max_request_body_size_middleware():
    """Before this, every endpoint accepted a JSON request body of any
    size -- unlike this tool's own dashboard /api/upload, which has
    always capped uploads at MAX_UPLOAD_BYTES (see routes/upload.py) for
    exactly this reason. A deployed generated app had no equivalent: one
    oversized request could consume unbounded memory building the body
    before Pydantic ever got a chance to validate or reject it.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "class MaxRequestBodySizeMiddleware:" in code
    assert 'os.getenv("NOTEBOOK_API_MAX_REQUEST_BYTES", "10485760")' in code
    assert "app.add_middleware(MaxRequestBodySizeMiddleware)" in code


def test_generated_app_stamps_security_headers_on_every_response():
    """Confirmed exploitable before this fix: the generated app set none
    of the baseline OWASP-recommended hardening headers (X-Content-Type-
    Options, X-Frame-Options, Referrer-Policy) on any response -- grepped
    for across the whole file, zero hits. Registered *after*
    MaxRequestBodySizeMiddleware/CORSMiddleware (see this middleware's
    own comment) so it ends up outermost -- these headers must land on
    every response, not just a successful one.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "@app.middleware('http')" in code
    assert "async def _add_security_headers(request, call_next):" in code
    assert "response.headers['X-Content-Type-Options'] = 'nosniff'" in code
    assert "response.headers['X-Frame-Options'] = 'DENY'" in code
    assert "response.headers['Referrer-Policy'] = 'no-referrer'" in code
    # Registered after (not before) MaxRequestBodySizeMiddleware -- the
    # middleware added last ends up outermost, so this must appear later
    # in the generated source than that registration.
    assert code.index("app.add_middleware(MaxRequestBodySizeMiddleware)") < code.index(
        "async def _add_security_headers"
    )


def test_generated_app_configures_gzip_response_compression():
    """Confirmed exploitable before this fix: the generated app never
    compressed any response -- grepped for across the whole file, zero
    hits -- even though a notebook function's own result, or GET /tasks'
    still-up-to-100-entries-per-page (see the status/limit/offset
    pagination this file's own list_tasks adds), can be large. Registered
    *after* _add_security_headers (see that middleware's own comment on
    why registration order determines outermost-ness) so it compresses
    the truly final response body, not something an inner layer might
    still rewrite.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "from fastapi.middleware.gzip import GZipMiddleware" in code
    assert "app.add_middleware(GZipMiddleware)" in code
    assert code.index("async def _add_security_headers") < code.index(
        "app.add_middleware(GZipMiddleware)"
    )


def test_generated_app_stamps_x_process_time_ms_on_every_response():
    """Confirmed exploitable before this fix: the generated app gave an
    operator no way to see per-request latency short of instrumenting it
    externally -- grepped for across the whole file, no X-Process-Time
    header anywhere. Registered *after* GZipMiddleware (see that
    middleware's own comment on registration order) so the timer spans
    every other layer too, not just handler time.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "async def _add_process_time_header(request, call_next):" in code
    assert "start_time = time.perf_counter()" in code
    assert "response.headers['X-Process-Time-Ms'] = " in code
    assert code.index("app.add_middleware(GZipMiddleware)") < code.index(
        "async def _add_process_time_header"
    )


def test_generated_app_stamps_x_request_id_and_honors_a_caller_supplied_one():
    """Confirmed exploitable before this fix: the generated app never
    surfaced a request-correlation id at all -- grepped for across the
    whole file, no X-Request-ID anywhere -- so a caller had no shared id
    to search server-side logs by when investigating a specific failed
    call after the fact. Must honor a caller-supplied X-Request-ID
    instead of always minting a fresh one, so a trace started upstream
    (e.g. by a gateway already assigning one) isn't forked into two
    disconnected ids.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "async def _add_request_id_header(request, call_next):" in code
    assert (
        "request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())"
        in code
    )
    assert "response.headers['X-Request-ID'] = request_id" in code
    assert code.index("async def _add_process_time_header") < code.index(
        "async def _add_request_id_header"
    )


def test_generated_app_exposes_get_config_reporting_its_own_runtime_limits(monkeypatch):
    """Confirmed exploitable before this fix: every NOTEBOOK_API_* limit
    this app enforces (MAX_REQUEST_BODY_BYTES, TASK_TTL_SECONDS,
    MAX_PENDING_TASKS, RATE_LIMIT_PER_MINUTE, ALLOWED_ORIGINS,
    DISABLE_DOCS, PUBLIC_URL) was only discoverable by reading the
    deployment's own environment directly -- shell access to the
    container -- with /auth/info's own "rate_limit_per_minute" the sole
    exception. The same "ask the running app what it's actually
    configured with" gap GET /api/config already closes for this
    dashboard's own configuration (routes/upload.py), never given an
    equivalent here. No secrets here (API_KEYS' own values are
    deliberately never returned), so this needs no authentication.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "@app.get('/config')" in code
    assert "def service_config():" in code
    for field in (
        "'max_request_body_bytes': MAX_REQUEST_BODY_BYTES,",
        "'task_ttl_seconds': TASK_TTL_SECONDS,",
        "'max_pending_tasks': MAX_PENDING_TASKS,",
        "'webhook_timeout_seconds': WEBHOOK_TIMEOUT_SECONDS,",
        "'webhook_signing_enabled': bool(WEBHOOK_SECRET),",
        "'rate_limit_per_minute': RATE_LIMIT_PER_MINUTE or None,",
        "'allowed_origins': ALLOWED_ORIGINS,",
        "'disable_docs': DISABLE_DOCS,",
        "'public_url': PUBLIC_URL,",
    ):
        assert field in code

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    resp = client.get("/config")

    assert resp.status_code == 200
    body = resp.json()
    assert body["max_request_body_bytes"] == 10 * 1024 * 1024
    assert body["task_ttl_seconds"] == 3600
    assert body["max_pending_tasks"] == 10000
    assert body["webhook_timeout_seconds"] == 5
    assert body["webhook_signing_enabled"] is False
    assert body["rate_limit_per_minute"] is None
    assert body["allowed_origins"] == ["*"]
    assert body["disable_docs"] is False
    assert body["public_url"] == "http://localhost:8000"


def test_notebook_function_named_service_config_is_rejected():
    """service_config is a reserved infrastructure name (GET /config) --
    same collision hazard class as service_info/metrics/uptime.
    """

    functions = [{"name": "service_config", "args": [], "return_type": "dict"}]

    with pytest.raises(ReservedFunctionNameError, match="service_config"):
        generate_fastapi_code(functions)


def test_notebook_function_named_max_request_body_bytes_is_rejected():
    """MAX_REQUEST_BODY_BYTES is a module-level name the generated app
    itself defines (see RESERVED_INFRASTRUCTURE_NAMES) -- same collision
    hazard class as API_KEYS, TASKS, or ALLOWED_ORIGINS.
    """

    functions = [
        {"name": "MAX_REQUEST_BODY_BYTES", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="MAX_REQUEST_BODY_BYTES"):
        generate_fastapi_code(functions)


def test_notebook_function_named_evict_expired_tasks_is_rejected():
    """_evict_expired_tasks is a module-level helper the generated app
    itself defines (see RESERVED_INFRASTRUCTURE_NAMES) -- same collision
    hazard class as ALLOWED_ORIGINS or MAX_PENDING_TASKS, but for a
    private function rather than a constant. Confirmed exploitable
    before this was added: a notebook function of this exact name
    compiled fine and silently overwrote the real helper at module-
    execution time (Python has no protection against redefining a name),
    breaking every *other* background endpoint's own submission too,
    since each one calls this same now-shadowed name before enqueuing a
    new task.
    """

    functions = [
        {"name": "_evict_expired_tasks", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="_evict_expired_tasks"):
        generate_fastapi_code(functions)


def test_notebook_function_named_run_background_task_is_rejected():
    """_run_background_task is a module-level helper the generated app
    itself defines -- every background endpoint's own submission passes
    this exact name to background_tasks.add_task(...) to actually run
    the task, so a notebook function shadowing it would silently break
    execution of *every* background task in the app, not just the
    notebook's own colliding endpoint.
    """

    functions = [
        {"name": "_run_background_task", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="_run_background_task"):
        generate_fastapi_code(functions)


def test_notebook_function_named_task_ttl_seconds_is_rejected():
    """TASK_TTL_SECONDS is read by name from inside _evict_expired_tasks'
    own body -- same collision hazard class as _evict_expired_tasks/
    _run_background_task themselves, just for a constant one of them
    reads rather than the helper itself.
    """

    functions = [
        {"name": "TASK_TTL_SECONDS", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="TASK_TTL_SECONDS"):
        generate_fastapi_code(functions)


def test_notebook_function_named_source_notebook_sha256_is_rejected():
    """SOURCE_NOTEBOOK_SHA256 is assigned this compile's own real content
    hash once, at module load, then read back verbatim by GET /info --
    same collision hazard class as every other module-level constant
    here.
    """

    functions = [
        {"name": "SOURCE_NOTEBOOK_SHA256", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="SOURCE_NOTEBOOK_SHA256"):
        generate_fastapi_code(functions)


def test_notebook_function_named_enforce_rate_limit_is_rejected():
    """_enforce_rate_limit is a module-level helper verify_api_key's own
    body calls by name on every single request (via Depends(verify_api_key)
    on literally every endpoint) -- same collision hazard class as
    _evict_expired_tasks/_run_background_task, just for the rate-limiting
    subsystem instead of the background-task one.
    """

    functions = [
        {"name": "_enforce_rate_limit", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="_enforce_rate_limit"):
        generate_fastapi_code(functions)


def test_notebook_function_named_rate_limit_per_minute_is_rejected():
    """RATE_LIMIT_PER_MINUTE is read by name from inside
    _enforce_rate_limit's own body -- one level removed from
    _enforce_rate_limit's own name, but the identical exposure: every
    endpoint's own Depends(verify_api_key) calls _enforce_rate_limit,
    which reads this constant.
    """

    functions = [
        {"name": "RATE_LIMIT_PER_MINUTE", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="RATE_LIMIT_PER_MINUTE"):
        generate_fastapi_code(functions)


def test_notebook_function_named_rate_limit_window_seconds_is_rejected():

    functions = [
        {"name": "RATE_LIMIT_WINDOW_SECONDS", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="RATE_LIMIT_WINDOW_SECONDS"):
        generate_fastapi_code(functions)


def test_notebook_function_named_rate_limit_windows_is_rejected():

    functions = [
        {"name": "_RATE_LIMIT_WINDOWS", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="_RATE_LIMIT_WINDOWS"):
        generate_fastapi_code(functions)


def test_background_endpoint_rejects_new_tasks_past_max_pending_tasks():
    """_evict_expired_tasks bounds TASKS' long-term growth, but a burst of
    background requests arriving faster than TASK_TTL_SECONDS still grew
    TASKS without limit in the meantime -- nothing stopped a client from
    submitting far more tasks than the process could ever get to,
    exhausting memory well before any of them would expire.
    """

    functions = [{"name": "train_model", "args": [], "return_type": "str"}]

    code = generate_fastapi_code(functions)

    assert 'os.getenv("NOTEBOOK_API_MAX_TASKS", "10000")' in code
    assert "if len(TASKS) >= MAX_PENDING_TASKS:" in code
    assert "status_code=503" in code


def test_non_background_endpoint_has_no_max_pending_tasks_check():
    """A synchronous endpoint never touches TASKS at all -- the check
    only belongs in a background endpoint's own body.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    assert "if len(TASKS) >= MAX_PENDING_TASKS:" not in code
    # The constant itself is still always defined at module level.
    assert "MAX_PENDING_TASKS = int(os.getenv(" in code


def test_notebook_function_named_max_pending_tasks_is_rejected():
    """MAX_PENDING_TASKS is a module-level name the generated app itself
    defines (see RESERVED_INFRASTRUCTURE_NAMES) -- same collision hazard
    class as TASKS, API_KEYS, or MAX_REQUEST_BODY_BYTES.
    """

    functions = [
        {"name": "MAX_PENDING_TASKS", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="MAX_PENDING_TASKS"):
        generate_fastapi_code(functions)


def test_notebook_function_named_webhook_timeout_seconds_is_rejected():
    """WEBHOOK_TIMEOUT_SECONDS is a module-level name the generated app
    itself defines (see RESERVED_INFRASTRUCTURE_NAMES) -- same collision
    hazard class as MAX_PENDING_TASKS or TASK_TTL_SECONDS.
    """

    functions = [
        {"name": "WEBHOOK_TIMEOUT_SECONDS", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="WEBHOOK_TIMEOUT_SECONDS"):
        generate_fastapi_code(functions)


def test_notebook_function_named_webhook_secret_is_rejected():
    """WEBHOOK_SECRET is a module-level name the generated app itself
    defines (see RESERVED_INFRASTRUCTURE_NAMES) -- same collision hazard
    class as WEBHOOK_TIMEOUT_SECONDS or MAX_PENDING_TASKS.
    """

    functions = [
        {"name": "WEBHOOK_SECRET", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="WEBHOOK_SECRET"):
        generate_fastapi_code(functions)


def test_notebook_function_named_deliver_task_webhook_is_rejected():
    """_deliver_task_webhook is a module-level helper the generated app
    itself defines -- same collision hazard class as _evict_expired_tasks
    or _run_background_task: a notebook function of this exact name would
    silently overwrite the real helper at module-execution time.
    """

    functions = [
        {"name": "_deliver_task_webhook", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="_deliver_task_webhook"):
        generate_fastapi_code(functions)


def test_background_endpoint_rejects_non_http_callback_url(monkeypatch):
    """A caller-supplied callback_url is fully caller-controlled input
    (unlike every other limit this app enforces, which is set by the
    operator) -- restricted to http(s) so a "file://" or other
    non-network scheme can't reach urllib.request.urlopen inside
    _deliver_task_webhook at all.
    """

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    namespace["notebook_module"].process_data = lambda: "ok"

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    rejected = client.post(
        "/process_data",
        json={},
        params={"callback_url": "file:///etc/passwd"},
        headers=headers,
    )
    assert rejected.status_code == 400
    assert "callback_url" in rejected.json()["detail"]
    assert namespace["TASKS"] == {}

    accepted = client.post(
        "/process_data",
        json={},
        params={"callback_url": "https://example.test/hook"},
        headers=headers,
    )
    assert accepted.status_code == 200


def test_background_task_delivers_webhook_on_completion_and_failure(monkeypatch):
    """Confirmed missing before this feature: a caller of a background
    endpoint had no way to learn a task finished short of polling
    get_task/wait_for_task -- there was no way to opt into being notified
    the moment it actually completes or fails.
    """
    import urllib.request

    functions = [
        {"name": "process_data", "args": [], "return_type": "dict"},
        {"name": "train_model", "args": [], "return_type": "dict"},
    ]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    namespace["notebook_module"].process_data = lambda: {"score": 0.9}

    def _blows_up():
        raise ValueError("training diverged")

    namespace["notebook_module"].train_model = _blows_up

    delivered = []

    class _FakeResponse:
        def close(self):
            pass

    def fake_urlopen(request, timeout=None):
        delivered.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    ok_response = client.post(
        "/process_data",
        json={},
        params={"callback_url": "https://example.test/hook-ok"},
        headers=headers,
    )
    assert ok_response.status_code == 200

    fail_response = client.post(
        "/train_model",
        json={},
        params={"callback_url": "https://example.test/hook-fail"},
        headers=headers,
    )
    assert fail_response.status_code == 200

    assert len(delivered) == 2

    ok_call = next(c for c in delivered if c["url"] == "https://example.test/hook-ok")
    assert ok_call["timeout"] == 5
    assert ok_call["body"]["status"] == "completed"
    assert ok_call["body"]["result"] == {"score": 0.9}
    assert ok_call["body"]["task_id"] == ok_response.json()["task_id"]

    fail_call = next(
        c for c in delivered if c["url"] == "https://example.test/hook-fail"
    )
    assert fail_call["body"]["status"] == "failed"
    assert "training diverged" in fail_call["body"]["error"]


def test_background_task_without_callback_url_never_touches_urlopen(monkeypatch):
    """The overwhelmingly common case (no callback_url given) must behave
    exactly as before this feature -- no network call attempted at all.
    """
    import urllib.request

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    namespace["notebook_module"].process_data = lambda: "ok"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("urlopen should never be called without callback_url")

    monkeypatch.setattr(urllib.request, "urlopen", fail_if_called)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    response = client.post("/process_data", json={}, headers=headers)
    assert response.status_code == 200


def test_webhook_delivery_failure_does_not_affect_task_result(monkeypatch):
    """Webhook delivery is purely best-effort -- an unreachable/erroring
    callback_url must never prevent the task's own real result from being
    recorded in TASKS.
    """
    import urllib.request

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    namespace["notebook_module"].process_data = lambda: "ok"

    def broken_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", broken_urlopen)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    response = client.post(
        "/process_data",
        json={},
        params={"callback_url": "https://example.test/unreachable"},
        headers=headers,
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    task = namespace["TASKS"][task_id]
    assert task["status"] == "completed"
    assert task["result"] == "ok"


def test_webhook_delivery_omits_signature_header_when_no_secret_configured(
    monkeypatch
):
    """The overwhelmingly common case (NOTEBOOK_API_WEBHOOK_SECRET unset)
    must behave exactly as before this feature -- no signature header at
    all, so an existing receiver that predates this feature keeps working
    unmodified.
    """
    import urllib.request

    monkeypatch.delenv("NOTEBOOK_API_WEBHOOK_SECRET", raising=False)

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    namespace["notebook_module"].process_data = lambda: "ok"

    captured = {}

    class _FakeResponse:
        def close(self):
            pass

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    response = client.post(
        "/process_data",
        json={},
        params={"callback_url": "https://example.test/hook"},
        headers=headers,
    )
    assert response.status_code == 200

    assert captured["request"].get_header("X-webhook-signature") is None


def test_webhook_delivery_includes_hmac_signature_when_secret_configured(
    monkeypatch
):
    """Confirmed missing before this feature: a receiver of a background
    task's own webhook delivery had no way to verify a request actually
    came from this app (rather than an attacker who guessed or leaked the
    callback_url) -- NOTEBOOK_API_WEBHOOK_SECRET, when set, now signs the
    exact request body with HMAC-SHA256, the same X-Hub-Signature-256
    contract GitHub/Stripe webhooks already use.
    """
    import hashlib
    import hmac
    import urllib.request

    monkeypatch.setenv("NOTEBOOK_API_WEBHOOK_SECRET", "s3cr3t")

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    namespace["notebook_module"].process_data = lambda: {"score": 0.9}

    captured = {}

    class _FakeResponse:
        def close(self):
            pass

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    response = client.post(
        "/process_data",
        json={},
        params={"callback_url": "https://example.test/hook"},
        headers=headers,
    )
    assert response.status_code == 200

    request = captured["request"]
    # urllib.request.Request.get_header does an exact lookup against its
    # own stored key, which add_header/the constructor already normalize
    # via str.capitalize() ("X-Webhook-Signature" -> "X-webhook-signature")
    # -- not a case-insensitive lookup, so the header name here must match
    # that exact casing.
    signature_header = request.get_header("X-webhook-signature")
    assert signature_header is not None

    expected_signature = "sha256=" + hmac.new(
        b"s3cr3t", request.data, hashlib.sha256
    ).hexdigest()
    assert signature_header == expected_signature


def test_webhook_delivery_signature_changes_when_secret_changes(monkeypatch):
    """A different NOTEBOOK_API_WEBHOOK_SECRET must produce a different
    signature over the exact same body -- otherwise the secret wouldn't
    actually be doing any verification work.
    """
    import urllib.request

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    def _deliver_with_secret(secret):
        monkeypatch.setenv("NOTEBOOK_API_WEBHOOK_SECRET", secret)

        code = generate_fastapi_code(functions)

        _register_fake_notebook_module(monkeypatch)
        namespace = {}
        exec(compile(code, "<generated>", "exec"), namespace)
        namespace["notebook_module"].process_data = lambda: "ok"

        captured = {}

        class _FakeResponse:
            def close(self):
                pass

        def fake_urlopen(request, timeout=None):
            captured["request"] = request
            return _FakeResponse()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        from fastapi.testclient import TestClient

        client = TestClient(namespace["app"])
        headers = {"X-API-Key": "notebook-to-api-dev-key"}

        client.post(
            "/process_data",
            json={},
            params={"callback_url": "https://example.test/hook"},
            headers=headers,
        )

        return captured["request"].get_header("X-webhook-signature")

    signature_a = _deliver_with_secret("secret-a")
    signature_b = _deliver_with_secret("secret-b")

    assert signature_a != signature_b


def test_notebook_function_named_task_execution_timeout_seconds_is_rejected():
    """TASK_EXECUTION_TIMEOUT_SECONDS is a module-level name the generated
    app itself defines (see RESERVED_INFRASTRUCTURE_NAMES) -- same
    collision hazard class as WEBHOOK_TIMEOUT_SECONDS or MAX_PENDING_TASKS.
    """

    functions = [
        {"name": "TASK_EXECUTION_TIMEOUT_SECONDS", "args": [], "return_type": "dict"}
    ]

    with pytest.raises(ReservedFunctionNameError, match="TASK_EXECUTION_TIMEOUT_SECONDS"):
        generate_fastapi_code(functions)


def test_background_sync_task_hanging_past_the_timeout_is_marked_failed(monkeypatch):
    """Confirmed exploitable before this feature: a hung or runaway
    notebook function (an infinite loop, a network call with no timeout
    of its own) tied up one of this process' limited worker threads
    forever -- the same threadpool every *synchronous* endpoint (GET
    /health included) also runs on, so enough hung tasks would eventually
    starve the entire app, not just background ones.
    """
    import time as time_module

    monkeypatch.setenv("NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS", "1")

    functions = [{"name": "train_model", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    namespace["notebook_module"].train_model = lambda: time_module.sleep(30)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    start = time_module.monotonic()
    response = client.post("/train_model", json={}, headers=headers)
    elapsed = time_module.monotonic() - start

    assert response.status_code == 200
    # abandon_on_cancel=True is what makes this assertion meaningful: the
    # orphaned thread itself keeps sleeping for the full 30s in the
    # background, but this request -- and the coroutine awaiting it --
    # must not be held up waiting for it.
    assert elapsed < 30

    task_id = response.json()["task_id"]
    task = namespace["TASKS"][task_id]
    assert task["status"] == "failed"
    assert "1s" in task["error"]
    assert "execution timeout" in task["error"]


def test_background_async_task_hanging_past_the_timeout_is_marked_failed(monkeypatch):

    import asyncio

    monkeypatch.setenv("NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS", "1")

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    async def _hangs_forever():
        await asyncio.sleep(30)

    namespace["notebook_module"].process_data = _hangs_forever

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    response = client.post("/process_data", json={}, headers=headers)

    assert response.status_code == 200
    task_id = response.json()["task_id"]
    task = namespace["TASKS"][task_id]
    assert task["status"] == "failed"
    assert "execution timeout" in task["error"]


def test_background_task_timeout_delivers_a_failed_webhook(monkeypatch):

    import time as time_module
    import urllib.request

    monkeypatch.setenv("NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS", "1")

    functions = [{"name": "train_model", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    namespace["notebook_module"].train_model = lambda: time_module.sleep(30)

    delivered = []

    class _FakeResponse:
        def close(self):
            pass

    def fake_urlopen(request, timeout=None):
        delivered.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    response = client.post(
        "/train_model",
        json={},
        params={"callback_url": "https://example.test/hook"},
        headers=headers,
    )
    assert response.status_code == 200

    assert len(delivered) == 1
    assert delivered[0]["status"] == "failed"
    assert "execution timeout" in delivered[0]["error"]


def test_background_task_disabled_timeout_preserves_unbounded_execution(monkeypatch):
    """0 (the default) must behave exactly as before this feature existed
    -- a slow-but-finite task still completes normally, never cut off.
    """
    import time as time_module

    monkeypatch.delenv("NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS", raising=False)

    functions = [{"name": "train_model", "args": [], "return_type": "str"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    namespace["notebook_module"].train_model = lambda: (
        time_module.sleep(0.3) or "done"
    )

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    response = client.post("/train_model", json={}, headers=headers)

    assert response.status_code == 200
    task_id = response.json()["task_id"]
    task = namespace["TASKS"][task_id]
    assert task["status"] == "completed"
    assert task["result"] == "done"


def test_generated_app_get_config_reports_task_execution_timeout_seconds(monkeypatch):

    monkeypatch.setenv("NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS", "45")

    functions = [{"name": "add", "args": [], "return_type": "int"}]
    code = generate_fastapi_code(functions)

    assert "'task_execution_timeout_seconds': TASK_EXECUTION_TIMEOUT_SECONDS or None," in code

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    resp = client.get("/config")

    assert resp.status_code == 200
    assert resp.json()["task_execution_timeout_seconds"] == 45


def test_generated_app_get_config_reports_null_task_execution_timeout_by_default(
    monkeypatch
):

    monkeypatch.delenv("NOTEBOOK_API_TASK_EXECUTION_TIMEOUT_SECONDS", raising=False)

    functions = [{"name": "add", "args": [], "return_type": "int"}]
    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    resp = client.get("/config")

    assert resp.status_code == 200
    assert resp.json()["task_execution_timeout_seconds"] is None


def test_route_generation():

    functions = [
        {
            "name": "predict",
            "args": [],
            "return_type": None
        }
    ]

    code = generate_fastapi_code(functions)

    assert "/predict" in code


def test_async_function_generates_awaited_async_endpoint():

    functions = [
        {
            "name": "fetch_data",
            "args": [{"name": "url", "type": "str"}],
            "return_type": "dict",
            "is_async": True,
        }
    ]

    code = generate_fastapi_code(functions)

    assert "async def fetch_data(" in code
    assert "await notebook_module.fetch_data(" in code


def test_sync_function_generates_unawaited_sync_endpoint():

    functions = [
        {
            "name": "add",
            "args": [{"name": "a", "type": "int"}],
            "return_type": "int",
            "is_async": False,
        }
    ]

    code = generate_fastapi_code(functions)

    assert "def add(" in code
    assert "async def add(" not in code
    assert "await notebook_module.add(" not in code
    assert "result = notebook_module.add(" in code


def test_keyword_only_arg_is_passed_by_keyword_in_generated_call():

    functions = [
        {
            # Deliberately not a LONG_RUNNING_KEYWORDS name, so this takes
            # the direct-call endpoint path rather than the background-task
            # path (which forwards args differently, through add_task).
            "name": "score",
            "args": [
                {"name": "data", "type": "list", "kind": "positional"},
                {"name": "epochs", "type": "int", "default": 10, "has_default": True, "kind": "keyword_only"},
            ],
            "return_type": "dict",
        }
    ]

    code = generate_fastapi_code(functions)

    assert "notebook_module.score(req.data, epochs=req.epochs)" in code


def test_tasks_endpoints_require_api_key_auth():
    """Confirmed exploitable before this fix: the /tasks family of
    endpoints (which return stored function call inputs/outputs, or let
    a caller wipe task state) omitted Depends(verify_api_key) even though
    every per-function endpoint and /auth/validate require it -- anyone
    could read past task results or delete all task state with no
    credentials at all.
    """

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    list_tasks_signature = code[code.index("def list_tasks("):code.index("):", code.index("def list_tasks(")) + 2]
    assert "_: None = Depends(verify_api_key)" in list_tasks_signature
    assert "def get_task(task_id: str, _: None = Depends(verify_api_key)):" in code
    assert "def delete_completed_tasks(_: None = Depends(verify_api_key)):" in code
    assert "def delete_failed_tasks(_: None = Depends(verify_api_key)):" in code
    assert "def cleanup_tasks(_: None = Depends(verify_api_key)):" in code
    assert "def reset_tasks(_: None = Depends(verify_api_key)):" in code
    assert "def delete_task(task_id: str, _: None = Depends(verify_api_key)):" in code


def test_background_task_creation_evicts_expired_tasks_and_stamps_created_at():
    """Confirmed exploitable before this fix: TASKS is an in-memory dict
    with no automatic eviction anywhere in the generated app -- nothing
    calls the manual /tasks/cleanup-style endpoints on its own, so a
    long-running deployment handling steady background-task traffic
    accumulates one entry per call forever. A new task's creation must
    both stamp a created_at timestamp (needed to determine expiry) and
    sweep out anything already past TASK_TTL_SECONDS.
    """

    functions = [
        {"name": "process_data", "args": [], "return_type": "dict"},
    ]

    code = generate_fastapi_code(functions)

    assert "TASK_TTL_SECONDS = int(os.getenv(" in code
    assert '"created_at": time.time()' in code
    assert "_evict_expired_tasks()" in code
    # Eviction must run before the new task is recorded, not after --
    # otherwise the brand new task could itself be swept if TTL is 0.
    assert code.index("_evict_expired_tasks()") < code.index('TASKS[task_id] = {"status": "processing"')


def test_list_tasks_supports_status_filter_and_pagination(monkeypatch):
    """Confirmed exploitable before this fix: GET /tasks always returned
    the *entire* TASKS dict with no filter and no pagination -- with
    NOTEBOOK_API_MAX_TASKS defaulting to 10000, and each task potentially
    carrying a large `result` payload, a single request could return an
    enormous response body, and there was no way to ask for just e.g. the
    failed tasks without fetching and filtering client-side.
    """

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    assert "status: Optional[str] = None," in code
    assert "limit: int = Query(default=100, ge=1, le=1000)," in code
    assert "offset: int = Query(default=0, ge=0)," in code
    assert "matching_items.sort(key=lambda item: item[1].get('created_at', 0), reverse=True)" in code
    assert "'matching_tasks': len(matching_items)," in code

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    namespace["TASKS"]["t-old-failed"] = {
        "status": "failed", "created_at": 1.0, "error": "boom",
    }
    namespace["TASKS"]["t-new-completed"] = {
        "status": "completed", "created_at": 3.0, "result": "ok",
    }
    namespace["TASKS"]["t-mid-processing"] = {
        "status": "processing", "created_at": 2.0,
    }

    response = client.get("/tasks", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["matching_tasks"] == 3
    assert body["limit"] == 100
    assert body["offset"] == 0
    # Most recently created task first.
    assert list(body["tasks"].keys()) == [
        "t-new-completed", "t-mid-processing", "t-old-failed",
    ]

    filtered = client.get("/tasks", params={"status": "failed"}, headers=headers)
    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert filtered_body["matching_tasks"] == 1
    assert list(filtered_body["tasks"].keys()) == ["t-old-failed"]

    paginated = client.get(
        "/tasks", params={"limit": 1, "offset": 1}, headers=headers
    )
    assert paginated.status_code == 200
    paginated_body = paginated.json()
    assert paginated_body["matching_tasks"] == 3
    assert list(paginated_body["tasks"].keys()) == ["t-mid-processing"]

    invalid_status = client.get(
        "/tasks", params={"status": "bogus"}, headers=headers
    )
    assert invalid_status.status_code == 400
    assert "Invalid status 'bogus'" in invalid_status.json()["detail"]

    invalid_limit = client.get("/tasks", params={"limit": 0}, headers=headers)
    assert invalid_limit.status_code == 422


def test_evict_expired_tasks_never_evicts_a_processing_task(monkeypatch):
    """Confirmed exploitable before this fix: _evict_expired_tasks swept
    out any task past TASK_TTL_SECONDS purely by created_at age, with no
    regard for whether it was still 'processing' -- a background function
    (train/process/generate/embed/scrape, routinely slow by design) that
    simply took longer than the TTL to finish had its own TASKS entry
    evicted while still running, out from under it.
    """

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    long_ago = namespace["time"].time() - namespace["TASK_TTL_SECONDS"] - 1

    namespace["TASKS"]["still-running"] = {
        "status": "processing", "created_at": long_ago,
    }
    namespace["TASKS"]["long-done"] = {
        "status": "completed", "created_at": long_ago, "result": "ok",
    }

    namespace["_evict_expired_tasks"]()

    assert "still-running" in namespace["TASKS"]
    assert "long-done" not in namespace["TASKS"]


def test_delete_task_rejects_deletion_of_a_processing_task(monkeypatch):
    """Confirmed exploitable before this fix: DELETE /tasks/{task_id}
    popped a task's TASKS entry regardless of its status -- there is no
    way to actually cancel work already handed to a background thread, so
    deleting a still-processing task just meant _run_background_task's
    own eventual TASKS[task_id][...] write raised a bare KeyError once
    the task finished, an unhandled exception silently losing that task's
    real result or error.
    """

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    headers = {"X-API-Key": "notebook-to-api-dev-key"}

    namespace["TASKS"]["still-running"] = {"status": "processing"}
    namespace["TASKS"]["already-done"] = {"status": "completed", "result": "ok"}

    conflict = client.delete("/tasks/still-running", headers=headers)
    assert conflict.status_code == 409
    assert "still processing" in conflict.json()["detail"]
    assert "still-running" in namespace["TASKS"]

    ok = client.delete("/tasks/already-done", headers=headers)
    assert ok.status_code == 200
    assert ok.json()["status"] == "completed"
    assert "already-done" not in namespace["TASKS"]

    missing = client.delete("/tasks/does-not-exist", headers=headers)
    assert missing.status_code == 404


def test_run_background_task_tolerates_a_missing_tasks_entry(monkeypatch):
    """_evict_expired_tasks (above) never removes a 'processing' task, but
    POST /tasks/reset still unconditionally clears every entry, in-flight
    or not. Confirmed exploitable before this fix: a task still running
    when /tasks/reset fired raised a bare KeyError from inside
    _run_background_task once it finished -- an unhandled exception in a
    fire-and-forget asyncio task -- for both the success path
    (TASKS[task_id]["status"] = "completed") and the failure path
    (TASKS[task_id]["error"] = ...), since the second raised again against
    the same missing key.
    """
    import asyncio

    functions = [{"name": "process_data", "args": [], "return_type": "dict"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    run_background_task = namespace["_run_background_task"]

    def succeeds():
        return "ok"

    def fails():
        raise ValueError("boom")

    # Neither task_id was ever added to TASKS -- simulating one removed
    # out from under a still-running task by POST /tasks/reset. Must not
    # raise, and must not resurrect the entry.
    asyncio.run(run_background_task(succeeds, "reset-away-success"))
    asyncio.run(run_background_task(fails, "reset-away-failure"))

    assert "reset-away-success" not in namespace["TASKS"]
    assert "reset-away-failure" not in namespace["TASKS"]


def test_background_endpoint_documents_the_task_response_it_actually_sends():
    """Confirmed wrong before this fix: a background endpoint's decorator
    documented `example_response`/the function's own return type (e.g.
    {"result": ""}) as its 200 response -- but the function body actually
    always `return`s {"task_id": ..., "status": "processing"} instead,
    with the real result only available later via GET /tasks/{task_id}.
    /docs, and any third-party tool generating a client from
    openapi.json, would be told to expect a response this endpoint never
    sends.
    """

    functions = [
        {
            "name": "train_model",
            "args": [],
            "return_type": "str",
            "example_response": {"result": "trained"},
        },
    ]

    code = generate_fastapi_code(functions)

    decorator_line = next(
        line for line in code.splitlines() if '@app.post("/train_model"' in line
    )

    assert "'task_id': '<uuid>'" in decorator_line
    assert "'status': 'processing'" in decorator_line
    assert "trained" not in decorator_line
    assert '"x-notebook-to-api-async": True' in decorator_line


def test_non_background_endpoint_is_not_marked_async_and_documents_its_own_result():

    functions = [
        {
            "name": "add",
            "args": [],
            "return_type": "int",
            "example_response": {"result": 3},
        },
    ]

    code = generate_fastapi_code(functions)

    decorator_line = next(
        line for line in code.splitlines() if '@app.post("/add"' in line
    )

    assert "x-notebook-to-api-async" not in decorator_line
    assert "'result': 3" in decorator_line


def test_sync_endpoint_documents_401_429_and_500_in_its_openapi_schema(monkeypatch):
    """Confirmed missing before this feature: a generated endpoint's own
    OpenAPI schema documented only its 200 response and FastAPI's own
    automatic 422 -- 401 (verify_api_key) and 429 (_enforce_rate_limit)
    run ahead of every endpoint's own body via Depends(verify_api_key),
    and a sync endpoint's own body can raise 500 (the notebook function's
    exception, or a non-JSON-serializable return value), but none of
    that was ever documented anywhere a caller (or a codegen tool reading
    openapi.json) could actually see it.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    schema = namespace["app"].openapi()
    responses = schema["paths"]["/add"]["post"]["responses"]

    assert set(responses) == {"200", "401", "429", "500", "422"}
    assert responses["401"]["description"] == "Missing or invalid X-API-Key header."
    assert "NOTEBOOK_API_RATE_LIMIT_PER_MINUTE" in responses["429"]["description"]
    assert "'add' raised" in responses["500"]["description"]


def test_background_endpoint_documents_400_401_429_and_503_in_its_openapi_schema(
    monkeypatch,
):
    """Same gap as the synchronous case above, plus the two extra
    failure modes only a background endpoint's own body can produce:
    503 (NOTEBOOK_API_MAX_TASKS already at capacity) and 400 (a
    caller-supplied ?callback_url= that isn't http(s)).
    """

    functions = [{"name": "train_model", "args": [], "return_type": "str"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    schema = namespace["app"].openapi()
    responses = schema["paths"]["/train_model"]["post"]["responses"]

    assert set(responses) == {"200", "400", "401", "429", "503", "422"}
    assert "callback_url" in responses["400"]["description"]
    assert "NOTEBOOK_API_MAX_TASKS" in responses["503"]["description"]
    assert responses["401"]["description"] == "Missing or invalid X-API-Key header."


def test_documented_401_response_matches_a_real_unauthenticated_request(monkeypatch):
    """The documented 401 example must actually match what a real,
    unauthenticated request gets back -- not just be a plausible-looking
    but disconnected description.
    """

    functions = [{"name": "add", "args": [], "return_type": "int"}]

    code = generate_fastapi_code(functions)

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    from fastapi.testclient import TestClient

    client = TestClient(namespace["app"])
    documented_example = (
        namespace["app"].openapi()["paths"]["/add"]["post"]["responses"]["401"]
        ["content"]["application/json"]["example"]
    )

    response = client.post("/add", json={})

    assert response.status_code == 401
    assert response.json() == documented_example


def test_keyword_only_arg_forwarded_by_keyword_through_background_task():

    functions = [
        {
            "name": "train",
            "args": [
                {"name": "data", "type": "list", "kind": "positional"},
                {"name": "epochs", "type": "int", "default": 10, "has_default": True, "kind": "keyword_only"},
            ],
            "return_type": "dict",
        }
    ]

    code = generate_fastapi_code(functions)

    assert (
        "background_tasks.add_task(_run_background_task, notebook_module.train, "
        "task_id, req.data, epochs=req.epochs, callback_url=callback_url)"
    ) in code


def test_field_with_explicit_none_default_is_not_required():
    """A default of None (has_default=True, default=None) must produce an
    optional Pydantic field, not a required one -- otherwise the generated
    endpoint 422s on any call that omits the field, even though the
    underlying notebook function has a perfectly valid default.
    """

    functions = [
        {
            "name": "greet",
            "args": [
                {"name": "name", "type": "str", "has_default": False, "kind": "positional"},
                {"name": "title", "type": "str", "default": None, "has_default": True, "kind": "positional"},
            ],
            "return_type": "str",
        }
    ]

    code = generate_fastapi_code(functions)

    assert "title: str = Field(default=None," in code
    assert "name: str = Field(description=" in code
    assert "name: str = Field(default=" not in code


def test_field_with_no_default_is_required():

    functions = [
        {
            "name": "greet",
            "args": [
                {"name": "name", "type": "str", "has_default": False, "kind": "positional"},
            ],
            "return_type": "str",
        }
    ]

    code = generate_fastapi_code(functions)

    assert "name: str = Field(description=" in code
    assert "default=" not in code.split("class GreetRequest(BaseModel):")[1].split("\n\n")[0]


def test_field_uses_docstring_arg_description_over_the_generic_fallback():
    """Confirmed missing before this feature: extract_functions_from_code
    (backend/parser/ast_parser.py) now attaches each parameter's own
    Google-style "Args:" description, but generate_fastapi_code ignored
    it entirely and always fell back to a generic "Parameter 'x' of type
    T" -- no matter how thoroughly the notebook author had actually
    documented the function.
    """

    functions = [
        {
            "name": "train",
            "args": [
                {
                    "name": "epochs", "type": "int", "has_default": False,
                    "kind": "positional",
                    "description": "Number of training passes.",
                },
            ],
            "return_type": "str",
        }
    ]

    code = generate_fastapi_code(functions)

    assert "description='Number of training passes.'" in code
    assert "Parameter 'epochs' of type int" not in code


def test_field_falls_back_to_generic_description_when_undocumented():

    functions = [
        {
            "name": "train",
            "args": [
                {
                    "name": "epochs", "type": "int", "has_default": False,
                    "kind": "positional", "description": None,
                },
            ],
            "return_type": "str",
        }
    ]

    code = generate_fastapi_code(functions)

    assert "description=\"Parameter 'epochs' of type int\"" in code


def test_field_does_not_override_an_annotations_own_field_description():
    """Confirmed exploitable before this fix: Pydantic merges an
    Annotated[...] metadata's own FieldInfo with the one assigned as the
    field's default value, and the *assigned* one's description wins on
    conflict -- so this generator's own unconditional
    description=repr(field_description) silently discarded a notebook
    author's own Annotated[int, Field(description=...)] description in
    the actual served OpenAPI schema, with nothing to indicate it had
    been overridden.
    """

    functions = [
        {
            "name": "compute",
            "args": [
                {
                    "name": "x",
                    "type": 'Annotated[int, Field(gt=0, description="must be positive")]',
                    "has_default": False, "kind": "positional",
                    "description": None,
                },
            ],
            "return_type": "int",
        }
    ]

    code = generate_fastapi_code(functions)

    class_body = code.split("class ComputeRequest(BaseModel):")[1].split("\n\n")[0]

    assert "must be positive" in class_body
    assert "Parameter 'x' of type" not in class_body
    # No redundant/overriding Field(...) assignment at all -- the
    # Annotated[...] metadata already carries everything meaningful.
    assert "= Field(" not in class_body


def test_field_with_default_still_assigns_default_when_annotation_has_own_description():
    """The no-assignment shortcut above only applies when there's no
    default to assign -- a default must still be attached via
    Field(default=...), but without an overriding description= alongside
    it.
    """

    functions = [
        {
            "name": "compute",
            "args": [
                {
                    "name": "x",
                    "type": 'Annotated[int, Field(gt=0, description="must be positive")]',
                    "has_default": True, "default": 5, "default_is_literal": True,
                    "kind": "positional", "description": None,
                },
            ],
            "return_type": "int",
        }
    ]

    code = generate_fastapi_code(functions)

    class_body = code.split("class ComputeRequest(BaseModel):")[1].split("\n\n")[0]

    assert "Field(default=5)" in class_body
    # Exactly one "description=" -- the annotation's own, embedded
    # inside Annotated[...]; none added by the outer assigned Field(...).
    assert class_body.count("description=") == 1


def test_field_prefers_annotations_own_description_over_docstring_description():
    """When both a docstring Args: entry and the annotation's own
    Annotated[..., Field(description=...)] document the same parameter,
    the more explicit, closer-to-usage annotation wins -- generating a
    redundant/conflicting outer description would be worse than just
    leaving the one already attached directly to the field's own type.
    """

    functions = [
        {
            "name": "compute",
            "args": [
                {
                    "name": "x",
                    "type": 'Annotated[int, Field(description="from annotation")]',
                    "has_default": False, "kind": "positional",
                    "description": "from docstring",
                },
            ],
            "return_type": "int",
        }
    ]

    code = generate_fastapi_code(functions)

    class_body = code.split("class ComputeRequest(BaseModel):")[1].split("\n\n")[0]

    assert "from docstring" not in class_body
    assert "= Field(" not in class_body


def test_typing_generic_argument_types_get_a_matching_typing_import(monkeypatch):
    """Confirmed exploitable before this fix: arg["type"] (a raw
    ast.unparse'd annotation like "List[float]" or "Optional[str]") was
    written straight into the generated Pydantic model with no matching
    `from typing import ...`, so building the model at runtime raised
    `PydanticUserError: 'PredictRequest' is not fully defined; you should
    define 'List', then call 'PredictRequest.model_rebuild()'` the first
    time FastAPI needed the schema (i.e. on the first request or /docs
    load, not at compile time).
    """

    functions = [
        {
            "name": "predict",
            "args": [
                {"name": "items", "type": "List[float]", "has_default": False, "kind": "positional"},
                {"name": "name", "type": "Optional[str]", "default": None, "has_default": True, "kind": "positional"},
                {"name": "meta", "type": "Dict[str, Any]", "default": None, "has_default": True, "kind": "positional"},
            ],
            "return_type": "str",
        }
    ]

    code = generate_fastapi_code(functions)

    assert "from typing import Any, Dict, List, Optional" in code
    assert "items: List[float] = Field(" in code
    assert "name: Optional[str] = Field(" in code
    assert "meta: Dict[str, Any] = Field(" in code

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    schema = namespace["PredictRequest"].model_json_schema()
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["properties"]["name"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


def test_untyped_argument_defaults_to_str_not_the_literal_none_type():
    """Confirmed exploitable before this fix: arg.get("type", "str") only
    falls back to "str" when the "type" key is *absent*, but the parser
    always sets it (to None when there's no annotation), so an untyped
    notebook parameter produced a field literally annotated `: None`,
    rejecting every value including its own default.
    """

    functions = [
        {
            "name": "greet",
            "args": [
                {"name": "name", "type": None, "has_default": True, "default": "world", "kind": "positional"},
            ],
            "return_type": "str",
        }
    ]

    code = generate_fastapi_code(functions)

    assert "name: str = Field(" in code
    assert ": None = Field(" not in code


def test_notebook_defined_type_is_qualified_with_notebook_module():
    """A bare class/Enum name from the notebook (e.g. a Status Enum used as
    a parameter type) isn't defined anywhere in the generated app's own
    namespace, so referencing it unqualified raises a NameError while
    building the model. It must be qualified as `notebook_module.<name>`,
    the alias the generated app already imports the notebook's runtime
    module under.
    """

    functions = [
        {
            "name": "set_status",
            "args": [
                {"name": "status", "type": "Status", "has_default": False, "kind": "positional"},
            ],
            "return_type": "str",
        }
    ]

    code = generate_fastapi_code(functions)

    assert "status: notebook_module.Status = Field(" in code
    assert "status: Status = Field(" not in code
    # The human-readable Field description should stay unqualified.
    assert "of type Status" in code


def test_non_literal_default_is_embedded_as_a_qualified_expression_not_a_string():
    """Confirmed exploitable before this fix: a default that isn't a
    literal_eval-able literal (e.g. a notebook-defined Enum member like
    `Priority.HIGH`) was repr()'d exactly like a real literal default,
    which silently turned it into the *string* "Priority.HIGH" in the
    generated Pydantic model instead of the actual enum member -- a
    caller omitting that field to take its default then passed the raw
    string straight into the notebook's own function, breaking whatever
    it did with the real enum member (e.g. `.value`).
    """

    functions = [
        {
            "name": "set_priority",
            "args": [
                {
                    "name": "priority",
                    "type": "Priority",
                    "has_default": True,
                    "default_is_literal": False,
                    "default": "Priority.HIGH",
                    "kind": "positional",
                },
            ],
            "return_type": "str",
        }
    ]

    code = generate_fastapi_code(functions)

    assert (
        "priority: notebook_module.Priority = Field("
        "default=notebook_module.Priority.HIGH, "
        in code
    )
    assert "default='Priority.HIGH'" not in code
    assert 'default="Priority.HIGH"' not in code


def test_literal_default_is_still_repr_embedded():
    """A real literal default (the common case) must keep going through
    repr(), not the qualification path above -- e.g. a plain string
    default must stay a quoted string literal, not be treated as a bare
    expression referencing a notebook name.
    """

    functions = [
        {
            "name": "greet",
            "args": [
                {
                    "name": "name",
                    "type": "str",
                    "has_default": True,
                    "default_is_literal": True,
                    "default": "world",
                    "kind": "positional",
                },
            ],
            "return_type": "str",
        }
    ]

    code = generate_fastapi_code(functions)

    assert "default='world'" in code


def test_pydantic_model_generation():

    functions = [
        {
            "name": "train_model",
            "args": [
                {
                    "name": "epochs",
                    "type": "int"
                }
            ],
            "return_type": None
        }
    ]

    code = generate_fastapi_code(functions)

    assert "BaseModel" in code


def test_zero_argument_function_produces_a_valid_request_model():
    """Confirmed exploitable before this fix: a zero-parameter notebook
    function (e.g. `def health(): ...`) produced `class HealthRequest
    (BaseModel):` with no fields and no model_config -- an empty class
    body, which is a SyntaxError that fails to compile the *entire*
    generated app, not just this one endpoint.
    """

    functions = [
        {"name": "get_status", "args": [], "return_type": "dict"},
    ]

    code = generate_fastapi_code(functions)

    compile(code, "<generated>", "exec")
    assert "class Get_statusRequest(BaseModel):\n    pass" in code


def test_notebook_function_named_verify_api_key_is_rejected():
    """Confirmed exploitable before this fix: a notebook function named
    verify_api_key was emitted as `def verify_api_key(...)`, rebinding the
    module-level name the real auth check is defined under. Since
    Depends(verify_api_key) defaults are resolved at def-statement
    execution time (top-to-bottom module load), every endpoint defined
    *after* the collision silently got Depends(verify_api_key) pointing
    at the notebook's own function instead of the real guard -- disabling
    API-key authentication for the rest of the app with no error.
    """

    functions = [
        {"name": "verify_api_key", "args": [], "return_type": "dict"},
    ]

    with pytest.raises(ReservedFunctionNameError, match="verify_api_key"):
        generate_fastapi_code(functions)


def test_notebook_function_named_after_other_reserved_infrastructure_is_rejected():

    for reserved_name in ["custom_openapi", "root", "health_check", "notebook_module", "TASKS"]:
        functions = [
            {"name": reserved_name, "args": [], "return_type": "dict"},
        ]

        with pytest.raises(ReservedFunctionNameError):
            generate_fastapi_code(functions)


def test_non_colliding_functions_alongside_a_reserved_name_still_raise():
    """The whole compile must fail clearly rather than silently dropping
    just the colliding function -- a silently-dropped endpoint could be
    just as confusing as a silent auth bypass, so this must be a loud,
    actionable error, not a silent skip.
    """

    functions = [
        {"name": "train_model", "args": [], "return_type": "dict"},
        {"name": "verify_api_key", "args": [], "return_type": "dict"},
    ]

    with pytest.raises(ReservedFunctionNameError):
        generate_fastapi_code(functions)


def test_functions_colliding_on_request_model_name_get_distinct_classes(monkeypatch):
    """Confirmed exploitable before this fix: model_name only uppercased
    the function name's first character, so "get_data" and "Get_data"
    (two distinct, valid Python function names) both produced the class
    name "Get_dataRequest". The second class definition silently shadowed
    the first, so BOTH endpoints resolved to the same class -- the first
    function's endpoint ended up validating requests against the
    *second* function's fields, with no compile-time or runtime error.
    """

    functions = [
        {
            "name": "get_data",
            "args": [{"name": "query", "type": "str", "has_default": False, "kind": "positional"}],
            "return_type": "dict",
        },
        {
            "name": "Get_data",
            "args": [{"name": "id", "type": "int", "has_default": False, "kind": "positional"}],
            "return_type": "dict",
        },
    ]

    code = generate_fastapi_code(functions)

    compile(code, "<generated>", "exec")

    assert code.count("class Get_dataRequest(BaseModel):") == 1
    assert code.count("class Get_dataRequest_2(BaseModel):") == 1
    assert "def get_data(req: Get_dataRequest, " in code
    assert "def Get_data(req: Get_dataRequest_2, " in code

    _register_fake_notebook_module(monkeypatch)
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)

    assert "query" in namespace["Get_dataRequest"].model_fields
    assert "id" in namespace["Get_dataRequest_2"].model_fields


def test_pipeline_model_generator():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineModelGenerator

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source", "config", "input_size"],
        output_fields=["result", "metric_count"],
        execution_stages=1,
        parallelism_score=1.0,
    )

    generator = PipelineModelGenerator()
    generated_code = generator.generate_request_model(spec)

    assert "class RunPipelineRequest(" in generated_code
    assert "source: str" in generated_code
    assert "config: str" in generated_code
    assert "input_size: int" in generated_code

    generated_resp = generator.generate_response_model(spec)
    assert "class RunPipelineResponse(" in generated_resp
    assert "result: str" in generated_resp
    assert "metric_count: int" in generated_resp

    from backend.generator.pipeline_route_generator import PipelineRouteGenerator
    route_gen = PipelineRouteGenerator()
    generated_route = route_gen.generate_route(spec)
    assert "response_model=\n        RunPipelineResponse" in generated_route or "response_model=RunPipelineResponse" in generated_route or "response_model=" in generated_route

    assert spec.metadata_name() == "RunPipelineMetadata"
    metadata = generator.schema_generator.generate_metadata(spec)
    assert metadata.input_count() == 3
    assert metadata.output_count() == 2
    assert len(metadata.all_fields()) == 5

    openapi_schema = generator.schema_generator.generate_openapi_schema(spec)
    assert openapi_schema["endpoint"] == "run_pipeline"
    assert openapi_schema["request"]["source"] == {"type": "str"}
    assert openapi_schema["request"]["input_size"] == {"type": "int"}
    assert openapi_schema["response"]["result"] == {"type": "str"}
    assert openapi_schema["response"]["metric_count"] == {"type": "int"}

    sdk_types = generator.schema_generator.generate_sdk_types(spec)
    assert sdk_types["request_types"]["source"] == "str"
    assert sdk_types["request_types"]["input_size"] == "int"
    assert sdk_types["response_types"]["result"] == "str"
    assert sdk_types["response_types"]["metric_count"] == "int"

    assert spec.typescript_request_name() == "RunPipelineRequest"
    assert spec.typescript_response_name() == "RunPipelineResponse"

    ts_interfaces = generator.schema_generator.generate_typescript_interfaces(spec)
    assert "export interface RunPipelineRequest {" in ts_interfaces["request"]
    assert "source: string;" in ts_interfaces["request"]
    assert "input_size: number;" in ts_interfaces["request"]
    assert "export interface RunPipelineResponse {" in ts_interfaces["response"]
    assert "result: string;" in ts_interfaces["response"]
    assert "metric_count: number;" in ts_interfaces["response"]

    assert spec.client_method_name() == "run_pipeline"
    ts_client = generator.schema_generator.generate_typescript_client(spec)
    assert "export async function run_pipeline(" in ts_client
    assert "request: RunPipelineRequest" in ts_client
    assert "Promise<RunPipelineResponse>" in ts_client
    assert '"/run_pipeline"' in ts_client

    assert spec.sdk_module_name() == "run_pipeline_sdk"
    assert spec.sdk_filename() == "run_pipeline_sdk.ts"
    ts_sdk = generator.schema_generator.generate_typescript_sdk(spec)
    assert "export interface RunPipelineRequest {" in ts_sdk
    assert "export interface RunPipelineResponse {" in ts_sdk
    assert "export async function run_pipeline(" in ts_sdk

    sdk_index = generator.schema_generator.generate_sdk_index([spec])
    assert 'export * from "./run_pipeline_sdk";' in sdk_index

    assert spec.npm_package_name() == "run-pipeline-sdk"
    assert spec.package_directory() == "run-pipeline-sdk"
    sdk_package = generator.schema_generator.generate_sdk_package(spec.npm_package_name())
    assert '"name": "run-pipeline-sdk"' in sdk_package["package_json"]
    assert '"compilerOptions": {' in sdk_package["tsconfig"]

    sdk_project = generator.schema_generator.generate_sdk_project([spec])
    assert sdk_project.file_count() == 4  # package.json, tsconfig.json, src/index.ts, src/run_pipeline_sdk.ts
    file_names = sdk_project.file_names()
    assert "package.json" in file_names
    assert "tsconfig.json" in file_names
    assert "src/index.ts" in file_names
    assert "src/run_pipeline_sdk.ts" in file_names
    assert "export interface RunPipelineRequest {" in sdk_project.files["src/run_pipeline_sdk.ts"]


def test_performance_report_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, PerformanceReportGenerator
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    report = PerformanceReportGenerator().generate()

    assert report.title == "Performance Report"
    assert report.section_count == 7
    assert report.sections == [
        "Performance Assessment",
        "Bottleneck Detection",
        "Scalability Analysis",
        "Capacity Planning",
        "Performance Optimization",
        "Performance Recommendations",
        "Performance Scorecard",
    ]

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.performance_report_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_report = generator.generate_performance_report()
    assert generated_report.title == "Performance Report"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.performance_report_manifest(report)
    assert manifest["title"] == "Performance Report"
    assert manifest["section_count"] == 7


def test_performance_intelligence_control_center_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, PerformanceIntelligenceControlCenterGenerator
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    control_center = PerformanceIntelligenceControlCenterGenerator().generate()

    assert control_center.performance_assessment_enabled is True
    assert control_center.bottleneck_detection_enabled is True
    assert control_center.scalability_analysis_enabled is True
    assert control_center.capacity_planning_enabled is True
    assert control_center.performance_optimization_enabled is True
    assert control_center.performance_recommendations_enabled is True
    assert control_center.performance_scorecard_enabled is True
    assert control_center.performance_report_enabled is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.performance_intelligence_control_center_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_control_center = generator.generate_performance_intelligence_control_center()
    assert generated_control_center.performance_report_enabled is True

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.performance_intelligence_manifest(control_center)
    assert manifest["performance_assessment_enabled"] is True
    assert manifest["performance_report_enabled"] is True


def test_performance_automation_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, PerformanceAutomationEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    automation = PerformanceAutomationEngine().generate()

    assert automation.workflow_name == "performance_monitoring"
    assert automation.triggers == [
        "latency_threshold_exceeded",
        "throughput_drop_detected",
        "bottleneck_identified",
    ]
    assert automation.actions == [
        "generate_performance_report",
        "notify_platform_team",
        "create_optimization_ticket",
    ]

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.performance_automation_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_automation = generator.generate_performance_automation()
    assert generated_automation.workflow_name == "performance_monitoring"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.performance_automation_manifest(automation)
    assert manifest["workflow_name"] == "performance_monitoring"
    assert manifest["trigger_count"] == 3
    assert manifest["action_count"] == 3


def test_performance_remediation_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, PerformanceRemediationEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    remediation = PerformanceRemediationEngine().generate()

    assert remediation.issue_type == "high_latency"
    assert remediation.priority == "high"
    assert remediation.remediation_actions == [
        "optimize_database_queries",
        "increase_cache_hit_rate",
        "scale_application_instances",
        "enable_connection_pooling",
    ]

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.performance_remediation_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_remediation = generator.generate_performance_remediation()
    assert generated_remediation.issue_type == "high_latency"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.performance_remediation_manifest(remediation)
    assert manifest["issue_type"] == "high_latency"
    assert manifest["action_count"] == 4
    assert manifest["priority"] == "high"


def test_performance_governance_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, PerformanceGovernanceEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    governance = PerformanceGovernanceEngine().generate()

    assert governance.performance_owner == "platform_team"
    assert governance.review_frequency == "monthly"
    assert governance.sla_review_required is True
    assert governance.benchmark_review_required is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.performance_governance_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_governance = generator.generate_performance_governance()
    assert generated_governance.performance_owner == "platform_team"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.performance_governance_manifest(governance)
    assert manifest["performance_owner"] == "platform_team"
    assert manifest["review_frequency"] == "monthly"
    assert manifest["sla_review_required"] is True
    assert manifest["benchmark_review_required"] is True


def test_autonomous_performance_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AutonomousPerformanceEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    performance = AutonomousPerformanceEngine().generate()

    assert performance.self_tuning_enabled is True
    assert performance.adaptive_scaling_enabled is True
    assert performance.performance_learning_enabled is True
    assert performance.continuous_optimization_enabled is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.autonomous_performance_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_performance = generator.generate_autonomous_performance()
    assert generated_performance.self_tuning_enabled is True

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.autonomous_performance_manifest(performance)
    assert manifest["self_tuning_enabled"] is True
    assert manifest["adaptive_scaling_enabled"] is True
    assert manifest["performance_learning_enabled"] is True
    assert manifest["continuous_optimization_enabled"] is True


def test_ai_readiness_assessment_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AIReadinessAssessmentEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    assessment = AIReadinessAssessmentEngine().generate()

    assert assessment.ai_readiness_score == 94.0
    assert assessment.llm_compatibility_score == 92.0
    assert assessment.agent_readiness_score == 90.0
    assert assessment.ai_readiness_grade == "A"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_readiness_assessment_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_assessment = generator.generate_ai_readiness_assessment()
    assert generated_assessment.ai_readiness_score == 94.0

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_readiness_assessment_manifest(assessment)
    assert manifest["ai_readiness_score"] == 94.0
    assert manifest["llm_compatibility_score"] == 92.0
    assert manifest["agent_readiness_score"] == 90.0
    assert manifest["ai_readiness_grade"] == "A"


def test_llm_integration_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, LLMIntegrationEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    integration = LLMIntegrationEngine().generate()

    assert integration.provider == "OpenAI"
    assert integration.interaction_pattern == "tool_calling"
    assert integration.recommended_model == "gpt-5.5"
    assert integration.prompt_strategy == "structured_system_prompt"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.llm_integration_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_integration = generator.generate_llm_integration()
    assert generated_integration.provider == "OpenAI"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.llm_integration_manifest(integration)
    assert manifest["provider"] == "OpenAI"
    assert manifest["interaction_pattern"] == "tool_calling"
    assert manifest["recommended_model"] == "gpt-5.5"
    assert manifest["prompt_strategy"] == "structured_system_prompt"


def test_rag_intelligence_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, RAGIntelligenceEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    rag = RAGIntelligenceEngine().generate()

    assert rag.retrieval_strategy == "hybrid_search"
    assert rag.embedding_model == "text-embedding-3-large"
    assert rag.vector_database == "Qdrant"
    assert rag.chunking_strategy == "semantic_chunking"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.rag_intelligence_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_rag = generator.generate_rag_intelligence()
    assert generated_rag.retrieval_strategy == "hybrid_search"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.rag_intelligence_manifest(rag)
    assert manifest["retrieval_strategy"] == "hybrid_search"
    assert manifest["embedding_model"] == "text-embedding-3-large"
    assert manifest["vector_database"] == "Qdrant"
    assert manifest["chunking_strategy"] == "semantic_chunking"


def test_ai_agent_architecture_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AIAgentArchitectureEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    architecture = AIAgentArchitectureEngine().generate()

    assert architecture.architecture_type == "multi_agent"
    assert architecture.orchestration_strategy == "planner_executor"
    assert architecture.tool_invocation_pattern == "function_calling"
    assert architecture.memory_strategy == "hybrid_memory"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_agent_architecture_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_architecture = generator.generate_ai_agent_architecture()
    assert generated_architecture.architecture_type == "multi_agent"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_agent_architecture_manifest(architecture)
    assert manifest["architecture_type"] == "multi_agent"
    assert manifest["orchestration_strategy"] == "planner_executor"
    assert manifest["tool_invocation_pattern"] == "function_calling"
    assert manifest["memory_strategy"] == "hybrid_memory"


def test_ai_workflow_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AIWorkflowEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    workflow = AIWorkflowEngine().generate()

    assert workflow.workflow_name == "agentic_request_processing"
    assert workflow.stages == [
        "request_analysis",
        "retrieval",
        "reasoning",
        "tool_execution",
        "response_generation",
    ]
    assert workflow.execution_strategy == "planner_executor"
    assert workflow.parallel_execution is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_workflow_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_workflow = generator.generate_ai_workflow()
    assert generated_workflow.workflow_name == "agentic_request_processing"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_workflow_manifest(workflow)
    assert manifest["workflow_name"] == "agentic_request_processing"
    assert manifest["stage_count"] == 5
    assert manifest["execution_strategy"] == "planner_executor"
    assert manifest["parallel_execution"] is True


def test_ai_recommendation_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AIRecommendationEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    recommendations = AIRecommendationEngine().generate()

    assert len(recommendations) == 3
    assert recommendations[0].recommendation == "introduce_long_term_memory"
    assert recommendations[0].category == "agent_memory"
    assert recommendations[0].priority == "high"
    assert recommendations[1].recommendation == "enable_semantic_routing"
    assert recommendations[2].recommendation == "implement_multi_agent_coordination"
    assert recommendations[2].priority == "medium"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_recommendations_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_recommendations = generator.generate_ai_recommendations()
    assert len(generated_recommendations) == 3

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_recommendation_manifest(recommendations)
    assert manifest["recommendation_count"] == 3


def test_ai_scorecard_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AIScorecardEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    scorecard = AIScorecardEngine().generate()

    assert scorecard.overall_score == 93.0
    assert scorecard.ai_grade == "A"
    assert scorecard.ai_readiness_score == 94.0
    assert scorecard.llm_compatibility_score == 92.0
    assert scorecard.agent_readiness_score == 90.0
    assert scorecard.recommendation_count == 3

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_scorecard_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_scorecard = generator.generate_ai_scorecard()
    assert generated_scorecard.overall_score == 93.0

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_scorecard_manifest(scorecard)
    assert manifest["overall_score"] == 93.0
    assert manifest["ai_grade"] == "A"
    assert manifest["ai_readiness_score"] == 94.0
    assert manifest["llm_compatibility_score"] == 92.0
    assert manifest["agent_readiness_score"] == 90.0


def test_ai_report_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AIReportGenerator
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    report = AIReportGenerator().generate()

    assert report.title == "AI Report"
    assert report.section_count == 7
    assert len(report.sections) == 7
    assert report.sections[0] == "AI Readiness Assessment"
    assert report.sections[-1] == "AI Scorecard"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_report_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_report = generator.generate_ai_report()
    assert generated_report.title == "AI Report"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_report_manifest(report)
    assert manifest["title"] == "AI Report"
    assert manifest["section_count"] == 7


def test_ai_intelligence_control_center_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        AIIntelligenceControlCenterGenerator,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    control_center = AIIntelligenceControlCenterGenerator().generate()

    assert control_center.ai_readiness_enabled is True
    assert control_center.llm_integration_enabled is True
    assert control_center.rag_intelligence_enabled is True
    assert control_center.ai_agent_architecture_enabled is True
    assert control_center.ai_workflow_enabled is True
    assert control_center.ai_recommendations_enabled is True
    assert control_center.ai_scorecard_enabled is True
    assert control_center.ai_report_enabled is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_intelligence_control_center_enabled() is True

    generator = PipelineSchemaGenerator()
    generated = generator.generate_ai_intelligence_control_center()
    assert generated.ai_readiness_enabled is True

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_intelligence_manifest(control_center)
    assert manifest["ai_readiness_enabled"] is True
    assert manifest["llm_integration_enabled"] is True
    assert manifest["rag_intelligence_enabled"] is True
    assert manifest["ai_agent_architecture_enabled"] is True
    assert manifest["ai_workflow_enabled"] is True
    assert manifest["ai_recommendations_enabled"] is True
    assert manifest["ai_scorecard_enabled"] is True
    assert manifest["ai_report_enabled"] is True


def test_ai_automation_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AIAutomationEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    automation = AIAutomationEngine().generate()

    assert automation.workflow_name == "agentic_ai_pipeline"
    assert automation.triggers == [
        "new_user_request",
        "knowledge_base_updated",
        "scheduled_reasoning_cycle",
    ]
    assert automation.actions == [
        "retrieve_context",
        "invoke_llm",
        "execute_tools",
        "generate_response",
    ]

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_automation_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_automation = generator.generate_ai_automation()
    assert generated_automation.workflow_name == "agentic_ai_pipeline"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_automation_manifest(automation)
    assert manifest["workflow_name"] == "agentic_ai_pipeline"
    assert manifest["trigger_count"] == 3
    assert manifest["action_count"] == 4


def test_ai_remediation_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AIRemediationEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    remediation = AIRemediationEngine().generate()

    assert remediation.issue_type == "llm_failure"
    assert remediation.remediation_actions == [
        "switch_to_backup_model",
        "retry_with_reduced_context",
        "fallback_to_cached_response",
        "notify_ai_operations",
    ]
    assert remediation.priority == "high"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_remediation_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_remediation = generator.generate_ai_remediation()
    assert generated_remediation.issue_type == "llm_failure"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_remediation_manifest(remediation)
    assert manifest["issue_type"] == "llm_failure"
    assert manifest["action_count"] == 4
    assert manifest["priority"] == "high"


def test_ai_governance_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AIGovernanceEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    governance = AIGovernanceEngine().generate()

    assert governance.ai_owner == "ai_platform_team"
    assert governance.model_review_frequency == "monthly"
    assert governance.responsible_ai_review_required is True
    assert governance.model_versioning_required is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.ai_governance_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_governance = generator.generate_ai_governance()
    assert generated_governance.ai_owner == "ai_platform_team"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.ai_governance_manifest(governance)
    assert manifest["ai_owner"] == "ai_platform_team"
    assert manifest["model_review_frequency"] == "monthly"
    assert manifest["responsible_ai_review_required"] is True
    assert manifest["model_versioning_required"] is True


def test_autonomous_ai_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator, AutonomousAIEngine
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    ai = AutonomousAIEngine().generate()

    assert ai.self_learning_enabled is True
    assert ai.adaptive_orchestration_enabled is True
    assert ai.autonomous_reasoning_enabled is True
    assert ai.continuous_improvement_enabled is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.autonomous_ai_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_ai = generator.generate_autonomous_ai()
    assert generated_ai.self_learning_enabled is True

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.autonomous_ai_manifest(ai)
    assert manifest["self_learning_enabled"] is True
    assert manifest["adaptive_orchestration_enabled"] is True
    assert manifest["autonomous_reasoning_enabled"] is True
    assert manifest["continuous_improvement_enabled"] is True


def test_enterprise_readiness_assessment_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        EnterpriseReadinessAssessmentEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    assessment = EnterpriseReadinessAssessmentEngine().generate()

    assert assessment.enterprise_readiness_score == 95.0
    assert assessment.business_readiness_score == 93.0
    assert assessment.organizational_maturity_score == 91.0
    assert assessment.enterprise_grade == "A"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.enterprise_readiness_assessment_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_assessment = generator.generate_enterprise_readiness_assessment()
    assert generated_assessment.enterprise_readiness_score == 95.0

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.enterprise_readiness_assessment_manifest(assessment)
    assert manifest["enterprise_readiness_score"] == 95.0
    assert manifest["business_readiness_score"] == 93.0
    assert manifest["organizational_maturity_score"] == 91.0
    assert manifest["enterprise_grade"] == "A"


def test_platform_readiness_assessment_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        PlatformReadinessAssessmentEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    assessment = PlatformReadinessAssessmentEngine().generate()

    assert assessment.platform_readiness_score == 95.0
    assert assessment.developer_experience_score == 93.0
    assert assessment.platform_maturity_score == 92.0
    assert assessment.platform_grade == "A"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.platform_readiness_assessment_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_assessment = generator.generate_platform_readiness_assessment()
    assert generated_assessment.platform_readiness_score == 95.0

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.platform_readiness_assessment_manifest(assessment)
    assert manifest["platform_readiness_score"] == 95.0
    assert manifest["developer_experience_score"] == 93.0
    assert manifest["platform_maturity_score"] == 92.0
    assert manifest["platform_grade"] == "A"


def test_developer_experience_intelligence_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        DeveloperExperienceIntelligenceEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    developer_experience = DeveloperExperienceIntelligenceEngine().generate()

    assert developer_experience.onboarding_experience == "excellent"
    assert developer_experience.self_service_score == 94.0
    assert developer_experience.documentation_quality == "high"
    assert developer_experience.golden_path_available is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.developer_experience_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_developer_experience = generator.generate_developer_experience()
    assert generated_developer_experience.self_service_score == 94.0

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.developer_experience_intelligence_manifest(
        developer_experience
    )
    assert manifest["onboarding_experience"] == "excellent"
    assert manifest["self_service_score"] == 94.0
    assert manifest["documentation_quality"] == "high"
    assert manifest["golden_path_available"] is True


def test_internal_developer_platform_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        InternalDeveloperPlatformEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    platform = InternalDeveloperPlatformEngine().generate()

    assert platform.platform_type == "internal_developer_platform"
    assert platform.developer_portal == "Backstage"
    assert platform.self_service_model == "golden_paths"
    assert platform.software_catalog_enabled is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.internal_developer_platform_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_platform = generator.generate_internal_developer_platform()
    assert generated_platform.platform_type == "internal_developer_platform"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.internal_developer_platform_manifest(platform)
    assert manifest["platform_type"] == "internal_developer_platform"
    assert manifest["developer_portal"] == "Backstage"
    assert manifest["self_service_model"] == "golden_paths"
    assert manifest["software_catalog_enabled"] is True


def test_platform_operations_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        PlatformOperationsIntelligenceEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    operations = PlatformOperationsIntelligenceEngine().generate()

    assert operations.operating_model == "platform_as_a_product"
    assert operations.service_ownership == "platform_team"
    assert operations.operational_health == "healthy"
    assert operations.incident_management == "sre_driven"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.platform_operations_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_operations = generator.generate_platform_operations()
    assert generated_operations.operating_model == "platform_as_a_product"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.platform_operations_manifest(operations)
    assert manifest["operating_model"] == "platform_as_a_product"
    assert manifest["service_ownership"] == "platform_team"
    assert manifest["operational_health"] == "healthy"
    assert manifest["incident_management"] == "sre_driven"


def test_platform_recommendation_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        PlatformRecommendationEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    recommendations = PlatformRecommendationEngine().generate()

    assert len(recommendations) == 3
    assert recommendations[0].recommendation == "expand_golden_path_templates"
    assert recommendations[0].category == "developer_experience"
    assert recommendations[0].priority == "high"
    assert recommendations[1].recommendation == "enable_self_service_provisioning"
    assert recommendations[1].category == "platform_operations"
    assert recommendations[1].priority == "high"
    assert recommendations[2].recommendation == "introduce_platform_scorecards"
    assert recommendations[2].category == "governance"
    assert recommendations[2].priority == "medium"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.platform_recommendations_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_recommendations = generator.generate_platform_recommendations()
    assert generated_recommendations[0].recommendation == "expand_golden_path_templates"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.platform_recommendation_manifest(recommendations)
    assert manifest["recommendation_count"] == 3


def test_platform_report_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        PlatformReportGenerator,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    report = PlatformReportGenerator().generate()

    assert report.title == "Platform Report"
    assert report.sections == [
        "Platform Readiness Assessment",
        "Developer Experience",
        "Internal Developer Platform",
        "Platform Engineering Architecture",
        "Platform Operations",
        "Platform Recommendations",
        "Platform Scorecard"
    ]
    assert report.section_count == 7

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.platform_report_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_report = generator.generate_platform_report()
    assert generated_report.title == "Platform Report"
    assert generated_report.section_count == 7

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.platform_report_manifest(report)
    assert manifest["title"] == "Platform Report"
    assert manifest["section_count"] == 7


def test_platform_automation_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        PlatformAutomationEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    automation = PlatformAutomationEngine().generate()

    assert automation.workflow_name == "platform_self_service"
    assert automation.triggers == [
        "developer_request",
        "repository_created",
        "service_registered"
    ]
    assert automation.actions == [
        "provision_infrastructure",
        "configure_ci_cd",
        "register_service",
        "notify_platform_team"
    ]

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.platform_automation_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_automation = generator.generate_platform_automation()
    assert generated_automation.workflow_name == "platform_self_service"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.platform_automation_manifest(automation)
    assert manifest["workflow_name"] == "platform_self_service"
    assert manifest["trigger_count"] == 3
    assert manifest["action_count"] == 4


def test_platform_remediation_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        PlatformRemediationEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    remediation = PlatformRemediationEngine().generate()

    assert remediation.issue_type == "developer_portal_unavailable"
    assert remediation.remediation_actions == [
        "restart_platform_services",
        "rebuild_service_catalog",
        "revalidate_platform_integrations",
        "notify_platform_operations"
    ]
    assert remediation.priority == "high"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.platform_remediation_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_remediation = generator.generate_platform_remediation()
    assert generated_remediation.issue_type == "developer_portal_unavailable"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.platform_remediation_manifest(remediation)
    assert manifest["issue_type"] == "developer_portal_unavailable"
    assert manifest["action_count"] == 4
    assert manifest["priority"] == "high"


def test_platform_governance_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        PlatformGovernanceEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    governance = PlatformGovernanceEngine().generate()

    assert governance.platform_owner == "platform_engineering_team"
    assert governance.governance_review_frequency == "monthly"
    assert governance.platform_standards_required is True
    assert governance.developer_experience_review_required is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.platform_governance_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_governance = generator.generate_platform_governance()
    assert generated_governance.platform_owner == "platform_engineering_team"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.platform_governance_manifest(governance)
    assert manifest["platform_owner"] == "platform_engineering_team"
    assert manifest["governance_review_frequency"] == "monthly"
    assert manifest["platform_standards_required"] is True
    assert manifest["developer_experience_review_required"] is True


def test_autonomous_platform_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        AutonomousPlatformEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    platform = AutonomousPlatformEngine().generate()

    assert platform.adaptive_platform_enabled is True
    assert platform.self_service_optimization_enabled is True
    assert platform.developer_experience_learning_enabled is True
    assert platform.continuous_platform_improvement_enabled is True

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.autonomous_platform_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_platform = generator.generate_autonomous_platform()
    assert generated_platform.adaptive_platform_enabled is True

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.autonomous_platform_manifest(platform)
    assert manifest["adaptive_platform_enabled"] is True
    assert manifest["self_service_optimization_enabled"] is True
    assert manifest["developer_experience_learning_enabled"] is True
    assert manifest["continuous_platform_improvement_enabled"] is True


def test_platform_engineering_architecture_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import (
        PipelineSchemaGenerator,
        PlatformEngineeringArchitectureEngine,
    )
    from backend.generator.sdk_release_generator import SDKReleaseGenerator

    architecture = PlatformEngineeringArchitectureEngine().generate()

    assert architecture.architecture_style == "platform_as_a_product"
    assert architecture.platform_services == [
        "developer_portal",
        "software_catalog",
        "ci_cd_platform",
        "observability_platform",
        "secrets_management"
    ]
    assert architecture.service_catalog_enabled is True
    assert architecture.platform_api_model == "self_service"

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.platform_engineering_architecture_enabled() is True

    generator = PipelineSchemaGenerator()
    generated_architecture = generator.generate_platform_engineering_architecture()
    assert generated_architecture.architecture_style == "platform_as_a_product"

    release_generator = SDKReleaseGenerator()
    manifest = release_generator.platform_engineering_architecture_manifest(architecture)
    assert manifest["architecture_style"] == "platform_as_a_product"
    assert manifest["platform_service_count"] == 5
    assert manifest["service_catalog_enabled"] is True
    assert manifest["platform_api_model"] == "self_service"


def test_pipeline_contract_validator():
    import pytest
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineContractValidator

    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )

    validator = PipelineContractValidator()

    # Valid schema
    valid_schema = {
        "request": {"source": {"type": "str"}},
        "response": {"result": {"type": "str"}}
    }
    assert validator.validate_schema(spec, valid_schema) is True

    # Invalid request schema
    invalid_req_schema = {
        "request": {"mismatch": {"type": "str"}},
        "response": {"result": {"type": "str"}}
    }
    with pytest.raises(ValueError, match="Request schema does not match endpoint spec"):
        validator.validate_schema(spec, invalid_req_schema)

    # Invalid response schema
    invalid_resp_schema = {
        "request": {"source": {"type": "str"}},
        "response": {"mismatch": {"type": "str"}}
    }
    with pytest.raises(ValueError, match="Response schema does not match endpoint spec"):
        validator.validate_schema(spec, invalid_resp_schema)


def test_python_sdk_generation():
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec
    from backend.generator import PipelineSchemaGenerator

    spec = PipelineEndpointSpec(
        endpoint_name="train_model",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )

    generator = PipelineSchemaGenerator()
    python_code = generator.generate_python_sdk(spec)

    assert "class TrainModelClient:" in python_code
    assert "def train_model(" in python_code
    assert "requests.post(" in python_code

    models = generator.generate_python_models(spec)
    assert "class TrainModelRequest(" in models["request"]
    assert "source: str" in models["request"]
    assert "class TrainModelResponse(" in models["response"]
    assert "result: str" in models["response"]

    assert spec.python_package_name() == "train_model_sdk"
    assert spec.python_async_client_name() == "TrainModelAsyncClient"

    assert spec.supports_authentication() is True

    package = generator.generate_python_package(spec)
    assert package.file_count() == 8
    assert package.file_names() == [
        "README.md",
        "__init__.py",
        "async_client.py",
        "client.py",
        "exceptions.py",
        "models.py",
        "pyproject.toml",
        "requirements.txt",
    ]
    assert package.contains_file("client.py") is True
    assert package.contains_file("async_client.py") is True
    assert package.contains_file("nonexistent.py") is False
    assert package.has_client() is True
    assert "from .client import *" in package.files["__init__.py"]
    assert "from .async_client import *" in package.files["__init__.py"]
    assert "from .exceptions import *" in package.files["__init__.py"]
    assert "class TrainModelClient:" in package.files["client.py"]
    assert "class TrainModelAsyncClient:" in package.files["async_client.py"]
    assert "api_key: str | None = None" in package.files["client.py"]
    assert "bearer_token: str | None = None" in package.files["client.py"]
    assert "def build_headers(" in package.files["client.py"]
    assert "api_key: str | None = None" in package.files["async_client.py"]
    assert "bearer_token: str | None = None" in package.files["async_client.py"]
    assert "def build_headers(" in package.files["async_client.py"]
    assert "from .exceptions import (\n    APIError\n)" in package.files["client.py"]
    assert "raise APIError(" in package.files["client.py"]
    assert "max_retries: int = 3" in package.files["client.py"]
    assert "timeout: int = 30" in package.files["client.py"]
    assert "for _ in range(" in package.files["client.py"]
    assert "class TrainModelRequest(" in package.files["models.py"]
    assert "class SDKError(" in package.files["exceptions.py"]
    assert "class RetryError(" in package.files["exceptions.py"]

    # Pagination: method signatures
    assert "page: int = 1" in package.files["client.py"]
    assert "limit: int = 100" in package.files["client.py"]
    assert "page: int = 1" in package.files["async_client.py"]
    assert "limit: int = 100" in package.files["async_client.py"]

    # Pagination: params dict in requests
    assert '"page"' in package.files["client.py"]
    assert '"limit"' in package.files["client.py"]
    assert '"page"' in package.files["async_client.py"]
    assert '"limit"' in package.files["async_client.py"]

    # Pagination: PaginationInfo model included in models.py
    assert "class PaginationInfo(" in package.files["models.py"]
    assert "page: int" in package.files["models.py"]
    assert "total: int" in package.files["models.py"]

    # generate_pagination_models standalone check
    pagination = generator.generate_pagination_models()
    assert "class PaginationInfo(" in pagination
    assert "page: int" in pagination
    assert "limit: int" in pagination
    assert "total: int" in pagination

    # README docs
    assert package.contains_file("README.md") is True
    assert "# train_model_sdk" in package.files["README.md"]
    assert "pip install train_model_sdk" in package.files["README.md"]
    assert "TrainModelClient" in package.files["README.md"]
    assert "POST /train_model" in package.files["README.md"]

    # generate_python_docs standalone check
    readme = generator.generate_python_docs(spec)
    assert "# train_model_sdk" in readme
    assert "pip install train_model_sdk" in readme
    assert "TrainModelClient" in readme

    # PyPI packaging
    assert package.contains_file("pyproject.toml") is True
    assert package.contains_file("requirements.txt") is True
    assert 'name =\n    "train_model_sdk"' in package.files["pyproject.toml"]
    assert "setuptools" in package.files["pyproject.toml"]
    assert "requests>=2.0.0" in package.files["requirements.txt"]
    assert "pydantic>=2.0.0" in package.files["requirements.txt"]
    assert "httpx>=0.25.0" in package.files["requirements.txt"]

    # generate_python_packaging standalone check
    packaging = generator.generate_python_packaging(spec)
    assert "pyproject" in packaging
    assert "requirements" in packaging
    assert "train_model_sdk" in packaging["pyproject"]
    assert "httpx" in packaging["requirements"]

    # PythonPackage.manifest()
    m = package.manifest()
    assert m["file_count"] == 8
    assert "client.py" in m["files"]
    assert "README.md" in m["files"]
    assert "pyproject.toml" in m["files"]

    # generate_release_metadata standalone check
    from backend.generator import SDKReleaseMetadata
    meta = generator.generate_release_metadata(spec, 8)
    assert isinstance(meta, SDKReleaseMetadata)
    assert meta.package_name == "train_model_sdk"
    assert meta.version == "1.0.0"
    assert meta.artifact_count == 8
    assert meta.generated_at != ""

    # generate_release_bundle end-to-end check
    bundle = generator.generate_release_bundle(spec)
    assert "package" in bundle
    assert "metadata" in bundle
    assert "manifest" in bundle
    assert bundle["metadata"].package_name == "train_model_sdk"
    assert bundle["metadata"].artifact_count == 8
    assert bundle["manifest"]["artifact_count"] == 8
    assert "client.py" in bundle["manifest"]["artifacts"]
    assert bundle["package"].has_client() is True

    # supported_sdk_targets on spec
    assert spec.supported_sdk_targets() == ["python", "typescript"]

    # generate_multilanguage_bundle end-to-end check
    from backend.generator import MultiLanguageRelease
    ml_bundle = generator.generate_multilanguage_bundle(spec)
    assert isinstance(ml_bundle, MultiLanguageRelease)

    # manifest structure
    assert "languages" in ml_bundle.manifest
    assert "python" in ml_bundle.manifest["languages"]
    assert "typescript" in ml_bundle.manifest["languages"]
    assert "artifacts" in ml_bundle.manifest
    assert "python" in ml_bundle.manifest["artifacts"]
    assert "typescript" in ml_bundle.manifest["artifacts"]

    # python artifacts nested correctly
    py_artifacts = ml_bundle.manifest["artifacts"]["python"]
    assert py_artifacts["artifact_count"] == 8
    assert "client.py" in py_artifacts["artifacts"]

    # typescript manifest nested correctly
    ts_manifest = ml_bundle.manifest["artifacts"]["typescript"]
    assert "module" in ts_manifest
    assert "package" in ts_manifest
    assert ts_manifest["package"] == "train-model-sdk"

    # metadata
    assert ml_bundle.metadata["release_version"] == "1.0.0"
    assert ml_bundle.metadata["sdk_count"] == 2

    # python and typescript bundles accessible on the release object
    assert ml_bundle.python_bundle["package"].has_client() is True
    assert "sdk" in ml_bundle.typescript_bundle


def test_governance_assessment_engine():
    from backend.generator import GovernanceAssessment, GovernanceAssessmentEngine
    from backend.generator.pipeline_schema_generator import PipelineSchemaGenerator
    from backend.generator.sdk_release_generator import SDKReleaseGenerator
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec

    # 1. Verify GovernanceAssessmentEngine
    engine = GovernanceAssessmentEngine()
    assessment = engine.generate()
    assert isinstance(assessment, GovernanceAssessment)
    assert assessment.governance_score == 91.0
    assert assessment.compliance_score == 89.0
    assert assessment.audit_readiness_score == 93.0
    assert assessment.governance_grade == "A"

    # 2. Verify PipelineSchemaGenerator
    schema_gen = PipelineSchemaGenerator()
    assert isinstance(schema_gen.governance_assessment_engine, GovernanceAssessmentEngine)
    gen_assessment = schema_gen.generate_governance_assessment()
    assert gen_assessment.governance_score == 91.0

    # 3. Verify SDKReleaseGenerator
    release_gen = SDKReleaseGenerator()
    manifest = release_gen.governance_assessment_manifest(assessment)
    assert manifest["governance_score"] == 91.0
    assert manifest["compliance_score"] == 89.0
    assert manifest["audit_readiness_score"] == 93.0
    assert manifest["governance_grade"] == "A"

    # 4. Verify PipelineEndpointSpec
    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.governance_assessment_enabled() is True


def test_compliance_intelligence_engine():
    from backend.generator import ComplianceFramework, ComplianceIntelligenceEngine
    from backend.generator.pipeline_schema_generator import PipelineSchemaGenerator
    from backend.generator.sdk_release_generator import SDKReleaseGenerator
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec

    # 1. Verify ComplianceIntelligenceEngine
    engine = ComplianceIntelligenceEngine()
    frameworks = engine.generate()
    assert len(frameworks) == 3
    assert all(isinstance(f, ComplianceFramework) for f in frameworks)
    assert frameworks[0].framework_name == "SOC2"
    assert frameworks[0].compliance_status == "partial"
    assert frameworks[0].coverage_percent == 82.0

    # 2. Verify PipelineSchemaGenerator
    schema_gen = PipelineSchemaGenerator()
    assert isinstance(schema_gen.compliance_intelligence_engine, ComplianceIntelligenceEngine)
    gen_frameworks = schema_gen.generate_compliance_frameworks()
    assert len(gen_frameworks) == 3

    # 3. Verify SDKReleaseGenerator
    release_gen = SDKReleaseGenerator()
    manifest = release_gen.compliance_framework_manifest(frameworks)
    assert manifest["framework_count"] == 3

    # 4. Verify PipelineEndpointSpec
    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.compliance_intelligence_enabled() is True


def test_policy_enforcement_engine():
    from backend.generator import PolicyControl, PolicyEnforcementEngine
    from backend.generator.pipeline_schema_generator import PipelineSchemaGenerator
    from backend.generator.sdk_release_generator import SDKReleaseGenerator
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec

    # 1. Verify PolicyEnforcementEngine
    engine = PolicyEnforcementEngine()
    controls = engine.generate()
    assert len(controls) == 3
    assert all(isinstance(c, PolicyControl) for c in controls)
    assert controls[0].policy_name == "authentication_required"
    assert controls[0].enforcement_status == "enforced"
    assert controls[0].severity == "critical"

    # 2. Verify PipelineSchemaGenerator
    schema_gen = PipelineSchemaGenerator()
    assert isinstance(schema_gen.policy_enforcement_engine, PolicyEnforcementEngine)
    gen_controls = schema_gen.generate_policy_controls()
    assert len(gen_controls) == 3

    # 3. Verify SDKReleaseGenerator
    release_gen = SDKReleaseGenerator()
    manifest = release_gen.policy_control_manifest(controls)
    assert manifest["control_count"] == 3

    # 4. Verify PipelineEndpointSpec
    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.policy_enforcement_enabled() is True


def test_governance_risk_analysis_engine():
    from backend.generator import GovernanceRisk, GovernanceRiskAnalysisEngine
    from backend.generator.pipeline_schema_generator import PipelineSchemaGenerator
    from backend.generator.sdk_release_generator import SDKReleaseGenerator
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec

    # 1. Verify GovernanceRiskAnalysisEngine
    engine = GovernanceRiskAnalysisEngine()
    risks = engine.generate()
    assert len(risks) == 3
    assert all(isinstance(r, GovernanceRisk) for r in risks)
    assert risks[0].risk_name == "incomplete_audit_logging"
    assert risks[0].probability == "medium"
    assert risks[0].impact == "high"

    # 2. Verify PipelineSchemaGenerator
    schema_gen = PipelineSchemaGenerator()
    assert isinstance(schema_gen.governance_risk_analysis_engine, GovernanceRiskAnalysisEngine)
    gen_risks = schema_gen.generate_governance_risks()
    assert len(gen_risks) == 3

    # 3. Verify SDKReleaseGenerator
    release_gen = SDKReleaseGenerator()
    manifest = release_gen.governance_risk_manifest(risks)
    assert manifest["risk_count"] == 3

    # 4. Verify PipelineEndpointSpec
    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.governance_risk_analysis_enabled() is True


def test_audit_readiness_engine():
    from backend.generator import AuditReadiness, AuditReadinessEngine
    from backend.generator.pipeline_schema_generator import PipelineSchemaGenerator
    from backend.generator.sdk_release_generator import SDKReleaseGenerator
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec

    # 1. Verify AuditReadinessEngine
    engine = AuditReadinessEngine()
    readiness = engine.generate()
    assert isinstance(readiness, AuditReadiness)
    assert readiness.readiness_score == 92.0
    assert readiness.audit_ready is True
    assert readiness.control_coverage_percent == 95.0
    assert readiness.open_findings_count == 2

    # 2. Verify PipelineSchemaGenerator
    schema_gen = PipelineSchemaGenerator()
    assert isinstance(schema_gen.audit_readiness_engine, AuditReadinessEngine)
    gen_readiness = schema_gen.generate_audit_readiness()
    assert gen_readiness.readiness_score == 92.0

    # 3. Verify SDKReleaseGenerator
    release_gen = SDKReleaseGenerator()
    manifest = release_gen.audit_readiness_manifest(readiness)
    assert manifest["readiness_score"] == 92.0
    assert manifest["audit_ready"] is True
    assert manifest["control_coverage_percent"] == 95.0
    assert manifest["open_findings_count"] == 2

    # 4. Verify PipelineEndpointSpec
    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.audit_readiness_enabled() is True


def test_governance_recommendation_engine():
    from backend.generator import GovernanceRecommendation, GovernanceRecommendationEngine
    from backend.generator.pipeline_schema_generator import PipelineSchemaGenerator
    from backend.generator.sdk_release_generator import SDKReleaseGenerator
    from backend.analyzer.pipeline_endpoint_spec import PipelineEndpointSpec

    # 1. Verify GovernanceRecommendationEngine
    engine = GovernanceRecommendationEngine()
    recommendations = engine.generate()
    assert len(recommendations) == 3
    assert all(isinstance(r, GovernanceRecommendation) for r in recommendations)
    assert recommendations[0].recommendation == "enable_comprehensive_audit_logging"
    assert recommendations[0].priority == "high"
    assert recommendations[0].impact == "high"

    # 2. Verify PipelineSchemaGenerator
    schema_gen = PipelineSchemaGenerator()
    assert isinstance(schema_gen.governance_recommendation_engine, GovernanceRecommendationEngine)
    gen_recs = schema_gen.generate_governance_recommendations()
    assert len(gen_recs) == 3

    # 3. Verify SDKReleaseGenerator
    release_gen = SDKReleaseGenerator()
    manifest = release_gen.governance_recommendation_manifest(recommendations)
    assert manifest["recommendation_count"] == 3

    # 4. Verify PipelineEndpointSpec
    spec = PipelineEndpointSpec(
        endpoint_name="run_pipeline",
        input_fields=["source"],
        output_fields=["result"],
        execution_stages=1,
        parallelism_score=1.0,
    )
    assert spec.governance_recommendations_enabled() is True


def test_dockerfile_content_is_a_pure_string_with_no_disk_access(tmp_path, monkeypatch):
    from backend.generator.docker_generator import dockerfile_content

    # Would fail loudly if dockerfile_content tried to open/write anything --
    # there's no writable cwd for it to do that in.
    monkeypatch.chdir(tmp_path)

    content = dockerfile_content(package_name="my_app", python_version="3.12")

    assert content.startswith("FROM python:3.12-slim")
    assert "COPY . my_app/" in content
    assert 'CMD ["sh", "-c", "uvicorn my_app.app:app' in content
    assert list(tmp_path.iterdir()) == []


def test_dockerfile_content_defaults_match_generate_dockerfiles_own_defaults():
    from backend.generator.docker_generator import dockerfile_content

    content = dockerfile_content()

    assert content.startswith("FROM python:3.11-slim")
    assert "COPY . generated/" in content


def test_generate_dockerfile_writes_exactly_what_dockerfile_content_returns(tmp_path):
    from backend.generator.docker_generator import dockerfile_content, generate_dockerfile

    output_path = tmp_path / "Dockerfile"

    generate_dockerfile(str(output_path), package_name="my_app", python_version="3.13")

    assert output_path.read_text(encoding="utf-8") == dockerfile_content("my_app", "3.13")


def test_dockerignore_content_is_a_pure_string_with_no_disk_access(tmp_path, monkeypatch):
    from backend.generator.docker_generator import dockerignore_content

    monkeypatch.chdir(tmp_path)

    content = dockerignore_content()

    assert ".git/" in content
    assert ".compile_metadata.json" in content
    assert list(tmp_path.iterdir()) == []


def test_generate_dockerignore_writes_exactly_what_dockerignore_content_returns(tmp_path):
    from backend.generator.docker_generator import dockerignore_content, generate_dockerignore

    output_path = tmp_path / ".dockerignore"

    generate_dockerignore(str(output_path))

    assert output_path.read_text(encoding="utf-8") == dockerignore_content()


def test_readme_content_is_a_pure_string_with_no_disk_access(tmp_path, monkeypatch):
    from backend.generator.docker_generator import readme_content

    monkeypatch.chdir(tmp_path)

    content = readme_content(
        package_name="my_app",
        functions=[{"name": "add", "args": [], "return_type": "int"}],
    )

    assert content.startswith("# my_app")
    assert "`POST /add`" in content
    assert list(tmp_path.iterdir()) == []


def test_readme_content_marks_a_background_function_as_such():
    from backend.generator.docker_generator import readme_content

    content = readme_content(
        functions=[{"name": "train_model", "args": [], "return_type": "dict"}],
    )

    assert (
        "`POST /train_model` -- enqueues a background task; poll "
        "`GET /tasks/{task_id}` for the result"
    ) in content


def test_readme_content_does_not_mark_a_synchronous_function_as_background():
    from backend.generator.docker_generator import readme_content

    content = readme_content(
        functions=[{"name": "add", "args": [], "return_type": "int"}],
    )

    assert "`POST /add`" in content
    assert "`POST /add` --" not in content


def test_readme_content_with_no_functions_says_so():
    from backend.generator.docker_generator import readme_content

    content = readme_content(functions=[])

    assert "doesn't expose any functions yet" in content


def test_readme_content_lists_every_env_var_with_its_own_default():
    from backend.generator.docker_generator import readme_content

    env_vars = [
        {
            "name": "NOTEBOOK_API_KEY",
            "default": "notebook-to-api-dev-key",
            "description": "API key(s) accepted on X-API-Key.",
        },
    ]

    content = readme_content(env_vars=env_vars)

    assert (
        "`NOTEBOOK_API_KEY` (default: `notebook-to-api-dev-key`) -- "
        "API key(s) accepted on X-API-Key."
    ) in content


def test_readme_content_defaults_match_generate_readmes_own_defaults():
    from backend.generator.docker_generator import readme_content

    content = readme_content()

    assert content.startswith("# generated")


def test_generate_readme_writes_exactly_what_readme_content_returns(tmp_path):
    from backend.generator.docker_generator import readme_content, generate_readme

    output_path = tmp_path / "README.md"
    functions = [{"name": "add", "args": [], "return_type": "int"}]
    env_vars = [
        {"name": "NOTEBOOK_API_KEY", "default": "dev-key", "description": "d"}
    ]

    generate_readme(str(output_path), "my_app", functions, env_vars)

    assert (
        output_path.read_text(encoding="utf-8")
        == readme_content("my_app", functions, env_vars)
    )