import hashlib
import http.server
import io
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nbformat
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.compiler import (
    COMPILE_LOCK,
    COMPILE_METADATA_FILENAME,
    NOTEBOOK_TO_API_VERSION,
    compiling_python_version,
    package_name_for_output_dir,
)
from backend.dashboard import app
from backend.routes.upload import (
    GENERATED_DIR,
    MAX_NOTEBOOK_VERSIONS,
    MAX_SEARCH_REGEX_LENGTH,
    UPLOAD_DIR,
    _bundle_sha256,
    _compile_search_regex,
    _description_sidecar_path,
    _notebook_versions_dir,
    _regex_has_nested_unbounded_repetition,
    _source_url_sidecar_path,
    _tags_sidecar_path,
    resolve_generated_path,
    resolve_upload_path,
)
import re._parser as _sre_parser

PROJECT_ROOT = Path(__file__).resolve().parents[1]

client = TestClient(app)


def _notebook_bytes(function_source):
    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(nbformat.v4.new_code_cell(function_source))
    return nbformat.writes(notebook).encode("utf-8")


def _non_python_notebook_bytes(cell_source="f <- function(x) x + 1"):
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "ir", "display_name": "R", "language": "R",
    }
    notebook.cells.append(nbformat.v4.new_code_cell(cell_source))
    return nbformat.writes(notebook).encode("utf-8")


@pytest.fixture(autouse=True)
def _cleanup_uploaded_files():
    # Backs up full file contents, not just names -- test_delete_all_notebooks_*
    # below exercises DELETE /api/notebooks, which (correctly, by design) removes
    # every ".ipynb" file directly in UPLOAD_DIR, including any that predate this
    # test run (e.g. uploads/sample.ipynb, checked into this repo). Restoring by
    # name alone (the previous behavior: remove whatever's new, ignore what's
    # missing) left a pre-existing file permanently deleted from disk the first
    # time a test actually exercised that endpoint -- confirmed: it happened.
    names_before = set(os.listdir(UPLOAD_DIR))
    backup = {
        name: (Path(UPLOAD_DIR) / name).read_bytes()
        for name in names_before
        if (Path(UPLOAD_DIR) / name).is_file()
    }
    yield
    names_after = set(os.listdir(UPLOAD_DIR))
    for name in names_after - names_before:
        path = Path(UPLOAD_DIR) / name
        # A test exercising notebook version history (see
        # _notebook_versions_dir in backend/routes/upload.py) creates the
        # ".versions" directory as a new top-level UPLOAD_DIR entry the
        # first time it runs -- os.remove alone can't remove a directory
        # (it raises IsADirectoryError), so this needs the same
        # dir-vs-file branch DELETE /api/notebooks/{filename} itself
        # already applies to that same directory.
        if path.is_dir():
            shutil.rmtree(path)
        else:
            os.remove(path)
    for name in names_before - names_after:
        if name in backup:
            (Path(UPLOAD_DIR) / name).write_bytes(backup[name])


def test_blocking_endpoints_are_declared_as_plain_def_not_async_def():
    """FastAPI only runs `async def` path operations directly on the
    single asyncio event loop; a handler that does purely synchronous,
    blocking work (file I/O, subprocess.run for `docker build`/`docker
    push` -- up to DEPLOY_SUBPROCESS_TIMEOUT_SECONDS, 600s by default --
    and compile_notebook's own file writes and per-dependency
    importlib.metadata.version() lookups) with no `await` inside it
    blocks *every* concurrent request this server is handling for as
    long as it runs -- not just the one caller who made it, including an
    unrelated GET /api/health from a completely different client.

    Confirmed against a real (non-TestClient) uvicorn server: an async
    def endpoint blocking for 1.5s with no await delayed a concurrent
    request to a trivial endpoint by the same 1.5s; the identical
    blocking call in a plain def endpoint (which FastAPI runs in its
    worker threadpool instead) added under 2ms. Every one of these
    handlers is purely synchronous internally already, so declaring them
    `def` instead of `async def` changes nothing about how they work --
    only how FastAPI schedules them -- which is also why this can be
    verified directly (is this a coroutine function or not) rather than
    through a timing-based test: TestClient's own threading model doesn't
    reproduce single-event-loop contention the way a real server does.
    """
    import inspect

    from backend.routes import upload as upload_module

    blocking_endpoints = [
        upload_module.list_notebooks,
        upload_module.delete_all_notebooks,
        upload_module.delete_notebook,
        upload_module.get_notebook,
        upload_module.rename_notebook,
        upload_module.inspect_notebook_endpoint,
        upload_module.compile_notebook_endpoint,
        upload_module.export_openapi_endpoint,
        upload_module.export_sdk_endpoint,
        upload_module.deploy_generated_app,
        upload_module.download_generated_app,
        upload_module.list_generated_files_endpoint,
        upload_module.delete_generated_app,
        upload_module.health_check,
    ]

    for endpoint in blocking_endpoints:
        assert not inspect.iscoroutinefunction(endpoint), (
            f"{endpoint.__name__} is declared async def but does no "
            "awaiting -- it blocks the whole event loop, not just its "
            "own caller, for as long as it runs."
        )

    # upload_notebook genuinely awaits UploadFile.read and must stay async.
    assert inspect.iscoroutinefunction(upload_module.upload_notebook)


def test_resolve_upload_path_rejects_absolute_path():

    with pytest.raises(Exception):
        resolve_upload_path("/etc/passwd")


def test_resolve_upload_path_rejects_relative_traversal():

    with pytest.raises(Exception):
        resolve_upload_path("../../../../etc/passwd")


def test_resolve_upload_path_rejects_an_embedded_null_byte():
    """Confirmed exploitable before this fix: Path("nb\x00.ipynb").is_absolute()
    doesn't raise (a null byte isn't special to pathlib's own parsing),
    so this sailed past the existing absolute-path guard clause -- but
    the later .resolve() call eventually hands it to the underlying
    os.path.realpath/lstat syscalls, which do reject it, as a bare
    ValueError ("embedded null character in path"), an unhandled 500
    instead of the same clean 400 every other malformed-path case in
    this file already gets.
    """

    with pytest.raises(Exception):
        resolve_upload_path("nb\x00.ipynb")


def test_resolve_upload_path_accepts_plain_filename():

    resolved = resolve_upload_path("my_notebook.ipynb")

    assert resolved.name == "my_notebook.ipynb"
    assert str(resolved).startswith(os.path.abspath(UPLOAD_DIR))


def test_resolve_generated_path_rejects_absolute_path():

    with pytest.raises(Exception):
        resolve_generated_path("/etc/passwd")


def test_resolve_generated_path_rejects_relative_traversal():

    with pytest.raises(Exception):
        resolve_generated_path("../../../../etc/passwd")


def test_resolve_generated_path_rejects_an_embedded_null_byte():

    with pytest.raises(Exception):
        resolve_generated_path("app\x00.py")


def test_resolve_generated_path_accepts_a_nested_path():

    resolved = resolve_generated_path("runtime/notebook_module.py")

    assert resolved.name == "notebook_module.py"
    assert str(resolved).startswith(os.path.abspath("generated"))


def test_upload_dir_defaults_to_uploads_without_the_env_var():
    """Run in a fresh subprocess (no NOTEBOOK_API_UPLOAD_DIR set) rather
    than asserting against the already-imported UPLOAD_DIR in this test
    process, which could be misleadingly "correct" simply because nothing
    in this test session happens to have set the env var -- the same
    care test_allowed_origins_env_var_overrides_default_list
    (test_dashboard_cors.py) already takes for GENERATED_DIR's sibling
    NOTEBOOK_API_ALLOWED_ORIGINS.
    """

    env = {k: v for k, v in os.environ.items() if k != "NOTEBOOK_API_UPLOAD_DIR"}

    proc = subprocess.run(
        [sys.executable, "-c", "from backend.routes.upload import UPLOAD_DIR; print(UPLOAD_DIR)"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "uploads"


def test_upload_dir_env_var_overrides_the_default(tmp_path):
    """Before this, UPLOAD_DIR was permanently fixed to "uploads" with no
    way for an operator to point the dashboard at a different uploads
    directory -- unlike its sibling GENERATED_DIR, which already supports
    exactly this via NOTEBOOK_API_GENERATED_DIR (see GENERATED_DIR's own
    comment in backend/routes/upload.py). A container deployment wanting
    to mount a persistent volume for uploads at a specific path, or avoid
    colliding with an "uploads" directory something else on the host
    already uses, had no way to configure that without editing source.

    Run end-to-end in a fresh subprocess (POST /api/upload through a real
    TestClient, then confirm the file landed on disk at the configured
    path) since UPLOAD_DIR's directory is created once, eagerly, at
    import time -- setting the env var only takes effect for a process
    that hasn't imported backend.routes.upload yet.
    """

    custom_dir = tmp_path / "custom_uploads_env_var_test"

    script = f"""
import io
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})

from fastapi.testclient import TestClient
from backend.dashboard import app
from backend.routes.upload import UPLOAD_DIR

assert UPLOAD_DIR == {str(custom_dir)!r}, UPLOAD_DIR

client = TestClient(app)
resp = client.post(
    "/api/upload",
    files={{"file": ("env_var_test.ipynb", io.BytesIO(b'{{"cells": [], "metadata": {{}}, "nbformat": 4, "nbformat_minor": 5}}'), "application/json")}},
)
assert resp.status_code == 200, resp.text

print("UPLOAD_DIR_ENV_OVERRIDE_OK")
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "NOTEBOOK_API_UPLOAD_DIR": str(custom_dir)},
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "UPLOAD_DIR_ENV_OVERRIDE_OK" in proc.stdout
    assert (custom_dir / "env_var_test.ipynb").is_file()
    # Must not have fallen back to the default "uploads" directory instead.
    assert not (PROJECT_ROOT / "uploads" / "env_var_test.ipynb").exists()


def test_upload_dir_and_generated_dir_configured_to_the_same_path_are_rejected_at_import(
    tmp_path
):
    """Confirmed catastrophic if left unchecked: UPLOAD_DIR and
    GENERATED_DIR are each read independently from their own env var, with
    nothing stopping an operator from configuring them to the same
    directory. Reproduced live before this fix: pointing both at the same
    path, uploading a notebook, compiling it, then calling DELETE
    /api/generated -- whose own docstring says it resets the dashboard's
    compiled-app state back to "nothing compiled yet" via
    shutil.rmtree(GENERATED_DIR) -- permanently destroyed the uploaded
    notebook right along with it: the whole shared directory vanished
    outright, not just the compiled output, with no way to recover it.

    Run in a fresh subprocess since both directories are only ever read
    once, at import time.
    """

    shared_dir = tmp_path / "shared_dir"

    proc = subprocess.run(
        [sys.executable, "-c", "from backend.routes import upload"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "NOTEBOOK_API_UPLOAD_DIR": str(shared_dir),
            "NOTEBOOK_API_GENERATED_DIR": str(shared_dir),
        },
    )

    assert proc.returncode != 0
    assert "must not be the same directory" in proc.stderr


def test_upload_dir_nested_inside_generated_dir_is_rejected_at_import(tmp_path):
    """Same class of destructive overlap as the identical-path case above,
    just reached the other way around: GENERATED_DIR nested inside
    UPLOAD_DIR (DELETE /api/generated's own shutil.rmtree(GENERATED_DIR)
    would still remove real uploaded notebooks sitting under it) or
    UPLOAD_DIR nested inside GENERATED_DIR (the reverse -- a recompile's
    own directory-wide writes could just as easily reach into it).
    """

    parent_dir = tmp_path / "parent_dir"
    nested_dir = parent_dir / "nested"

    proc = subprocess.run(
        [sys.executable, "-c", "from backend.routes import upload"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "NOTEBOOK_API_UPLOAD_DIR": str(parent_dir),
            "NOTEBOOK_API_GENERATED_DIR": str(nested_dir),
        },
    )

    assert proc.returncode != 0
    assert "must not be the same directory" in proc.stderr


def test_upload_dir_and_generated_dir_configured_to_separate_paths_import_cleanly(
    tmp_path
):
    """The common, correct case -- two distinct, non-nested directories --
    must be completely unaffected by the collision check above.
    """

    proc = subprocess.run(
        [sys.executable, "-c", "from backend.routes import upload; print('IMPORT_OK')"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "NOTEBOOK_API_UPLOAD_DIR": str(tmp_path / "separate_uploads"),
            "NOTEBOOK_API_GENERATED_DIR": str(tmp_path / "separate_generated"),
        },
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "IMPORT_OK" in proc.stdout


def test_upload_rejects_filename_that_escapes_upload_dir():
    """Confirmed exploitable before this fix: an uploaded file named
    '../poc.ipynb' was written one directory above uploads/, outside the
    intended storage location, with status 200 "success".
    """

    resp = client.post(
        "/api/upload",
        files={"file": ("../escape_test.ipynb", io.BytesIO(b"data"), "application/json")},
    )

    assert resp.status_code == 400
    assert not os.path.exists("escape_test.ipynb")


def test_upload_rejects_a_filename_containing_a_nested_directory():
    """Confirmed exploitable before this fix: file.filename staying
    within UPLOAD_DIR (so the traversal check above lets it through) does
    not mean it has no directory component of its own. "subdir/nb.ipynb"
    crashed upload_notebook's own os.replace(temp_path, file_path) call
    with an unhandled FileNotFoundError -- an uncaught 500, not a clean
    400 -- since nothing ever creates the intermediate "subdir/"
    directory, and even if it did, every other route that operates on an
    uploaded notebook by name (list_notebooks, get_notebook,
    delete_notebook) already assumes a flat, single-segment filename.
    """

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "nested_dir_test/nb.ipynb",
                io.BytesIO(b"data"),
                "application/json",
            )
        },
    )

    assert resp.status_code == 400
    assert not os.path.isdir("uploads/nested_dir_test")


def test_compile_rejects_a_notebook_path_containing_a_nested_directory():
    """The same flat-filename restriction applies to notebook_path on
    every route that resolves it via resolve_upload_path, not just
    upload's own file.filename -- a caller can't route around it by
    typing a nested path directly into the JSON body instead.
    """

    resp = client.post(
        "/api/compile", json={"notebook_path": "some_dir/nb.ipynb"}
    )

    assert resp.status_code == 400


def test_upload_still_accepts_a_normal_flat_filename():
    """Sanity check alongside the nested-directory rejection above: an
    ordinary, single-segment filename must still upload successfully.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "flat_filename_sanity_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    assert resp.status_code == 200


def test_upload_rejects_content_that_is_not_a_valid_notebook():
    """Before this fix, /api/upload only checked the filename ended in
    ".ipynb" -- literally any content was accepted onto disk with a 200
    "success", and only failed later, opaquely, whenever /api/inspect or
    /api/compile next tried to parse it.
    """

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "garbage.ipynb",
                io.BytesIO(b"this is not json, let alone a notebook"),
                "application/json",
            )
        },
    )

    assert resp.status_code == 400
    assert not os.path.exists(os.path.join(UPLOAD_DIR, "garbage.ipynb"))


def test_upload_rejects_a_notebook_with_a_non_python_kernel():
    """Confirmed exploitable before this fix: a genuinely non-Python
    notebook (its own kernelspec.language declaring "R", say) uploaded
    and later "compiled" cleanly -- every one of its code cells simply
    failed is_parseable_python and was silently dropped, producing a
    working-but-endpoint-less API with nothing anywhere explaining why it
    exposed nothing. Rejecting it at upload, the earliest point this
    project's own MALFORMED_NOTEBOOK_ERRORS docstring already establishes
    for a not-even-valid-JSON notebook, surfaces the real, specific
    reason immediately instead.
    """

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "r_notebook.ipynb",
                io.BytesIO(_non_python_notebook_bytes()),
                "application/json",
            )
        },
    )

    assert resp.status_code == 400
    assert "not Python" in resp.json()["detail"]
    assert not os.path.exists(os.path.join(UPLOAD_DIR, "r_notebook.ipynb"))


def test_compile_rejects_a_non_python_notebook_already_on_disk():
    """POST /api/compile already validates with its own dedicated
    load_notebook() call before doing anything else (see MALFORMED_NOTEBOOK_ERRORS'
    own docstring) -- this notebook is written directly into UPLOAD_DIR,
    bypassing POST /api/upload's own identical check, specifically to
    confirm compile's own pre-check independently catches a non-Python
    kernel too, not only upload's.
    """

    filename = "compile_non_python.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(_non_python_notebook_bytes())

    compile_resp = client.post("/api/compile", json={"notebook_path": filename})

    assert compile_resp.status_code == 400
    assert "not Python" in compile_resp.json()["detail"]


def test_upload_rejects_a_notebook_exceeding_the_configured_max_size(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_UPLOAD_BYTES", 10)

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    assert len(content) > 10

    resp = client.post(
        "/api/upload",
        files={"file": ("too_big.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 413
    assert not os.path.exists(os.path.join(UPLOAD_DIR, "too_big.ipynb"))


def test_upload_accepts_a_notebook_within_a_raised_size_limit(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    resp = client.post(
        "/api/upload",
        files={"file": ("within_limit.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200
    assert os.path.exists(os.path.join(UPLOAD_DIR, "within_limit.ipynb"))


def test_upload_rejects_a_new_notebook_once_max_notebooks_is_reached(monkeypatch):

    from backend.routes import upload as upload_module

    current_count = upload_module._current_notebook_count()
    monkeypatch.setattr(upload_module, "MAX_NOTEBOOKS", current_count + 1)

    first_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "max_notebooks_first.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert first_resp.status_code == 200

    second_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "max_notebooks_second.ipynb",
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    assert second_resp.status_code == 400
    assert "maximum" in second_resp.json()["detail"].lower()
    assert not (Path(UPLOAD_DIR) / "max_notebooks_second.ipynb").exists()


def test_upload_overwrite_is_never_blocked_by_max_notebooks(monkeypatch):

    from backend.routes import upload as upload_module

    filename = "max_notebooks_overwrite_target.ipynb"

    setup_resp = client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert setup_resp.status_code == 200

    # The catalog is already "full" by this cap's own accounting -- an
    # overwrite of an already-existing filename must still succeed,
    # since it never changes how many distinct notebooks UPLOAD_DIR holds.
    monkeypatch.setattr(
        upload_module, "MAX_NOTEBOOKS", upload_module._current_notebook_count()
    )

    overwrite_resp = client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    assert overwrite_resp.status_code == 200
    assert overwrite_resp.json()["overwritten"] is True


def test_upload_dry_run_reports_the_max_notebooks_rejection_without_writing(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "MAX_NOTEBOOKS", upload_module._current_notebook_count()
    )

    resp = client.post(
        "/api/upload",
        params={"dry_run": "true"},
        files={
            "file": (
                "max_notebooks_dry_run.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )

    assert resp.status_code == 400
    assert not (Path(UPLOAD_DIR) / "max_notebooks_dry_run.ipynb").exists()


def test_upload_batch_reports_errors_for_files_beyond_max_notebooks(monkeypatch):

    from backend.routes import upload as upload_module

    current_count = upload_module._current_notebook_count()
    monkeypatch.setattr(upload_module, "MAX_NOTEBOOKS", current_count + 1)

    resp = client.post(
        "/api/upload/batch",
        files=[
            (
                "files",
                (
                    "max_notebooks_batch_a.ipynb",
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                ),
            ),
            (
                "files",
                (
                    "max_notebooks_batch_b.ipynb",
                    io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                    "application/json",
                ),
            ),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    statuses = {r["filename"]: r["status"] for r in body["results"]}
    assert statuses["max_notebooks_batch_a.ipynb"] == "success"
    assert statuses["max_notebooks_batch_b.ipynb"] == "error"
    assert not (Path(UPLOAD_DIR) / "max_notebooks_batch_b.ipynb").exists()


def test_upload_max_notebooks_disabled_by_default_allows_unbounded_uploads():

    from backend.routes.upload import MAX_NOTEBOOKS

    assert MAX_NOTEBOOKS == 0


def test_get_config_reports_the_max_notebooks_default_of_zero():

    from backend.routes.upload import MAX_NOTEBOOKS

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["max_notebooks"] == MAX_NOTEBOOKS


def test_get_config_reflects_a_configured_max_notebooks(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_NOTEBOOKS", 5)

    resp = client.get("/api/config")

    assert resp.json()["max_notebooks"] == 5


def test_upload_reports_overwritten_false_for_a_brand_new_notebook():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    resp = client.post(
        "/api/upload",
        files={"file": ("fresh.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200
    assert resp.json()["overwritten"] is False


def test_upload_overwrite_reports_was_currently_compiled_true_for_the_compiled_source():
    """An overwrite has exactly the same staleness effect on GENERATED_DIR
    a delete/rename already has -- DELETE /api/notebooks/{filename} (and
    friends) already report this same fact via "was_currently_compiled";
    this closes the identical gap for POST /api/upload?overwrite=true.
    """

    filename = "was_currently_compiled_upload_test.ipynb"
    _compile_a_notebook(filename)

    resp = client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    assert resp.status_code == 200
    assert resp.json()["was_currently_compiled"] is True


def test_upload_overwrite_reports_was_currently_compiled_false_for_an_unrelated_notebook():

    compiled_filename = "was_currently_compiled_other_test.ipynb"
    _compile_a_notebook(compiled_filename)

    unrelated_filename = "was_currently_compiled_unrelated_test.ipynb"
    client.post(
        "/api/upload",
        files={
            "file": (
                unrelated_filename,
                io.BytesIO(_notebook_bytes("def h() -> int:\n    return 3\n")),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                unrelated_filename,
                io.BytesIO(_notebook_bytes("def h2() -> int:\n    return 4\n")),
                "application/json",
            )
        },
    )

    assert resp.status_code == 200
    assert resp.json()["was_currently_compiled"] is False

    os.remove(Path(UPLOAD_DIR) / unrelated_filename)


def test_upload_dry_run_reports_was_currently_compiled_true_without_writing():

    filename = "was_currently_compiled_upload_dry_run_test.ipynb"
    _compile_a_notebook(filename)

    resp = client.post(
        "/api/upload",
        params={"overwrite": "true", "dry_run": "true"},
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["was_currently_compiled"] is True


def test_upload_reports_the_sha256_of_the_uploaded_content():

    import hashlib

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    expected = hashlib.sha256(content).hexdigest()

    resp = client.post(
        "/api/upload",
        files={"file": ("upload_sha256.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200
    assert resp.json()["sha256"] == expected


def test_upload_dry_run_does_not_write_the_file():
    """_save_uploaded_notebook's own "dry_run" already provides this
    preview -- reused by POST /api/notebooks/import -- but POST
    /api/upload, the endpoint every one of those ultimately exists to
    preview an upload *for*, never itself exposed it.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    resp = client.post(
        "/api/upload",
        params={"dry_run": "true"},
        files={"file": ("upload_dry_run.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["overwritten"] is False
    assert body["sha256"] == hashlib.sha256(content).hexdigest()

    assert not (Path(UPLOAD_DIR) / "upload_dry_run.ipynb").exists()


def test_upload_dry_run_reports_a_collision_without_writing_anything():

    original_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("upload_dry_run_collide.ipynb", io.BytesIO(original_content), "application/json")},
    )

    resp = client.post(
        "/api/upload",
        params={"dry_run": "true"},
        files={
            "file": (
                "upload_dry_run_collide.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )

    assert resp.status_code == 409
    assert (Path(UPLOAD_DIR) / "upload_dry_run_collide.ipynb").read_bytes() == original_content


def test_upload_dry_run_does_not_apply_tags_or_description():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    resp = client.post(
        "/api/upload",
        params={"dry_run": "true", "tags": "prod", "description": "hello"},
        files={"file": ("upload_dry_run_tags.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True
    assert not (Path(UPLOAD_DIR) / "upload_dry_run_tags.ipynb").exists()


def test_upload_batch_dry_run_does_not_write_any_file():

    content_a = _notebook_bytes("def f() -> int:\n    return 1\n")
    content_b = _notebook_bytes("def g() -> int:\n    return 2\n")

    resp = client.post(
        "/api/upload/batch",
        params={"dry_run": "true"},
        files=[
            ("files", ("batch_dry_run_a.ipynb", io.BytesIO(content_a), "application/json")),
            ("files", ("batch_dry_run_b.ipynb", io.BytesIO(content_b), "application/json")),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 2
    assert all(r["status"] == "success" for r in body["results"])

    assert not (Path(UPLOAD_DIR) / "batch_dry_run_a.ipynb").exists()
    assert not (Path(UPLOAD_DIR) / "batch_dry_run_b.ipynb").exists()


def test_upload_batch_dry_run_does_not_apply_tags_or_description():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    resp = client.post(
        "/api/upload/batch",
        params={"dry_run": "true", "tags": "prod", "description": "hello"},
        files=[
            ("files", ("batch_dry_run_tags.ipynb", io.BytesIO(content), "application/json")),
        ],
    )

    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True
    assert not (Path(UPLOAD_DIR) / "batch_dry_run_tags.ipynb").exists()


def test_upload_expected_sha256_matching_succeeds():

    import hashlib

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    expected = hashlib.sha256(content).hexdigest()

    resp = client.post(
        "/api/upload",
        params={"expected_sha256": expected},
        files={"file": ("upload_expected_sha256_match.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200
    assert resp.json()["sha256"] == expected


def test_upload_expected_sha256_is_case_insensitive():

    import hashlib

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    expected_upper = hashlib.sha256(content).hexdigest().upper()

    resp = client.post(
        "/api/upload",
        params={"expected_sha256": expected_upper},
        files={"file": ("upload_expected_sha256_upper.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200


def test_upload_expected_sha256_mismatch_is_rejected_and_saves_nothing():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    filename = "upload_expected_sha256_mismatch.ipynb"

    resp = client.post(
        "/api/upload",
        params={"expected_sha256": "0" * 64},
        files={"file": (filename, io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 400
    assert "does not match expected_sha256" in resp.json()["detail"]
    assert not (Path(UPLOAD_DIR) / filename).exists()


def test_upload_expected_sha256_mismatch_is_checked_after_notebook_validity():

    resp = client.post(
        "/api/upload",
        params={"expected_sha256": "0" * 64},
        files={
            "file": (
                "upload_expected_sha256_malformed.ipynb",
                io.BytesIO(b"not valid json"),
                "application/json",
            )
        },
    )

    assert resp.status_code == 400
    assert "not a valid Jupyter notebook" in resp.json()["detail"]


def test_upload_batch_reports_a_sha256_per_successful_file():

    import hashlib

    content_a = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    content_b = _notebook_bytes("def sub(a: int, b: int) -> int:\n    return a - b\n")

    resp = client.post(
        "/api/upload/batch",
        files=[
            ("files", ("batch_sha256_a.ipynb", io.BytesIO(content_a), "application/json")),
            ("files", ("batch_sha256_b.ipynb", io.BytesIO(content_b), "application/json")),
        ],
    )

    assert resp.status_code == 200
    results_by_filename = {r["filename"]: r for r in resp.json()["results"]}
    assert results_by_filename["batch_sha256_a.ipynb"]["sha256"] == hashlib.sha256(content_a).hexdigest()
    assert results_by_filename["batch_sha256_b.ipynb"]["sha256"] == hashlib.sha256(content_b).hexdigest()


def test_upload_tags_sets_the_notebooks_tags_on_success():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    resp = client.post(
        "/api/upload",
        params={"tags": "prod, reviewed"},
        files={"file": ("upload_tags.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200
    assert client.get("/api/notebooks/upload_tags.ipynb/tags").json()["tags"] == [
        "prod", "reviewed",
    ]


def test_upload_tags_is_not_applied_when_the_upload_itself_fails():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("upload_tags_collision.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/upload",
        params={"tags": "prod"},
        files={
            "file": (
                "upload_tags_collision.ipynb", io.BytesIO(content), "application/json",
            )
        },
    )

    assert resp.status_code == 409
    assert client.get(
        "/api/notebooks/upload_tags_collision.ipynb/tags"
    ).json()["tags"] == []


def test_upload_rejects_an_invalid_tags_value_before_reading_the_file(monkeypatch):

    resp = client.post(
        "/api/upload",
        params={"tags": "x" * 51},
        files={
            "file": (
                "upload_tags_bad.ipynb", io.BytesIO(b"not valid json"), "application/json",
            )
        },
    )

    assert resp.status_code == 400
    assert "tags" in resp.json()["detail"]
    assert not (Path(UPLOAD_DIR) / "upload_tags_bad.ipynb").exists()


def test_upload_description_sets_the_notebooks_description_on_success():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    resp = client.post(
        "/api/upload",
        params={"description": "adds two numbers"},
        files={"file": ("upload_description.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200
    assert client.get(
        "/api/notebooks/upload_description.ipynb/description"
    ).json()["description"] == "adds two numbers"


def test_upload_without_description_leaves_the_notebook_undescribed():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    resp = client.post(
        "/api/upload",
        files={"file": ("upload_no_description.ipynb", io.BytesIO(content), "application/json")},
    )

    assert resp.status_code == 200
    assert client.get(
        "/api/notebooks/upload_no_description.ipynb/description"
    ).json()["description"] == ""


def test_upload_description_is_not_applied_when_the_upload_itself_fails():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "upload_description_collision.ipynb", io.BytesIO(content), "application/json",
            )
        },
    )

    resp = client.post(
        "/api/upload",
        params={"description": "should not be set"},
        files={
            "file": (
                "upload_description_collision.ipynb", io.BytesIO(content), "application/json",
            )
        },
    )

    assert resp.status_code == 409
    assert client.get(
        "/api/notebooks/upload_description_collision.ipynb/description"
    ).json()["description"] == ""


def test_upload_rejects_an_invalid_description_value_before_reading_the_file():

    resp = client.post(
        "/api/upload",
        params={"description": "x" * 2001},
        files={
            "file": (
                "upload_description_bad.ipynb", io.BytesIO(b"not valid json"), "application/json",
            )
        },
    )

    assert resp.status_code == 400
    assert "description" in resp.json()["detail"]
    assert not (Path(UPLOAD_DIR) / "upload_description_bad.ipynb").exists()


def test_upload_batch_description_applies_uniformly_to_every_successful_file():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    resp = client.post(
        "/api/upload/batch",
        params={"description": "batch-uploaded"},
        files=[
            ("files", ("upload_batch_description_a.ipynb", io.BytesIO(content), "application/json")),
            ("files", ("upload_batch_description_b.ipynb", io.BytesIO(content), "application/json")),
        ],
    )

    assert resp.status_code == 200
    assert client.get(
        "/api/notebooks/upload_batch_description_a.ipynb/description"
    ).json()["description"] == "batch-uploaded"
    assert client.get(
        "/api/notebooks/upload_batch_description_b.ipynb/description"
    ).json()["description"] == "batch-uploaded"


def test_upload_batch_description_is_not_applied_to_a_file_that_failed_to_upload():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={
            "file": (
                "upload_batch_description_collision.ipynb", io.BytesIO(content), "application/json",
            )
        },
    )

    resp = client.post(
        "/api/upload/batch",
        params={"description": "batch-uploaded"},
        files=[
            ("files", ("upload_batch_description_collision.ipynb", io.BytesIO(content), "application/json")),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["status"] == "error"
    assert client.get(
        "/api/notebooks/upload_batch_description_collision.ipynb/description"
    ).json()["description"] == ""


def test_upload_batch_rejects_an_invalid_description_value_as_a_whole_request_400():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    resp = client.post(
        "/api/upload/batch",
        params={"description": "x" * 2001},
        files=[
            ("files", ("upload_batch_description_bad.ipynb", io.BytesIO(content), "application/json")),
        ],
    )

    assert resp.status_code == 400
    assert not (Path(UPLOAD_DIR) / "upload_batch_description_bad.ipynb").exists()


def test_import_notebooks_description_applies_to_every_successfully_imported_entry():

    content_a = _notebook_bytes("def f() -> int:\n    return 1\n")
    content_b = _notebook_bytes("def g() -> int:\n    return 2\n")

    archive_bytes = _zip_bytes({
        "import_description_a.ipynb": content_a,
        "import_description_b.ipynb": content_b,
    })

    resp = client.post(
        "/api/notebooks/import",
        params={"description": "imported from backup"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    assert client.get(
        "/api/notebooks/import_description_a.ipynb/description"
    ).json()["description"] == "imported from backup"
    assert client.get(
        "/api/notebooks/import_description_b.ipynb/description"
    ).json()["description"] == "imported from backup"


def test_import_notebooks_description_is_not_applied_to_a_failed_entry():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={
            "file": (
                "import_description_collision.ipynb", io.BytesIO(content), "application/json",
            )
        },
    )

    archive_bytes = _zip_bytes({"import_description_collision.ipynb": content})

    resp = client.post(
        "/api/notebooks/import",
        params={"description": "should not be set"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["status"] == "error"
    assert client.get(
        "/api/notebooks/import_description_collision.ipynb/description"
    ).json()["description"] == ""


def test_import_notebooks_rejects_an_invalid_description_value_before_reading_the_archive():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")
    archive_bytes = _zip_bytes({"import_description_bad.ipynb": content})

    resp = client.post(
        "/api/notebooks/import",
        params={"description": "x" * 2001},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 400
    assert not (Path(UPLOAD_DIR) / "import_description_bad.ipynb").exists()


def test_upload_sweeps_a_stale_leftover_temp_file_from_a_previous_crashed_upload(
    monkeypatch,
):
    """A hard process crash/restart between upload_notebook creating its
    hidden ".part" temp file and finishing the request skips every one of
    upload_notebook's own cleanup paths, leaving that file behind
    permanently -- it doesn't end in ".ipynb", so GET /api/notebooks never
    lists it, and nothing else ever looked at UPLOAD_DIR for one again.
    Confirmed: before this fix, such a file just sat there forever with
    no way to reclaim the disk space short of an operator finding and
    deleting a hidden dot-file on the server by hand.
    """

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "STALE_UPLOAD_TEMP_FILE_SECONDS", 1)

    stale_temp_path = Path(UPLOAD_DIR) / ".crash_leftover_test.ipynb.deadbeef.part"
    stale_temp_path.write_text("leftover from a crashed upload", encoding="utf-8")

    old_time = os.path.getmtime(stale_temp_path) - 100
    os.utime(stale_temp_path, (old_time, old_time))

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "stale_temp_sweep_trigger.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    assert resp.status_code == 200
    assert not stale_temp_path.exists()


def test_upload_leaves_a_recent_in_flight_temp_file_alone(monkeypatch):
    """The sweep must be age-gated, not indiscriminate -- a ".part" file
    younger than the staleness threshold could belong to a large upload
    that is itself still genuinely streaming on another concurrent
    request, and must not be deleted out from under it.
    """

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "STALE_UPLOAD_TEMP_FILE_SECONDS", 3600)

    recent_temp_path = Path(UPLOAD_DIR) / ".still_in_flight_test.ipynb.deadbeef.part"
    recent_temp_path.write_text("still streaming", encoding="utf-8")

    try:
        content = _notebook_bytes(
            "def add(a: int, b: int) -> int:\n    return a + b\n"
        )
        resp = client.post(
            "/api/upload",
            files={
                "file": (
                    "recent_temp_sweep_trigger.ipynb",
                    io.BytesIO(content),
                    "application/json",
                )
            },
        )

        assert resp.status_code == 200
        assert recent_temp_path.exists()
    finally:
        recent_temp_path.unlink(missing_ok=True)


def test_upload_rejects_a_same_named_reupload_without_overwrite():
    """Before overwrite protection existed, re-uploading an existing
    filename silently replaced its bytes as they streamed in -- with no
    conflict response and no way to opt out of the collision.
    """

    original_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    first_resp = client.post(
        "/api/upload",
        files={"file": ("collide.ipynb", io.BytesIO(original_content), "application/json")},
    )
    assert first_resp.status_code == 200

    conflicting_content = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    second_resp = client.post(
        "/api/upload",
        files={"file": ("collide.ipynb", io.BytesIO(conflicting_content), "application/json")},
    )

    assert second_resp.status_code == 409
    assert "already exists" in second_resp.json()["detail"]

    # The original file must be completely untouched.
    on_disk = Path(UPLOAD_DIR, "collide.ipynb").read_bytes()
    assert on_disk == original_content


def test_upload_replaces_a_same_named_notebook_when_overwrite_is_requested():

    original_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    first_resp = client.post(
        "/api/upload",
        files={"file": ("replace_me.ipynb", io.BytesIO(original_content), "application/json")},
    )
    assert first_resp.status_code == 200

    new_content = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    second_resp = client.post(
        "/api/upload?overwrite=true",
        files={"file": ("replace_me.ipynb", io.BytesIO(new_content), "application/json")},
    )

    assert second_resp.status_code == 200
    assert second_resp.json()["overwritten"] is True

    on_disk = Path(UPLOAD_DIR, "replace_me.ipynb").read_bytes()
    assert on_disk == new_content


def test_upload_with_overwrite_leaves_original_untouched_if_replacement_is_invalid():
    """The critical data-loss case this fix closes: even with
    ?overwrite=true, an invalid or corrupt re-upload must not destroy the
    existing good notebook. Before streaming to a temp file first, the
    write to the final path happened before validation, so this exact
    scenario silently lost the original.
    """

    original_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    first_resp = client.post(
        "/api/upload",
        files={"file": ("dont_lose_me.ipynb", io.BytesIO(original_content), "application/json")},
    )
    assert first_resp.status_code == 200

    second_resp = client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                "dont_lose_me.ipynb",
                io.BytesIO(b"this is not json, let alone a notebook"),
                "application/json",
            )
        },
    )

    assert second_resp.status_code == 400

    on_disk = Path(UPLOAD_DIR, "dont_lose_me.ipynb").read_bytes()
    assert on_disk == original_content


def test_upload_leaves_no_temp_files_behind_after_a_rejected_reupload():

    original_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("no_debris.ipynb", io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("no_debris.ipynb", io.BytesIO(original_content), "application/json")},
    )

    leftover = [
        name for name in os.listdir(UPLOAD_DIR)
        if "no_debris" in name and not name.endswith(".ipynb")
    ]
    assert leftover == []


def test_upload_batch_uploads_multiple_notebooks_in_one_request():

    first_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    second_content = _notebook_bytes("def subtract(a: int, b: int) -> int:\n    return a - b\n")

    resp = client.post(
        "/api/upload/batch",
        files=[
            ("files", ("batch_a.ipynb", io.BytesIO(first_content), "application/json")),
            ("files", ("batch_b.ipynb", io.BytesIO(second_content), "application/json")),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0
    assert [r["filename"] for r in body["results"]] == ["batch_a.ipynb", "batch_b.ipynb"]
    assert all(r["status"] == "success" for r in body["results"])
    assert all(r["overwritten"] is False for r in body["results"])

    assert (Path(UPLOAD_DIR) / "batch_a.ipynb").read_bytes() == first_content
    assert (Path(UPLOAD_DIR) / "batch_b.ipynb").read_bytes() == second_content


def test_upload_batch_continues_past_a_single_invalid_file():
    """One bad file in the batch must not abort the rest -- each file is
    processed independently, unlike a naive loop of individual POST
    /api/upload calls a caller might otherwise have to write, where an
    unhandled error on file N could leave files after it never attempted.
    """

    good_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    resp = client.post(
        "/api/upload/batch",
        files=[
            ("files", ("batch_good.ipynb", io.BytesIO(good_content), "application/json")),
            ("files", ("batch_bad.ipynb", io.BytesIO(b"not a notebook"), "application/json")),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    good_result, bad_result = body["results"]
    assert good_result == {
        "status": "success",
        "filename": "batch_good.ipynb",
        "path": str(resolve_upload_path("batch_good.ipynb")),
        "overwritten": False,
        "sha256": hashlib.sha256(good_content).hexdigest(),
        "was_currently_compiled": False,
    }
    assert bad_result["filename"] == "batch_bad.ipynb"
    assert bad_result["status"] == "error"
    assert "not a valid Jupyter notebook" in bad_result["detail"]

    assert (Path(UPLOAD_DIR) / "batch_good.ipynb").is_file()
    assert not (Path(UPLOAD_DIR) / "batch_bad.ipynb").exists()


def test_upload_batch_reports_a_collision_error_without_overwrite():

    original_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("batch_collide.ipynb", io.BytesIO(original_content), "application/json")},
    )

    resp = client.post(
        "/api/upload/batch",
        files=[
            (
                "files",
                (
                    "batch_collide.ipynb",
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                ),
            ),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["succeeded_count"] == 0
    assert body["failed_count"] == 1
    assert body["results"][0]["status"] == "error"
    assert "already exists" in body["results"][0]["detail"]

    # The original file must be completely untouched.
    assert (Path(UPLOAD_DIR) / "batch_collide.ipynb").read_bytes() == original_content


def test_upload_batch_overwrite_applies_to_every_file():

    original_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("batch_overwrite.ipynb", io.BytesIO(original_content), "application/json")},
    )

    replacement_content = _notebook_bytes("def subtract(a: int, b: int) -> int:\n    return a - b\n")

    resp = client.post(
        "/api/upload/batch?overwrite=true",
        files=[
            (
                "files",
                ("batch_overwrite.ipynb", io.BytesIO(replacement_content), "application/json"),
            ),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["succeeded_count"] == 1
    assert body["results"][0]["overwritten"] is True
    assert (Path(UPLOAD_DIR) / "batch_overwrite.ipynb").read_bytes() == replacement_content


def test_upload_batch_tags_applies_uniformly_to_every_successful_file():

    first_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    second_content = _notebook_bytes("def sub(a: int, b: int) -> int:\n    return a - b\n")

    resp = client.post(
        "/api/upload/batch",
        params={"tags": "prod,batch"},
        files=[
            ("files", ("batch_tags_a.ipynb", io.BytesIO(first_content), "application/json")),
            ("files", ("batch_tags_b.ipynb", io.BytesIO(second_content), "application/json")),
        ],
    )

    assert resp.status_code == 200
    assert resp.json()["succeeded_count"] == 2

    assert client.get("/api/notebooks/batch_tags_a.ipynb/tags").json()["tags"] == [
        "batch", "prod",
    ]
    assert client.get("/api/notebooks/batch_tags_b.ipynb/tags").json()["tags"] == [
        "batch", "prod",
    ]


def test_upload_batch_tags_is_not_applied_to_a_file_that_failed_to_upload():

    resp = client.post(
        "/api/upload/batch",
        params={"tags": "prod"},
        files=[
            (
                "files",
                (
                    "batch_tags_bad.ipynb", io.BytesIO(b"not a notebook"), "application/json",
                ),
            ),
        ],
    )

    assert resp.status_code == 200
    assert resp.json()["failed_count"] == 1
    assert not (Path(UPLOAD_DIR) / "batch_tags_bad.ipynb").exists()


def test_upload_batch_rejects_an_invalid_tags_value_as_a_whole_request_400():

    resp = client.post(
        "/api/upload/batch",
        params={"tags": "x" * 51},
        files=[
            ("files", ("batch_tags_invalid.ipynb", io.BytesIO(b"{}"), "application/json")),
        ],
    )

    assert resp.status_code == 400
    assert not (Path(UPLOAD_DIR) / "batch_tags_invalid.ipynb").exists()


def test_upload_batch_rejects_more_files_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    resp = client.post(
        "/api/upload/batch",
        files=[
            ("files", ("batch_max_a.ipynb", io.BytesIO(content), "application/json")),
            ("files", ("batch_max_b.ipynb", io.BytesIO(content), "application/json")),
        ],
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]
    assert not (Path(UPLOAD_DIR) / "batch_max_a.ipynb").exists()
    assert not (Path(UPLOAD_DIR) / "batch_max_b.ipynb").exists()


def _zip_bytes(entries):
    """Build an in-memory .zip archive from {entry_name: content_bytes}."""

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry_name, content in entries.items():
            archive.writestr(entry_name, content)

    return buffer.getvalue()


def test_import_notebooks_uploads_every_ipynb_entry_in_the_zip():

    content_a = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    content_b = _notebook_bytes("def sub(a: int, b: int) -> int:\n    return a - b\n")

    archive_bytes = _zip_bytes({
        "import_a.ipynb": content_a,
        "import_b.ipynb": content_b,
    })

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0
    assert [r["filename"] for r in body["results"]] == ["import_a.ipynb", "import_b.ipynb"]
    assert all(r["status"] == "success" for r in body["results"])

    assert (Path(UPLOAD_DIR) / "import_a.ipynb").read_bytes() == content_a
    assert (Path(UPLOAD_DIR) / "import_b.ipynb").read_bytes() == content_b


def test_import_notebooks_flattens_nested_paths_to_their_basename():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    archive_bytes = _zip_bytes({"nested/dir/import_nested.ipynb": content})

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["filename"] == "import_nested.ipynb"
    assert (Path(UPLOAD_DIR) / "import_nested.ipynb").read_bytes() == content


def test_import_notebooks_skips_non_ipynb_entries():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    archive_bytes = _zip_bytes({
        "import_readme.ipynb": content,
        "README.md": b"not a notebook",
    })

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert [r["filename"] for r in body["results"]] == ["import_readme.ipynb"]


def test_import_notebooks_continues_past_a_single_invalid_entry():

    good_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    archive_bytes = _zip_bytes({
        "import_good.ipynb": good_content,
        "import_bad.ipynb": b"not a notebook",
    })

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    good_result, bad_result = body["results"]
    assert good_result["status"] == "success"
    assert bad_result["filename"] == "import_bad.ipynb"
    assert bad_result["status"] == "error"
    assert "not a valid Jupyter notebook" in bad_result["detail"]

    assert (Path(UPLOAD_DIR) / "import_good.ipynb").is_file()
    assert not (Path(UPLOAD_DIR) / "import_bad.ipynb").exists()


def test_import_notebooks_reports_a_collision_error_without_overwrite():

    original_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("import_collide.ipynb", io.BytesIO(original_content), "application/json")},
    )

    archive_bytes = _zip_bytes({
        "import_collide.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
    })

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["failed_count"] == 1
    assert "already exists" in body["results"][0]["detail"]

    assert (Path(UPLOAD_DIR) / "import_collide.ipynb").read_bytes() == original_content


def test_import_notebooks_overwrite_applies_to_every_entry():

    original_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("import_overwrite.ipynb", io.BytesIO(original_content), "application/json")},
    )

    replacement_content = _notebook_bytes("def sub(a: int, b: int) -> int:\n    return a - b\n")

    archive_bytes = _zip_bytes({"import_overwrite.ipynb": replacement_content})

    resp = client.post(
        "/api/notebooks/import?overwrite=true",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["overwritten"] is True
    assert (Path(UPLOAD_DIR) / "import_overwrite.ipynb").read_bytes() == replacement_content


def test_import_notebooks_tags_applies_to_every_successfully_imported_entry():

    content_a = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    content_b = _notebook_bytes("def sub(a: int, b: int) -> int:\n    return a - b\n")

    archive_bytes = _zip_bytes({
        "import_tags_a.ipynb": content_a,
        "import_tags_b.ipynb": content_b,
    })

    resp = client.post(
        "/api/notebooks/import",
        params={"tags": "imported,reviewed"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    assert resp.json()["succeeded_count"] == 2

    assert client.get(
        "/api/notebooks/import_tags_a.ipynb/tags"
    ).json()["tags"] == ["imported", "reviewed"]
    assert client.get(
        "/api/notebooks/import_tags_b.ipynb/tags"
    ).json()["tags"] == ["imported", "reviewed"]


def test_import_notebooks_tags_is_not_applied_to_a_failed_entry():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("import_tags_collision.ipynb", io.BytesIO(content), "application/json")},
    )

    archive_bytes = _zip_bytes({"import_tags_collision.ipynb": content})

    resp = client.post(
        "/api/notebooks/import",
        params={"tags": "imported"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["status"] == "error"

    assert client.get(
        "/api/notebooks/import_tags_collision.ipynb/tags"
    ).json()["tags"] == []


def test_import_notebooks_rejects_an_invalid_tags_value_before_reading_the_archive():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    archive_bytes = _zip_bytes({"import_tags_bad.ipynb": content})

    resp = client.post(
        "/api/notebooks/import",
        params={"tags": "x" * 51},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 400
    assert not (Path(UPLOAD_DIR) / "import_tags_bad.ipynb").exists()


def test_import_notebooks_rejects_a_non_zip_file():

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("notes.ipynb", io.BytesIO(b"{}"), "application/json")},
    )

    assert resp.status_code == 400
    assert "must be a .zip archive" in resp.json()["detail"]


def test_import_notebooks_rejects_a_corrupt_zip_file():

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(b"not a real zip"), "application/zip")},
    )

    assert resp.status_code == 400
    assert "not a valid zip archive" in resp.json()["detail"]


def test_import_notebooks_succeeds_with_a_matching_expected_sha256():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")
    archive_bytes = _zip_bytes({"import_expected_sha256_ok.ipynb": content})
    expected = hashlib.sha256(archive_bytes).hexdigest()

    resp = client.post(
        "/api/notebooks/import",
        params={"expected_sha256": expected},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["status"] == "success"
    assert (Path(UPLOAD_DIR) / "import_expected_sha256_ok.ipynb").is_file()


def test_import_notebooks_rejects_a_mismatched_expected_sha256_before_writing_anything():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")
    archive_bytes = _zip_bytes({"import_expected_sha256_bad.ipynb": content})

    resp = client.post(
        "/api/notebooks/import",
        params={"expected_sha256": "0" * 64},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 400
    assert "does not match expected_sha256" in resp.json()["detail"]
    assert not (Path(UPLOAD_DIR) / "import_expected_sha256_bad.ipynb").is_file()


def test_import_notebooks_expected_sha256_is_case_insensitive():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")
    archive_bytes = _zip_bytes({"import_expected_sha256_case.ipynb": content})
    expected = hashlib.sha256(archive_bytes).hexdigest().upper()

    resp = client.post(
        "/api/notebooks/import",
        params={"expected_sha256": expected},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200


def test_import_notebooks_rejects_a_mismatched_expected_sha256_before_a_bad_zip_check():
    """A malformed upload is still reported as that specific, more
    actionable error, not a bare hash mismatch, regardless of whether
    "expected_sha256" was given -- the identical ordering
    _save_uploaded_notebook's own check already follows for a single
    notebook.
    """

    resp = client.post(
        "/api/notebooks/import",
        params={"expected_sha256": "0" * 64},
        files={"file": ("bundle.zip", io.BytesIO(b"not a real zip"), "application/zip")},
    )

    assert resp.status_code == 400
    assert "not a valid zip archive" in resp.json()["detail"]


def test_import_notebooks_rejects_a_zip_with_no_ipynb_files():

    archive_bytes = _zip_bytes({"README.md": b"nothing to import here"})

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 400
    assert "no .ipynb files" in resp.json()["detail"]


def test_import_notebooks_rejects_more_entries_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    archive_bytes = _zip_bytes({
        "import_max_a.ipynb": content,
        "import_max_b.ipynb": content,
    })

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]
    assert not (Path(UPLOAD_DIR) / "import_max_a.ipynb").exists()
    assert not (Path(UPLOAD_DIR) / "import_max_b.ipynb").exists()


def test_import_notebooks_round_trips_an_include_versions_export_archive():

    client.delete("/api/notebooks?confirm=true")

    original_a = _notebook_bytes("def f() -> int:\n    return 1\n")
    current_a = _notebook_bytes("def f() -> int:\n    return 2\n")
    current_b = _notebook_bytes("def g() -> int:\n    return 3\n")

    client.post(
        "/api/upload",
        files={"file": ("import_versions_a.ipynb", io.BytesIO(original_a), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": ("import_versions_a.ipynb", io.BytesIO(current_a), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("import_versions_b.ipynb", io.BytesIO(current_b), "application/json")},
    )

    version_id = client.get(
        "/api/notebooks/import_versions_a.ipynb/versions"
    ).json()["versions"][0]["version_id"]

    export_bytes = client.get(
        "/api/notebooks/export", params={"include_versions": "true"}
    ).content

    client.delete("/api/notebooks?confirm=true")

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 2

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["import_versions_a.ipynb"]["restored_version_count"] == 1
    assert results_by_filename["import_versions_b.ipynb"]["restored_version_count"] == 0

    assert client.get("/api/notebooks/import_versions_a.ipynb").content == current_a
    assert client.get("/api/notebooks/import_versions_b.ipynb").content == current_b

    restored_versions = client.get(
        "/api/notebooks/import_versions_a.ipynb/versions"
    ).json()["versions"]
    assert [v["version_id"] for v in restored_versions] == [version_id]
    assert (
        client.get(
            f"/api/notebooks/import_versions_a.ipynb/versions/{version_id}"
        ).content
        == original_a
    )

    assert (
        client.get("/api/notebooks/import_versions_b.ipynb/versions").json()["versions"]
        == []
    )


def test_import_notebooks_ignores_version_entries_with_no_matching_notebook_entry():

    archive_bytes = _zip_bytes({
        "import_orphan_versions.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
        "versions/some_other_notebook.ipynb/20260101T000000000000_abcd1234.ipynb": b"{}",
    })

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["results"][0]["filename"] == "import_orphan_versions.ipynb"
    assert body["results"][0]["restored_version_count"] == 0

    assert client.get(
        "/api/notebooks/import_orphan_versions.ipynb/versions"
    ).json()["versions"] == []


def test_import_notebooks_restores_each_entrys_own_archived_tags_and_description():

    archive_bytes = _zip_bytes({
        "import_meta_a.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
        "import_meta_b.ipynb": _notebook_bytes("def g() -> int:\n    return 2\n"),
        "tags/import_meta_a.ipynb.json": json.dumps(["production", "bug"]),
        "description/import_meta_a.ipynb.txt": "the first notebook",
        "tags/import_meta_b.ipynb.json": json.dumps(["staging"]),
    })

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 2

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["import_meta_a.ipynb"]["restored_tags"] == ["bug", "production"]
    assert results_by_filename["import_meta_a.ipynb"]["restored_description"] == "the first notebook"
    assert results_by_filename["import_meta_b.ipynb"]["restored_tags"] == ["staging"]
    assert results_by_filename["import_meta_b.ipynb"]["restored_description"] is None

    assert client.get(
        "/api/notebooks/import_meta_a.ipynb/tags"
    ).json()["tags"] == ["bug", "production"]
    assert client.get(
        "/api/notebooks/import_meta_a.ipynb/description"
    ).json()["description"] == "the first notebook"
    assert client.get(
        "/api/notebooks/import_meta_b.ipynb/tags"
    ).json()["tags"] == ["staging"]
    assert client.get(
        "/api/notebooks/import_meta_b.ipynb/description"
    ).json()["description"] == ""


def test_import_notebooks_explicit_tags_and_description_override_the_archived_ones():

    archive_bytes = _zip_bytes({
        "import_meta_override.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
        "tags/import_meta_override.ipynb.json": json.dumps(["archived"]),
        "description/import_meta_override.ipynb.txt": "archived description",
    })

    resp = client.post(
        "/api/notebooks/import",
        params={"tags": "explicit", "description": "explicit description"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    result = body["results"][0]

    # The explicit query params win -- and since they did, nothing was
    # actually *restored from the archive* for this entry.
    assert result["restored_tags"] is None
    assert result["restored_description"] is None

    assert client.get(
        "/api/notebooks/import_meta_override.ipynb/tags"
    ).json()["tags"] == ["explicit"]
    assert client.get(
        "/api/notebooks/import_meta_override.ipynb/description"
    ).json()["description"] == "explicit description"


def test_import_notebooks_dry_run_predicts_restored_tags_and_description_without_writing():

    archive_bytes = _zip_bytes({
        "import_meta_dry_run.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
        "tags/import_meta_dry_run.ipynb.json": json.dumps(["production"]),
        "description/import_meta_dry_run.ipynb.txt": "would be restored",
    })

    resp = client.post(
        "/api/notebooks/import",
        params={"dry_run": "true"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["restored_tags"] == ["production"]
    assert result["restored_description"] == "would be restored"

    assert not (Path(UPLOAD_DIR) / "import_meta_dry_run.ipynb").exists()


def test_import_notebooks_restores_each_entrys_own_archived_source_url():

    archive_bytes = _zip_bytes({
        "import_source_url_a.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
        "import_source_url_b.ipynb": _notebook_bytes("def g() -> int:\n    return 2\n"),
        "source_url/import_source_url_a.ipynb.json": json.dumps(
            {"source_url": "https://example.com/a.ipynb"}
        ),
    })

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 2

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert (
        results_by_filename["import_source_url_a.ipynb"]["restored_source_url"]
        == "https://example.com/a.ipynb"
    )
    assert results_by_filename["import_source_url_b.ipynb"]["restored_source_url"] is None

    assert client.get(
        "/api/notebooks/import_source_url_a.ipynb/info"
    ).json()["source_url"] == "https://example.com/a.ipynb"
    assert client.get(
        "/api/notebooks/import_source_url_b.ipynb/info"
    ).json()["source_url"] is None


def test_import_notebooks_dry_run_predicts_restored_source_url_without_writing():

    archive_bytes = _zip_bytes({
        "import_source_url_dry_run.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
        "source_url/import_source_url_dry_run.ipynb.json": json.dumps(
            {"source_url": "https://example.com/dry-run.ipynb"}
        ),
    })

    resp = client.post(
        "/api/notebooks/import",
        params={"dry_run": "true"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["restored_source_url"] == "https://example.com/dry-run.ipynb"

    assert not (Path(UPLOAD_DIR) / "import_source_url_dry_run.ipynb").exists()
    assert not _source_url_sidecar_path("import_source_url_dry_run.ipynb").exists()


def test_export_notebooks_round_trips_each_notebooks_own_tags_and_description():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("export_meta_a.ipynb", io.BytesIO(content), "application/json")},
    )
    client.put(
        "/api/notebooks/export_meta_a.ipynb/tags", json={"tags": ["prod", "bug"]}
    )
    client.put(
        "/api/notebooks/export_meta_a.ipynb/description",
        json={"description": "tagged and described"},
    )
    client.post(
        "/api/upload",
        files={"file": ("export_meta_untagged.ipynb", io.BytesIO(content), "application/json")},
    )

    export_bytes = client.get("/api/notebooks/export").content

    with zipfile.ZipFile(io.BytesIO(export_bytes)) as archive:
        assert json.loads(
            archive.read("tags/export_meta_a.ipynb.json")
        ) == ["bug", "prod"]
        assert (
            archive.read("description/export_meta_a.ipynb.txt").decode("utf-8")
            == "tagged and described"
        )
        # An untagged, undescribed notebook contributes no entries at all.
        assert "tags/export_meta_untagged.ipynb.json" not in archive.namelist()
        assert "description/export_meta_untagged.ipynb.txt" not in archive.namelist()

    client.delete("/api/notebooks?confirm=true")

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    results_by_filename = {r["filename"]: r for r in resp.json()["results"]}
    assert results_by_filename["export_meta_a.ipynb"]["restored_tags"] == ["bug", "prod"]
    assert (
        results_by_filename["export_meta_a.ipynb"]["restored_description"]
        == "tagged and described"
    )

    assert client.get(
        "/api/notebooks/export_meta_a.ipynb/tags"
    ).json()["tags"] == ["bug", "prod"]
    assert client.get(
        "/api/notebooks/export_meta_a.ipynb/description"
    ).json()["description"] == "tagged and described"


def test_export_notebooks_round_trips_source_url(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    client.delete("/api/notebooks?confirm=true")

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/export_source_url_a.ipynb"},
    )
    client.post(
        "/api/upload",
        files={
            "file": (
                "export_source_url_untagged.ipynb",
                io.BytesIO(handler.content),
                "application/json",
            )
        },
    )

    export_bytes = client.get("/api/notebooks/export").content

    with zipfile.ZipFile(io.BytesIO(export_bytes)) as archive:
        assert json.loads(
            archive.read("source_url/export_source_url_a.ipynb.json")
        ) == {"source_url": f"{base_url}/export_source_url_a.ipynb"}
        # A notebook that was never imported from a URL contributes no
        # entry at all -- the same "empty/absent is a valid state, not
        # an error" reasoning "tags/"/"description/" already follow.
        assert (
            "source_url/export_source_url_untagged.ipynb.json"
            not in archive.namelist()
        )

    client.delete("/api/notebooks?confirm=true")

    resp = client.post(
        "/api/notebooks/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    results_by_filename = {r["filename"]: r for r in resp.json()["results"]}
    assert (
        results_by_filename["export_source_url_a.ipynb"]["restored_source_url"]
        == f"{base_url}/export_source_url_a.ipynb"
    )
    assert (
        results_by_filename["export_source_url_untagged.ipynb"]["restored_source_url"]
        is None
    )

    assert client.get(
        "/api/notebooks/export_source_url_a.ipynb/info"
    ).json()["source_url"] == f"{base_url}/export_source_url_a.ipynb"


def test_import_notebooks_dry_run_does_not_write_any_file():

    archive_bytes = _zip_bytes({
        "import_dry_run.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
    })

    resp = client.post(
        "/api/notebooks/import",
        params={"dry_run": "true"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1
    assert body["results"][0]["status"] == "success"
    assert body["results"][0]["filename"] == "import_dry_run.ipynb"
    assert body["results"][0]["overwritten"] is False

    assert not (Path(UPLOAD_DIR) / "import_dry_run.ipynb").exists()


def test_import_notebooks_dry_run_reports_a_collision_without_writing_anything():

    original_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("import_dry_run_collide.ipynb", io.BytesIO(original_content), "application/json")},
    )

    archive_bytes = _zip_bytes({
        "import_dry_run_collide.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
    })

    resp = client.post(
        "/api/notebooks/import",
        params={"dry_run": "true"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["failed_count"] == 1
    assert "already exists" in body["results"][0]["detail"]

    assert (Path(UPLOAD_DIR) / "import_dry_run_collide.ipynb").read_bytes() == original_content


def test_import_notebooks_dry_run_reports_an_invalid_entry_without_writing_the_good_one():

    good_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    archive_bytes = _zip_bytes({
        "import_dry_run_good.ipynb": good_content,
        "import_dry_run_bad.ipynb": b"not a notebook",
    })

    resp = client.post(
        "/api/notebooks/import",
        params={"dry_run": "true"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    good_result, bad_result = body["results"]
    assert good_result["status"] == "success"
    assert bad_result["status"] == "error"
    assert "not a valid Jupyter notebook" in bad_result["detail"]

    assert not (Path(UPLOAD_DIR) / "import_dry_run_good.ipynb").exists()
    assert not (Path(UPLOAD_DIR) / "import_dry_run_bad.ipynb").exists()


def test_import_notebooks_dry_run_does_not_apply_tags_or_description():

    archive_bytes = _zip_bytes({
        "import_dry_run_tags.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
    })

    resp = client.post(
        "/api/notebooks/import",
        params={"dry_run": "true", "tags": "production", "description": "a preview"},
        files={"file": ("bundle.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    assert resp.json()["succeeded_count"] == 1
    assert not (Path(UPLOAD_DIR) / "import_dry_run_tags.ipynb").exists()


def test_import_notebooks_dry_run_predicts_restored_version_count_without_restoring():

    client.delete("/api/notebooks?confirm=true")

    original_a = _notebook_bytes("def f() -> int:\n    return 1\n")
    current_a = _notebook_bytes("def f() -> int:\n    return 2\n")

    client.post(
        "/api/upload",
        files={"file": ("import_dry_run_versions.ipynb", io.BytesIO(original_a), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": ("import_dry_run_versions.ipynb", io.BytesIO(current_a), "application/json")},
    )

    export_bytes = client.get(
        "/api/notebooks/export", params={"include_versions": "true"}
    ).content

    client.delete("/api/notebooks?confirm=true")

    resp = client.post(
        "/api/notebooks/import",
        params={"dry_run": "true"},
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["restored_version_count"] == 1

    assert not (Path(UPLOAD_DIR) / "import_dry_run_versions.ipynb").exists()
    assert client.get(
        "/api/notebooks/import_dry_run_versions.ipynb/versions"
    ).status_code == 404


class _NotebookUrlHandler(http.server.BaseHTTPRequestHandler):
    """A minimal HTTP server standing in for wherever a notebook actually
    lives (a GitHub raw URL, an S3 object URL, ...) for POST
    /api/notebooks/import-url's own tests below -- serves fixed bytes at
    a fixed path, and (only when `redirect_to` is set) a redirect at
    "/redirect" instead, for the tests exercising import-url's own
    redirect handling.
    """

    content = b""
    redirect_to = None
    # One entry per request this handler (or any other server sharing
    # this same class -- see test_import_url_drops_custom_headers_after_a_cross_origin_redirect
    # below, which spins up a second server from this identical class
    # specifically so its own request headers land in this same shared
    # log) actually received, in request order -- lets a test tell hop 1's
    # own headers apart from hop 2's after a redirect, rather than only
    # ever seeing whichever request happened to run last.
    received_headers_log = None

    def do_GET(self):

        type(self).received_headers_log.append(dict(self.headers))

        if self.path == "/redirect" and type(self).redirect_to:
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
            return

        payload = type(self).content
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ipynb+json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def notebook_url_server():
    _NotebookUrlHandler.content = b""
    _NotebookUrlHandler.redirect_to = None
    _NotebookUrlHandler.received_headers_log = []

    server = http.server.HTTPServer(("127.0.0.1", 0), _NotebookUrlHandler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        yield f"http://127.0.0.1:{port}", _NotebookUrlHandler
    finally:
        server.shutdown()
        server_thread.join(timeout=5)


@pytest.fixture
def _bypass_import_url_ssrf_guard(monkeypatch):
    """POST /api/notebooks/import-url's own real _reject_unsafe_import_url_host
    correctly refuses every loopback address -- including notebook_url_server's
    own 127.0.0.1 -- so every test below that needs an actual successful fetch
    against that local server has to bypass it deliberately. The guard's own
    rejection behavior is tested directly, against the real function, in
    test_import_url_rejects_a_private_address/test_import_url_rejects_a_non_http_scheme
    below instead.
    """
    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "_reject_unsafe_import_url_host", lambda url: None
    )


def test_import_url_fetches_and_saves_a_notebook(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/nb.ipynb"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "nb.ipynb"
    assert body["overwritten"] is False
    assert body["dry_run"] is False
    assert body["source_url"] == f"{base_url}/nb.ipynb"
    assert body["sha256"] == hashlib.sha256(handler.content).hexdigest()

    assert (Path(UPLOAD_DIR) / "nb.ipynb").read_bytes() == handler.content


def test_import_url_persists_source_url_to_notebook_info(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    """Before this, "source_url" was only ever visible in that one
    request's own response -- gone the moment it scrolled off a
    terminal, with no way to later ask "which notebooks came from a URL,
    and from where".
    """

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/source_url_test.ipynb"},
    )
    assert resp.status_code == 200, resp.text

    info = client.get("/api/notebooks/source_url_test.ipynb/info").json()
    assert info["source_url"] == f"{base_url}/source_url_test.ipynb"


def test_source_url_is_null_for_a_directly_uploaded_notebook():

    _upload_sample_notebook("source_url_direct_upload_test.ipynb")

    info = client.get("/api/notebooks/source_url_direct_upload_test.ipynb/info").json()
    assert info["source_url"] is None


def test_import_url_does_not_persist_source_url_under_dry_run(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/source_url_dry_run.ipynb", "dry_run": True},
    )
    assert resp.status_code == 200, resp.text

    assert not _source_url_sidecar_path("source_url_dry_run.ipynb").exists()


def test_import_url_overwrite_updates_source_url_to_the_new_url(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    """The notebook's own current content only ever came from the most
    recent import -- recording anything else about "source_url" would be
    misleading.
    """

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/source_url_overwrite_a.ipynb", "filename": "source_url_overwrite.ipynb"},
    )

    resp = client.post(
        "/api/notebooks/import-url",
        json={
            "url": f"{base_url}/source_url_overwrite_b.ipynb",
            "filename": "source_url_overwrite.ipynb",
            "overwrite": True,
        },
    )
    assert resp.status_code == 200, resp.text

    info = client.get("/api/notebooks/source_url_overwrite.ipynb/info").json()
    assert info["source_url"] == f"{base_url}/source_url_overwrite_b.ipynb"


def test_delete_notebook_removes_its_source_url_sidecar_file(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/source_url_delete_test.ipynb"},
    )
    assert _source_url_sidecar_path("source_url_delete_test.ipynb").is_file()

    resp = client.delete("/api/notebooks/source_url_delete_test.ipynb")
    assert resp.status_code == 200

    assert not _source_url_sidecar_path("source_url_delete_test.ipynb").exists()


def test_notebooks_info_batch_reports_source_url(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/source_url_info_batch.ipynb"},
    )

    resp = client.post(
        "/api/notebooks/info-batch",
        json={"filenames": ["source_url_info_batch.ipynb"]},
    )

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["source_url"] == f"{base_url}/source_url_info_batch.ipynb"


def test_list_notebooks_reports_source_url_for_an_imported_notebook(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/source_url_list_test.ipynb"},
    )

    notebooks = client.get("/api/notebooks").json()["notebooks"]
    entry = next(n for n in notebooks if n["filename"] == "source_url_list_test.ipynb")
    assert entry["source_url"] == f"{base_url}/source_url_list_test.ipynb"


def test_import_url_explicit_filename_overrides_the_derived_one(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/nb.ipynb", "filename": "custom.ipynb"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["filename"] == "custom.ipynb"
    assert (Path(UPLOAD_DIR) / "custom.ipynb").exists()
    assert not (Path(UPLOAD_DIR) / "nb.ipynb").exists()


def test_import_url_rejects_a_non_ipynb_filename(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/notes.txt"},
    )

    assert resp.status_code == 400
    assert "filename" in resp.json()["detail"]


def test_import_url_requires_a_url_field():

    resp = client.post("/api/notebooks/import-url", json={})

    assert resp.status_code == 400
    assert "url" in resp.json()["detail"]


def test_import_url_rejects_a_non_http_scheme():

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": "ftp://example.com/nb.ipynb"},
    )

    assert resp.status_code == 400
    assert "scheme" in resp.json()["detail"]
    assert not (Path(UPLOAD_DIR) / "nb.ipynb").exists()


def test_import_url_rejects_a_private_address():
    """Run against the real _reject_unsafe_import_url_host (no bypass
    fixture) -- 127.0.0.1 is a numeric literal, so this resolves with no
    real DNS lookup and is exactly the kind of address (loopback) POST
    /api/notebooks/import-url must never let a caller reach through this
    server.
    """

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": "http://127.0.0.1:1/nb.ipynb"},
    )

    assert resp.status_code == 400
    assert "non-public" in resp.json()["detail"]


def test_import_url_rechecks_the_ssrf_guard_on_every_redirect_hop(
    notebook_url_server, monkeypatch
):
    """A URL that itself resolves safely can still redirect to one that
    doesn't -- _reject_unsafe_import_url_host must be re-run against the
    redirect's own target, not only the original URL, or a public-looking
    URL could be used to reach an address only this server's own network
    can.

    Swaps in a fake guard (rather than the real one) so this test can
    assert re-invocation happened for *both* hops without depending on
    the real function's own private-address classification, which is
    already covered directly by test_import_url_rejects_a_private_address
    above.
    """
    from backend.routes import upload as upload_module

    base_url, handler = notebook_url_server
    handler.redirect_to = "http://127.0.0.1:1/evil.ipynb"

    checked_urls = []

    def fake_guard(url):
        checked_urls.append(url)
        if "evil" in url:
            raise HTTPException(status_code=400, detail="blocked by test guard")

    monkeypatch.setattr(upload_module, "_reject_unsafe_import_url_host", fake_guard)

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/redirect", "filename": "nb.ipynb"},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "blocked by test guard"
    assert checked_urls == [f"{base_url}/redirect", "http://127.0.0.1:1/evil.ipynb"]
    assert not (Path(UPLOAD_DIR) / "evil.ipynb").exists()


def test_import_url_forwards_custom_headers_to_the_fetch(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    """Before this feature, there was no way to authenticate the fetch
    POST /api/notebooks/import-url performs at all -- a private GitHub
    raw URL, or an internal artifact server behind its own API key, had
    no way to be imported short of the two-step "download it yourself,
    then POST /api/upload it" round trip this endpoint's own docstring
    already says it exists to avoid.
    """

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={
            "url": f"{base_url}/nb.ipynb",
            "filename": "custom_headers.ipynb",
            "headers": {"Authorization": "Bearer secret-token"},
        },
    )

    assert resp.status_code == 200, resp.text
    [received] = handler.received_headers_log
    assert received.get("Authorization") == "Bearer secret-token"


def test_import_url_forwards_headers_across_a_same_origin_redirect(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")
    handler.redirect_to = f"{base_url}/final.ipynb"

    resp = client.post(
        "/api/notebooks/import-url",
        json={
            "url": f"{base_url}/redirect",
            "filename": "same_origin_headers.ipynb",
            "headers": {"Authorization": "Bearer secret-token"},
        },
    )

    assert resp.status_code == 200, resp.text
    redirect_hop, final_hop = handler.received_headers_log
    assert redirect_hop.get("Authorization") == "Bearer secret-token"
    assert final_hop.get("Authorization") == "Bearer secret-token"


def test_import_url_drops_custom_headers_after_a_cross_origin_redirect(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    """A redirect can land anywhere -- a host under an attacker's control,
    or simply somewhere the caller never intended -- so a caller-supplied
    credential meant for the *original* host must never follow it there,
    the same one-way "never forward credentials across an origin change"
    discipline requests/browsers already apply to a redirected
    Authorization header.

    The second server below is a distinct process-level listener on its
    own port, but deliberately shares _NotebookUrlHandler's own class
    (not a second, independent handler) specifically so its own request
    lands in the exact same received_headers_log the first server's own
    requests do -- letting this test tell hop 1's headers (sent to the
    original, trusted host) apart from hop 2's (sent to a different host
    entirely) from that one shared log, in request order.
    """

    base_url, handler = notebook_url_server
    handler.content = b"unused -- the redirect target serves the real content"

    second_server = http.server.HTTPServer(("127.0.0.1", 0), _NotebookUrlHandler)
    second_port = second_server.server_address[1]
    second_thread = threading.Thread(target=second_server.serve_forever, daemon=True)
    second_thread.start()

    try:
        _NotebookUrlHandler.content = _notebook_bytes("def f(): return 1\n")
        handler.redirect_to = f"http://127.0.0.1:{second_port}/nb.ipynb"

        resp = client.post(
            "/api/notebooks/import-url",
            json={
                "url": f"{base_url}/redirect",
                "filename": "cross_origin_headers.ipynb",
                "headers": {"Authorization": "Bearer secret-token"},
            },
        )

        assert resp.status_code == 200, resp.text
        redirect_hop, final_hop = handler.received_headers_log
        assert redirect_hop.get("Authorization") == "Bearer secret-token"
        assert "Authorization" not in final_hop
    finally:
        second_server.shutdown()
        second_thread.join(timeout=5)


def test_import_url_rejects_a_non_object_headers_value(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/nb.ipynb", "headers": "not-an-object"},
    )

    assert resp.status_code == 400
    assert "headers must be an object" in resp.json()["detail"]


def test_import_url_rejects_a_headers_value_with_a_non_string_entry(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/nb.ipynb", "headers": {"X-Count": 5}},
    )

    assert resp.status_code == 400
    assert "headers must be an object" in resp.json()["detail"]


def test_import_url_rejects_too_many_headers(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    too_many_headers = {f"X-Header-{i}": "value" for i in range(21)}

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/nb.ipynb", "headers": too_many_headers},
    )

    assert resp.status_code == 400
    assert "must not contain more than" in resp.json()["detail"]


def test_import_url_rejects_an_oversized_header_value(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={
            "url": f"{base_url}/nb.ipynb",
            "headers": {"X-Big": "a" * 4097},
        },
    )

    assert resp.status_code == 400
    assert "must not exceed" in resp.json()["detail"]


def test_import_url_without_headers_field_behaves_exactly_as_before(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/nb.ipynb", "filename": "no_headers.ipynb"},
    )

    assert resp.status_code == 200, resp.text
    [received] = handler.received_headers_log
    assert "Authorization" not in received


def test_import_url_rejects_content_over_the_configured_max_size(
    notebook_url_server, _bypass_import_url_ssrf_guard, monkeypatch
):
    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_UPLOAD_BYTES", 10)

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")
    assert len(handler.content) > 10

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/nb.ipynb"},
    )

    assert resp.status_code == 413
    assert not (Path(UPLOAD_DIR) / "nb.ipynb").exists()


def test_import_url_rejects_a_sha256_mismatch(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={
            "url": f"{base_url}/nb.ipynb",
            "expected_sha256": "0" * 64,
        },
    )

    assert resp.status_code == 400
    assert "expected_sha256" in resp.json()["detail"]
    assert not (Path(UPLOAD_DIR) / "nb.ipynb").exists()


def test_import_url_dry_run_does_not_write_anything(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/nb.ipynb", "dry_run": True},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dry_run"] is True
    assert not (Path(UPLOAD_DIR) / "nb.ipynb").exists()


def test_import_url_applies_tags_and_description_on_success(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={
            "url": f"{base_url}/tagged.ipynb",
            "tags": "a,b",
            "description": "fetched from a url",
        },
    )

    assert resp.status_code == 200, resp.text

    info = client.get("/api/notebooks/tagged.ipynb/info").json()
    assert sorted(info["tags"]) == ["a", "b"]
    assert info["description"] == "fetched from a url"


def test_import_url_does_not_apply_tags_under_dry_run(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    resp = client.post(
        "/api/notebooks/import-url",
        json={
            "url": f"{base_url}/dry_tagged.ipynb",
            "tags": "a,b",
            "dry_run": True,
        },
    )

    assert resp.status_code == 200, resp.text
    assert not (Path(UPLOAD_DIR) / "dry_tagged.ipynb").exists()


def test_import_url_requires_overwrite_to_replace_an_existing_notebook(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("collide.ipynb", io.BytesIO(handler.content), "application/json")},
    )

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/collide.ipynb"},
    )

    assert resp.status_code == 409

    resp = client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/collide.ipynb", "overwrite": True},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["overwritten"] is True


def test_upload_lock_for_returns_the_same_lock_for_the_same_filename():
    """_upload_lock_for must hand back the *same* Lock instance for the
    same filename across separate calls (separate requests, in practice)
    -- otherwise two concurrent uploads of the same filename would each
    acquire their own independent Lock and never actually exclude each
    other at all, silently defeating the whole point of it.
    """

    from backend.routes.upload import _upload_lock_for

    assert (
        _upload_lock_for("same_lock_test.ipynb")
        is _upload_lock_for("same_lock_test.ipynb")
    )


def test_upload_lock_for_returns_different_locks_for_different_filenames():
    """Scoped per filename, not a single global lock -- two concurrent
    uploads of two *different* notebooks (the overwhelmingly common case)
    must stay fully concurrent; only genuinely colliding same-name
    uploads should ever need to serialize.
    """

    from backend.routes.upload import _upload_lock_for

    assert (
        _upload_lock_for("different_lock_test_a.ipynb")
        is not _upload_lock_for("different_lock_test_b.ipynb")
    )


def test_upload_lock_for_enforces_mutual_exclusion_for_the_same_filename():
    """The actual property upload_notebook depends on to close the race
    this fix exists for: two coroutines contending for the same
    filename's lock can never both be inside the critical section at
    once, and the second only ever enters after the first has fully
    exited (not merely "started exiting") -- reproduced deterministically
    via two coroutines racing the identical lock, driven by
    asyncio.gather on a single event loop, rather than trying to force
    this specific interleaving through two full, independent HTTP
    requests (whose exact timing an in-memory ASGI transport doesn't
    reliably reproduce -- confirmed while writing this test: even a
    deliberately delayed UploadFile.read() didn't reliably interleave
    two concurrent POST /api/upload calls to the same filename through
    TestClient/httpx's in-memory transport, unlike a real network
    connection's genuine I/O wait). Testing the lock's own guarantee
    directly is both deterministic and exactly what upload_notebook
    actually relies on.
    """

    import asyncio

    from backend.routes.upload import _upload_lock_for

    async def scenario():

        filename = "mutual_exclusion_test.ipynb"

        currently_inside = 0
        max_concurrent = 0
        events = []

        async def critical_section(tag):
            nonlocal currently_inside, max_concurrent

            async with _upload_lock_for(filename):

                currently_inside += 1
                max_concurrent = max(max_concurrent, currently_inside)
                events.append((tag, "enter"))

                # Stands in for upload_notebook's own streaming/validation
                # work while holding the lock -- long enough that, if the
                # lock weren't actually excluding the other coroutine, it
                # would have every opportunity to interleave its own
                # "enter" in between.
                await asyncio.sleep(0.05)

                events.append((tag, "exit"))
                currently_inside -= 1

        await asyncio.gather(critical_section("A"), critical_section("B"))

        return max_concurrent, events

    max_concurrent, events = asyncio.run(scenario())

    assert max_concurrent == 1

    # Whichever coroutine goes first must fully enter *and exit* before
    # the other ever enters -- not just start before the other starts.
    assert events in (
        [("A", "enter"), ("A", "exit"), ("B", "enter"), ("B", "exit")],
        [("B", "enter"), ("B", "exit"), ("A", "enter"), ("A", "exit")],
    )


def test_upload_lock_for_does_not_serialize_different_filenames():
    """The flip side of the mutual-exclusion test above: two coroutines
    holding *different* filenames' locks must be able to run fully
    concurrently, with neither waiting on the other at all.
    """

    import asyncio

    from backend.routes.upload import _upload_lock_for

    async def scenario():

        both_entered = asyncio.Event()
        entered_count = 0
        events = []

        async def critical_section(tag, filename):
            nonlocal entered_count

            async with _upload_lock_for(filename):

                events.append((tag, "enter"))
                entered_count += 1

                if entered_count == 2:
                    both_entered.set()

                # If these two were sharing a lock, this wait would never
                # be satisfied within the timeout below -- the second
                # coroutine couldn't have entered while the first still
                # holds a shared lock.
                await asyncio.wait_for(both_entered.wait(), timeout=1)

                events.append((tag, "exit"))

        await asyncio.gather(
            critical_section("A", "concurrent_lock_test_a.ipynb"),
            critical_section("B", "concurrent_lock_test_b.ipynb"),
        )

        return events

    events = asyncio.run(scenario())

    # Both must have entered before either exited -- true concurrency,
    # not one waiting for the other to finish first.
    assert events[0][1] == "enter"
    assert events[1][1] == "enter"


def test_list_notebooks_includes_uploaded_files():
    """/api/upload was previously a one-way door: nothing in the API let
    a caller see what had already been uploaded, or remove it again.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("list_test.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    list_resp = client.get("/api/notebooks")
    assert list_resp.status_code == 200

    notebooks = list_resp.json()["notebooks"]
    entry = next(nb for nb in notebooks if nb["filename"] == "list_test.ipynb")

    assert entry["size_bytes"] == len(content)
    assert "modified_at" in entry
    assert entry["currently_compiled"] is False


def test_list_notebooks_marks_the_currently_compiled_notebook():
    """Nothing previously recorded which uploaded notebook (if any)
    produced whatever's currently in GENERATED_DIR -- a dashboard
    frontend had to track that itself client-side, which is fragile
    (lost on refresh) and wrong the moment a second compile happens
    without it finding out.

    /api/compile always targets the real "generated" directory (like
    /api/export-openapi, /api/export-sdk, /api/download and /api/deploy
    already do -- see their own docstrings), so this exercises that same
    shared directory rather than an isolated one, matching how the other
    compile-flow tests in this file already operate.
    """

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("currently_compiled_a.ipynb", content_a),
        ("currently_compiled_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "currently_compiled_a.ipynb"}
    )
    assert compile_resp.status_code == 200

    after = {
        nb["filename"]: nb["currently_compiled"]
        for nb in client.get("/api/notebooks").json()["notebooks"]
    }
    assert after["currently_compiled_a.ipynb"] is True
    assert after["currently_compiled_b.ipynb"] is False

    # Recompiling a different notebook must flip which one is flagged.
    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "currently_compiled_b.ipynb"}
    )
    assert compile_resp.status_code == 200

    final = {
        nb["filename"]: nb["currently_compiled"]
        for nb in client.get("/api/notebooks").json()["notebooks"]
    }
    assert final["currently_compiled_a.ipynb"] is False
    assert final["currently_compiled_b.ipynb"] is True


def test_list_notebooks_currently_compiled_is_false_when_metadata_is_missing(
    monkeypatch
):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_no_metadata_test_dir"
    )

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    resp = client.post(
        "/api/upload",
        files={"file": ("no_metadata_test.ipynb", io.BytesIO(content), "application/json")},
    )
    assert resp.status_code == 200

    notebooks = client.get("/api/notebooks").json()["notebooks"]
    entry = next(nb for nb in notebooks if nb["filename"] == "no_metadata_test.ipynb")

    assert entry["currently_compiled"] is False
    # Not the currently-compiled notebook at all -- staleness is only
    # meaningful (and only reported) for the one that is.
    assert "notebook_changed_since_compile" not in entry


def test_list_notebooks_reports_the_currently_compiled_notebook_as_unchanged_right_after_compile():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("freshly_compiled.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "freshly_compiled.ipynb"}
    )
    assert compile_resp.status_code == 200

    notebooks = client.get("/api/notebooks").json()["notebooks"]
    entry = next(nb for nb in notebooks if nb["filename"] == "freshly_compiled.ipynb")

    assert entry["currently_compiled"] is True
    assert entry["notebook_changed_since_compile"] is False


def test_list_notebooks_flags_a_notebook_changed_since_its_last_compile():
    """The gap this closes: /api/notebooks could already say "this is the
    notebook that produced generated/" (see
    test_list_notebooks_marks_the_currently_compiled_notebook), but had
    no way to tell a caller that notebook had since been edited and
    re-uploaded -- e.g. via /api/upload?overwrite=true -- *after* that
    compile, silently leaving the currently-served app stale relative to
    what a caller might reasonably assume it still matches exactly.
    """

    original_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("edited_after_compile.ipynb", io.BytesIO(original_content), "application/json")},
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "edited_after_compile.ipynb"}
    )
    assert compile_resp.status_code == 200

    changed_content = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    overwrite_resp = client.post(
        "/api/upload?overwrite=true",
        files={"file": ("edited_after_compile.ipynb", io.BytesIO(changed_content), "application/json")},
    )
    assert overwrite_resp.status_code == 200

    notebooks = client.get("/api/notebooks").json()["notebooks"]
    entry = next(nb for nb in notebooks if nb["filename"] == "edited_after_compile.ipynb")

    # Still the notebook that produced the current generated/ output --
    # just no longer an exact match for what's actually on disk now.
    assert entry["currently_compiled"] is True
    assert entry["notebook_changed_since_compile"] is True


def test_list_notebooks_reports_the_compiled_at_timestamp():
    """.compile_metadata.json already records when the compile that
    produced the current generated/ output happened (see
    write_compile_metadata in backend/compiler.py), and this endpoint
    already reads that same file to resolve currently_compiled and
    notebook_changed_since_compile -- but previously discarded
    "compiled_at" rather than returning it. Without it, a caller could
    tell *that* the currently running app might be stale (via
    notebook_changed_since_compile) but not *how* stale -- e.g. to show
    "last compiled 3 minutes ago" -- without a separate, redundant read
    of the same file.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": ("compiled_at_test.ipynb", io.BytesIO(content), "application/json")
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "compiled_at_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    with open(
        Path("generated") / COMPILE_METADATA_FILENAME, "r", encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    notebooks = client.get("/api/notebooks").json()["notebooks"]
    entry = next(nb for nb in notebooks if nb["filename"] == "compiled_at_test.ipynb")

    assert entry["compiled_at"] == metadata["compiled_at"]


def test_list_notebooks_omits_compiled_at_for_a_notebook_that_is_not_currently_compiled():

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("compiled_at_a.ipynb", content_a),
        ("compiled_at_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "compiled_at_a.ipynb"}
    )
    assert compile_resp.status_code == 200

    notebooks = client.get("/api/notebooks").json()["notebooks"]
    entry_b = next(nb for nb in notebooks if nb["filename"] == "compiled_at_b.ipynb")

    assert entry_b["currently_compiled"] is False
    assert "compiled_at" not in entry_b


def test_list_notebooks_search_filters_by_a_case_insensitive_filename_substring():

    for filename in ("search_apple.ipynb", "search_banana.ipynb"):
        resp = client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                )
            },
        )
        assert resp.status_code == 200

    notebooks = client.get("/api/notebooks?search=APPLE").json()["notebooks"]
    filenames = {nb["filename"] for nb in notebooks}

    assert filenames == {"search_apple.ipynb"}


def test_list_notebooks_search_matching_nothing_returns_an_empty_list():

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "search_only_this.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert resp.status_code == 200

    notebooks = client.get(
        "/api/notebooks?search=this_substring_matches_nothing_at_all"
    ).json()["notebooks"]

    assert notebooks == []


def test_list_notebooks_sorts_by_name_ascending_by_default():
    """Preserves the previous, and still default, behavior -- a plain GET
    /api/notebooks with no query string -- so an existing caller relying
    on alphabetical-by-filename order sees no change.
    """

    for filename in ("sort_default_b.ipynb", "sort_default_a.ipynb"):
        resp = client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                )
            },
        )
        assert resp.status_code == 200

    notebooks = client.get("/api/notebooks?search=sort_default_").json()["notebooks"]
    filenames = [nb["filename"] for nb in notebooks]

    assert filenames == ["sort_default_a.ipynb", "sort_default_b.ipynb"]


def test_list_notebooks_sorts_by_size_ascending_and_descending():

    for filename, function_source in (
        ("sort_size_small.ipynb", "def f() -> int:\n    return 1\n"),
        (
            "sort_size_large.ipynb",
            "def f() -> int:\n    return 1  # padding to make this cell bigger\n",
        ),
    ):
        resp = client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes(function_source)),
                    "application/json",
                )
            },
        )
        assert resp.status_code == 200

    asc = client.get("/api/notebooks?search=sort_size_&sort=size&order=asc").json()["notebooks"]
    assert [nb["filename"] for nb in asc] == ["sort_size_small.ipynb", "sort_size_large.ipynb"]

    desc = client.get("/api/notebooks?search=sort_size_&sort=size&order=desc").json()["notebooks"]
    assert [nb["filename"] for nb in desc] == ["sort_size_large.ipynb", "sort_size_small.ipynb"]


def test_list_notebooks_sorts_by_modified_descending_shows_the_newest_first():

    resp_older = client.post(
        "/api/upload",
        files={
            "file": (
                "sort_modified_older.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert resp_older.status_code == 200

    older_path = Path(UPLOAD_DIR) / "sort_modified_older.ipynb"
    older_stat = older_path.stat()
    os.utime(older_path, (older_stat.st_atime, older_stat.st_mtime - 3600))

    resp_newer = client.post(
        "/api/upload",
        files={
            "file": (
                "sort_modified_newer.ipynb",
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    assert resp_newer.status_code == 200

    notebooks = client.get(
        "/api/notebooks?search=sort_modified_&sort=modified&order=desc"
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == [
        "sort_modified_newer.ipynb",
        "sort_modified_older.ipynb",
    ]


def test_list_notebooks_rejects_an_invalid_sort_value():

    resp = client.get("/api/notebooks?sort=not_a_real_field")

    assert resp.status_code == 400


def test_list_notebooks_rejects_an_invalid_order_value():

    resp = client.get("/api/notebooks?order=sideways")

    assert resp.status_code == 400


def test_list_notebooks_paginates_with_limit_and_offset():

    for filename in (
        "page_a.ipynb",
        "page_b.ipynb",
        "page_c.ipynb",
    ):
        resp = client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                )
            },
        )
        assert resp.status_code == 200

    first_page = client.get("/api/notebooks?search=page_&limit=2&offset=0").json()
    assert [nb["filename"] for nb in first_page["notebooks"]] == [
        "page_a.ipynb",
        "page_b.ipynb",
    ]
    assert first_page["total_count"] == 3
    assert first_page["limit"] == 2
    assert first_page["offset"] == 0

    second_page = client.get("/api/notebooks?search=page_&limit=2&offset=2").json()
    assert [nb["filename"] for nb in second_page["notebooks"]] == ["page_c.ipynb"]
    assert second_page["total_count"] == 3
    assert second_page["limit"] == 2
    assert second_page["offset"] == 2


def test_list_notebooks_without_limit_returns_every_matching_notebook():
    """Preserves the previous, still-default behavior -- a plain GET
    /api/notebooks with no "limit" returns everything matching "search",
    not just some implicit page size.
    """

    for filename in ("nolimit_a.ipynb", "nolimit_b.ipynb"):
        resp = client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                )
            },
        )
        assert resp.status_code == 200

    body = client.get("/api/notebooks?search=nolimit_").json()

    assert [nb["filename"] for nb in body["notebooks"]] == [
        "nolimit_a.ipynb",
        "nolimit_b.ipynb",
    ]
    assert body["total_count"] == 2
    assert body["limit"] is None
    assert body["offset"] == 0


def test_list_notebooks_offset_past_the_end_returns_an_empty_list():

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "offset_only_one.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert resp.status_code == 200

    body = client.get(
        "/api/notebooks?search=offset_only_one&offset=5"
    ).json()

    assert body["notebooks"] == []
    assert body["total_count"] == 1


def test_list_notebooks_rejects_a_negative_offset():

    resp = client.get("/api/notebooks?offset=-1")

    assert resp.status_code == 400


def test_list_notebooks_rejects_a_non_positive_limit():

    resp = client.get("/api/notebooks?limit=0")

    assert resp.status_code == 400

    resp = client.get("/api/notebooks?limit=-5")

    assert resp.status_code == 400


def test_list_notebooks_rejects_an_unknown_format():

    resp = client.get("/api/notebooks?format=xml")

    assert resp.status_code == 400


def test_list_notebooks_csv_format_returns_a_csv_response():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("list_csv_a.ipynb", io.BytesIO(content), "application/json")},
    )
    client.put(
        "/api/notebooks/list_csv_a.ipynb/tags", json={"tags": ["prod", "beta"]}
    )
    client.put(
        "/api/notebooks/list_csv_a.ipynb/description",
        json={"description": "a sample notebook"},
    )

    resp = client.get("/api/notebooks", params={"format": "csv"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="notebooks.csv"' in resp.headers["content-disposition"]

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == (
        "filename,size_bytes,modified_at,currently_compiled,tags,"
        "description,notebook_changed_since_compile,compiled_at,"
        "compiled_version_id"
    )
    assert len(rows) == 2

    fields = rows[1].split(",")
    assert fields[0] == "list_csv_a.ipynb"
    assert fields[1] == str(len(content))
    assert fields[3] == "False"
    assert "beta;prod" in rows[1]
    assert "a sample notebook" in rows[1]


def test_list_notebooks_csv_format_leaves_staleness_columns_blank_for_a_non_compiled_notebook():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    client.post(
        "/api/upload",
        files={"file": ("list_csv_no_compile.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get("/api/notebooks", params={"format": "csv"})

    rows = resp.text.strip().split("\r\n")
    assert rows[1] == "list_csv_no_compile.ipynb," + rows[1].split(",", 1)[1]
    assert rows[1].endswith(",,,")


def test_list_notebooks_csv_format_composes_with_tag_and_limit():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("list_csv_tagged.ipynb", io.BytesIO(content), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("list_csv_untagged.ipynb", io.BytesIO(content), "application/json")},
    )
    client.put(
        "/api/notebooks/list_csv_tagged.ipynb/tags", json={"tags": ["prod"]}
    )

    resp = client.get(
        "/api/notebooks",
        params={"format": "csv", "tag": "prod", "limit": 1},
    )

    assert resp.status_code == 200
    rows = resp.text.strip().split("\r\n")
    assert len(rows) == 2
    assert rows[1].startswith("list_csv_tagged.ipynb,")


def test_delete_notebook_removes_an_uploaded_file():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("delete_test.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200
    assert os.path.exists(os.path.join(UPLOAD_DIR, "delete_test.ipynb"))

    delete_resp = client.delete("/api/notebooks/delete_test.ipynb")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["filename"] == "delete_test.ipynb"
    # Never compiled -- deleting it can't have orphaned anything.
    assert delete_resp.json()["was_currently_compiled"] is False

    assert not os.path.exists(os.path.join(UPLOAD_DIR, "delete_test.ipynb"))

    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "delete_test.ipynb" not in filenames


def test_delete_notebook_flags_was_currently_compiled_true_for_the_compiled_notebook():
    """Deleting the notebook that produced whatever's currently running in
    GENERATED_DIR doesn't touch the compiled app itself -- it keeps
    running exactly as before -- but silently orphans it: there's no
    longer an uploaded notebook a caller could re-inspect, diff, or
    recompile from to confirm what's currently being served. Before this,
    a caller had no way to know that had just happened short of a
    separate GET /api/notebooks call beforehand to check.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "delete_currently_compiled_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile",
        json={"notebook_path": "delete_currently_compiled_test.ipynb"},
    )
    assert compile_resp.status_code == 200

    delete_resp = client.delete(
        "/api/notebooks/delete_currently_compiled_test.ipynb"
    )
    assert delete_resp.status_code == 200
    assert delete_resp.json()["was_currently_compiled"] is True


def test_delete_notebook_flags_was_currently_compiled_false_for_an_unrelated_notebook():

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("delete_unrelated_a.ipynb", content_a),
        ("delete_unrelated_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "delete_unrelated_a.ipynb"}
    )
    assert compile_resp.status_code == 200

    delete_resp = client.delete("/api/notebooks/delete_unrelated_b.ipynb")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["was_currently_compiled"] is False


def test_delete_notebook_returns_404_for_missing_file():

    resp = client.delete("/api/notebooks/does_not_exist_at_all.ipynb")

    assert resp.status_code == 404


def test_delete_notebook_dry_run_reports_success_without_deleting():

    _upload_sample_notebook("delete_dry_run.ipynb")

    resp = client.delete(
        "/api/notebooks/delete_dry_run.ipynb", params={"dry_run": "true"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["filename"] == "delete_dry_run.ipynb"
    assert body["was_currently_compiled"] is False

    # Nothing was actually deleted.
    assert (Path(UPLOAD_DIR) / "delete_dry_run.ipynb").is_file()

    os.remove(Path(UPLOAD_DIR) / "delete_dry_run.ipynb")


def test_delete_notebook_dry_run_still_returns_404_for_missing_file():

    resp = client.delete(
        "/api/notebooks/delete_dry_run_missing.ipynb", params={"dry_run": "true"}
    )

    assert resp.status_code == 404


def test_get_notebook_returns_the_uploaded_content():
    """GET /api/notebooks lists what's been uploaded and DELETE removes
    it, but there was previously no way to retrieve a specific notebook's
    actual content again -- only re-upload a fresh copy from scratch.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("get_test.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    get_resp = client.get("/api/notebooks/get_test.ipynb")

    assert get_resp.status_code == 200
    assert get_resp.headers["content-type"] == "application/x-ipynb+json"
    assert json.loads(get_resp.content) == json.loads(content)


def test_get_notebook_reports_the_content_sha256_header():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("get_sha256.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get("/api/notebooks/get_sha256.ipynb")

    assert resp.status_code == 200
    assert resp.headers["x-content-sha256"] == hashlib.sha256(resp.content).hexdigest()


def test_get_notebook_content_sha256_header_matches_the_upload_responses_own_sha256():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("get_sha256_matches_upload.ipynb", io.BytesIO(content), "application/json")},
    )

    get_resp = client.get("/api/notebooks/get_sha256_matches_upload.ipynb")

    assert get_resp.headers["x-content-sha256"] == upload_resp.json()["sha256"]


def test_get_notebook_content_sha256_header_changes_after_overwrite():

    filename = "get_sha256_overwrite.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    first_sha256 = client.get(f"/api/notebooks/{filename}").headers["x-content-sha256"]

    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    second_sha256 = client.get(f"/api/notebooks/{filename}").headers["x-content-sha256"]

    assert first_sha256 != second_sha256


def test_get_notebook_reports_a_quoted_etag_matching_the_content_sha256():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("get_etag.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get("/api/notebooks/get_etag.ipynb")

    assert resp.status_code == 200
    assert resp.headers["etag"] == f'"{resp.headers["x-content-sha256"]}"'


def test_get_notebook_returns_304_when_if_none_match_matches_the_current_etag():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("get_conditional.ipynb", io.BytesIO(content), "application/json")},
    )

    first = client.get("/api/notebooks/get_conditional.ipynb")
    etag = first.headers["etag"]

    second = client.get(
        "/api/notebooks/get_conditional.ipynb",
        headers={"If-None-Match": etag},
    )

    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["etag"] == etag
    assert second.headers["x-content-sha256"] == first.headers["x-content-sha256"]


def test_get_notebook_sends_cache_control_no_cache_on_both_200_and_304():
    """Confirmed exploitable before this fix: this endpoint sent an ETag
    and honored If-None-Match, but never sent Cache-Control -- without
    it, a standard HTTP cache (browser, CDN, caching proxy) has no
    reliable signal that this response is even cacheable at all, so it
    has nothing telling it to store the response and revalidate via
    If-None-Match on a later request, the entire point of an ETag.
    "no-cache" (not "no-store") means "cache this, but always revalidate
    first" -- correct here since PATCH/overwrite can change this content
    at any time, so a plain max-age freshness window would risk serving
    stale content.
    """

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("get_cache_control.ipynb", io.BytesIO(content), "application/json")},
    )

    first = client.get("/api/notebooks/get_cache_control.ipynb")
    assert first.headers["cache-control"] == "no-cache"

    second = client.get(
        "/api/notebooks/get_cache_control.ipynb",
        headers={"If-None-Match": first.headers["etag"]},
    )
    assert second.status_code == 304
    assert second.headers["cache-control"] == "no-cache"


def test_get_notebook_returns_304_for_a_wildcard_if_none_match():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("get_conditional_wildcard.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get(
        "/api/notebooks/get_conditional_wildcard.ipynb",
        headers={"If-None-Match": "*"},
    )

    assert resp.status_code == 304


def test_get_notebook_returns_304_for_an_unquoted_or_weak_matching_etag():
    """A caller's own cached copy of a prior ETag may be echoed back
    unquoted, or prefixed "W/" (a weak-validator marker) -- neither
    changes whether the underlying content actually matches, so neither
    should make an otherwise-matching request miss.
    """

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("get_conditional_weak.ipynb", io.BytesIO(content), "application/json")},
    )

    content_sha256 = client.get(
        "/api/notebooks/get_conditional_weak.ipynb"
    ).headers["x-content-sha256"]

    unquoted_resp = client.get(
        "/api/notebooks/get_conditional_weak.ipynb",
        headers={"If-None-Match": content_sha256},
    )
    assert unquoted_resp.status_code == 304

    weak_resp = client.get(
        "/api/notebooks/get_conditional_weak.ipynb",
        headers={"If-None-Match": f'W/"{content_sha256}"'},
    )
    assert weak_resp.status_code == 304


def test_get_notebook_returns_200_when_if_none_match_is_stale():

    filename = "get_conditional_stale.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    stale_etag = client.get(f"/api/notebooks/{filename}").headers["etag"]

    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    resp = client.get(
        f"/api/notebooks/{filename}", headers={"If-None-Match": stale_etag}
    )

    assert resp.status_code == 200
    assert resp.headers["etag"] != stale_etag
    assert resp.content


def test_get_notebook_returns_304_for_one_matching_entry_in_a_multi_valued_if_none_match():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("get_conditional_multi.ipynb", io.BytesIO(content), "application/json")},
    )

    etag = client.get("/api/notebooks/get_conditional_multi.ipynb").headers["etag"]

    resp = client.get(
        "/api/notebooks/get_conditional_multi.ipynb",
        headers={"If-None-Match": f'"not-a-real-etag", {etag}'},
    )

    assert resp.status_code == 304


def test_get_notebook_returns_404_for_missing_file():

    resp = client.get("/api/notebooks/does_not_exist_at_all.ipynb")

    assert resp.status_code == 404


def test_get_notebook_rejects_absolute_filename():

    resp = client.get("/api/notebooks/%2Fetc%2Fpasswd")

    assert resp.status_code in (400, 404)
    assert "root:" not in resp.text


def test_get_notebook_rejects_a_filename_with_an_embedded_null_byte():
    """Confirmed exploitable before this fix: a null byte in the filename
    sailed past resolve_upload_path's absolute-path guard clause (a null
    byte isn't special to pathlib's own parsing), but the later
    .resolve() call raised a bare ValueError from the underlying
    os.path.realpath/lstat syscalls, an unhandled 500 instead of a clean
    400.
    """

    resp = client.get("/api/notebooks/nb%00.ipynb")

    assert resp.status_code == 400


def test_list_notebooks_checksums_flag_adds_a_sha256_field():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("list_checksums.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get("/api/notebooks", params={"checksums": "true"})

    assert resp.status_code == 200
    entry = next(
        nb for nb in resp.json()["notebooks"] if nb["filename"] == "list_checksums.ipynb"
    )
    assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_list_notebooks_without_checksums_omits_sha256():

    _upload_sample_notebook("list_no_checksums.ipynb")

    resp = client.get("/api/notebooks")

    assert resp.status_code == 200
    entry = next(
        nb for nb in resp.json()["notebooks"] if nb["filename"] == "list_no_checksums.ipynb"
    )
    assert "sha256" not in entry


def test_list_notebooks_csv_format_checksums_flag_adds_a_sha256_column():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("list_csv_checksums.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get("/api/notebooks", params={"format": "csv", "checksums": "true"})

    assert resp.status_code == 200
    rows = resp.text.strip().split("\r\n")
    assert rows[0] == (
        "filename,size_bytes,modified_at,currently_compiled,tags,"
        "description,notebook_changed_since_compile,compiled_at,"
        "compiled_version_id,sha256"
    )
    assert rows[1].endswith(hashlib.sha256(content).hexdigest())


def test_get_notebook_info_matches_the_notebooks_own_entry_in_the_list():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("info_test.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200
    client.put("/api/notebooks/info_test.ipynb/tags", json={"tags": ["scratch"]})

    info_resp = client.get("/api/notebooks/info_test.ipynb/info")
    assert info_resp.status_code == 200

    info_body = info_resp.json()
    assert info_body["status"] == "success"
    assert info_body["filename"] == "info_test.ipynb"
    assert info_body["tags"] == ["scratch"]
    assert info_body["currently_compiled"] is False
    assert "notebook_changed_since_compile" not in info_body
    assert "compiled_at" not in info_body

    # GET /api/notebooks/{filename}/info always includes "sha256" (a
    # single-notebook fetch can afford to hash unconditionally); GET
    # /api/notebooks' own bulk listing only does under "checksums=true" --
    # excluded here from the structural comparison below for that reason,
    # not because the two entries actually disagree on its value.
    list_entry = next(
        nb for nb in client.get("/api/notebooks").json()["notebooks"]
        if nb["filename"] == "info_test.ipynb"
    )
    assert {
        k: v for k, v in info_body.items() if k not in ("status", "sha256")
    } == list_entry

    list_entry_with_checksum = next(
        nb for nb in client.get(
            "/api/notebooks", params={"checksums": "true"}
        ).json()["notebooks"]
        if nb["filename"] == "info_test.ipynb"
    )
    assert info_body["sha256"] == list_entry_with_checksum["sha256"]


def test_get_notebook_info_reports_currently_compiled_fields():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("info_compiled_test.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "info_compiled_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    info_resp = client.get("/api/notebooks/info_compiled_test.ipynb/info")
    assert info_resp.status_code == 200

    info_body = info_resp.json()
    assert info_body["currently_compiled"] is True
    assert info_body["notebook_changed_since_compile"] is False
    assert info_body["compiled_at"] is not None


def test_get_notebook_info_returns_404_for_missing_file():

    resp = client.get("/api/notebooks/does_not_exist_at_all.ipynb/info")

    assert resp.status_code == 404


def test_get_notebook_info_always_includes_sha256():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("info_sha256_test.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get("/api/notebooks/info_sha256_test.ipynb/info")

    assert resp.status_code == 200
    assert resp.json()["sha256"] == hashlib.sha256(content).hexdigest()


def test_get_notebooks_info_batch_always_includes_sha256():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("info_batch_sha256_test.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/notebooks/info-batch",
        json={"filenames": ["info_batch_sha256_test.ipynb"]},
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["sha256"] == hashlib.sha256(content).hexdigest()


def test_get_notebook_info_rejects_absolute_filename():

    resp = client.get("/api/notebooks/%2Fetc%2Fpasswd/info")

    assert resp.status_code in (400, 404)
    assert "root:" not in resp.text


def test_get_notebooks_info_batch_matches_each_notebooks_own_info_entry():

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("info_batch_a.ipynb", content_a),
        ("info_batch_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    client.put("/api/notebooks/info_batch_a.ipynb/tags", json={"tags": ["scratch"]})

    resp = client.post(
        "/api/notebooks/info-batch",
        json={"filenames": ["info_batch_a.ipynb", "info_batch_b.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["info_batch_a.ipynb"]["status"] == "success"
    assert results_by_filename["info_batch_a.ipynb"]["tags"] == ["scratch"]

    single_info = client.get("/api/notebooks/info_batch_a.ipynb/info").json()
    assert results_by_filename["info_batch_a.ipynb"] == single_info


def test_get_notebooks_info_batch_reports_a_missing_filename_without_aborting_the_rest():

    _upload_sample_notebook("info_batch_partial.ipynb")

    resp = client.post(
        "/api/notebooks/info-batch",
        json={"filenames": ["info_batch_partial.ipynb", "does_not_exist.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["info_batch_partial.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]


def test_get_notebooks_info_batch_rejects_a_non_list_filenames_value():

    resp = client.post("/api/notebooks/info-batch", json={"filenames": "not-a-list"})

    assert resp.status_code == 400


def test_get_notebooks_info_batch_rejects_an_empty_filenames_list():

    resp = client.post("/api/notebooks/info-batch", json={"filenames": []})

    assert resp.status_code == 400


def test_get_notebooks_info_batch_rejects_more_filenames_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/notebooks/info-batch",
        json={"filenames": ["a.ipynb", "b.ipynb"]},
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_export_notebooks_zips_the_named_filenames():

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("export_a.ipynb", content_a),
        ("export_b.ipynb", content_b),
        ("export_c.ipynb", content_a),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    export_resp = client.get(
        "/api/notebooks/export", params={"filenames": "export_a.ipynb,export_b.ipynb"}
    )

    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:
        assert sorted(archive.namelist()) == ["export_a.ipynb", "export_b.ipynb"]
        assert json.loads(archive.read("export_a.ipynb")) == json.loads(content_a)
        assert json.loads(archive.read("export_b.ipynb")) == json.loads(content_b)


def test_export_notebooks_reports_a_bundle_sha256_over_the_exported_notebooks():
    """"X-Bundle-SHA256" summarizes just the exported notebooks' own
    content -- not the "tags/"/"description/" sidecar entries also
    bundled in alongside them -- the same _bundle_sha256 GET
    /api/download's own identical header already uses for a compiled
    bundle's own file set.
    """

    content_a = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    content_b = _notebook_bytes("def sub(a: int, b: int) -> int:\n    return a - b\n")

    for filename, content in (
        ("bundle_sha_a.ipynb", content_a),
        ("bundle_sha_b.ipynb", content_b),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    export_resp = client.get(
        "/api/notebooks/export",
        params={"filenames": "bundle_sha_a.ipynb,bundle_sha_b.ipynb"},
    )

    assert export_resp.status_code == 200
    bundle_sha256 = export_resp.headers["x-bundle-sha256"]
    assert bundle_sha256

    expected = _bundle_sha256([
        {"filename": "bundle_sha_a.ipynb", "sha256": hashlib.sha256(content_a).hexdigest()},
        {"filename": "bundle_sha_b.ipynb", "sha256": hashlib.sha256(content_b).hexdigest()},
    ])
    assert bundle_sha256 == expected


def test_export_notebooks_without_filenames_exports_every_uploaded_notebook():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("export_all_a.ipynb", "export_all_b.ipynb"):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    export_resp = client.get("/api/notebooks/export")

    assert export_resp.status_code == 200

    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:
        assert sorted(archive.namelist()) == ["export_all_a.ipynb", "export_all_b.ipynb"]


def test_export_notebooks_returns_404_naming_every_missing_filename():

    _upload_sample_notebook("export_missing_present.ipynb")

    resp = client.get(
        "/api/notebooks/export",
        params={"filenames": "export_missing_present.ipynb,does_not_exist.ipynb"},
    )

    assert resp.status_code == 404
    assert "does_not_exist.ipynb" in resp.json()["detail"]


def test_export_notebooks_returns_404_when_nothing_uploaded():

    client.delete("/api/notebooks?confirm=true")

    resp = client.get("/api/notebooks/export")

    assert resp.status_code == 404


def test_export_notebooks_rejects_a_blank_filenames_value():

    resp = client.get("/api/notebooks/export", params={"filenames": " , , "})

    assert resp.status_code == 400


def test_export_notebooks_by_tag_bundles_only_matching_notebooks():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("export_tag_a.ipynb", "export_tag_b.ipynb", "export_tag_c.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    client.put("/api/notebooks/export_tag_a.ipynb/tags", json={"tags": ["prod"]})
    client.put("/api/notebooks/export_tag_b.ipynb/tags", json={"tags": ["prod"]})
    client.put("/api/notebooks/export_tag_c.ipynb/tags", json={"tags": ["staging"]})

    resp = client.get("/api/notebooks/export", params={"tag": "prod"})

    assert resp.status_code == 200

    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        ipynb_entries = [n for n in archive.namelist() if n.endswith(".ipynb")]
        assert sorted(ipynb_entries) == ["export_tag_a.ipynb", "export_tag_b.ipynb"]


def test_export_notebooks_by_tag_returns_404_when_nothing_matches():

    client.delete("/api/notebooks?confirm=true")

    _upload_sample_notebook("export_tag_unmatched.ipynb")

    resp = client.get("/api/notebooks/export", params={"tag": "does-not-exist"})

    assert resp.status_code == 404


def test_export_notebooks_rejects_both_filenames_and_tag():

    _upload_sample_notebook("export_both_a.ipynb")

    resp = client.get(
        "/api/notebooks/export",
        params={"filenames": "export_both_a.ipynb", "tag": "prod"},
    )

    assert resp.status_code == 400


def test_export_notebooks_include_versions_bundles_each_notebooks_own_history():

    client.delete("/api/notebooks?confirm=true")

    original_a = _notebook_bytes("def f() -> int:\n    return 1\n")
    current_a = _notebook_bytes("def f() -> int:\n    return 2\n")
    current_b = _notebook_bytes("def g() -> int:\n    return 3\n")

    client.post(
        "/api/upload",
        files={"file": ("export_versions_a.ipynb", io.BytesIO(original_a), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": ("export_versions_a.ipynb", io.BytesIO(current_a), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("export_versions_b.ipynb", io.BytesIO(current_b), "application/json")},
    )

    version_id = client.get(
        "/api/notebooks/export_versions_a.ipynb/versions"
    ).json()["versions"][0]["version_id"]

    export_resp = client.get(
        "/api/notebooks/export", params={"include_versions": "true"}
    )

    assert export_resp.status_code == 200

    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:

        names = set(archive.namelist())
        assert names == {
            "export_versions_a.ipynb",
            "export_versions_b.ipynb",
            f"versions/export_versions_a.ipynb/{version_id}",
        }

        assert archive.read("export_versions_a.ipynb") == current_a
        assert archive.read("export_versions_b.ipynb") == current_b
        assert (
            archive.read(f"versions/export_versions_a.ipynb/{version_id}")
            == original_a
        )


def test_export_notebooks_without_include_versions_omits_version_history():

    client.delete("/api/notebooks?confirm=true")

    original = _notebook_bytes("def f() -> int:\n    return 1\n")
    current = _notebook_bytes("def f() -> int:\n    return 2\n")

    client.post(
        "/api/upload",
        files={"file": ("export_no_versions.ipynb", io.BytesIO(original), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": ("export_no_versions.ipynb", io.BytesIO(current), "application/json")},
    )

    export_resp = client.get("/api/notebooks/export")

    assert export_resp.status_code == 200

    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:
        assert archive.namelist() == ["export_no_versions.ipynb"]


def test_export_notebooks_include_versions_with_no_history_adds_no_entries():

    _upload_sample_notebook("export_versions_none.ipynb")

    export_resp = client.get(
        "/api/notebooks/export",
        params={"filenames": "export_versions_none.ipynb", "include_versions": "true"},
    )

    assert export_resp.status_code == 200

    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:
        assert archive.namelist() == ["export_versions_none.ipynb"]


def test_find_duplicate_notebooks_groups_byte_identical_uploads():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("dup_a1.ipynb", content_a),
        ("dup_a2.ipynb", content_a),
        ("dup_a3.ipynb", content_a),
        ("dup_b1.ipynb", content_b),
        ("dup_unique.ipynb", content_b + b" "),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.get("/api/notebooks/duplicates")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["group_count"] == 1
    assert body["duplicate_notebook_count"] == 3

    group = body["duplicate_groups"][0]
    assert group["filenames"] == ["dup_a1.ipynb", "dup_a2.ipynb", "dup_a3.ipynb"]
    assert group["size_bytes"] == len(content_a)
    assert len(group["sha256"]) == 64


def test_find_duplicate_notebooks_csv_format_returns_one_row_per_filename():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("dup_csv_a1.ipynb", content_a),
        ("dup_csv_a2.ipynb", content_a),
        ("dup_csv_b1.ipynb", content_b),
        ("dup_csv_unique.ipynb", content_b + b" "),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get("/api/notebooks/duplicates", params={"format": "csv"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="duplicates.csv"' in resp.headers["content-disposition"]

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == "sha256,filename,size_bytes"
    assert len(rows) == 3

    sha256 = rows[1].split(",")[0]
    assert rows[1] == f"{sha256},dup_csv_a1.ipynb,{len(content_a)}"
    assert rows[2] == f"{sha256},dup_csv_a2.ipynb,{len(content_a)}"


def test_find_duplicate_notebooks_csv_format_respects_limit_and_offset():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes("def a() -> int:\n    return 1\n")
    content_b = _notebook_bytes("def b() -> int:\n    return 2\n")

    for filename, content in (
        ("dup_csv_page_a1.ipynb", content_a),
        ("dup_csv_page_a2.ipynb", content_a),
        ("dup_csv_page_b1.ipynb", content_b),
        ("dup_csv_page_b2.ipynb", content_b),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    unpaginated_groups = client.get("/api/notebooks/duplicates").json()["duplicate_groups"]
    assert len(unpaginated_groups) == 2
    second_group = unpaginated_groups[1]

    resp = client.get(
        "/api/notebooks/duplicates",
        params={"format": "csv", "limit": 1, "offset": 1},
    )

    assert resp.status_code == 200
    rows = resp.text.strip().split("\r\n")
    assert rows[0] == "sha256,filename,size_bytes"
    # Groups are sorted by "sha256" (the same order the unpaginated JSON
    # call above already reports them in); offset=1 skips the first group
    # entirely, limit=1 keeps only the second -- so only that group's own
    # two filenames appear, never the first group's.
    assert len(rows) == 3
    for filename in second_group["filenames"]:
        assert any(row.startswith(f"{second_group['sha256']},{filename},") for row in rows[1:])


def test_find_duplicate_notebooks_rejects_an_invalid_format():

    resp = client.get("/api/notebooks/duplicates", params={"format": "xml"})

    assert resp.status_code == 400
    assert "format must be" in resp.json()["detail"]


def test_find_duplicate_notebooks_reports_no_groups_when_nothing_duplicated():

    client.delete("/api/notebooks?confirm=true")

    _upload_sample_notebook("dup_none_a.ipynb")

    resp = client.get("/api/notebooks/duplicates")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "duplicate_groups": [],
        "group_count": 0,
        "duplicate_notebook_count": 0,
        "limit": None,
        "offset": 0,
    }


def test_find_duplicate_notebooks_reports_multiple_independent_groups():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("dup_group1_a.ipynb", content_a),
        ("dup_group1_b.ipynb", content_a),
        ("dup_group2_a.ipynb", content_b),
        ("dup_group2_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.get("/api/notebooks/duplicates")

    assert resp.status_code == 200
    body = resp.json()
    assert body["group_count"] == 2
    assert body["duplicate_notebook_count"] == 4


def test_find_duplicate_notebooks_limit_paginates_the_groups_but_not_the_totals():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("dup_page_group1_a.ipynb", content_a),
        ("dup_page_group1_b.ipynb", content_a),
        ("dup_page_group2_a.ipynb", content_b),
        ("dup_page_group2_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.get("/api/notebooks/duplicates", params={"limit": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["duplicate_groups"]) == 1
    # The totals still reflect every matching group, not just this page.
    assert body["group_count"] == 2
    assert body["duplicate_notebook_count"] == 4
    assert body["limit"] == 1
    assert body["offset"] == 0

    # Groups are sorted by their own sha256, so paging is stable: the
    # second page's one group is whichever the first page's own group
    # wasn't.
    first_page_sha256 = body["duplicate_groups"][0]["sha256"]

    second_page = client.get(
        "/api/notebooks/duplicates", params={"limit": 1, "offset": 1}
    ).json()

    assert len(second_page["duplicate_groups"]) == 1
    assert second_page["duplicate_groups"][0]["sha256"] != first_page_sha256


def test_find_duplicate_notebooks_offset_past_the_end_yields_no_groups():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    for filename in ("dup_offset_a.ipynb", "dup_offset_b.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get("/api/notebooks/duplicates", params={"offset": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["duplicate_groups"] == []
    assert body["group_count"] == 1


def test_find_duplicate_notebooks_rejects_a_negative_offset():

    resp = client.get("/api/notebooks/duplicates", params={"offset": -1})

    assert resp.status_code == 400


def test_find_duplicate_notebooks_rejects_a_negative_limit():

    resp = client.get("/api/notebooks/duplicates", params={"limit": -1})

    assert resp.status_code == 400


def test_find_duplicate_notebooks_scopes_to_a_tag():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("dup_tag_prod_a.ipynb", "dup_tag_prod_b.ipynb", "dup_tag_scratch.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )

    client.put("/api/notebooks/dup_tag_prod_a.ipynb/tags", json={"tags": ["production"]})
    client.put("/api/notebooks/dup_tag_prod_b.ipynb/tags", json={"tags": ["production"]})

    body = client.get("/api/notebooks/duplicates", params={"tag": "production"}).json()

    assert body["group_count"] == 1
    assert body["duplicate_groups"][0]["filenames"] == [
        "dup_tag_prod_a.ipynb", "dup_tag_prod_b.ipynb",
    ]


def test_find_duplicate_notebooks_tag_with_only_one_matching_member_yields_no_group():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("dup_tag_solo_a.ipynb", "dup_tag_solo_b.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )

    client.put("/api/notebooks/dup_tag_solo_a.ipynb/tags", json={"tags": ["production"]})

    body = client.get("/api/notebooks/duplicates", params={"tag": "production"}).json()

    assert body["group_count"] == 0
    assert body["duplicate_groups"] == []


def test_find_duplicate_notebooks_sha256_narrows_to_the_matching_group():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename in ("dup_sha_a1.ipynb", "dup_sha_a2.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )
    for filename in ("dup_sha_b1.ipynb", "dup_sha_b2.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_b), "application/json")},
        )

    unfiltered = client.get("/api/notebooks/duplicates").json()
    assert unfiltered["group_count"] == 2

    target_sha256 = next(
        g["sha256"] for g in unfiltered["duplicate_groups"]
        if g["filenames"] == ["dup_sha_a1.ipynb", "dup_sha_a2.ipynb"]
    )

    body = client.get(
        "/api/notebooks/duplicates", params={"sha256": target_sha256}
    ).json()

    assert body["group_count"] == 1
    assert body["duplicate_groups"][0]["sha256"] == target_sha256
    assert body["duplicate_groups"][0]["filenames"] == [
        "dup_sha_a1.ipynb", "dup_sha_a2.ipynb",
    ]


def test_find_duplicate_notebooks_sha256_matching_no_notebook_yields_no_group():

    _upload_sample_notebook("dup_sha_unknown.ipynb")

    body = client.get(
        "/api/notebooks/duplicates", params={"sha256": "0" * 64}
    ).json()

    assert body["group_count"] == 0
    assert body["duplicate_groups"] == []


def test_find_duplicate_notebooks_sha256_matching_a_single_notebook_yields_no_group():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def f() -> int:\n    return 1\n")
    client.post(
        "/api/upload",
        files={"file": ("dup_sha_solo.ipynb", io.BytesIO(content), "application/json")},
    )

    solo_sha256 = hashlib.sha256(content).hexdigest()

    body = client.get(
        "/api/notebooks/duplicates", params={"sha256": solo_sha256}
    ).json()

    assert body["group_count"] == 0
    assert body["duplicate_groups"] == []


def test_resolve_duplicate_notebooks_keeps_alphabetically_first_by_default():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("resolve_z.ipynb", "resolve_a.ipynb", "resolve_m.ipynb"):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.post("/api/notebooks/duplicates/resolve", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 0

    result = body["results"][0]
    assert result["status"] == "success"
    assert result["kept_filename"] == "resolve_a.ipynb"
    assert sorted(e["filename"] for e in result["deleted_filenames"]) == [
        "resolve_m.ipynb", "resolve_z.ipynb",
    ]

    remaining = {n["filename"] for n in client.get("/api/notebooks").json()["notebooks"]}
    assert remaining == {"resolve_a.ipynb"}


def test_resolve_duplicate_notebooks_dry_run_reports_the_plan_without_deleting():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("resolve_dry_z.ipynb", "resolve_dry_a.ipynb", "resolve_dry_m.ipynb"):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.post(
        "/api/notebooks/duplicates/resolve", json={"dry_run": True}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 0

    result = body["results"][0]
    assert result["status"] == "success"
    assert result["kept_filename"] == "resolve_dry_a.ipynb"
    assert sorted(e["filename"] for e in result["deleted_filenames"]) == [
        "resolve_dry_m.ipynb", "resolve_dry_z.ipynb",
    ]

    # Nothing was actually deleted.
    remaining = {n["filename"] for n in client.get("/api/notebooks").json()["notebooks"]}
    assert remaining == {
        "resolve_dry_a.ipynb", "resolve_dry_m.ipynb", "resolve_dry_z.ipynb",
    }


def test_resolve_duplicate_notebooks_non_dry_run_reports_dry_run_false():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("resolve_real_a.ipynb", "resolve_real_b.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )

    resp = client.post("/api/notebooks/duplicates/resolve", json={})

    assert resp.status_code == 200
    assert resp.json()["dry_run"] is False


def test_resolve_duplicate_notebooks_honors_keep_override():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("resolve_keep_a.ipynb", "resolve_keep_b.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )

    sha256 = client.get("/api/notebooks/duplicates").json()["duplicate_groups"][0]["sha256"]

    resp = client.post(
        "/api/notebooks/duplicates/resolve",
        json={"keep": {sha256: "resolve_keep_b.ipynb"}},
    )

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["kept_filename"] == "resolve_keep_b.ipynb"
    assert [e["filename"] for e in result["deleted_filenames"]] == ["resolve_keep_a.ipynb"]

    remaining = {n["filename"] for n in client.get("/api/notebooks").json()["notebooks"]}
    assert remaining == {"resolve_keep_b.ipynb"}


def test_resolve_duplicate_notebooks_reports_an_invalid_keep_filename_for_just_that_group():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("resolve_bad_a.ipynb", "resolve_bad_b.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )

    sha256 = client.get("/api/notebooks/duplicates").json()["duplicate_groups"][0]["sha256"]

    resp = client.post(
        "/api/notebooks/duplicates/resolve",
        json={"keep": {sha256: "not_in_this_group.ipynb"}},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 0
    assert body["failed_count"] == 1
    assert body["results"][0]["status"] == "error"
    assert "not a member" in body["results"][0]["detail"]

    # Nothing was deleted -- both duplicates remain.
    remaining = {n["filename"] for n in client.get("/api/notebooks").json()["notebooks"]}
    assert remaining == {"resolve_bad_a.ipynb", "resolve_bad_b.ipynb"}


def test_resolve_duplicate_notebooks_is_a_no_op_success_when_nothing_is_duplicated():

    client.delete("/api/notebooks?confirm=true")

    _upload_sample_notebook("resolve_none.ipynb")

    resp = client.post("/api/notebooks/duplicates/resolve", json={})

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success", "dry_run": False, "results": [],
        "succeeded_count": 0, "failed_count": 0,
    }

    remaining = {n["filename"] for n in client.get("/api/notebooks").json()["notebooks"]}
    assert remaining == {"resolve_none.ipynb"}


def test_resolve_duplicate_notebooks_also_removes_tags_description_and_versions(
    tmp_path
):

    from backend.routes import upload as upload_module

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in ("resolve_cleanup_a.ipynb", "resolve_cleanup_z.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )

    client.put(
        "/api/notebooks/resolve_cleanup_z.ipynb/tags", json={"tags": ["stale"]}
    )
    client.put(
        "/api/notebooks/resolve_cleanup_z.ipynb/description",
        json={"description": "about to be resolved away"},
    )

    resp = client.post("/api/notebooks/duplicates/resolve", json={})
    assert resp.status_code == 200

    assert not upload_module._tags_sidecar_path("resolve_cleanup_z.ipynb").is_file()
    assert not upload_module._description_sidecar_path(
        "resolve_cleanup_z.ipynb"
    ).is_file()


def test_resolve_duplicate_notebooks_rejects_a_non_object_keep_value():

    resp = client.post("/api/notebooks/duplicates/resolve", json={"keep": "not-an-object"})

    assert resp.status_code == 400


def test_resolve_duplicate_notebooks_scopes_to_a_tag():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename in (
        "resolve_tag_prod_a.ipynb", "resolve_tag_prod_b.ipynb", "resolve_tag_scratch.ipynb",
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )

    client.put("/api/notebooks/resolve_tag_prod_a.ipynb/tags", json={"tags": ["production"]})
    client.put("/api/notebooks/resolve_tag_prod_b.ipynb/tags", json={"tags": ["production"]})

    resp = client.post(
        "/api/notebooks/duplicates/resolve", json={"tag": "production"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["results"][0]["kept_filename"] == "resolve_tag_prod_a.ipynb"
    assert [e["filename"] for e in body["results"][0]["deleted_filenames"]] == [
        "resolve_tag_prod_b.ipynb",
    ]

    # The untagged byte-identical notebook was never touched.
    remaining = {n["filename"] for n in client.get("/api/notebooks").json()["notebooks"]}
    assert remaining == {"resolve_tag_prod_a.ipynb", "resolve_tag_scratch.ipynb"}


def test_resolve_duplicate_notebooks_rejects_a_non_string_tag():

    resp = client.post("/api/notebooks/duplicates/resolve", json={"tag": 123})

    assert resp.status_code == 400


def test_resolve_duplicate_notebooks_sha256_resolves_only_the_matching_group():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename in ("resolve_sha_a1.ipynb", "resolve_sha_a2.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_a), "application/json")},
        )
    for filename in ("resolve_sha_b1.ipynb", "resolve_sha_b2.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content_b), "application/json")},
        )

    target_sha256 = hashlib.sha256(content_a).hexdigest()

    resp = client.post(
        "/api/notebooks/duplicates/resolve", json={"sha256": target_sha256}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["results"][0]["sha256"] == target_sha256
    assert body["results"][0]["kept_filename"] == "resolve_sha_a1.ipynb"
    assert [e["filename"] for e in body["results"][0]["deleted_filenames"]] == [
        "resolve_sha_a2.ipynb",
    ]

    # The other duplicate group was never touched.
    remaining = {n["filename"] for n in client.get("/api/notebooks").json()["notebooks"]}
    assert remaining == {
        "resolve_sha_a1.ipynb", "resolve_sha_b1.ipynb", "resolve_sha_b2.ipynb",
    }


def test_resolve_duplicate_notebooks_sha256_matching_no_group_resolves_nothing():

    client.delete("/api/notebooks?confirm=true")

    _upload_sample_notebook("resolve_sha_no_match.ipynb")

    resp = client.post(
        "/api/notebooks/duplicates/resolve", json={"sha256": "0" * 64}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"] == []
    assert body["succeeded_count"] == 0
    assert body["failed_count"] == 0

    remaining = {n["filename"] for n in client.get("/api/notebooks").json()["notebooks"]}
    assert remaining == {"resolve_sha_no_match.ipynb"}


def test_resolve_duplicate_notebooks_rejects_a_non_string_sha256():

    resp = client.post("/api/notebooks/duplicates/resolve", json={"sha256": 123})

    assert resp.status_code == 400


def test_search_notebook_content_finds_notebooks_with_a_matching_cell():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "import pandas as pd\n\n"
        "def load() -> str:\n    df = pd.read_csv('data.csv')\n    return 'done'\n"
    )
    content_b = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    for filename, content in (
        ("search_content_a.ipynb", content_a),
        ("search_content_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.get(
        "/api/notebooks/search-content", params={"search": "read_csv"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["search"] == "read_csv"
    assert body["notebook_count"] == 1
    assert body["matches"][0]["filename"] == "search_content_a.ipynb"

    cell_match = body["matches"][0]["matches"][0]
    assert cell_match["cell_index"] == 0
    assert "read_csv" in cell_match["snippet"]


def test_search_notebook_content_filters_by_tag():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes(
        "import pandas as pd\n\n"
        "def load() -> str:\n    df = pd.read_csv('data.csv')\n    return 'done'\n"
    )

    for filename in ("search_content_tag_a.ipynb", "search_content_tag_b.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    client.put(
        "/api/notebooks/search_content_tag_a.ipynb/tags", json={"tags": ["prod"]}
    )

    resp = client.get(
        "/api/notebooks/search-content", params={"search": "read_csv", "tag": "prod"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["notebook_count"] == 1
    assert body["matches"][0]["filename"] == "search_content_tag_a.ipynb"


def test_search_notebook_content_is_case_insensitive():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes(
        "def f() -> int:\n    # TODO: fix this\n    return 1\n"
    )

    resp = client.post(
        "/api/upload",
        files={"file": ("search_content_case.ipynb", io.BytesIO(content), "application/json")},
    )
    assert resp.status_code == 200

    resp = client.get(
        "/api/notebooks/search-content", params={"search": "todo"}
    )

    assert resp.status_code == 200
    assert resp.json()["notebook_count"] == 1


def test_search_notebook_content_reports_no_matches():

    client.delete("/api/notebooks?confirm=true")

    resp = client.get(
        "/api/notebooks/search-content", params={"search": "nonexistent_xyz"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["matches"] == []
    assert body["notebook_count"] == 0


def test_search_notebook_content_requires_a_search_value():

    resp = client.get("/api/notebooks/search-content")

    assert resp.status_code == 400


def test_search_notebook_content_rejects_an_unknown_format():

    resp = client.get("/api/notebooks/search-content?search=foo&format=xml")

    assert resp.status_code == 400


def test_search_notebook_content_csv_format_returns_one_row_per_matching_cell():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes(
        "import pandas as pd\ndf = pd.read_csv('a.csv')\n"
    )
    client.post(
        "/api/upload",
        files={"file": ("search_content_csv.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get(
        "/api/notebooks/search-content", params={"search": "pd.", "format": "csv"}
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="search_content.csv"' in resp.headers["content-disposition"]

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == "filename,cell_index,snippet"
    assert len(rows) == 2
    assert rows[1] == "search_content_csv.ipynb,0,df = pd.read_csv('a.csv')"


def test_search_notebook_content_skips_a_malformed_notebook_file():

    client.delete("/api/notebooks?confirm=true")

    filename = "search_content_malformed.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.get(
        "/api/notebooks/search-content", params={"search": "anything"}
    )

    assert resp.status_code == 200
    assert resp.json()["matches"] == []


def test_search_notebook_content_reports_multiple_matching_cells_in_one_notebook():

    client.delete("/api/notebooks?confirm=true")

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(nbformat.v4.new_code_cell("MARKER = 1\n"))
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "def f() -> int:\n    MARKER_VALUE = 2\n    return MARKER_VALUE\n"
        )
    )
    content = nbformat.writes(notebook).encode("utf-8")

    resp = client.post(
        "/api/upload",
        files={"file": ("search_content_multi.ipynb", io.BytesIO(content), "application/json")},
    )
    assert resp.status_code == 200

    resp = client.get(
        "/api/notebooks/search-content", params={"search": "MARKER"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["notebook_count"] == 1
    assert len(body["matches"][0]["matches"]) == 2
    assert [m["cell_index"] for m in body["matches"][0]["matches"]] == [0, 1]


def test_search_notebook_content_limit_and_offset_page_the_matching_notebooks():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("SEARCH_CONTENT_PAGE_MARKER = 1\n")

    for filename in (
        "search_content_page_a.ipynb",
        "search_content_page_b.ipynb",
        "search_content_page_c.ipynb",
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    all_matches = client.get(
        "/api/notebooks/search-content", params={"search": "SEARCH_CONTENT_PAGE_MARKER"}
    ).json()["matches"]
    assert len(all_matches) == 3

    resp = client.get(
        "/api/notebooks/search-content",
        params={"search": "SEARCH_CONTENT_PAGE_MARKER", "limit": 2},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [m["filename"] for m in body["matches"]] == [
        m["filename"] for m in all_matches[:2]
    ]
    assert body["notebook_count"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0

    resp = client.get(
        "/api/notebooks/search-content",
        params={"search": "SEARCH_CONTENT_PAGE_MARKER", "offset": 1},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [m["filename"] for m in body["matches"]] == [
        m["filename"] for m in all_matches[1:]
    ]
    assert body["notebook_count"] == 3
    assert body["offset"] == 1


def test_search_notebook_content_rejects_a_negative_offset():

    resp = client.get(
        "/api/notebooks/search-content", params={"search": "anything", "offset": -1}
    )

    assert resp.status_code == 400


def test_search_notebook_content_rejects_a_non_positive_limit():

    resp = client.get(
        "/api/notebooks/search-content", params={"search": "anything", "limit": 0}
    )

    assert resp.status_code == 400


def test_search_notebook_content_regex_matches_a_pattern_not_a_literal_substring():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes(
        "import pandas as pd\n\n"
        "def load() -> str:\n    df = pd.read_csv('data.csv', index_col=0)\n    return 'done'\n"
    )
    content_b = _notebook_bytes(
        "import pandas as pd\n\n"
        "def load() -> str:\n    df = pd.read_csv('data.csv')\n    return 'done'\n"
    )

    for filename, content in (
        ("search_regex_a.ipynb", content_a),
        ("search_regex_b.ipynb", content_b),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    # A plain substring for "read_csv(" alone matches both notebooks; the
    # pattern below only matches the one that also passes "index_col=".
    resp = client.get(
        "/api/notebooks/search-content",
        params={"search": r"read_csv\([^)]*index_col=", "regex": "true"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["regex"] is True
    assert body["notebook_count"] == 1
    assert body["matches"][0]["filename"] == "search_regex_a.ipynb"
    assert "index_col" in body["matches"][0]["matches"][0]["snippet"]


def test_search_notebook_content_regex_is_case_insensitive():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def f() -> int:\n    # TODO: fix this\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("search_regex_case.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get(
        "/api/notebooks/search-content",
        params={"search": "todo:.*fix", "regex": "true"},
    )

    assert resp.status_code == 200
    assert resp.json()["notebook_count"] == 1


def test_search_notebook_content_regex_false_treats_search_as_a_plain_substring():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes(
        "def f() -> int:\n    return 1  # not a real regex a.b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("search_no_regex.ipynb", io.BytesIO(content), "application/json")},
    )

    # "a.b" would match "a.b" literally here even without regex=true, so
    # use a pattern that only matches as a *regex* (any-character "."),
    # never as a literal substring the source doesn't actually contain.
    resp = client.get(
        "/api/notebooks/search-content",
        params={"search": "a.c", "regex": "false"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["regex"] is False
    assert body["notebook_count"] == 0


def test_search_notebook_content_regex_rejects_an_invalid_pattern():

    resp = client.get(
        "/api/notebooks/search-content",
        params={"search": "(unclosed", "regex": "true"},
    )

    assert resp.status_code == 400
    assert "regular expression" in resp.json()["detail"]


def test_search_notebook_content_regex_rejects_a_catastrophically_backtracking_pattern():
    """Matches this endpoint's raw code-cell *source* -- potentially many
    megabytes per notebook, across the whole catalog -- the highest-value
    target of the three endpoints sharing _compile_search_regex's own
    nested-unbounded-repetition check (see its own docstring).
    """

    resp = client.get(
        "/api/notebooks/search-content",
        params={"search": "(a+)+", "regex": "true"},
    )

    assert resp.status_code == 400
    assert "nested" in resp.json()["detail"].lower()


def test_diff_notebooks_reports_added_removed_changed_and_unchanged():

    old_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def remove_me() -> int:\n    return 0\n\n"
        "def unchanged_fn() -> int:\n    return 1\n"
    )
    new_content = _notebook_bytes(
        "def add(a: int, b: int, c: int) -> int:\n    return a + b + c\n\n"
        "def add_me() -> int:\n    return 2\n\n"
        "def unchanged_fn() -> int:\n    return 1\n"
    )

    for filename, content in (
        ("diff_old.ipynb", old_content),
        ("diff_new.ipynb", new_content),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.get(
        "/api/notebooks/diff", params={"old": "diff_old.ipynb", "new": "diff_new.ipynb"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["old"] == "diff_old.ipynb"
    assert body["new"] == "diff_new.ipynb"
    assert [f["name"] for f in body["added"]] == ["add_me"]
    assert [f["name"] for f in body["removed"]] == ["remove_me"]
    assert [c["name"] for c in body["changed"]] == ["add"]
    assert body["unchanged"] == ["unchanged_fn"]
    assert body["compatible"] is False
    breaking_types = {c["type"] for c in body["breaking_changes"]}
    assert "removed_endpoint" in breaking_types
    assert "required_parameter_added" in breaking_types


def test_diff_notebooks_reports_compatible_when_nothing_would_break_callers():

    old_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    new_content = _notebook_bytes(
        "def add(a: int, b: int, c: int = 0) -> int:\n    return a + b + c\n"
    )

    for filename, content in (
        ("diff_compat_old.ipynb", old_content),
        ("diff_compat_new.ipynb", new_content),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.get(
        "/api/notebooks/diff",
        params={"old": "diff_compat_old.ipynb", "new": "diff_compat_new.ipynb"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["compatible"] is True
    assert body["breaking_changes"] == []


def test_diff_notebooks_omits_content_diff_by_default():

    old_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    new_content = _notebook_bytes(
        'def add(a: int, b: int) -> int:\n    """docstring only"""\n    return a + b\n'
    )

    for filename, content in (
        ("diff_no_content_old.ipynb", old_content),
        ("diff_no_content_new.ipynb", new_content),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get(
        "/api/notebooks/diff",
        params={"old": "diff_no_content_old.ipynb", "new": "diff_no_content_new.ipynb"},
    )

    assert resp.status_code == 200
    assert "content_diff" not in resp.json()


def test_diff_notebooks_content_true_reports_a_docstring_only_edit():
    """The one case the structural "changed"/"unchanged" report deliberately
    doesn't surface -- content_diff must still show it.
    """

    old_content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    new_content = _notebook_bytes(
        'def add(a: int, b: int) -> int:\n    """docstring only"""\n    return a + b\n'
    )

    for filename, content in (
        ("diff_content_old.ipynb", old_content),
        ("diff_content_new.ipynb", new_content),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get(
        "/api/notebooks/diff",
        params={
            "old": "diff_content_old.ipynb", "new": "diff_content_new.ipynb",
            "content": "true",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["unchanged"] == ["add"]
    assert any(
        line.startswith("+") and "docstring only" in line
        for line in body["content_diff"]
    )
    assert "diff_content_old.ipynb" in body["content_diff"][0]
    assert "diff_content_new.ipynb" in body["content_diff"][1]


def test_diff_notebooks_requires_both_filenames():

    resp = client.get("/api/notebooks/diff", params={"old": "a.ipynb"})

    assert resp.status_code == 400


def test_diff_notebooks_returns_404_naming_the_missing_old_notebook():

    _upload_sample_notebook("diff_missing_new_target.ipynb")

    resp = client.get(
        "/api/notebooks/diff",
        params={"old": "does_not_exist.ipynb", "new": "diff_missing_new_target.ipynb"},
    )

    assert resp.status_code == 404
    assert "does_not_exist.ipynb" in resp.json()["detail"]


def test_diff_notebooks_returns_404_naming_the_missing_new_notebook():

    _upload_sample_notebook("diff_missing_old_target.ipynb")

    resp = client.get(
        "/api/notebooks/diff",
        params={"old": "diff_missing_old_target.ipynb", "new": "does_not_exist.ipynb"},
    )

    assert resp.status_code == 404
    assert "does_not_exist.ipynb" in resp.json()["detail"]


def test_diff_notebooks_returns_400_for_a_malformed_notebook():

    _upload_sample_notebook("diff_malformed_valid_side.ipynb")

    filename = "diff_malformed_bad_side.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.get(
        "/api/notebooks/diff",
        params={"old": "diff_malformed_valid_side.ipynb", "new": filename},
    )

    assert resp.status_code == 400
    assert "'new' notebook" in resp.json()["detail"]


def test_diff_notebooks_old_version_compares_a_snapshot_against_the_other_sides_current_content():

    old_original = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    old_current = _notebook_bytes(
        "def add(a: int, b: int, c: int) -> int:\n    return a + b + c\n"
    )
    new_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("diff_ov_old.ipynb", io.BytesIO(old_original), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": ("diff_ov_old.ipynb", io.BytesIO(old_current), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("diff_ov_new.ipynb", io.BytesIO(new_content), "application/json")},
    )

    old_version_id = client.get(
        "/api/notebooks/diff_ov_old.ipynb/versions"
    ).json()["versions"][0]["version_id"]

    # Comparing the *current* content of both notebooks would report
    # "add" as changed (3-arg vs 2-arg) -- pinning "old" to the
    # snapshotted version taken *before* that edit instead should report
    # them as identical.
    resp = client.get(
        "/api/notebooks/diff",
        params={
            "old": "diff_ov_old.ipynb", "new": "diff_ov_new.ipynb",
            "old_version": old_version_id,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["old_version"] == old_version_id
    assert body["new_version"] is None
    assert body["changed"] == []
    assert body["unchanged"] == ["add"]


def test_diff_notebooks_both_sides_can_be_pinned_to_versions_of_different_notebooks():

    a_v1 = _notebook_bytes("def f() -> int:\n    return 1\n")
    a_v2 = _notebook_bytes("def f() -> int:\n    return 2\n")
    b_v1 = _notebook_bytes("def f() -> int:\n    return 1\n")
    b_v2 = _notebook_bytes("def f() -> int:\n    return 3\n")

    client.post(
        "/api/upload",
        files={"file": ("diff_both_a.ipynb", io.BytesIO(a_v1), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": ("diff_both_a.ipynb", io.BytesIO(a_v2), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("diff_both_b.ipynb", io.BytesIO(b_v1), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": ("diff_both_b.ipynb", io.BytesIO(b_v2), "application/json")},
    )

    a_version_id = client.get(
        "/api/notebooks/diff_both_a.ipynb/versions"
    ).json()["versions"][0]["version_id"]
    b_version_id = client.get(
        "/api/notebooks/diff_both_b.ipynb/versions"
    ).json()["versions"][0]["version_id"]

    # Both notebooks' own *first* version returned "1" -- identical.
    resp = client.get(
        "/api/notebooks/diff",
        params={
            "old": "diff_both_a.ipynb", "new": "diff_both_b.ipynb",
            "old_version": a_version_id, "new_version": b_version_id,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["old_version"] == a_version_id
    assert body["new_version"] == b_version_id
    assert body["changed"] == []
    assert body["unchanged"] == ["f"]


def test_diff_notebooks_returns_404_for_an_unknown_old_version():

    _upload_sample_notebook("diff_unknown_version_old.ipynb")
    _upload_sample_notebook("diff_unknown_version_new.ipynb")

    resp = client.get(
        "/api/notebooks/diff",
        params={
            "old": "diff_unknown_version_old.ipynb",
            "new": "diff_unknown_version_new.ipynb",
            "old_version": "does-not-exist.ipynb",
        },
    )

    assert resp.status_code == 404
    assert "diff_unknown_version_old.ipynb" in resp.json()["detail"]


def test_delete_notebook_rejects_a_filename_with_an_embedded_null_byte():

    resp = client.delete("/api/notebooks/nb%00.ipynb")

    assert resp.status_code == 400


def test_notebook_storage_reports_per_notebook_and_total_bytes():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    content_b = _notebook_bytes(
        "def subtract_two_numbers(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("storage_a.ipynb", content_a),
        ("storage_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.get("/api/notebooks/storage")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["notebook_count"] == 2
    assert body["total_version_bytes"] == 0
    assert body["total_version_count"] == 0
    assert body["total_notebook_bytes"] == len(content_a) + len(content_b)
    assert body["total_bytes"] == body["total_notebook_bytes"]

    entries_by_filename = {n["filename"]: n for n in body["notebooks"]}
    assert entries_by_filename["storage_a.ipynb"] == {
        "filename": "storage_a.ipynb",
        "notebook_bytes": len(content_a),
        "version_bytes": 0,
        "version_count": 0,
        "total_bytes": len(content_a),
    }
    assert entries_by_filename["storage_b.ipynb"]["notebook_bytes"] == len(content_b)


def test_notebook_storage_reports_max_notebooks_disabled_by_default():

    resp = client.get("/api/notebooks/storage")

    assert resp.status_code == 200
    body = resp.json()
    assert body["max_notebooks"] == 0
    assert body["notebooks_remaining"] is None


def test_notebook_storage_reports_notebooks_remaining_with_a_configured_cap(monkeypatch):

    from backend.routes import upload as upload_module

    client.delete("/api/notebooks?confirm=true")

    client.post(
        "/api/upload",
        files={
            "file": (
                "storage_cap_a.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )

    current_count = upload_module._current_notebook_count()
    monkeypatch.setattr(upload_module, "MAX_NOTEBOOKS", current_count + 5)

    resp = client.get("/api/notebooks/storage")

    assert resp.status_code == 200
    body = resp.json()
    assert body["max_notebooks"] == current_count + 5
    assert body["notebooks_remaining"] == 5


def test_notebook_storage_notebooks_remaining_can_go_negative(monkeypatch):
    """Honest, not clamped to 0 -- a cap lowered after the catalog already
    exceeded it must still report the true (negative) remaining figure.
    """

    from backend.routes import upload as upload_module

    client.delete("/api/notebooks?confirm=true")

    for filename in ("storage_over_cap_a.ipynb", "storage_over_cap_b.ipynb"):
        client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                )
            },
        )

    current_count = upload_module._current_notebook_count()
    assert current_count >= 2
    monkeypatch.setattr(upload_module, "MAX_NOTEBOOKS", current_count - 1)

    resp = client.get("/api/notebooks/storage")

    body = resp.json()
    assert body["notebooks_remaining"] == -1


def test_notebook_storage_notebooks_remaining_ignores_the_tag_filter(monkeypatch):
    """"notebooks_remaining" must reflect the *whole* catalog against
    MAX_NOTEBOOKS, never just the "tag"-scoped subset this endpoint's own
    "notebook_count" narrows to when "tag" is given -- comparing a
    tag-scoped count against a catalog-wide cap would silently understate
    how close the whole catalog actually is to it.
    """

    from backend.routes import upload as upload_module

    client.delete("/api/notebooks?confirm=true")

    client.post(
        "/api/upload",
        files={
            "file": (
                "storage_tag_scope_tagged.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.put(
        "/api/notebooks/storage_tag_scope_tagged.ipynb/tags",
        json={"tags": ["prod"]},
    )
    client.post(
        "/api/upload",
        files={
            "file": (
                "storage_tag_scope_untagged.ipynb",
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    current_count = upload_module._current_notebook_count()
    assert current_count >= 2
    monkeypatch.setattr(upload_module, "MAX_NOTEBOOKS", current_count + 3)

    resp = client.get("/api/notebooks/storage", params={"tag": "prod"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["notebook_count"] == 1  # tag-scoped, unaffected by this feature
    assert body["notebooks_remaining"] == 3  # catalog-wide, not tag-scoped


def test_notebook_storage_csv_format_returns_a_csv_response():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    content_b = _notebook_bytes(
        "def subtract_two_numbers(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("storage_csv_a.ipynb", content_a),
        ("storage_csv_b.ipynb", content_b),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get("/api/notebooks/storage", params={"format": "csv"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="notebook_storage.csv"' in resp.headers["content-disposition"]

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == "filename,notebook_bytes,version_bytes,version_count,total_bytes"
    # Biggest-first, exactly like the "json" response's own "notebooks".
    assert rows[1] == f"storage_csv_b.ipynb,{len(content_b)},0,0,{len(content_b)}"
    assert rows[2] == f"storage_csv_a.ipynb,{len(content_a)},0,0,{len(content_a)}"
    assert len(rows) == 3


def test_notebook_storage_csv_format_composes_with_tag_and_limit():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("storage_csv_tagged.ipynb", io.BytesIO(content), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("storage_csv_untagged.ipynb", io.BytesIO(content), "application/json")},
    )
    client.put(
        "/api/notebooks/storage_csv_tagged.ipynb/tags", json={"tags": ["prod"]}
    )

    resp = client.get(
        "/api/notebooks/storage",
        params={"format": "csv", "tag": "prod", "limit": 1},
    )

    assert resp.status_code == 200
    rows = resp.text.strip().split("\r\n")
    assert len(rows) == 2
    assert rows[1].startswith("storage_csv_tagged.ipynb,")


def test_notebook_storage_rejects_an_unknown_format():

    resp = client.get("/api/notebooks/storage", params={"format": "xml"})

    assert resp.status_code == 400
    assert "format" in resp.json()["detail"]


def test_notebook_storage_filters_by_tag():

    client.delete("/api/notebooks?confirm=true")

    content_a = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")
    content_b = _notebook_bytes(
        "def subtract_two_numbers(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("storage_tag_prod.ipynb", content_a),
        ("storage_tag_other.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    client.put(
        "/api/notebooks/storage_tag_prod.ipynb/tags", json={"tags": ["prod"]}
    )

    resp = client.get("/api/notebooks/storage", params={"tag": "prod"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["notebook_count"] == 1
    assert [n["filename"] for n in body["notebooks"]] == ["storage_tag_prod.ipynb"]
    assert body["total_notebook_bytes"] == len(content_a)
    assert body["total_bytes"] == len(content_a)


def test_notebook_storage_unknown_tag_reports_zeros():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    resp = client.post(
        "/api/upload",
        files={"file": ("storage_no_tag.ipynb", io.BytesIO(content), "application/json")},
    )
    assert resp.status_code == 200

    resp = client.get("/api/notebooks/storage", params={"tag": "no-such-tag"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["notebooks"] == []
    assert body["notebook_count"] == 0
    assert body["total_bytes"] == 0


def test_notebook_storage_includes_version_history_bytes():

    client.delete("/api/notebooks?confirm=true")

    filename = "storage_versions.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")
    new_content = _notebook_bytes(
        "def f() -> int:\n    return 1\n\ndef g() -> int:\n    return 2\n"
    )

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(new_content), "application/json")},
    )

    resp = client.get("/api/notebooks/storage")

    assert resp.status_code == 200
    body = resp.json()

    entry = body["notebooks"][0]
    assert entry["filename"] == filename
    assert entry["notebook_bytes"] == len(new_content)
    assert entry["version_bytes"] == len(original_content)
    assert entry["version_count"] == 1
    assert entry["total_bytes"] == len(new_content) + len(original_content)

    assert body["total_version_bytes"] == len(original_content)
    assert body["total_version_count"] == 1
    assert body["total_bytes"] == entry["total_bytes"]


def test_notebook_storage_sorts_by_total_bytes_descending():

    client.delete("/api/notebooks?confirm=true")

    small = _notebook_bytes("def f() -> int:\n    return 1\n")
    large = _notebook_bytes(
        "def a_much_longer_function_name_for_a_bigger_notebook() -> int:\n"
        "    return 1\n"
    )
    assert len(large) > len(small)

    for filename, content in (
        ("storage_small.ipynb", small),
        ("storage_large.ipynb", large),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get("/api/notebooks/storage")

    assert resp.status_code == 200
    filenames_in_order = [n["filename"] for n in resp.json()["notebooks"]]
    assert filenames_in_order == ["storage_large.ipynb", "storage_small.ipynb"]


def test_notebook_storage_limit_caps_the_biggest_first_notebooks():

    client.delete("/api/notebooks?confirm=true")

    small = _notebook_bytes("def f() -> int:\n    return 1\n")
    medium = _notebook_bytes("def medium_sized_function_name() -> int:\n    return 1\n")
    large = _notebook_bytes(
        "def a_much_longer_function_name_for_a_bigger_notebook() -> int:\n"
        "    return 1\n"
    )

    for filename, content in (
        ("storage_limit_small.ipynb", small),
        ("storage_limit_medium.ipynb", medium),
        ("storage_limit_large.ipynb", large),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get("/api/notebooks/storage", params={"limit": 2})

    assert resp.status_code == 200
    body = resp.json()
    assert [n["filename"] for n in body["notebooks"]] == [
        "storage_limit_large.ipynb", "storage_limit_medium.ipynb",
    ]
    assert body["notebook_count"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0

    # Running totals still cover every notebook, not just the returned page.
    assert body["total_notebook_bytes"] == len(small) + len(medium) + len(large)


def test_notebook_storage_offset_skips_the_biggest_first_notebooks():

    client.delete("/api/notebooks?confirm=true")

    small = _notebook_bytes("def f() -> int:\n    return 1\n")
    large = _notebook_bytes(
        "def a_much_longer_function_name_for_a_bigger_notebook() -> int:\n"
        "    return 1\n"
    )

    for filename, content in (
        ("storage_offset_small.ipynb", small),
        ("storage_offset_large.ipynb", large),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get("/api/notebooks/storage", params={"offset": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert [n["filename"] for n in body["notebooks"]] == ["storage_offset_small.ipynb"]
    assert body["notebook_count"] == 2
    assert body["offset"] == 1


def test_notebook_storage_rejects_a_negative_offset():

    resp = client.get("/api/notebooks/storage", params={"offset": -1})

    assert resp.status_code == 400


def test_notebook_storage_rejects_a_non_positive_limit():

    resp = client.get("/api/notebooks/storage", params={"limit": 0})

    assert resp.status_code == 400


def test_notebook_storage_reports_zeros_for_an_empty_catalog():

    client.delete("/api/notebooks?confirm=true")

    resp = client.get("/api/notebooks/storage")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "notebooks": [],
        "notebook_count": 0,
        "limit": None,
        "offset": 0,
        "total_notebook_bytes": 0,
        "total_version_bytes": 0,
        "total_version_count": 0,
        "total_bytes": 0,
        "max_notebooks": 0,
        "notebooks_remaining": None,
    }


def test_delete_all_notebooks_requires_confirm_true():
    """A bulk delete with real, hard-to-undo consequences (the notebooks
    in UPLOAD_DIR are the only copy of a user's original uploaded source
    on this server) must not run without the same explicit opt-in
    /api/upload's own "overwrite" and /api/deploy's "force"/"push"
    already require elsewhere in this file.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("bulk_delete_no_confirm.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    resp = client.delete("/api/notebooks")

    assert resp.status_code == 400

    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_no_confirm.ipynb" in filenames


def test_delete_all_notebooks_dry_run_does_not_require_confirm_true():

    _upload_sample_notebook("bulk_delete_dry_run.ipynb")

    resp = client.delete("/api/notebooks", params={"dry_run": "true"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert "bulk_delete_dry_run.ipynb" in body["deleted_filenames"]

    # Nothing was actually deleted.
    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_dry_run.ipynb" in filenames

    os.remove(Path(UPLOAD_DIR) / "bulk_delete_dry_run.ipynb")


def test_delete_all_notebooks_dry_run_scoped_by_tag_reports_only_matching_notebooks():

    _upload_sample_notebook("bulk_delete_dry_run_match.ipynb")
    _upload_sample_notebook("bulk_delete_dry_run_other.ipynb")

    client.put(
        "/api/notebooks/bulk_delete_dry_run_match.ipynb/tags",
        json={"tags": ["scratch"]},
    )

    resp = client.delete(
        "/api/notebooks", params={"dry_run": "true", "tag": "scratch"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["deleted_filenames"] == ["bulk_delete_dry_run_match.ipynb"]

    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_dry_run_match.ipynb" in filenames
    assert "bulk_delete_dry_run_other.ipynb" in filenames

    os.remove(Path(UPLOAD_DIR) / "bulk_delete_dry_run_match.ipynb")
    os.remove(Path(UPLOAD_DIR) / "bulk_delete_dry_run_other.ipynb")


def test_delete_all_notebooks_removes_every_uploaded_notebook():

    content_a = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    content_b = _notebook_bytes(
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    for filename, content in (
        ("bulk_delete_a.ipynb", content_a),
        ("bulk_delete_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    delete_resp = client.delete("/api/notebooks?confirm=true")

    assert delete_resp.status_code == 200
    body = delete_resp.json()
    assert body["deleted_count"] >= 2
    assert "bulk_delete_a.ipynb" in body["deleted_filenames"]
    assert "bulk_delete_b.ipynb" in body["deleted_filenames"]
    assert body["currently_compiled_notebook_deleted"] is False

    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_a.ipynb" not in filenames
    assert "bulk_delete_b.ipynb" not in filenames


def test_delete_all_notebooks_scoped_by_tag_deletes_only_matching_notebooks():

    _upload_sample_notebook("bulk_delete_tag_match.ipynb")
    _upload_sample_notebook("bulk_delete_tag_other.ipynb")

    client.put(
        "/api/notebooks/bulk_delete_tag_match.ipynb/tags",
        json={"tags": ["bulk-delete-scope"]},
    )

    delete_resp = client.delete(
        "/api/notebooks", params={"confirm": "true", "tag": "bulk-delete-scope"}
    )

    assert delete_resp.status_code == 200
    body = delete_resp.json()
    assert body["deleted_filenames"] == ["bulk_delete_tag_match.ipynb"]
    assert body["deleted_count"] == 1

    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_tag_match.ipynb" not in filenames
    # The untagged notebook is left completely untouched.
    assert "bulk_delete_tag_other.ipynb" in filenames


def test_delete_all_notebooks_scoped_by_an_unknown_tag_deletes_nothing():

    _upload_sample_notebook("bulk_delete_unknown_tag.ipynb")

    delete_resp = client.delete(
        "/api/notebooks", params={"confirm": "true", "tag": "no-such-tag-anywhere"}
    )

    assert delete_resp.status_code == 200
    body = delete_resp.json()
    assert body["deleted_count"] == 0
    assert body["deleted_filenames"] == []

    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_unknown_tag.ipynb" in filenames


def test_delete_all_notebooks_scoped_by_tag_still_requires_confirm_true():

    resp = client.delete("/api/notebooks", params={"tag": "bulk-delete-scope"})

    assert resp.status_code == 400


def test_delete_all_notebooks_scoped_by_sha256_deletes_only_matching_notebooks():
    """Confirmed missing before this fix: GET /api/notebooks and GET
    /api/notebooks/duplicates both already support an exact-content
    "sha256" filter, and DELETE /api/notebooks's own docstring already
    cites both as siblings reusing its own "tag" filter -- but never
    itself gained the identical "sha256" filter those two siblings
    already have. An operator who finds a bad/duplicate content hash via
    GET /api/notebooks/duplicates (which reports every filename sharing
    it, since a notebook can be renamed or re-uploaded under a
    completely different name while keeping the same content) had no
    way to remove every copy of that exact content in one call.
    """

    # Computed once and reused for both uploads -- nbformat.v4.new_code_cell
    # stamps each cell with a random id, so two separate _notebook_bytes(...)
    # calls with identical source text still produce different bytes (and
    # so a different sha256); only reusing the exact same bytes object
    # guarantees the byte-identical content this test actually needs.
    shared_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    upload_a = client.post(
        "/api/upload",
        files={
            "file": (
                "bulk_delete_sha256_match_a.ipynb",
                io.BytesIO(shared_content),
                "application/json",
            )
        },
    )
    upload_b = client.post(
        "/api/upload",
        files={
            "file": (
                "bulk_delete_sha256_match_b.ipynb",
                io.BytesIO(shared_content),
                "application/json",
            )
        },
    )
    _upload_sample_notebook("bulk_delete_sha256_other.ipynb")

    matching_sha256 = upload_a.json()["sha256"]
    assert matching_sha256 == upload_b.json()["sha256"]

    delete_resp = client.delete(
        "/api/notebooks", params={"confirm": "true", "sha256": matching_sha256}
    )

    assert delete_resp.status_code == 200
    body = delete_resp.json()
    assert sorted(body["deleted_filenames"]) == [
        "bulk_delete_sha256_match_a.ipynb",
        "bulk_delete_sha256_match_b.ipynb",
    ]
    assert body["deleted_count"] == 2

    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_sha256_match_a.ipynb" not in filenames
    assert "bulk_delete_sha256_match_b.ipynb" not in filenames
    # A notebook with different content is left completely untouched,
    # even though "bulk_delete_sha256_other.ipynb" happens to share the
    # exact same content as every other _upload_sample_notebook() call
    # elsewhere in this file -- it just doesn't match *this* sha256.
    assert "bulk_delete_sha256_other.ipynb" in filenames


def test_delete_all_notebooks_scoped_by_an_unknown_sha256_deletes_nothing():

    _upload_sample_notebook("bulk_delete_unknown_sha256.ipynb")

    delete_resp = client.delete(
        "/api/notebooks",
        params={"confirm": "true", "sha256": "0" * 64},
    )

    assert delete_resp.status_code == 200
    body = delete_resp.json()
    assert body["deleted_count"] == 0
    assert body["deleted_filenames"] == []

    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_unknown_sha256.ipynb" in filenames


def test_delete_all_notebooks_sha256_composes_with_tag():
    """Both "tag" and "sha256" given must act as an AND, not an OR -- a
    notebook matching only one of the two must survive, the same
    "matches every given filter" composition GET /api/notebooks already
    gives its own "tag"/"sha256" query params together.
    """

    shared_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    upload_tagged = client.post(
        "/api/upload",
        files={
            "file": (
                "bulk_delete_sha256_and_tag_match.ipynb",
                io.BytesIO(shared_content),
                "application/json",
            )
        },
    )
    matching_sha256 = upload_tagged.json()["sha256"]

    client.put(
        "/api/notebooks/bulk_delete_sha256_and_tag_match.ipynb/tags",
        json={"tags": ["bulk-delete-sha256-and-tag"]},
    )

    # Same content (matches "sha256"), but never tagged -- must survive:
    # matching only one of the two given filters isn't enough.
    client.post(
        "/api/upload",
        files={
            "file": (
                "bulk_delete_sha256_and_tag_untagged.ipynb",
                io.BytesIO(shared_content),
                "application/json",
            )
        },
    )

    delete_resp = client.delete(
        "/api/notebooks",
        params={
            "confirm": "true",
            "sha256": matching_sha256,
            "tag": "bulk-delete-sha256-and-tag",
        },
    )

    assert delete_resp.status_code == 200
    body = delete_resp.json()
    assert body["deleted_filenames"] == ["bulk_delete_sha256_and_tag_match.ipynb"]

    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_sha256_and_tag_match.ipynb" not in filenames
    assert "bulk_delete_sha256_and_tag_untagged.ipynb" in filenames


def test_delete_all_notebooks_dry_run_scoped_by_sha256_reports_only_matching_notebooks():

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "bulk_delete_sha256_dry_run.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    matching_sha256 = upload_resp.json()["sha256"]

    dry_run_resp = client.delete(
        "/api/notebooks",
        params={"dry_run": "true", "sha256": matching_sha256},
    )

    assert dry_run_resp.status_code == 200
    body = dry_run_resp.json()
    assert body["dry_run"] is True
    assert "bulk_delete_sha256_dry_run.ipynb" in body["deleted_filenames"]

    # Nothing was actually deleted.
    list_resp = client.get("/api/notebooks")
    filenames = {nb["filename"] for nb in list_resp.json()["notebooks"]}
    assert "bulk_delete_sha256_dry_run.ipynb" in filenames


def test_delete_all_notebooks_flags_currently_compiled_notebook_deleted():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "bulk_delete_currently_compiled.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile",
        json={"notebook_path": "bulk_delete_currently_compiled.ipynb"},
    )
    assert compile_resp.status_code == 200

    delete_resp = client.delete("/api/notebooks?confirm=true")

    assert delete_resp.status_code == 200
    assert delete_resp.json()["currently_compiled_notebook_deleted"] is True


def test_delete_all_notebooks_does_not_touch_generated_dir():
    """Mirrors DELETE /api/notebooks/{filename}'s own behavior: the
    compiled app currently running must keep running exactly as before --
    this only ever clears UPLOAD_DIR, never GENERATED_DIR.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "bulk_delete_keeps_generated.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile",
        json={"notebook_path": "bulk_delete_keeps_generated.ipynb"},
    )
    assert compile_resp.status_code == 200

    delete_resp = client.delete("/api/notebooks?confirm=true")
    assert delete_resp.status_code == 200

    generated_resp = client.get("/api/generated")
    assert generated_resp.status_code == 200
    assert "app.py" in generated_resp.json()["generated_files"]


def test_delete_all_notebooks_leaves_stale_part_files_alone():
    """Only ever removes ".ipynb" files directly inside UPLOAD_DIR -- the
    same set GET /api/notebooks already lists -- so an in-flight upload's
    own hidden ".part" temp file must never be touched by this.
    """

    stale_part_path = Path(UPLOAD_DIR) / ".bulk_delete_in_flight.ipynb.abc123.part"
    stale_part_path.write_bytes(b"not yet a real notebook")

    try:
        resp = client.delete("/api/notebooks?confirm=true")
        assert resp.status_code in (200, 400)
        assert stale_part_path.exists()
    finally:
        stale_part_path.unlink(missing_ok=True)


def test_delete_all_notebooks_returns_zero_when_nothing_uploaded():
    """UPLOAD_DIR isn't guaranteed empty at test time -- e.g. this repo
    ships uploads/sample.ipynb -- so this drains it first via the same
    endpoint under test rather than assuming a pristine directory, then
    confirms a second call against the now-genuinely-empty directory
    reports zero.
    """

    client.delete("/api/notebooks?confirm=true")

    resp = client.delete("/api/notebooks?confirm=true")

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_count"] == 0
    assert body["deleted_filenames"] == []
    assert body["currently_compiled_notebook_deleted"] is False


def test_delete_notebooks_batch_removes_only_the_named_notebooks():

    _upload_sample_notebook("delete_batch_a.ipynb")
    _upload_sample_notebook("delete_batch_b.ipynb")
    _upload_sample_notebook("delete_batch_c.ipynb")

    resp = client.post(
        "/api/notebooks/delete-batch",
        json={"filenames": ["delete_batch_a.ipynb", "delete_batch_b.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["delete_batch_a.ipynb"]["status"] == "success"
    assert results_by_filename["delete_batch_b.ipynb"]["status"] == "success"

    assert client.get("/api/notebooks/delete_batch_a.ipynb").status_code == 404
    assert client.get("/api/notebooks/delete_batch_b.ipynb").status_code == 404
    # Untouched -- not named in the batch.
    assert client.get("/api/notebooks/delete_batch_c.ipynb").status_code == 200


def test_delete_notebooks_batch_dry_run_reports_the_plan_without_deleting():

    _upload_sample_notebook("delete_batch_dry_a.ipynb")
    _upload_sample_notebook("delete_batch_dry_b.ipynb")

    resp = client.post(
        "/api/notebooks/delete-batch",
        json={
            "filenames": [
                "delete_batch_dry_a.ipynb", "delete_batch_dry_b.ipynb",
                "does_not_exist_dry.ipynb",
            ],
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["delete_batch_dry_a.ipynb"]["status"] == "success"
    assert results_by_filename["delete_batch_dry_b.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist_dry.ipynb"]["status"] == "error"

    # Nothing was actually deleted.
    assert client.get("/api/notebooks/delete_batch_dry_a.ipynb").status_code == 200
    assert client.get("/api/notebooks/delete_batch_dry_b.ipynb").status_code == 200


def test_delete_notebooks_batch_non_dry_run_reports_dry_run_false():

    _upload_sample_notebook("delete_batch_real_run.ipynb")

    resp = client.post(
        "/api/notebooks/delete-batch",
        json={"filenames": ["delete_batch_real_run.ipynb"]},
    )

    assert resp.status_code == 200
    assert resp.json()["dry_run"] is False


def test_delete_notebooks_batch_reports_a_missing_filename_without_aborting_the_rest():

    _upload_sample_notebook("delete_batch_partial.ipynb")

    resp = client.post(
        "/api/notebooks/delete-batch",
        json={"filenames": ["delete_batch_partial.ipynb", "does_not_exist.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["delete_batch_partial.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]

    assert client.get("/api/notebooks/delete_batch_partial.ipynb").status_code == 404


def test_delete_notebooks_batch_flags_was_currently_compiled():

    _upload_sample_notebook("delete_batch_compiled.ipynb")

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "delete_batch_compiled.ipynb"}
    )
    assert compile_resp.status_code == 200

    resp = client.post(
        "/api/notebooks/delete-batch",
        json={"filenames": ["delete_batch_compiled.ipynb"]},
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["was_currently_compiled"] is True


def test_delete_notebooks_batch_removes_tags_and_version_history():

    _upload_sample_notebook("delete_batch_cleanup.ipynb")
    client.put(
        "/api/notebooks/delete_batch_cleanup.ipynb/tags",
        json={"tags": ["scratch"]},
    )
    assert _tags_sidecar_path("delete_batch_cleanup.ipynb").is_file()

    resp = client.post(
        "/api/notebooks/delete-batch",
        json={"filenames": ["delete_batch_cleanup.ipynb"]},
    )

    assert resp.status_code == 200
    assert not _tags_sidecar_path("delete_batch_cleanup.ipynb").is_file()


def test_delete_notebooks_batch_rejects_a_non_list_filenames_value():

    resp = client.post(
        "/api/notebooks/delete-batch", json={"filenames": "not-a-list"}
    )

    assert resp.status_code == 400


def test_delete_notebooks_batch_rejects_an_empty_filenames_list():

    resp = client.post("/api/notebooks/delete-batch", json={"filenames": []})

    assert resp.status_code == 400


def test_delete_notebooks_batch_rejects_more_filenames_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/notebooks/delete-batch",
        json={"filenames": ["a.ipynb", "b.ipynb"]},
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_rename_notebook_renames_the_file_on_disk():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("rename_source.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    rename_resp = client.patch(
        "/api/notebooks/rename_source.ipynb",
        json={"new_filename": "rename_target.ipynb"},
    )

    assert rename_resp.status_code == 200
    body = rename_resp.json()
    assert body["filename"] == "rename_source.ipynb"
    assert body["new_filename"] == "rename_target.ipynb"
    assert body["was_currently_compiled"] is False

    assert not (Path(UPLOAD_DIR) / "rename_source.ipynb").exists()
    assert (Path(UPLOAD_DIR) / "rename_target.ipynb").is_file()

    filenames = {nb["filename"] for nb in client.get("/api/notebooks").json()["notebooks"]}
    assert "rename_source.ipynb" not in filenames
    assert "rename_target.ipynb" in filenames

    os.remove(Path(UPLOAD_DIR) / "rename_target.ipynb")


def test_rename_notebook_returns_404_for_missing_file():

    resp = client.patch(
        "/api/notebooks/does_not_exist_at_all.ipynb",
        json={"new_filename": "whatever.ipynb"},
    )

    assert resp.status_code == 404


def test_rename_notebook_requires_new_filename():

    content = _notebook_bytes("def add(a, b):\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_missing_target.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.patch(
        "/api/notebooks/rename_missing_target.ipynb",
        json={},
    )

    assert resp.status_code == 400


def test_rename_notebook_rejects_a_non_ipynb_target_name():

    content = _notebook_bytes("def add(a, b):\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_bad_ext.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.patch(
        "/api/notebooks/rename_bad_ext.ipynb",
        json={"new_filename": "rename_bad_ext.txt"},
    )

    assert resp.status_code == 400


def test_rename_notebook_rejects_a_traversal_target_name():

    content = _notebook_bytes("def add(a, b):\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_traversal_source.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.patch(
        "/api/notebooks/rename_traversal_source.ipynb",
        json={"new_filename": "../../../../etc/passwd.ipynb"},
    )

    assert resp.status_code == 400
    assert (Path(UPLOAD_DIR) / "rename_traversal_source.ipynb").is_file()


def test_rename_notebook_rejects_a_new_filename_with_an_embedded_null_byte():

    content = _notebook_bytes("def add(a, b):\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_null_byte_source.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.patch(
        "/api/notebooks/rename_null_byte_source.ipynb",
        json={"new_filename": "evil\x00.ipynb"},
    )

    assert resp.status_code == 400
    assert (Path(UPLOAD_DIR) / "rename_null_byte_source.ipynb").is_file()


def test_rename_notebook_to_its_own_name_is_a_no_op():

    content = _notebook_bytes("def add(a, b):\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_noop.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.patch(
        "/api/notebooks/rename_noop.ipynb",
        json={"new_filename": "rename_noop.ipynb"},
    )

    assert resp.status_code == 200
    assert resp.json()["was_currently_compiled"] is False
    assert (Path(UPLOAD_DIR) / "rename_noop.ipynb").is_file()


def test_rename_notebook_rejects_collision_without_overwrite():

    content_a = _notebook_bytes("def add(a, b):\n    return a + b\n")
    content_b = _notebook_bytes("def sub(a, b):\n    return a - b\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_collision_a.ipynb", io.BytesIO(content_a), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("rename_collision_b.ipynb", io.BytesIO(content_b), "application/json")},
    )

    resp = client.patch(
        "/api/notebooks/rename_collision_a.ipynb",
        json={"new_filename": "rename_collision_b.ipynb"},
    )

    assert resp.status_code == 409
    # Neither file should have moved.
    assert (Path(UPLOAD_DIR) / "rename_collision_a.ipynb").is_file()
    assert json.loads((Path(UPLOAD_DIR) / "rename_collision_b.ipynb").read_bytes()) == json.loads(content_b)


def test_rename_notebook_dry_run_reports_the_new_filename_without_renaming():

    _upload_sample_notebook("rename_dry_run_source.ipynb")

    resp = client.patch(
        "/api/notebooks/rename_dry_run_source.ipynb",
        json={"new_filename": "rename_dry_run_target.ipynb", "dry_run": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["new_filename"] == "rename_dry_run_target.ipynb"

    # Nothing was actually renamed.
    assert (Path(UPLOAD_DIR) / "rename_dry_run_source.ipynb").is_file()
    assert not (Path(UPLOAD_DIR) / "rename_dry_run_target.ipynb").exists()


def test_rename_notebook_dry_run_still_reports_a_same_name_collision():

    _upload_sample_notebook("rename_dry_run_collision_source.ipynb")
    _upload_sample_notebook("rename_dry_run_collision_target.ipynb")

    resp = client.patch(
        "/api/notebooks/rename_dry_run_collision_source.ipynb",
        json={"new_filename": "rename_dry_run_collision_target.ipynb", "dry_run": True},
    )

    assert resp.status_code == 409
    assert (Path(UPLOAD_DIR) / "rename_dry_run_collision_source.ipynb").is_file()


def test_rename_notebook_serializes_two_concurrent_renames_onto_the_same_destination():
    """Before _rename_lock_for existed, two concurrent renames of two
    different existing notebooks onto the same new_filename raced this
    endpoint's own check-then-write sequence: both requests' "does the
    destination already exist" check could observe "not yet" for both,
    since neither had written new_path yet when either checked -- and
    unlike upload_notebook (which at least re-checks immediately before
    its own swap), this endpoint had *no* re-check at all before
    os.replace(), so both proceeded straight through with no 409 raised
    by either, one silently clobbered the other's just-renamed file, and
    *both* callers saw "status": "success". Confirmed exploitable,
    reproduced directly before this fix: two threads racing this exact
    scenario against a live server produced two 200s in 19 of 20 single
    trials -- rename_notebook is a plain `def`, not `async def` (see
    test_blocking_endpoints_are_declared_as_plain_def_not_async_def), so
    FastAPI runs concurrent calls to it in its worker threadpool with
    genuine OS-thread parallelism, which is exactly why this reproduces
    far more reliably via plain `threading.Thread`s than the identical
    class of race in upload_notebook (an `async def` on a single event
    loop) needed a deterministic asyncio.gather-driven test for instead.
    Repeated here across several iterations (rather than a single trial)
    since even a ~95% single-trial hit rate leaves a real chance of a
    false pass; failing on any iteration is enough to catch a regression.
    """

    content_a = _notebook_bytes("def a():\n    return 1\n")
    content_b = _notebook_bytes("def b():\n    return 2\n")

    source_a = "rename_race_source_a.ipynb"
    source_b = "rename_race_source_b.ipynb"
    target = "rename_race_target.ipynb"
    target_path = Path(UPLOAD_DIR) / target

    try:

        for _ in range(15):

            for name, content in ((source_a, content_a), (source_b, content_b)):

                if not (Path(UPLOAD_DIR) / name).exists():
                    resp = client.post(
                        "/api/upload",
                        files={"file": (name, io.BytesIO(content), "application/json")},
                    )
                    assert resp.status_code == 200

            if target_path.exists():
                os.remove(target_path)

            results = []

            def do_rename(source_name, tag):
                resp = client.patch(
                    f"/api/notebooks/{source_name}",
                    json={"new_filename": target},
                )
                results.append((tag, resp.status_code))

            t1 = threading.Thread(target=do_rename, args=(source_a, "A"))
            t2 = threading.Thread(target=do_rename, args=(source_b, "B"))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert not t1.is_alive() and not t2.is_alive(), "a request never returned -- deadlock"
            assert len(results) == 2

            statuses = sorted(r[1] for r in results)
            # Exactly one must succeed and the other must be rejected with
            # the same 409 a sequential rename onto an existing filename
            # without "overwrite": true already gets -- never two silent
            # 200s.
            assert statuses == [200, 409], results

            # Whichever source lost the race is still sitting where it
            # started -- the collision was rejected outright, not
            # silently clobbered.
            remaining_sources = [
                name for name in (source_a, source_b)
                if (Path(UPLOAD_DIR) / name).exists()
            ]
            assert len(remaining_sources) == 1
            assert target_path.is_file()

    finally:
        for name in (source_a, source_b, target):
            path = Path(UPLOAD_DIR) / name
            if path.exists():
                os.remove(path)


def test_rename_notebook_overwrites_when_requested():

    content_a = _notebook_bytes("def add(a, b):\n    return a + b\n")
    content_b = _notebook_bytes("def sub(a, b):\n    return a - b\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_overwrite_a.ipynb", io.BytesIO(content_a), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("rename_overwrite_b.ipynb", io.BytesIO(content_b), "application/json")},
    )

    resp = client.patch(
        "/api/notebooks/rename_overwrite_a.ipynb",
        json={"new_filename": "rename_overwrite_b.ipynb", "overwrite": True},
    )

    assert resp.status_code == 200
    assert not (Path(UPLOAD_DIR) / "rename_overwrite_a.ipynb").exists()
    assert json.loads((Path(UPLOAD_DIR) / "rename_overwrite_b.ipynb").read_bytes()) == json.loads(content_a)


def test_rename_notebook_keeps_currently_compiled_tracking_under_the_new_name():
    """The gap this closes: deleting and re-uploading the currently-
    compiled notebook under a new name left .compile_metadata.json's
    "source_notebook" pointing at a path that no longer existed, so every
    uploaded notebook -- including the freshly re-uploaded one -- reported
    "currently_compiled": false afterward, with no way to tell which
    notebook (if any) actually produced what's still running in
    GENERATED_DIR. Renaming in place must not have the same failure mode.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("rename_compiled_source.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "rename_compiled_source.ipynb"}
    )
    assert compile_resp.status_code == 200

    rename_resp = client.patch(
        "/api/notebooks/rename_compiled_source.ipynb",
        json={"new_filename": "rename_compiled_target.ipynb"},
    )

    assert rename_resp.status_code == 200
    assert rename_resp.json()["was_currently_compiled"] is True

    notebooks = {
        nb["filename"]: nb for nb in client.get("/api/notebooks").json()["notebooks"]
    }
    assert notebooks["rename_compiled_target.ipynb"]["currently_compiled"] is True
    assert (
        notebooks["rename_compiled_target.ipynb"]["notebook_changed_since_compile"]
        is False
    )

    os.remove(Path(UPLOAD_DIR) / "rename_compiled_target.ipynb")


def test_rename_notebooks_batch_renames_each_different_source_to_its_own_new_name():

    content_a = _notebook_bytes("def a() -> int:\n    return 1\n")
    content_b = _notebook_bytes("def b() -> int:\n    return 2\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_many_a.ipynb", io.BytesIO(content_a), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("rename_many_b.ipynb", io.BytesIO(content_b), "application/json")},
    )
    client.put("/api/notebooks/rename_many_a.ipynb/tags", json={"tags": ["template"]})

    resp = client.post(
        "/api/notebooks/rename-batch",
        json={
            "entries": [
                {"filename": "rename_many_a.ipynb", "new_filename": "rename_many_a2.ipynb"},
                {"filename": "rename_many_b.ipynb", "new_filename": "rename_many_b2.ipynb"},
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["rename_many_a.ipynb"]["status"] == "success"
    assert results_by_filename["rename_many_a.ipynb"]["new_filename"] == "rename_many_a2.ipynb"
    assert results_by_filename["rename_many_a.ipynb"]["was_currently_compiled"] is False
    assert results_by_filename["rename_many_b.ipynb"]["status"] == "success"
    assert results_by_filename["rename_many_b.ipynb"]["new_filename"] == "rename_many_b2.ipynb"

    assert (Path(UPLOAD_DIR) / "rename_many_a2.ipynb").read_bytes() == content_a
    assert (Path(UPLOAD_DIR) / "rename_many_b2.ipynb").read_bytes() == content_b
    assert not (Path(UPLOAD_DIR) / "rename_many_a.ipynb").exists()
    assert not (Path(UPLOAD_DIR) / "rename_many_b.ipynb").exists()
    # Tags moved along with the rename.
    assert client.get(
        "/api/notebooks/rename_many_a2.ipynb/tags"
    ).json()["tags"] == ["template"]


def test_rename_notebooks_batch_reports_a_bad_entry_without_aborting_the_rest():

    _upload_sample_notebook("rename_many_partial_source.ipynb")

    resp = client.post(
        "/api/notebooks/rename-batch",
        json={
            "entries": [
                {"filename": "rename_many_partial_source.ipynb", "new_filename": "rename_many_partial_target.ipynb"},
                {"filename": "does_not_exist.ipynb", "new_filename": "whatever.ipynb"},
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["rename_many_partial_source.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]

    assert (Path(UPLOAD_DIR) / "rename_many_partial_target.ipynb").is_file()


def test_rename_notebooks_batch_per_entry_overwrite_does_not_apply_to_other_entries():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_many_overwrite_source_a.ipynb", io.BytesIO(content), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("rename_many_overwrite_source_b.ipynb", io.BytesIO(content), "application/json")},
    )
    _upload_sample_notebook("rename_many_overwrite_existing_a.ipynb")
    _upload_sample_notebook("rename_many_overwrite_existing_b.ipynb")

    resp = client.post(
        "/api/notebooks/rename-batch",
        json={
            "entries": [
                {
                    "filename": "rename_many_overwrite_source_a.ipynb",
                    "new_filename": "rename_many_overwrite_existing_a.ipynb",
                    "overwrite": True,
                },
                {
                    "filename": "rename_many_overwrite_source_b.ipynb",
                    "new_filename": "rename_many_overwrite_existing_b.ipynb",
                },
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results = {r["filename"]: r for r in body["results"]}
    assert results["rename_many_overwrite_source_a.ipynb"]["status"] == "success"
    assert results["rename_many_overwrite_source_b.ipynb"]["status"] == "error"
    assert "already exists" in results["rename_many_overwrite_source_b.ipynb"]["detail"]

    assert (Path(UPLOAD_DIR) / "rename_many_overwrite_existing_a.ipynb").read_bytes() == content
    # The source that failed to rename was left in place.
    assert (Path(UPLOAD_DIR) / "rename_many_overwrite_source_b.ipynb").is_file()


def test_rename_notebooks_batch_keeps_currently_compiled_tracking_under_the_new_name():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("rename_many_compiled_source.ipynb", io.BytesIO(content), "application/json")},
    )
    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "rename_many_compiled_source.ipynb"}
    )
    assert compile_resp.status_code == 200

    resp = client.post(
        "/api/notebooks/rename-batch",
        json={
            "entries": [
                {
                    "filename": "rename_many_compiled_source.ipynb",
                    "new_filename": "rename_many_compiled_target.ipynb",
                },
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["was_currently_compiled"] is True

    notebooks = {
        nb["filename"]: nb for nb in client.get("/api/notebooks").json()["notebooks"]
    }
    assert notebooks["rename_many_compiled_target.ipynb"]["currently_compiled"] is True

    os.remove(Path(UPLOAD_DIR) / "rename_many_compiled_target.ipynb")


def test_rename_notebooks_batch_dry_run_reports_the_plan_without_renaming():

    _upload_sample_notebook("rename_many_dry_run_a.ipynb")
    _upload_sample_notebook("rename_many_dry_run_b.ipynb")
    _upload_sample_notebook("rename_many_dry_run_conflict.ipynb")

    resp = client.post(
        "/api/notebooks/rename-batch",
        json={
            "entries": [
                {"filename": "rename_many_dry_run_a.ipynb", "new_filename": "rename_many_dry_run_new.ipynb"},
                {"filename": "rename_many_dry_run_b.ipynb", "new_filename": "rename_many_dry_run_conflict.ipynb"},
            ],
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results = {r["filename"]: r for r in body["results"]}
    assert results["rename_many_dry_run_a.ipynb"]["status"] == "success"
    assert results["rename_many_dry_run_b.ipynb"]["status"] == "error"

    # Nothing was actually renamed.
    assert (Path(UPLOAD_DIR) / "rename_many_dry_run_a.ipynb").is_file()
    assert (Path(UPLOAD_DIR) / "rename_many_dry_run_b.ipynb").is_file()
    assert (Path(UPLOAD_DIR) / "rename_many_dry_run_conflict.ipynb").is_file()
    assert not (Path(UPLOAD_DIR) / "rename_many_dry_run_new.ipynb").exists()


def test_rename_notebooks_batch_rejects_a_non_list_entries_value():

    resp = client.post(
        "/api/notebooks/rename-batch",
        json={"entries": "not-a-list"},
    )

    assert resp.status_code == 400


def test_rename_notebooks_batch_rejects_an_empty_entries_list():

    resp = client.post(
        "/api/notebooks/rename-batch",
        json={"entries": []},
    )

    assert resp.status_code == 400


def test_rename_notebooks_batch_rejects_more_entries_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/notebooks/rename-batch",
        json={
            "entries": [
                {"filename": "a.ipynb", "new_filename": "a2.ipynb"},
                {"filename": "b.ipynb", "new_filename": "b2.ipynb"},
            ]
        },
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_rename_notebooks_batch_rejects_an_entry_missing_new_filename():

    _upload_sample_notebook("rename_many_missing_field.ipynb")

    resp = client.post(
        "/api/notebooks/rename-batch",
        json={"entries": [{"filename": "rename_many_missing_field.ipynb"}]},
    )

    assert resp.status_code == 400


def _upload_sample_notebook(filename):
    resp = client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert resp.status_code == 200


def test_copy_notebook_duplicates_the_file_and_keeps_the_source():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("copy_source.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    copy_resp = client.post(
        "/api/notebooks/copy_source.ipynb/copy",
        json={"new_filename": "copy_target.ipynb"},
    )

    assert copy_resp.status_code == 200
    assert copy_resp.json() == {
        "status": "success",
        "dry_run": False,
        "filename": "copy_source.ipynb",
        "new_filename": "copy_target.ipynb",
    }

    assert (Path(UPLOAD_DIR) / "copy_source.ipynb").is_file()
    assert (Path(UPLOAD_DIR) / "copy_target.ipynb").is_file()
    assert (
        (Path(UPLOAD_DIR) / "copy_source.ipynb").read_bytes()
        == (Path(UPLOAD_DIR) / "copy_target.ipynb").read_bytes()
    )

    filenames = {nb["filename"] for nb in client.get("/api/notebooks").json()["notebooks"]}
    assert "copy_source.ipynb" in filenames
    assert "copy_target.ipynb" in filenames

    os.remove(Path(UPLOAD_DIR) / "copy_target.ipynb")


def test_copy_notebook_returns_404_for_missing_source():

    resp = client.post(
        "/api/notebooks/does_not_exist_at_all.ipynb/copy",
        json={"new_filename": "whatever.ipynb"},
    )

    assert resp.status_code == 404


def test_copy_notebook_requires_new_filename():

    _upload_sample_notebook("copy_missing_target.ipynb")

    resp = client.post(
        "/api/notebooks/copy_missing_target.ipynb/copy",
        json={},
    )

    assert resp.status_code == 400


def test_copy_notebook_rejects_a_non_ipynb_target_name():

    _upload_sample_notebook("copy_bad_ext.ipynb")

    resp = client.post(
        "/api/notebooks/copy_bad_ext.ipynb/copy",
        json={"new_filename": "copy_bad_ext.txt"},
    )

    assert resp.status_code == 400


def test_copy_notebook_rejects_a_traversal_target_name():

    _upload_sample_notebook("copy_traversal_source.ipynb")

    resp = client.post(
        "/api/notebooks/copy_traversal_source.ipynb/copy",
        json={"new_filename": "../../../../etc/passwd.ipynb"},
    )

    assert resp.status_code == 400


def test_copy_notebook_rejects_copying_onto_its_own_name():

    _upload_sample_notebook("copy_self.ipynb")

    resp = client.post(
        "/api/notebooks/copy_self.ipynb/copy",
        json={"new_filename": "copy_self.ipynb"},
    )

    assert resp.status_code == 400


def test_copy_notebook_rejects_collision_without_overwrite():

    _upload_sample_notebook("copy_collision_source.ipynb")
    _upload_sample_notebook("copy_collision_target.ipynb")

    resp = client.post(
        "/api/notebooks/copy_collision_source.ipynb/copy",
        json={"new_filename": "copy_collision_target.ipynb"},
    )

    assert resp.status_code == 409
    os.remove(Path(UPLOAD_DIR) / "copy_collision_target.ipynb")


def test_copy_notebook_overwrites_when_requested():

    content_a = _notebook_bytes("def a() -> int:\n    return 1\n")
    content_b = _notebook_bytes("def b() -> int:\n    return 2\n")

    client.post(
        "/api/upload",
        files={"file": ("copy_overwrite_source.ipynb", io.BytesIO(content_a), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("copy_overwrite_target.ipynb", io.BytesIO(content_b), "application/json")},
    )

    copy_resp = client.post(
        "/api/notebooks/copy_overwrite_source.ipynb/copy",
        json={"new_filename": "copy_overwrite_target.ipynb", "overwrite": True},
    )

    assert copy_resp.status_code == 200
    assert (
        (Path(UPLOAD_DIR) / "copy_overwrite_target.ipynb").read_bytes() == content_a
    )

    os.remove(Path(UPLOAD_DIR) / "copy_overwrite_target.ipynb")


def test_copy_notebook_copies_tags_from_the_source():

    _upload_sample_notebook("copy_tags_source.ipynb")
    client.put(
        "/api/notebooks/copy_tags_source.ipynb/tags", json={"tags": ["bug"]}
    )

    copy_resp = client.post(
        "/api/notebooks/copy_tags_source.ipynb/copy",
        json={"new_filename": "copy_tags_target.ipynb"},
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/copy_tags_source.ipynb/tags"
    ).json()["tags"] == ["bug"]
    assert client.get(
        "/api/notebooks/copy_tags_target.ipynb/tags"
    ).json()["tags"] == ["bug"]

    os.remove(Path(UPLOAD_DIR) / "copy_tags_target.ipynb")
    _tags_sidecar_path("copy_tags_target.ipynb").unlink(missing_ok=True)


def test_copy_notebook_overwrite_discards_the_destinations_previous_tags():

    _upload_sample_notebook("copy_tags_overwrite_source.ipynb")
    _upload_sample_notebook("copy_tags_overwrite_target.ipynb")
    client.put(
        "/api/notebooks/copy_tags_overwrite_target.ipynb/tags",
        json={"tags": ["stale"]},
    )

    copy_resp = client.post(
        "/api/notebooks/copy_tags_overwrite_source.ipynb/copy",
        json={
            "new_filename": "copy_tags_overwrite_target.ipynb",
            "overwrite": True,
        },
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/copy_tags_overwrite_target.ipynb/tags"
    ).json()["tags"] == []

    os.remove(Path(UPLOAD_DIR) / "copy_tags_overwrite_target.ipynb")


def test_copy_notebook_overrides_tags_and_description_instead_of_inheriting():

    _upload_sample_notebook("copy_override_source.ipynb")
    client.put(
        "/api/notebooks/copy_override_source.ipynb/tags", json={"tags": ["production"]}
    )
    client.put(
        "/api/notebooks/copy_override_source.ipynb/description",
        json={"description": "the prod one"},
    )

    copy_resp = client.post(
        "/api/notebooks/copy_override_source.ipynb/copy",
        json={
            "new_filename": "copy_override_target.ipynb",
            "tags": ["scratch"],
            "description": "a scratch copy",
        },
    )
    assert copy_resp.status_code == 200

    target_info = client.get("/api/notebooks/copy_override_target.ipynb/info").json()
    assert target_info["tags"] == ["scratch"]
    assert target_info["description"] == "a scratch copy"

    # The source itself is untouched.
    source_info = client.get("/api/notebooks/copy_override_source.ipynb/info").json()
    assert source_info["tags"] == ["production"]
    assert source_info["description"] == "the prod one"

    os.remove(Path(UPLOAD_DIR) / "copy_override_target.ipynb")
    _tags_sidecar_path("copy_override_target.ipynb").unlink(missing_ok=True)
    _description_sidecar_path("copy_override_target.ipynb").unlink(missing_ok=True)


def test_copy_notebook_rejects_an_invalid_tags_override():

    _upload_sample_notebook("copy_bad_tags_source.ipynb")

    resp = client.post(
        "/api/notebooks/copy_bad_tags_source.ipynb/copy",
        json={"new_filename": "copy_bad_tags_target.ipynb", "tags": "not-a-list"},
    )

    assert resp.status_code == 400
    assert not (Path(UPLOAD_DIR) / "copy_bad_tags_target.ipynb").exists()


def test_copy_notebook_dry_run_reports_the_new_filename_without_copying():

    _upload_sample_notebook("copy_dry_run_source.ipynb")

    resp = client.post(
        "/api/notebooks/copy_dry_run_source.ipynb/copy",
        json={"new_filename": "copy_dry_run_target.ipynb", "dry_run": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["new_filename"] == "copy_dry_run_target.ipynb"

    # Nothing was actually copied.
    assert not (Path(UPLOAD_DIR) / "copy_dry_run_target.ipynb").exists()


def test_copy_notebook_dry_run_still_reports_a_same_name_collision():

    _upload_sample_notebook("copy_dry_run_collision_source.ipynb")
    _upload_sample_notebook("copy_dry_run_collision_target.ipynb")

    resp = client.post(
        "/api/notebooks/copy_dry_run_collision_source.ipynb/copy",
        json={"new_filename": "copy_dry_run_collision_target.ipynb", "dry_run": True},
    )

    assert resp.status_code == 409


def test_copy_notebook_does_not_copy_version_history():

    _upload_sample_notebook("copy_versions_source.ipynb")
    # Overwriting a notebook snapshots its previous content -- see
    # _snapshot_current_notebook_version -- giving copy_versions_source.ipynb
    # a non-empty version history to (deliberately) not copy.
    client.post(
        "/api/upload",
        files={
            "file": (
                "copy_versions_source.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 2\n")),
                "application/json",
            )
        },
        params={"overwrite": "true"},
    )
    assert client.get(
        "/api/notebooks/copy_versions_source.ipynb/versions"
    ).json()["versions"]

    copy_resp = client.post(
        "/api/notebooks/copy_versions_source.ipynb/copy",
        json={"new_filename": "copy_versions_target.ipynb"},
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/copy_versions_target.ipynb/versions"
    ).json()["versions"] == []

    os.remove(Path(UPLOAD_DIR) / "copy_versions_target.ipynb")


def test_copy_notebook_overwrite_discards_the_destinations_previous_version_history():

    _upload_sample_notebook("copy_versions_overwrite_source.ipynb")
    _upload_sample_notebook("copy_versions_overwrite_target.ipynb")
    client.post(
        "/api/upload",
        files={
            "file": (
                "copy_versions_overwrite_target.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 2\n")),
                "application/json",
            )
        },
        params={"overwrite": "true"},
    )
    assert client.get(
        "/api/notebooks/copy_versions_overwrite_target.ipynb/versions"
    ).json()["versions"]

    copy_resp = client.post(
        "/api/notebooks/copy_versions_overwrite_source.ipynb/copy",
        json={
            "new_filename": "copy_versions_overwrite_target.ipynb",
            "overwrite": True,
        },
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/copy_versions_overwrite_target.ipynb/versions"
    ).json()["versions"] == []

    os.remove(Path(UPLOAD_DIR) / "copy_versions_overwrite_target.ipynb")


def test_copy_notebook_does_not_affect_the_currently_compiled_source():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("copy_compiled_source.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "copy_compiled_source.ipynb"}
    )
    assert compile_resp.status_code == 200

    copy_resp = client.post(
        "/api/notebooks/copy_compiled_source.ipynb/copy",
        json={"new_filename": "copy_compiled_target.ipynb"},
    )
    assert copy_resp.status_code == 200

    notebooks = {
        nb["filename"]: nb for nb in client.get("/api/notebooks").json()["notebooks"]
    }
    assert notebooks["copy_compiled_source.ipynb"]["currently_compiled"] is True
    assert notebooks["copy_compiled_target.ipynb"]["currently_compiled"] is False

    os.remove(Path(UPLOAD_DIR) / "copy_compiled_target.ipynb")


def test_copy_notebook_batch_duplicates_the_source_under_every_new_filename():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("copy_batch_source.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200
    client.put(
        "/api/notebooks/copy_batch_source.ipynb/tags", json={"tags": ["template"]}
    )

    resp = client.post(
        "/api/notebooks/copy_batch_source.ipynb/copy-batch",
        json={"new_filenames": ["copy_batch_a.ipynb", "copy_batch_b.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["filename"] == "copy_batch_source.ipynb"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0
    assert [r["new_filename"] for r in body["results"]] == [
        "copy_batch_a.ipynb", "copy_batch_b.ipynb",
    ]
    assert all(r["status"] == "success" for r in body["results"])

    assert (Path(UPLOAD_DIR) / "copy_batch_a.ipynb").read_bytes() == content
    assert (Path(UPLOAD_DIR) / "copy_batch_b.ipynb").read_bytes() == content
    # Source's own tags are inherited by each copy.
    assert client.get(
        "/api/notebooks/copy_batch_a.ipynb/tags"
    ).json()["tags"] == ["template"]
    # Source itself is untouched.
    assert (Path(UPLOAD_DIR) / "copy_batch_source.ipynb").read_bytes() == content


def test_copy_notebook_batch_reports_a_collision_without_aborting_the_rest():

    _upload_sample_notebook("copy_batch_partial_source.ipynb")
    _upload_sample_notebook("copy_batch_partial_existing.ipynb")

    resp = client.post(
        "/api/notebooks/copy_batch_partial_source.ipynb/copy-batch",
        json={
            "new_filenames": [
                "copy_batch_partial_new.ipynb", "copy_batch_partial_existing.ipynb",
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["new_filename"]: r for r in body["results"]}
    assert results_by_filename["copy_batch_partial_new.ipynb"]["status"] == "success"
    assert results_by_filename["copy_batch_partial_existing.ipynb"]["status"] == "error"
    assert "already exists" in results_by_filename["copy_batch_partial_existing.ipynb"]["detail"]

    assert (Path(UPLOAD_DIR) / "copy_batch_partial_new.ipynb").is_file()


def test_copy_notebook_batch_overwrite_applies_to_every_destination():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("copy_batch_overwrite_source.ipynb", io.BytesIO(content), "application/json")},
    )
    _upload_sample_notebook("copy_batch_overwrite_existing.ipynb")

    resp = client.post(
        "/api/notebooks/copy_batch_overwrite_source.ipynb/copy-batch",
        json={
            "new_filenames": ["copy_batch_overwrite_existing.ipynb"],
            "overwrite": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert (Path(UPLOAD_DIR) / "copy_batch_overwrite_existing.ipynb").read_bytes() == content


def test_copy_notebook_batch_tags_and_description_apply_to_every_destination():

    _upload_sample_notebook("copy_batch_override_source.ipynb")
    client.put(
        "/api/notebooks/copy_batch_override_source.ipynb/tags",
        json={"tags": ["production"]},
    )

    resp = client.post(
        "/api/notebooks/copy_batch_override_source.ipynb/copy-batch",
        json={
            "new_filenames": [
                "copy_batch_override_a.ipynb", "copy_batch_override_b.ipynb",
            ],
            "tags": ["scratch"],
            "description": "batch scratch copy",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["succeeded_count"] == 2

    for filename in ("copy_batch_override_a.ipynb", "copy_batch_override_b.ipynb"):

        info = client.get(f"/api/notebooks/{filename}/info").json()
        assert info["tags"] == ["scratch"]
        assert info["description"] == "batch scratch copy"

        os.remove(Path(UPLOAD_DIR) / filename)
        _tags_sidecar_path(filename).unlink(missing_ok=True)
        _description_sidecar_path(filename).unlink(missing_ok=True)


def test_copy_notebook_batch_returns_404_for_missing_source():

    resp = client.post(
        "/api/notebooks/does_not_exist_at_all.ipynb/copy-batch",
        json={"new_filenames": ["whatever.ipynb"]},
    )

    assert resp.status_code == 404


def test_copy_notebook_batch_rejects_a_non_list_new_filenames_value():

    _upload_sample_notebook("copy_batch_bad_input.ipynb")

    resp = client.post(
        "/api/notebooks/copy_batch_bad_input.ipynb/copy-batch",
        json={"new_filenames": "not-a-list"},
    )

    assert resp.status_code == 400


def test_copy_notebook_batch_rejects_an_empty_new_filenames_list():

    _upload_sample_notebook("copy_batch_empty_input.ipynb")

    resp = client.post(
        "/api/notebooks/copy_batch_empty_input.ipynb/copy-batch",
        json={"new_filenames": []},
    )

    assert resp.status_code == 400


def test_copy_notebook_batch_rejects_more_new_filenames_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    _upload_sample_notebook("copy_batch_too_many_input.ipynb")

    resp = client.post(
        "/api/notebooks/copy_batch_too_many_input.ipynb/copy-batch",
        json={"new_filenames": ["a.ipynb", "b.ipynb"]},
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_copy_notebook_batch_dry_run_reports_the_plan_without_copying():

    _upload_sample_notebook("copy_batch_dry_run_source.ipynb")
    _upload_sample_notebook("copy_batch_dry_run_existing.ipynb")

    resp = client.post(
        "/api/notebooks/copy_batch_dry_run_source.ipynb/copy-batch",
        json={
            "new_filenames": [
                "copy_batch_dry_run_new.ipynb",
                "copy_batch_dry_run_existing.ipynb",
            ],
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results = {r["new_filename"]: r for r in body["results"]}
    assert results["copy_batch_dry_run_new.ipynb"]["status"] == "success"
    assert results["copy_batch_dry_run_existing.ipynb"]["status"] == "error"

    # Nothing was actually copied.
    assert not (Path(UPLOAD_DIR) / "copy_batch_dry_run_new.ipynb").exists()


def test_copy_notebook_batch_does_not_copy_version_history():

    filename = "copy_batch_versions_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    assert len(client.get(f"/api/notebooks/{filename}/versions").json()["versions"]) == 1

    resp = client.post(
        f"/api/notebooks/{filename}/copy-batch",
        json={"new_filenames": ["copy_batch_versions_target.ipynb"]},
    )

    assert resp.status_code == 200
    assert client.get(
        "/api/notebooks/copy_batch_versions_target.ipynb/versions"
    ).json()["versions"] == []


def test_copy_notebooks_batch_duplicates_each_different_source_to_its_own_destination():

    content_a = _notebook_bytes("def a() -> int:\n    return 1\n")
    content_b = _notebook_bytes("def b() -> int:\n    return 2\n")

    client.post(
        "/api/upload",
        files={"file": ("copy_many_a.ipynb", io.BytesIO(content_a), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("copy_many_b.ipynb", io.BytesIO(content_b), "application/json")},
    )
    client.put("/api/notebooks/copy_many_a.ipynb/tags", json={"tags": ["template"]})

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={
            "entries": [
                {"filename": "copy_many_a.ipynb", "new_filename": "copy_many_a_copy.ipynb"},
                {"filename": "copy_many_b.ipynb", "new_filename": "copy_many_b_copy.ipynb"},
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["copy_many_a.ipynb"]["status"] == "success"
    assert results_by_filename["copy_many_a.ipynb"]["new_filename"] == "copy_many_a_copy.ipynb"
    assert results_by_filename["copy_many_b.ipynb"]["status"] == "success"
    assert results_by_filename["copy_many_b.ipynb"]["new_filename"] == "copy_many_b_copy.ipynb"

    assert (Path(UPLOAD_DIR) / "copy_many_a_copy.ipynb").read_bytes() == content_a
    assert (Path(UPLOAD_DIR) / "copy_many_b_copy.ipynb").read_bytes() == content_b
    # Source's own tags are inherited by its copy.
    assert client.get(
        "/api/notebooks/copy_many_a_copy.ipynb/tags"
    ).json()["tags"] == ["template"]
    # Sources themselves are untouched.
    assert (Path(UPLOAD_DIR) / "copy_many_a.ipynb").read_bytes() == content_a
    assert (Path(UPLOAD_DIR) / "copy_many_b.ipynb").read_bytes() == content_b


def test_copy_notebooks_batch_reports_a_bad_entry_without_aborting_the_rest():

    _upload_sample_notebook("copy_many_partial_source.ipynb")
    _upload_sample_notebook("copy_many_partial_existing.ipynb")

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={
            "entries": [
                {"filename": "copy_many_partial_source.ipynb", "new_filename": "copy_many_partial_new.ipynb"},
                {"filename": "does_not_exist.ipynb", "new_filename": "copy_many_partial_existing.ipynb"},
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["copy_many_partial_source.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]

    assert (Path(UPLOAD_DIR) / "copy_many_partial_new.ipynb").is_file()


def test_copy_notebooks_batch_per_entry_overwrite_does_not_apply_to_other_entries():

    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("copy_many_overwrite_source.ipynb", io.BytesIO(content), "application/json")},
    )
    _upload_sample_notebook("copy_many_overwrite_existing_a.ipynb")
    _upload_sample_notebook("copy_many_overwrite_existing_b.ipynb")

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={
            "entries": [
                {
                    "filename": "copy_many_overwrite_source.ipynb",
                    "new_filename": "copy_many_overwrite_existing_a.ipynb",
                    "overwrite": True,
                },
                {
                    "filename": "copy_many_overwrite_source.ipynb",
                    "new_filename": "copy_many_overwrite_existing_b.ipynb",
                },
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results = {r["new_filename"]: r for r in body["results"]}
    assert results["copy_many_overwrite_existing_a.ipynb"]["status"] == "success"
    assert results["copy_many_overwrite_existing_b.ipynb"]["status"] == "error"
    assert "already exists" in results["copy_many_overwrite_existing_b.ipynb"]["detail"]

    assert (Path(UPLOAD_DIR) / "copy_many_overwrite_existing_a.ipynb").read_bytes() == content


def test_copy_notebooks_batch_per_entry_tags_and_description_do_not_apply_to_other_entries():

    _upload_sample_notebook("copy_many_override_source_a.ipynb")
    _upload_sample_notebook("copy_many_override_source_b.ipynb")

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={
            "entries": [
                {
                    "filename": "copy_many_override_source_a.ipynb",
                    "new_filename": "copy_many_override_target_a.ipynb",
                    "tags": ["alpha"],
                    "description": "entry a",
                },
                {
                    "filename": "copy_many_override_source_b.ipynb",
                    "new_filename": "copy_many_override_target_b.ipynb",
                },
            ]
        },
    )

    assert resp.status_code == 200
    assert resp.json()["succeeded_count"] == 2

    info_a = client.get("/api/notebooks/copy_many_override_target_a.ipynb/info").json()
    assert info_a["tags"] == ["alpha"]
    assert info_a["description"] == "entry a"

    info_b = client.get("/api/notebooks/copy_many_override_target_b.ipynb/info").json()
    assert info_b["tags"] == []
    assert info_b["description"] == ""

    for filename in (
        "copy_many_override_target_a.ipynb", "copy_many_override_target_b.ipynb",
    ):
        os.remove(Path(UPLOAD_DIR) / filename)
        _tags_sidecar_path(filename).unlink(missing_ok=True)
        _description_sidecar_path(filename).unlink(missing_ok=True)


def test_copy_notebooks_batch_bad_entry_tags_is_reported_without_aborting_the_rest():

    _upload_sample_notebook("copy_many_bad_tags_source_a.ipynb")
    _upload_sample_notebook("copy_many_bad_tags_source_b.ipynb")

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={
            "entries": [
                {
                    "filename": "copy_many_bad_tags_source_a.ipynb",
                    "new_filename": "copy_many_bad_tags_target_a.ipynb",
                    "tags": "not-a-list",
                },
                {
                    "filename": "copy_many_bad_tags_source_b.ipynb",
                    "new_filename": "copy_many_bad_tags_target_b.ipynb",
                },
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results = {r["new_filename"]: r for r in body["results"]}
    assert results["copy_many_bad_tags_target_a.ipynb"]["status"] == "error"
    assert results["copy_many_bad_tags_target_b.ipynb"]["status"] == "success"
    assert not (Path(UPLOAD_DIR) / "copy_many_bad_tags_target_a.ipynb").exists()

    os.remove(Path(UPLOAD_DIR) / "copy_many_bad_tags_target_b.ipynb")
    _tags_sidecar_path("copy_many_bad_tags_target_b.ipynb").unlink(missing_ok=True)
    _description_sidecar_path("copy_many_bad_tags_target_b.ipynb").unlink(missing_ok=True)


def test_copy_notebooks_batch_rejects_a_non_list_entries_value():

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={"entries": "not-a-list"},
    )

    assert resp.status_code == 400


def test_copy_notebooks_batch_rejects_an_empty_entries_list():

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={"entries": []},
    )

    assert resp.status_code == 400


def test_copy_notebooks_batch_rejects_more_entries_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={
            "entries": [
                {"filename": "a.ipynb", "new_filename": "a2.ipynb"},
                {"filename": "b.ipynb", "new_filename": "b2.ipynb"},
            ]
        },
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_copy_notebooks_batch_rejects_an_entry_missing_new_filename():

    _upload_sample_notebook("copy_many_missing_field.ipynb")

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={"entries": [{"filename": "copy_many_missing_field.ipynb"}]},
    )

    assert resp.status_code == 400


def test_copy_notebooks_batch_dry_run_reports_the_plan_without_copying():

    _upload_sample_notebook("copy_many_dry_run_a.ipynb")
    _upload_sample_notebook("copy_many_dry_run_existing.ipynb")

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={
            "entries": [
                {"filename": "copy_many_dry_run_a.ipynb", "new_filename": "copy_many_dry_run_new.ipynb"},
                {"filename": "copy_many_dry_run_a.ipynb", "new_filename": "copy_many_dry_run_existing.ipynb"},
            ],
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results = {r["new_filename"]: r for r in body["results"]}
    assert results["copy_many_dry_run_new.ipynb"]["status"] == "success"
    assert results["copy_many_dry_run_existing.ipynb"]["status"] == "error"

    # Nothing was actually copied.
    assert not (Path(UPLOAD_DIR) / "copy_many_dry_run_new.ipynb").exists()


def test_copy_notebooks_batch_does_not_copy_version_history():

    filename = "copy_many_versions_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    assert len(client.get(f"/api/notebooks/{filename}/versions").json()["versions"]) == 1

    resp = client.post(
        "/api/notebooks/copy-batch",
        json={"entries": [{"filename": filename, "new_filename": "copy_many_versions_target.ipynb"}]},
    )

    assert resp.status_code == 200
    assert client.get(
        "/api/notebooks/copy_many_versions_target.ipynb/versions"
    ).json()["versions"] == []


def test_get_notebook_tags_is_empty_for_a_never_tagged_notebook():

    _upload_sample_notebook("tags_untagged.ipynb")

    resp = client.get("/api/notebooks/tags_untagged.ipynb/tags")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "filename": "tags_untagged.ipynb",
        "tags": [],
    }


def test_get_notebook_tags_returns_404_for_missing_file():

    resp = client.get("/api/notebooks/tags_does_not_exist.ipynb/tags")

    assert resp.status_code == 404


def test_set_notebook_tags_returns_404_for_missing_file():

    resp = client.put(
        "/api/notebooks/tags_does_not_exist.ipynb/tags",
        json={"tags": ["bug"]},
    )

    assert resp.status_code == 404


def test_set_notebook_tags_persists_and_is_readable_back():

    _upload_sample_notebook("tags_persist.ipynb")

    set_resp = client.put(
        "/api/notebooks/tags_persist.ipynb/tags",
        json={"tags": ["production", "bug"]},
    )

    assert set_resp.status_code == 200
    assert set_resp.json() == {
        "status": "success",
        "dry_run": False,
        "filename": "tags_persist.ipynb",
        "tags": ["bug", "production"],
    }

    get_resp = client.get("/api/notebooks/tags_persist.ipynb/tags")

    assert get_resp.json()["tags"] == ["bug", "production"]


def test_set_notebook_tags_dry_run_reports_the_normalized_tags_without_writing():

    _upload_sample_notebook("tags_dry_run.ipynb")
    client.put("/api/notebooks/tags_dry_run.ipynb/tags", json={"tags": ["stale"]})

    resp = client.put(
        "/api/notebooks/tags_dry_run.ipynb/tags",
        json={"tags": [" Production ", "Production"], "dry_run": True},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "dry_run": True,
        "filename": "tags_dry_run.ipynb",
        "tags": ["Production"],
    }

    # Nothing was actually written.
    assert client.get(
        "/api/notebooks/tags_dry_run.ipynb/tags"
    ).json()["tags"] == ["stale"]


def test_set_notebook_tags_strips_whitespace_and_deduplicates():

    _upload_sample_notebook("tags_dedupe.ipynb")

    resp = client.put(
        "/api/notebooks/tags_dedupe.ipynb/tags",
        json={"tags": ["bug", "  bug  ", "feature"]},
    )

    assert resp.status_code == 200
    assert resp.json()["tags"] == ["bug", "feature"]


def test_set_notebook_tags_with_empty_list_clears_tags_and_removes_the_sidecar_file():

    _upload_sample_notebook("tags_clear.ipynb")

    client.put("/api/notebooks/tags_clear.ipynb/tags", json={"tags": ["bug"]})
    assert _tags_sidecar_path("tags_clear.ipynb").is_file()

    clear_resp = client.put("/api/notebooks/tags_clear.ipynb/tags", json={"tags": []})

    assert clear_resp.status_code == 200
    assert clear_resp.json()["tags"] == []
    assert not _tags_sidecar_path("tags_clear.ipynb").is_file()


def test_set_notebook_tags_rejects_a_non_list_tags_value():

    _upload_sample_notebook("tags_not_a_list.ipynb")

    resp = client.put(
        "/api/notebooks/tags_not_a_list.ipynb/tags",
        json={"tags": "bug"},
    )

    assert resp.status_code == 400


def test_set_notebook_tags_rejects_a_non_string_tag():

    _upload_sample_notebook("tags_non_string.ipynb")

    resp = client.put(
        "/api/notebooks/tags_non_string.ipynb/tags",
        json={"tags": ["bug", 5]},
    )

    assert resp.status_code == 400


def test_set_notebook_tags_rejects_an_empty_or_whitespace_only_tag():

    _upload_sample_notebook("tags_blank.ipynb")

    resp = client.put(
        "/api/notebooks/tags_blank.ipynb/tags",
        json={"tags": ["   "]},
    )

    assert resp.status_code == 400


def test_set_notebook_tags_rejects_a_tag_over_the_max_length():

    _upload_sample_notebook("tags_too_long.ipynb")

    resp = client.put(
        "/api/notebooks/tags_too_long.ipynb/tags",
        json={"tags": ["x" * 51]},
    )

    assert resp.status_code == 400


def test_set_notebook_tags_rejects_more_than_the_max_distinct_tags():

    _upload_sample_notebook("tags_too_many.ipynb")

    resp = client.put(
        "/api/notebooks/tags_too_many.ipynb/tags",
        json={"tags": [f"tag{i}" for i in range(21)]},
    )

    assert resp.status_code == 400


def test_set_notebook_tags_batch_sets_each_notebooks_own_distinct_tags():

    _upload_sample_notebook("tags_batch_a.ipynb")
    _upload_sample_notebook("tags_batch_b.ipynb")
    client.put(
        "/api/notebooks/tags_batch_a.ipynb/tags", json={"tags": ["stale"]}
    )

    resp = client.post(
        "/api/notebooks/tags-batch",
        json={
            "entries": [
                {"filename": "tags_batch_a.ipynb", "tags": ["production", "v2"]},
                {"filename": "tags_batch_b.ipynb", "tags": ["bug"]},
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["tags_batch_a.ipynb"]["status"] == "success"
    assert results_by_filename["tags_batch_a.ipynb"]["tags"] == ["production", "v2"]
    assert results_by_filename["tags_batch_b.ipynb"]["tags"] == ["bug"]

    # A full replace, not a merge -- "stale" is gone.
    assert client.get(
        "/api/notebooks/tags_batch_a.ipynb/tags"
    ).json()["tags"] == ["production", "v2"]


def test_set_notebook_tags_batch_dry_run_reports_the_plan_without_writing():

    _upload_sample_notebook("tags_batch_dry_run.ipynb")
    client.put(
        "/api/notebooks/tags_batch_dry_run.ipynb/tags", json={"tags": ["stale"]}
    )

    resp = client.post(
        "/api/notebooks/tags-batch",
        json={
            "entries": [
                {"filename": "tags_batch_dry_run.ipynb", "tags": ["production"]},
                {"filename": "does_not_exist.ipynb", "tags": ["x"]},
            ],
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["tags_batch_dry_run.ipynb"]["status"] == "success"
    assert results_by_filename["tags_batch_dry_run.ipynb"]["tags"] == ["production"]

    # Nothing was actually written.
    assert client.get(
        "/api/notebooks/tags_batch_dry_run.ipynb/tags"
    ).json()["tags"] == ["stale"]


def test_set_notebook_tags_batch_with_an_empty_tags_list_clears_that_entry():

    _upload_sample_notebook("tags_batch_clear.ipynb")
    client.put("/api/notebooks/tags_batch_clear.ipynb/tags", json={"tags": ["bug"]})

    resp = client.post(
        "/api/notebooks/tags-batch",
        json={"entries": [{"filename": "tags_batch_clear.ipynb", "tags": []}]},
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["tags"] == []
    assert client.get(
        "/api/notebooks/tags_batch_clear.ipynb/tags"
    ).json()["tags"] == []


def test_set_notebook_tags_batch_reports_a_missing_filename_without_aborting_the_rest():

    _upload_sample_notebook("tags_batch_partial.ipynb")

    resp = client.post(
        "/api/notebooks/tags-batch",
        json={
            "entries": [
                {"filename": "tags_batch_partial.ipynb", "tags": ["urgent"]},
                {"filename": "does_not_exist.ipynb", "tags": ["urgent"]},
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["tags_batch_partial.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]


def test_set_notebook_tags_batch_reports_an_invalid_tags_value_for_just_that_entry():

    _upload_sample_notebook("tags_batch_bad_tag_a.ipynb")
    _upload_sample_notebook("tags_batch_bad_tag_b.ipynb")

    resp = client.post(
        "/api/notebooks/tags-batch",
        json={
            "entries": [
                {"filename": "tags_batch_bad_tag_a.ipynb", "tags": ["ok"]},
                {"filename": "tags_batch_bad_tag_b.ipynb", "tags": ["   "]},
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["tags_batch_bad_tag_a.ipynb"]["status"] == "success"
    assert results_by_filename["tags_batch_bad_tag_b.ipynb"]["status"] == "error"
    assert "whitespace-only" in results_by_filename["tags_batch_bad_tag_b.ipynb"]["detail"]

    # The failing entry never got its tags touched.
    assert client.get(
        "/api/notebooks/tags_batch_bad_tag_b.ipynb/tags"
    ).json()["tags"] == []


def test_set_notebook_tags_batch_rejects_a_non_list_entries_value():

    resp = client.post("/api/notebooks/tags-batch", json={"entries": "not-a-list"})

    assert resp.status_code == 400


def test_set_notebook_tags_batch_rejects_an_empty_entries_list():

    resp = client.post("/api/notebooks/tags-batch", json={"entries": []})

    assert resp.status_code == 400


def test_set_notebook_tags_batch_rejects_more_entries_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/notebooks/tags-batch",
        json={
            "entries": [
                {"filename": "a.ipynb", "tags": ["x"]},
                {"filename": "b.ipynb", "tags": ["y"]},
            ]
        },
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_set_notebook_tags_batch_rejects_an_entry_missing_a_filename():

    resp = client.post(
        "/api/notebooks/tags-batch",
        json={"entries": [{"tags": ["bug"]}]},
    )

    assert resp.status_code == 400


def test_get_notebook_description_is_empty_for_a_never_described_notebook():

    _upload_sample_notebook("description_unset.ipynb")

    resp = client.get("/api/notebooks/description_unset.ipynb/description")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "filename": "description_unset.ipynb",
        "description": "",
    }


def test_get_notebook_description_returns_404_for_missing_file():

    resp = client.get("/api/notebooks/description_does_not_exist.ipynb/description")

    assert resp.status_code == 404


def test_set_notebook_description_returns_404_for_missing_file():

    resp = client.put(
        "/api/notebooks/description_does_not_exist.ipynb/description",
        json={"description": "hello"},
    )

    assert resp.status_code == 404


def test_set_notebook_description_persists_and_is_readable_back():

    _upload_sample_notebook("description_persist.ipynb")

    set_resp = client.put(
        "/api/notebooks/description_persist.ipynb/description",
        json={"description": "The quarterly churn model, retrained monthly."},
    )

    assert set_resp.status_code == 200
    assert set_resp.json() == {
        "status": "success",
        "dry_run": False,
        "filename": "description_persist.ipynb",
        "description": "The quarterly churn model, retrained monthly.",
    }

    get_resp = client.get("/api/notebooks/description_persist.ipynb/description")
    assert get_resp.json()["description"] == "The quarterly churn model, retrained monthly."


def test_set_notebook_description_dry_run_reports_the_normalized_value_without_writing():

    _upload_sample_notebook("description_dry_run.ipynb")
    client.put(
        "/api/notebooks/description_dry_run.ipynb/description",
        json={"description": "stale"},
    )

    resp = client.put(
        "/api/notebooks/description_dry_run.ipynb/description",
        json={"description": "  a new description  ", "dry_run": True},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "dry_run": True,
        "filename": "description_dry_run.ipynb",
        "description": "a new description",
    }

    # Nothing was actually written.
    assert client.get(
        "/api/notebooks/description_dry_run.ipynb/description"
    ).json()["description"] == "stale"


def test_set_notebook_description_strips_surrounding_whitespace():

    _upload_sample_notebook("description_strip.ipynb")

    resp = client.put(
        "/api/notebooks/description_strip.ipynb/description",
        json={"description": "   needs whitespace stripped   "},
    )

    assert resp.status_code == 200
    assert resp.json()["description"] == "needs whitespace stripped"


def test_set_notebook_description_with_empty_string_clears_it_and_removes_the_sidecar_file():

    _upload_sample_notebook("description_clear.ipynb")

    client.put(
        "/api/notebooks/description_clear.ipynb/description",
        json={"description": "temporary"},
    )
    assert _description_sidecar_path("description_clear.ipynb").is_file()

    clear_resp = client.put(
        "/api/notebooks/description_clear.ipynb/description",
        json={"description": ""},
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json()["description"] == ""
    assert not _description_sidecar_path("description_clear.ipynb").is_file()


def test_set_notebook_description_defaults_to_clearing_when_omitted():

    _upload_sample_notebook("description_omitted.ipynb")

    client.put(
        "/api/notebooks/description_omitted.ipynb/description",
        json={"description": "temporary"},
    )

    resp = client.put("/api/notebooks/description_omitted.ipynb/description", json={})

    assert resp.status_code == 200
    assert resp.json()["description"] == ""


def test_set_notebook_description_rejects_a_non_string_value():

    _upload_sample_notebook("description_not_a_string.ipynb")

    resp = client.put(
        "/api/notebooks/description_not_a_string.ipynb/description",
        json={"description": 5},
    )

    assert resp.status_code == 400


def test_set_notebook_description_rejects_a_description_over_the_max_length():

    _upload_sample_notebook("description_too_long.ipynb")

    resp = client.put(
        "/api/notebooks/description_too_long.ipynb/description",
        json={"description": "x" * 2001},
    )

    assert resp.status_code == 400


def test_get_notebook_source_url_is_null_for_a_notebook_with_none_recorded():

    _upload_sample_notebook("source_url_field_unset.ipynb")

    resp = client.get("/api/notebooks/source_url_field_unset.ipynb/source-url")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "filename": "source_url_field_unset.ipynb",
        "source_url": None,
    }


def test_get_notebook_source_url_returns_404_for_missing_file():

    resp = client.get("/api/notebooks/source_url_does_not_exist.ipynb/source-url")

    assert resp.status_code == 404


def test_set_notebook_source_url_returns_404_for_missing_file():

    resp = client.put(
        "/api/notebooks/source_url_does_not_exist.ipynb/source-url",
        json={"source_url": "https://example.com/a.ipynb"},
    )

    assert resp.status_code == 404


def test_set_notebook_source_url_persists_and_is_readable_back():

    _upload_sample_notebook("source_url_field_persist.ipynb")

    set_resp = client.put(
        "/api/notebooks/source_url_field_persist.ipynb/source-url",
        json={"source_url": "https://example.com/original.ipynb"},
    )

    assert set_resp.status_code == 200
    assert set_resp.json() == {
        "status": "success",
        "dry_run": False,
        "filename": "source_url_field_persist.ipynb",
        "source_url": "https://example.com/original.ipynb",
    }

    get_resp = client.get("/api/notebooks/source_url_field_persist.ipynb/source-url")
    assert get_resp.json()["source_url"] == "https://example.com/original.ipynb"

    assert client.get(
        "/api/notebooks/source_url_field_persist.ipynb/info"
    ).json()["source_url"] == "https://example.com/original.ipynb"


def test_set_notebook_source_url_dry_run_reports_the_normalized_value_without_writing():

    _upload_sample_notebook("source_url_field_dry_run.ipynb")
    client.put(
        "/api/notebooks/source_url_field_dry_run.ipynb/source-url",
        json={"source_url": "https://example.com/stale.ipynb"},
    )

    resp = client.put(
        "/api/notebooks/source_url_field_dry_run.ipynb/source-url",
        json={"source_url": "  https://example.com/new.ipynb  ", "dry_run": True},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "dry_run": True,
        "filename": "source_url_field_dry_run.ipynb",
        "source_url": "https://example.com/new.ipynb",
    }

    # Nothing was actually written.
    assert client.get(
        "/api/notebooks/source_url_field_dry_run.ipynb/source-url"
    ).json()["source_url"] == "https://example.com/stale.ipynb"


def test_set_notebook_source_url_strips_surrounding_whitespace():

    _upload_sample_notebook("source_url_field_strip.ipynb")

    resp = client.put(
        "/api/notebooks/source_url_field_strip.ipynb/source-url",
        json={"source_url": "   https://example.com/strip.ipynb   "},
    )

    assert resp.status_code == 200
    assert resp.json()["source_url"] == "https://example.com/strip.ipynb"


def test_set_notebook_source_url_with_empty_string_clears_it_and_removes_the_sidecar_file():

    _upload_sample_notebook("source_url_field_clear.ipynb")

    client.put(
        "/api/notebooks/source_url_field_clear.ipynb/source-url",
        json={"source_url": "https://example.com/temporary.ipynb"},
    )
    assert _source_url_sidecar_path("source_url_field_clear.ipynb").is_file()

    clear_resp = client.put(
        "/api/notebooks/source_url_field_clear.ipynb/source-url",
        json={"source_url": ""},
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json()["source_url"] is None
    assert not _source_url_sidecar_path("source_url_field_clear.ipynb").is_file()


def test_set_notebook_source_url_defaults_to_clearing_when_omitted():

    _upload_sample_notebook("source_url_field_omitted.ipynb")

    client.put(
        "/api/notebooks/source_url_field_omitted.ipynb/source-url",
        json={"source_url": "https://example.com/temporary.ipynb"},
    )

    resp = client.put("/api/notebooks/source_url_field_omitted.ipynb/source-url", json={})

    assert resp.status_code == 200
    assert resp.json()["source_url"] is None


def test_set_notebook_source_url_rejects_a_non_string_value():

    _upload_sample_notebook("source_url_field_not_a_string.ipynb")

    resp = client.put(
        "/api/notebooks/source_url_field_not_a_string.ipynb/source-url",
        json={"source_url": 5},
    )

    assert resp.status_code == 400


def test_set_notebook_source_url_rejects_a_value_over_the_max_length():

    _upload_sample_notebook("source_url_field_too_long.ipynb")

    resp = client.put(
        "/api/notebooks/source_url_field_too_long.ipynb/source-url",
        json={"source_url": "https://example.com/" + ("x" * 2048)},
    )

    assert resp.status_code == 400


def test_set_notebook_source_url_rejects_a_non_http_scheme():

    _upload_sample_notebook("source_url_field_bad_scheme.ipynb")

    resp = client.put(
        "/api/notebooks/source_url_field_bad_scheme.ipynb/source-url",
        json={"source_url": "ftp://example.com/a.ipynb"},
    )

    assert resp.status_code == 400


def test_set_notebook_source_url_rejects_a_url_with_no_host():

    _upload_sample_notebook("source_url_field_no_host.ipynb")

    resp = client.put(
        "/api/notebooks/source_url_field_no_host.ipynb/source-url",
        json={"source_url": "https://"},
    )

    assert resp.status_code == 400


def test_set_notebook_source_url_overwrites_one_recorded_by_a_real_import(
    notebook_url_server, _bypass_import_url_ssrf_guard
):
    """A manual PUT can correct a URL previously recorded automatically
    by a real POST /api/notebooks/import-url fetch -- and vice versa, a
    later real import-url fetch (on overwrite) still wins, since neither
    write path is otherwise special relative to the other.
    """

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/notebooks/import-url",
        json={
            "url": f"{base_url}/source_url_field_overwrite.ipynb",
            "filename": "source_url_field_overwrite.ipynb",
        },
    )

    resp = client.put(
        "/api/notebooks/source_url_field_overwrite.ipynb/source-url",
        json={"source_url": "https://example.com/corrected.ipynb"},
    )

    assert resp.status_code == 200
    assert resp.json()["source_url"] == "https://example.com/corrected.ipynb"
    assert client.get(
        "/api/notebooks/source_url_field_overwrite.ipynb/info"
    ).json()["source_url"] == "https://example.com/corrected.ipynb"


def test_set_notebook_source_url_batch_sets_each_notebooks_own_distinct_value():

    _upload_sample_notebook("source_url_batch_a.ipynb")
    _upload_sample_notebook("source_url_batch_b.ipynb")
    client.put(
        "/api/notebooks/source_url_batch_a.ipynb/source-url",
        json={"source_url": "https://example.com/stale.ipynb"},
    )

    resp = client.post(
        "/api/notebooks/source-url-batch",
        json={
            "entries": [
                {
                    "filename": "source_url_batch_a.ipynb",
                    "source_url": "https://example.com/a.ipynb",
                },
                {
                    "filename": "source_url_batch_b.ipynb",
                    "source_url": "https://example.com/b.ipynb",
                },
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["source_url_batch_a.ipynb"]["status"] == "success"
    assert (
        results_by_filename["source_url_batch_a.ipynb"]["source_url"]
        == "https://example.com/a.ipynb"
    )
    assert (
        results_by_filename["source_url_batch_b.ipynb"]["source_url"]
        == "https://example.com/b.ipynb"
    )

    # A full replace -- the stale value is gone.
    assert client.get(
        "/api/notebooks/source_url_batch_a.ipynb/source-url"
    ).json()["source_url"] == "https://example.com/a.ipynb"


def test_set_notebook_source_url_batch_dry_run_reports_the_plan_without_writing():

    _upload_sample_notebook("source_url_batch_dry_run.ipynb")
    client.put(
        "/api/notebooks/source_url_batch_dry_run.ipynb/source-url",
        json={"source_url": "https://example.com/stale.ipynb"},
    )

    resp = client.post(
        "/api/notebooks/source-url-batch",
        json={
            "entries": [
                {
                    "filename": "source_url_batch_dry_run.ipynb",
                    "source_url": "https://example.com/new.ipynb",
                },
                {"filename": "does_not_exist.ipynb", "source_url": "https://example.com/x.ipynb"},
            ],
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["source_url_batch_dry_run.ipynb"]["status"] == "success"
    assert (
        results_by_filename["source_url_batch_dry_run.ipynb"]["source_url"]
        == "https://example.com/new.ipynb"
    )

    # Nothing was actually written.
    assert client.get(
        "/api/notebooks/source_url_batch_dry_run.ipynb/source-url"
    ).json()["source_url"] == "https://example.com/stale.ipynb"


def test_set_notebook_source_url_batch_with_an_empty_value_clears_that_entry():

    _upload_sample_notebook("source_url_batch_clear.ipynb")
    client.put(
        "/api/notebooks/source_url_batch_clear.ipynb/source-url",
        json={"source_url": "https://example.com/temporary.ipynb"},
    )

    resp = client.post(
        "/api/notebooks/source-url-batch",
        json={"entries": [{"filename": "source_url_batch_clear.ipynb", "source_url": ""}]},
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["source_url"] is None
    assert client.get(
        "/api/notebooks/source_url_batch_clear.ipynb/source-url"
    ).json()["source_url"] is None


def test_set_notebook_source_url_batch_defaults_to_clearing_when_omitted():

    _upload_sample_notebook("source_url_batch_omitted.ipynb")
    client.put(
        "/api/notebooks/source_url_batch_omitted.ipynb/source-url",
        json={"source_url": "https://example.com/temporary.ipynb"},
    )

    resp = client.post(
        "/api/notebooks/source-url-batch",
        json={"entries": [{"filename": "source_url_batch_omitted.ipynb"}]},
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["source_url"] is None


def test_set_notebook_source_url_batch_reports_a_missing_filename_without_aborting_the_rest():

    _upload_sample_notebook("source_url_batch_partial.ipynb")

    resp = client.post(
        "/api/notebooks/source-url-batch",
        json={
            "entries": [
                {
                    "filename": "source_url_batch_partial.ipynb",
                    "source_url": "https://example.com/ok.ipynb",
                },
                {"filename": "does_not_exist.ipynb", "source_url": "https://example.com/ok.ipynb"},
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["source_url_batch_partial.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]


def test_set_notebook_source_url_batch_reports_an_invalid_value_for_just_that_entry():

    _upload_sample_notebook("source_url_batch_bad_a.ipynb")
    _upload_sample_notebook("source_url_batch_bad_b.ipynb")

    resp = client.post(
        "/api/notebooks/source-url-batch",
        json={
            "entries": [
                {
                    "filename": "source_url_batch_bad_a.ipynb",
                    "source_url": "https://example.com/ok.ipynb",
                },
                {"filename": "source_url_batch_bad_b.ipynb", "source_url": "ftp://example.com/x.ipynb"},
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["source_url_batch_bad_a.ipynb"]["status"] == "success"
    assert results_by_filename["source_url_batch_bad_b.ipynb"]["status"] == "error"
    assert (
        "only http:// and https:// URLs are accepted"
        in results_by_filename["source_url_batch_bad_b.ipynb"]["detail"]
    )

    # The failing entry never got its source_url touched.
    assert client.get(
        "/api/notebooks/source_url_batch_bad_b.ipynb/source-url"
    ).json()["source_url"] is None


def test_set_notebook_source_url_batch_rejects_a_non_list_entries_value():

    resp = client.post("/api/notebooks/source-url-batch", json={"entries": "not-a-list"})

    assert resp.status_code == 400


def test_set_notebook_source_url_batch_rejects_an_empty_entries_list():

    resp = client.post("/api/notebooks/source-url-batch", json={"entries": []})

    assert resp.status_code == 400


def test_set_notebook_source_url_batch_rejects_more_entries_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/notebooks/source-url-batch",
        json={
            "entries": [
                {"filename": "a.ipynb", "source_url": "https://example.com/a.ipynb"},
                {"filename": "b.ipynb", "source_url": "https://example.com/b.ipynb"},
            ]
        },
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_set_notebook_source_url_batch_rejects_an_entry_missing_a_filename():

    resp = client.post(
        "/api/notebooks/source-url-batch",
        json={"entries": [{"source_url": "https://example.com/a.ipynb"}]},
    )

    assert resp.status_code == 400


def test_notebook_list_and_info_include_the_description_field():

    _upload_sample_notebook("description_in_list.ipynb")
    client.put(
        "/api/notebooks/description_in_list.ipynb/description",
        json={"description": "shown in listings"},
    )

    list_entry = next(
        nb for nb in client.get("/api/notebooks").json()["notebooks"]
        if nb["filename"] == "description_in_list.ipynb"
    )
    assert list_entry["description"] == "shown in listings"

    info_resp = client.get("/api/notebooks/description_in_list.ipynb/info")
    assert info_resp.json()["description"] == "shown in listings"


def test_set_notebook_description_batch_sets_each_notebooks_own_distinct_description():

    _upload_sample_notebook("description_batch_a.ipynb")
    _upload_sample_notebook("description_batch_b.ipynb")
    client.put(
        "/api/notebooks/description_batch_a.ipynb/description",
        json={"description": "stale"},
    )

    resp = client.post(
        "/api/notebooks/description-batch",
        json={
            "entries": [
                {"filename": "description_batch_a.ipynb", "description": "The churn model."},
                {"filename": "description_batch_b.ipynb", "description": "The pricing model."},
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["description_batch_a.ipynb"]["status"] == "success"
    assert results_by_filename["description_batch_a.ipynb"]["description"] == "The churn model."
    assert results_by_filename["description_batch_b.ipynb"]["description"] == "The pricing model."

    # A full replace -- "stale" is gone.
    assert client.get(
        "/api/notebooks/description_batch_a.ipynb/description"
    ).json()["description"] == "The churn model."


def test_set_notebook_description_batch_dry_run_reports_the_plan_without_writing():

    _upload_sample_notebook("description_batch_dry_run.ipynb")
    client.put(
        "/api/notebooks/description_batch_dry_run.ipynb/description",
        json={"description": "stale"},
    )

    resp = client.post(
        "/api/notebooks/description-batch",
        json={
            "entries": [
                {"filename": "description_batch_dry_run.ipynb", "description": "new description"},
                {"filename": "does_not_exist.ipynb", "description": "x"},
            ],
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["description_batch_dry_run.ipynb"]["status"] == "success"
    assert results_by_filename["description_batch_dry_run.ipynb"]["description"] == "new description"

    # Nothing was actually written.
    assert client.get(
        "/api/notebooks/description_batch_dry_run.ipynb/description"
    ).json()["description"] == "stale"


def test_set_notebook_description_batch_with_an_empty_description_clears_that_entry():

    _upload_sample_notebook("description_batch_clear.ipynb")
    client.put(
        "/api/notebooks/description_batch_clear.ipynb/description",
        json={"description": "temporary"},
    )

    resp = client.post(
        "/api/notebooks/description-batch",
        json={"entries": [{"filename": "description_batch_clear.ipynb", "description": ""}]},
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["description"] == ""
    assert client.get(
        "/api/notebooks/description_batch_clear.ipynb/description"
    ).json()["description"] == ""


def test_set_notebook_description_batch_defaults_to_clearing_when_omitted():

    _upload_sample_notebook("description_batch_omitted.ipynb")
    client.put(
        "/api/notebooks/description_batch_omitted.ipynb/description",
        json={"description": "temporary"},
    )

    resp = client.post(
        "/api/notebooks/description-batch",
        json={"entries": [{"filename": "description_batch_omitted.ipynb"}]},
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["description"] == ""


def test_set_notebook_description_batch_strips_surrounding_whitespace():

    _upload_sample_notebook("description_batch_strip.ipynb")

    resp = client.post(
        "/api/notebooks/description-batch",
        json={
            "entries": [
                {"filename": "description_batch_strip.ipynb", "description": "  padded  "},
            ],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["description"] == "padded"


def test_set_notebook_description_batch_reports_a_missing_filename_without_aborting_the_rest():

    _upload_sample_notebook("description_batch_partial.ipynb")

    resp = client.post(
        "/api/notebooks/description-batch",
        json={
            "entries": [
                {"filename": "description_batch_partial.ipynb", "description": "ok"},
                {"filename": "does_not_exist.ipynb", "description": "ok"},
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["description_batch_partial.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]


def test_set_notebook_description_batch_reports_an_invalid_description_for_just_that_entry():

    _upload_sample_notebook("description_batch_bad_a.ipynb")
    _upload_sample_notebook("description_batch_bad_b.ipynb")

    resp = client.post(
        "/api/notebooks/description-batch",
        json={
            "entries": [
                {"filename": "description_batch_bad_a.ipynb", "description": "ok"},
                {"filename": "description_batch_bad_b.ipynb", "description": 5},
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["description_batch_bad_a.ipynb"]["status"] == "success"
    assert results_by_filename["description_batch_bad_b.ipynb"]["status"] == "error"
    assert "must be a string" in results_by_filename["description_batch_bad_b.ipynb"]["detail"]

    # The failing entry never got its description touched.
    assert client.get(
        "/api/notebooks/description_batch_bad_b.ipynb/description"
    ).json()["description"] == ""


def test_set_notebook_description_batch_reports_a_too_long_description_for_just_that_entry():

    _upload_sample_notebook("description_batch_too_long.ipynb")

    resp = client.post(
        "/api/notebooks/description-batch",
        json={
            "entries": [
                {"filename": "description_batch_too_long.ipynb", "description": "x" * 2001},
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["failed_count"] == 1
    assert "at most" in body["results"][0]["detail"]


def test_set_notebook_description_batch_rejects_a_non_list_entries_value():

    resp = client.post("/api/notebooks/description-batch", json={"entries": "not-a-list"})

    assert resp.status_code == 400


def test_set_notebook_description_batch_rejects_an_empty_entries_list():

    resp = client.post("/api/notebooks/description-batch", json={"entries": []})

    assert resp.status_code == 400


def test_set_notebook_description_batch_rejects_more_entries_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/notebooks/description-batch",
        json={
            "entries": [
                {"filename": "a.ipynb", "description": "a"},
                {"filename": "b.ipynb", "description": "b"},
            ]
        },
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_set_notebook_description_batch_rejects_an_entry_missing_a_filename():

    resp = client.post(
        "/api/notebooks/description-batch",
        json={"entries": [{"description": "ok"}]},
    )

    assert resp.status_code == 400


def test_rename_notebook_moves_its_description():

    _upload_sample_notebook("description_rename_source.ipynb")
    client.put(
        "/api/notebooks/description_rename_source.ipynb/description",
        json={"description": "moves with the rename"},
    )

    rename_resp = client.patch(
        "/api/notebooks/description_rename_source.ipynb",
        json={"new_filename": "description_rename_target.ipynb"},
    )
    assert rename_resp.status_code == 200

    assert client.get(
        "/api/notebooks/description_rename_target.ipynb/description"
    ).json()["description"] == "moves with the rename"
    assert not _description_sidecar_path("description_rename_source.ipynb").is_file()

    os.remove(Path(UPLOAD_DIR) / "description_rename_target.ipynb")
    _description_sidecar_path("description_rename_target.ipynb").unlink(missing_ok=True)


def test_copy_notebook_copies_the_description_from_the_source():

    _upload_sample_notebook("description_copy_source.ipynb")
    client.put(
        "/api/notebooks/description_copy_source.ipynb/description",
        json={"description": "copied along with the content"},
    )

    copy_resp = client.post(
        "/api/notebooks/description_copy_source.ipynb/copy",
        json={"new_filename": "description_copy_target.ipynb"},
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/description_copy_target.ipynb/description"
    ).json()["description"] == "copied along with the content"

    os.remove(Path(UPLOAD_DIR) / "description_copy_target.ipynb")
    _description_sidecar_path("description_copy_target.ipynb").unlink(missing_ok=True)


def test_copy_notebook_version_does_not_inherit_the_current_description():

    filename = "description_versions_copy_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    client.put(
        f"/api/notebooks/{filename}/description",
        json={"description": "the live notebook's own description"},
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    copy_resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={"new_filename": "description_versions_copy_target.ipynb"},
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/description_versions_copy_target.ipynb/description"
    ).json()["description"] == ""

    os.remove(Path(UPLOAD_DIR) / "description_versions_copy_target.ipynb")


def test_delete_notebook_removes_its_description_sidecar_file():

    _upload_sample_notebook("description_delete.ipynb")
    client.put(
        "/api/notebooks/description_delete.ipynb/description",
        json={"description": "goes away with the notebook"},
    )
    assert _description_sidecar_path("description_delete.ipynb").is_file()

    delete_resp = client.delete("/api/notebooks/description_delete.ipynb")
    assert delete_resp.status_code == 200

    assert not _description_sidecar_path("description_delete.ipynb").is_file()


def test_list_tags_response_has_the_expected_shape():

    resp = client.get("/api/tags")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert isinstance(body["tags"], list)
    for entry in body["tags"]:
        assert set(entry) == {"tag", "notebook_count"}


def test_list_tags_reports_distinct_tags_with_notebook_counts():

    _upload_sample_notebook("tags_catalog_one.ipynb")
    _upload_sample_notebook("tags_catalog_two.ipynb")
    _upload_sample_notebook("tags_catalog_three.ipynb")

    client.put(
        "/api/notebooks/tags_catalog_one.ipynb/tags",
        json={"tags": ["production", "bug"]},
    )
    client.put(
        "/api/notebooks/tags_catalog_two.ipynb/tags",
        json={"tags": ["production"]},
    )
    # tags_catalog_three.ipynb is left untagged -- it should contribute
    # nothing to the catalog.

    resp = client.get("/api/tags")

    assert resp.status_code == 200

    by_tag = {entry["tag"]: entry["notebook_count"] for entry in resp.json()["tags"]}

    assert by_tag["production"] == 2
    assert by_tag["bug"] == 1


def test_list_tags_reflects_a_tag_set_being_cleared():

    _upload_sample_notebook("tags_catalog_cleared.ipynb")

    client.put(
        "/api/notebooks/tags_catalog_cleared.ipynb/tags",
        json={"tags": ["scratch"]},
    )
    assert "scratch" in {
        entry["tag"] for entry in client.get("/api/tags").json()["tags"]
    }

    client.put(
        "/api/notebooks/tags_catalog_cleared.ipynb/tags",
        json={"tags": []},
    )

    assert "scratch" not in {
        entry["tag"] for entry in client.get("/api/tags").json()["tags"]
    }


def test_list_tags_are_sorted_alphabetically():

    _upload_sample_notebook("tags_catalog_sort.ipynb")

    client.put(
        "/api/notebooks/tags_catalog_sort.ipynb/tags",
        json={"tags": ["zeta", "alpha", "mu"]},
    )

    resp = client.get("/api/tags")

    tag_names = [entry["tag"] for entry in resp.json()["tags"]]

    assert tag_names == sorted(tag_names)
    assert {"zeta", "alpha", "mu"}.issubset(set(tag_names))


def test_list_tags_csv_format_returns_a_csv_response():

    _upload_sample_notebook("tags_csv_a.ipynb")
    _upload_sample_notebook("tags_csv_b.ipynb")

    client.put(
        "/api/notebooks/tags_csv_a.ipynb/tags", json={"tags": ["csvtag"]}
    )
    client.put(
        "/api/notebooks/tags_csv_b.ipynb/tags", json={"tags": ["csvtag"]}
    )

    resp = client.get("/api/tags", params={"format": "csv"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="tags.csv"' in resp.headers["content-disposition"]

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == "tag,notebook_count"
    assert "csvtag,2" in rows


def test_list_tags_rejects_an_unknown_format():

    resp = client.get("/api/tags", params={"format": "xml"})

    assert resp.status_code == 400
    assert "format" in resp.json()["detail"]


def test_delete_tag_removes_it_from_every_notebook_that_has_it():

    _upload_sample_notebook("tags_bulk_delete_a.ipynb")
    _upload_sample_notebook("tags_bulk_delete_b.ipynb")
    _upload_sample_notebook("tags_bulk_delete_c.ipynb")

    client.put(
        "/api/notebooks/tags_bulk_delete_a.ipynb/tags",
        json={"tags": ["scratch", "bug"]},
    )
    client.put(
        "/api/notebooks/tags_bulk_delete_b.ipynb/tags",
        json={"tags": ["scratch"]},
    )
    client.put(
        "/api/notebooks/tags_bulk_delete_c.ipynb/tags",
        json={"tags": ["production"]},
    )

    delete_resp = client.delete("/api/tags/scratch")

    assert delete_resp.status_code == 200
    body = delete_resp.json()
    assert body["status"] == "success"
    assert body["tag"] == "scratch"
    assert sorted(body["affected_notebooks"]) == [
        "tags_bulk_delete_a.ipynb", "tags_bulk_delete_b.ipynb",
    ]
    assert body["notebook_count"] == 2

    assert client.get(
        "/api/notebooks/tags_bulk_delete_a.ipynb/tags"
    ).json()["tags"] == ["bug"]
    assert client.get(
        "/api/notebooks/tags_bulk_delete_b.ipynb/tags"
    ).json()["tags"] == []
    # Untouched -- never carried "scratch" at all.
    assert client.get(
        "/api/notebooks/tags_bulk_delete_c.ipynb/tags"
    ).json()["tags"] == ["production"]

    assert "scratch" not in {
        entry["tag"] for entry in client.get("/api/tags").json()["tags"]
    }


def test_delete_tag_dry_run_reports_the_plan_without_removing_it():

    _upload_sample_notebook("tags_bulk_delete_dry_run_a.ipynb")
    _upload_sample_notebook("tags_bulk_delete_dry_run_b.ipynb")

    client.put(
        "/api/notebooks/tags_bulk_delete_dry_run_a.ipynb/tags",
        json={"tags": ["scratch-dry-run", "bug"]},
    )
    client.put(
        "/api/notebooks/tags_bulk_delete_dry_run_b.ipynb/tags",
        json={"tags": ["production"]},
    )

    delete_resp = client.delete(
        "/api/tags/scratch-dry-run", params={"dry_run": True}
    )

    assert delete_resp.status_code == 200
    body = delete_resp.json()
    assert body["dry_run"] is True
    assert body["affected_notebooks"] == ["tags_bulk_delete_dry_run_a.ipynb"]
    assert body["notebook_count"] == 1

    # Nothing was actually removed.
    assert client.get(
        "/api/notebooks/tags_bulk_delete_dry_run_a.ipynb/tags"
    ).json()["tags"] == ["bug", "scratch-dry-run"]
    assert "scratch-dry-run" in {
        entry["tag"] for entry in client.get("/api/tags").json()["tags"]
    }


def test_delete_tag_is_a_no_op_success_when_nothing_carries_it():

    resp = client.delete("/api/tags/this-tag-does-not-exist-anywhere")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "dry_run": False,
        "tag": "this-tag-does-not-exist-anywhere",
        "affected_notebooks": [],
        "notebook_count": 0,
    }


def test_delete_tag_removes_the_sidecar_file_when_it_was_the_only_tag():

    _upload_sample_notebook("tags_bulk_delete_only_tag.ipynb")
    client.put(
        "/api/notebooks/tags_bulk_delete_only_tag.ipynb/tags",
        json={"tags": ["temporary"]},
    )
    assert _tags_sidecar_path("tags_bulk_delete_only_tag.ipynb").is_file()

    delete_resp = client.delete("/api/tags/temporary")

    assert delete_resp.status_code == 200
    assert delete_resp.json()["affected_notebooks"] == [
        "tags_bulk_delete_only_tag.ipynb"
    ]
    assert not _tags_sidecar_path("tags_bulk_delete_only_tag.ipynb").is_file()


def test_apply_tag_adds_it_to_every_named_notebook_preserving_existing_tags():

    _upload_sample_notebook("tags_apply_a.ipynb")
    _upload_sample_notebook("tags_apply_b.ipynb")
    client.put(
        "/api/notebooks/tags_apply_a.ipynb/tags", json={"tags": ["bug"]}
    )

    resp = client.post(
        "/api/tags/production/apply",
        json={"filenames": ["tags_apply_a.ipynb", "tags_apply_b.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["tag"] == "production"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["tags_apply_a.ipynb"]["status"] == "success"
    assert results_by_filename["tags_apply_a.ipynb"]["tags"] == ["bug", "production"]
    assert results_by_filename["tags_apply_b.ipynb"]["tags"] == ["production"]

    # Existing tags weren't clobbered -- "bug" survives alongside the newly
    # applied "production".
    assert client.get(
        "/api/notebooks/tags_apply_a.ipynb/tags"
    ).json()["tags"] == ["bug", "production"]


def test_apply_tag_reports_a_missing_filename_without_aborting_the_rest():

    _upload_sample_notebook("tags_apply_partial.ipynb")

    resp = client.post(
        "/api/tags/urgent/apply",
        json={"filenames": ["tags_apply_partial.ipynb", "does_not_exist.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["tags_apply_partial.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]

    assert client.get(
        "/api/notebooks/tags_apply_partial.ipynb/tags"
    ).json()["tags"] == ["urgent"]


def test_apply_tag_is_idempotent_for_a_notebook_that_already_has_it():

    _upload_sample_notebook("tags_apply_idempotent.ipynb")
    client.put(
        "/api/notebooks/tags_apply_idempotent.ipynb/tags",
        json={"tags": ["production"]},
    )

    resp = client.post(
        "/api/tags/production/apply",
        json={"filenames": ["tags_apply_idempotent.ipynb"]},
    )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["tags"] == ["production"]


def test_apply_tag_rejects_a_non_list_filenames_value():

    resp = client.post("/api/tags/production/apply", json={"filenames": "not-a-list"})

    assert resp.status_code == 400


def test_apply_tag_rejects_an_empty_filenames_list():

    resp = client.post("/api/tags/production/apply", json={"filenames": []})

    assert resp.status_code == 400


def test_apply_tag_rejects_more_filenames_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/tags/production/apply",
        json={"filenames": ["a.ipynb", "b.ipynb"]},
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_apply_tag_rejects_an_empty_tag():

    _upload_sample_notebook("tags_apply_empty_tag.ipynb")

    resp = client.post(
        "/api/tags/%20/apply",
        json={"filenames": ["tags_apply_empty_tag.ipynb"]},
    )

    assert resp.status_code == 400


def test_apply_tag_dry_run_reports_the_merged_tags_without_writing_them():

    _upload_sample_notebook("tags_apply_dry_run.ipynb")
    client.put(
        "/api/notebooks/tags_apply_dry_run.ipynb/tags", json={"tags": ["bug"]}
    )

    resp = client.post(
        "/api/tags/production/apply",
        json={"filenames": ["tags_apply_dry_run.ipynb"], "dry_run": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["results"][0]["status"] == "success"
    assert body["results"][0]["tags"] == ["bug", "production"]

    # The dry run's own predicted merge was never actually written.
    assert client.get(
        "/api/notebooks/tags_apply_dry_run.ipynb/tags"
    ).json()["tags"] == ["bug"]


def test_apply_tag_dry_run_still_reports_a_missing_filename_as_an_error():

    resp = client.post(
        "/api/tags/production/apply",
        json={"filenames": ["does_not_exist_dry_run.ipynb"], "dry_run": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["failed_count"] == 1
    assert "not found" in body["results"][0]["detail"]


def test_remove_tag_batch_removes_it_from_named_notebooks_only():

    _upload_sample_notebook("tags_remove_a.ipynb")
    _upload_sample_notebook("tags_remove_b.ipynb")
    _upload_sample_notebook("tags_remove_untouched.ipynb")

    client.put(
        "/api/notebooks/tags_remove_a.ipynb/tags",
        json={"tags": ["production", "bug"]},
    )
    client.put(
        "/api/notebooks/tags_remove_b.ipynb/tags",
        json={"tags": ["production"]},
    )
    client.put(
        "/api/notebooks/tags_remove_untouched.ipynb/tags",
        json={"tags": ["production"]},
    )

    resp = client.post(
        "/api/tags/production/remove",
        json={"filenames": ["tags_remove_a.ipynb", "tags_remove_b.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["tag"] == "production"
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["tags_remove_a.ipynb"]["status"] == "success"
    assert results_by_filename["tags_remove_a.ipynb"]["tags"] == ["bug"]
    assert results_by_filename["tags_remove_b.ipynb"]["tags"] == []

    # A notebook not named in "filenames" keeps the tag untouched, even
    # though it also carries it.
    assert client.get(
        "/api/notebooks/tags_remove_untouched.ipynb/tags"
    ).json()["tags"] == ["production"]


def test_remove_tag_batch_reports_a_missing_filename_without_aborting_the_rest():

    _upload_sample_notebook("tags_remove_partial.ipynb")
    client.put(
        "/api/notebooks/tags_remove_partial.ipynb/tags",
        json={"tags": ["urgent"]},
    )

    resp = client.post(
        "/api/tags/urgent/remove",
        json={"filenames": ["tags_remove_partial.ipynb", "does_not_exist.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["tags_remove_partial.ipynb"]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]

    assert client.get(
        "/api/notebooks/tags_remove_partial.ipynb/tags"
    ).json()["tags"] == []


def test_remove_tag_batch_is_idempotent_for_a_notebook_that_never_had_it():

    _upload_sample_notebook("tags_remove_idempotent.ipynb")

    resp = client.post(
        "/api/tags/production/remove",
        json={"filenames": ["tags_remove_idempotent.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["results"][0]["status"] == "success"
    assert body["results"][0]["tags"] == []


def test_remove_tag_batch_rejects_a_non_list_filenames_value():

    resp = client.post("/api/tags/production/remove", json={"filenames": "not-a-list"})

    assert resp.status_code == 400


def test_remove_tag_batch_rejects_an_empty_filenames_list():

    resp = client.post("/api/tags/production/remove", json={"filenames": []})

    assert resp.status_code == 400


def test_remove_tag_batch_rejects_more_filenames_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/tags/production/remove",
        json={"filenames": ["a.ipynb", "b.ipynb"]},
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_remove_tag_batch_rejects_an_empty_tag():

    _upload_sample_notebook("tags_remove_empty_tag.ipynb")

    resp = client.post(
        "/api/tags/%20/remove",
        json={"filenames": ["tags_remove_empty_tag.ipynb"]},
    )

    assert resp.status_code == 400


def test_remove_tag_batch_dry_run_reports_the_remaining_tags_without_writing_them():

    _upload_sample_notebook("tags_remove_dry_run.ipynb")
    client.put(
        "/api/notebooks/tags_remove_dry_run.ipynb/tags",
        json={"tags": ["production", "bug"]},
    )

    resp = client.post(
        "/api/tags/production/remove",
        json={"filenames": ["tags_remove_dry_run.ipynb"], "dry_run": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["results"][0]["status"] == "success"
    assert body["results"][0]["tags"] == ["bug"]

    # The dry run's own predicted removal was never actually written.
    assert client.get(
        "/api/notebooks/tags_remove_dry_run.ipynb/tags"
    ).json()["tags"] == ["bug", "production"]


def test_remove_tag_batch_dry_run_still_reports_a_missing_filename_as_an_error():

    resp = client.post(
        "/api/tags/production/remove",
        json={"filenames": ["does_not_exist_dry_run.ipynb"], "dry_run": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["failed_count"] == 1
    assert "not found" in body["results"][0]["detail"]


def test_rename_tag_renames_it_on_every_notebook_that_has_it():

    _upload_sample_notebook("tags_rename_a.ipynb")
    _upload_sample_notebook("tags_rename_b.ipynb")
    _upload_sample_notebook("tags_rename_c.ipynb")

    client.put(
        "/api/notebooks/tags_rename_a.ipynb/tags",
        json={"tags": ["prod", "bug"]},
    )
    client.put(
        "/api/notebooks/tags_rename_b.ipynb/tags",
        json={"tags": ["prod"]},
    )
    client.put(
        "/api/notebooks/tags_rename_c.ipynb/tags",
        json={"tags": ["staging"]},
    )

    rename_resp = client.patch(
        "/api/tags/prod", json={"new_tag": "production"}
    )

    assert rename_resp.status_code == 200
    body = rename_resp.json()
    assert body["status"] == "success"
    assert body["tag"] == "prod"
    assert body["new_tag"] == "production"
    assert sorted(body["affected_notebooks"]) == [
        "tags_rename_a.ipynb", "tags_rename_b.ipynb",
    ]
    assert body["notebook_count"] == 2

    assert client.get(
        "/api/notebooks/tags_rename_a.ipynb/tags"
    ).json()["tags"] == ["bug", "production"]
    assert client.get(
        "/api/notebooks/tags_rename_b.ipynb/tags"
    ).json()["tags"] == ["production"]
    # Untouched -- never carried "prod" at all.
    assert client.get(
        "/api/notebooks/tags_rename_c.ipynb/tags"
    ).json()["tags"] == ["staging"]

    tag_names = {entry["tag"] for entry in client.get("/api/tags").json()["tags"]}
    assert "prod" not in tag_names
    assert "production" in tag_names


def test_rename_tag_dry_run_reports_the_plan_without_renaming_it():

    _upload_sample_notebook("tags_rename_dry_run_a.ipynb")
    _upload_sample_notebook("tags_rename_dry_run_b.ipynb")

    client.put(
        "/api/notebooks/tags_rename_dry_run_a.ipynb/tags",
        json={"tags": ["prod-dry-run", "bug"]},
    )
    client.put(
        "/api/notebooks/tags_rename_dry_run_b.ipynb/tags",
        json={"tags": ["staging"]},
    )

    rename_resp = client.patch(
        "/api/tags/prod-dry-run",
        json={"new_tag": "production-dry-run", "dry_run": True},
    )

    assert rename_resp.status_code == 200
    body = rename_resp.json()
    assert body["dry_run"] is True
    assert body["affected_notebooks"] == ["tags_rename_dry_run_a.ipynb"]
    assert body["notebook_count"] == 1

    # Nothing was actually renamed.
    assert client.get(
        "/api/notebooks/tags_rename_dry_run_a.ipynb/tags"
    ).json()["tags"] == ["bug", "prod-dry-run"]
    tag_names = {entry["tag"] for entry in client.get("/api/tags").json()["tags"]}
    assert "prod-dry-run" in tag_names
    assert "production-dry-run" not in tag_names


def test_rename_tag_is_a_no_op_success_when_nothing_carries_it():

    resp = client.patch(
        "/api/tags/this-tag-does-not-exist-anywhere",
        json={"new_tag": "something-else"},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "dry_run": False,
        "tag": "this-tag-does-not-exist-anywhere",
        "new_tag": "something-else",
        "affected_notebooks": [],
        "notebook_count": 0,
    }


def test_rename_tag_merges_into_an_existing_new_tag_without_duplicating():

    _upload_sample_notebook("tags_rename_merge.ipynb")
    client.put(
        "/api/notebooks/tags_rename_merge.ipynb/tags",
        json={"tags": ["prod", "production"]},
    )

    resp = client.patch("/api/tags/prod", json={"new_tag": "production"})

    assert resp.status_code == 200
    assert resp.json()["affected_notebooks"] == ["tags_rename_merge.ipynb"]

    assert client.get(
        "/api/notebooks/tags_rename_merge.ipynb/tags"
    ).json()["tags"] == ["production"]


def test_rename_tag_rejects_a_missing_new_tag():

    resp = client.patch("/api/tags/prod", json={})

    assert resp.status_code == 400


def test_rename_tag_rejects_an_empty_new_tag():

    resp = client.patch("/api/tags/prod", json={"new_tag": "   "})

    assert resp.status_code == 400


def test_rename_tag_rejects_renaming_a_tag_to_itself():

    resp = client.patch("/api/tags/prod", json={"new_tag": "prod"})

    assert resp.status_code == 400


def test_search_functions_finds_notebooks_with_a_matching_function_name():

    content_a = _notebook_bytes(
        "def train_model(epochs: int) -> str:\n    return 'done'\n"
    )
    content_b = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("search_functions_a.ipynb", io.BytesIO(content_a), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("search_functions_b.ipynb", io.BytesIO(content_b), "application/json")},
    )

    resp = client.get("/api/functions?search=train")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["search"] == "train"
    assert body["notebook_count"] == 1
    assert body["matches"][0]["filename"] == "search_functions_a.ipynb"
    assert [f["name"] for f in body["matches"][0]["functions"]] == ["train_model"]


def test_search_functions_filters_by_tag():

    content = _notebook_bytes(
        "def train_model(epochs: int) -> str:\n    return 'done'\n"
    )

    for filename in ("search_functions_tag_a.ipynb", "search_functions_tag_b.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    client.put(
        "/api/notebooks/search_functions_tag_a.ipynb/tags", json={"tags": ["prod"]}
    )

    resp = client.get("/api/functions?search=train&tag=prod")

    assert resp.status_code == 200
    body = resp.json()
    assert body["notebook_count"] == 1
    assert body["matches"][0]["filename"] == "search_functions_tag_a.ipynb"


def test_search_functions_is_case_insensitive():

    content = _notebook_bytes(
        "def TrainModel(epochs: int) -> str:\n    return 'done'\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("search_functions_case.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get("/api/functions?search=trainmodel")

    assert resp.status_code == 200
    assert [m["filename"] for m in resp.json()["matches"]] == ["search_functions_case.ipynb"]


def test_search_functions_reports_no_matches():

    resp = client.get("/api/functions?search=this_function_name_does_not_exist_anywhere")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "search": "this_function_name_does_not_exist_anywhere",
        "regex": False,
        "matches": [],
        "notebook_count": 0,
        "limit": None,
        "offset": 0,
    }


def test_search_functions_requires_a_search_value():

    resp = client.get("/api/functions")

    assert resp.status_code == 400


def test_search_functions_rejects_an_unknown_format():

    resp = client.get("/api/functions?search=train&format=xml")

    assert resp.status_code == 400


def test_search_functions_csv_format_returns_one_row_per_matching_function():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes(
        "def train_model(epochs: int, lr: float = 0.01) -> str:\n    return 'done'\n\n"
        "async def train_async() -> None:\n    pass\n"
    )
    client.post(
        "/api/upload",
        files={"file": ("search_functions_csv.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get("/api/functions", params={"search": "train", "format": "csv"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="functions.csv"' in resp.headers["content-disposition"]

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == "filename,function_name,args,return_type,is_async"
    assert len(rows) == 3

    train_model_row = next(r for r in rows[1:] if "train_model" in r)
    assert train_model_row.startswith("search_functions_csv.ipynb,train_model,")
    assert "epochs:int" in train_model_row
    assert "lr:float" in train_model_row

    train_async_row = next(r for r in rows[1:] if "train_async" in r)
    assert train_async_row == "search_functions_csv.ipynb,train_async,,None,True"


def test_search_functions_skips_a_malformed_notebook_file():

    filename = "search_functions_malformed.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    content = _notebook_bytes(
        "def search_functions_clean_marker_fn(a: int, b: int) -> int:\n    return a + b\n"
    )
    client.post(
        "/api/upload",
        files={"file": ("search_functions_clean.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get("/api/functions?search=search_functions_clean_marker_fn")

    assert resp.status_code == 200
    assert [m["filename"] for m in resp.json()["matches"]] == ["search_functions_clean.ipynb"]

    os.remove(file_path)


def test_search_functions_limit_and_offset_page_the_matching_notebooks():

    content = _notebook_bytes(
        "def search_functions_page_marker() -> int:\n    return 1\n"
    )

    for filename in (
        "search_functions_page_a.ipynb",
        "search_functions_page_b.ipynb",
        "search_functions_page_c.ipynb",
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    all_matches = client.get(
        "/api/functions?search=search_functions_page_marker"
    ).json()["matches"]
    assert len(all_matches) == 3

    resp = client.get(
        "/api/functions",
        params={"search": "search_functions_page_marker", "limit": 2},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [m["filename"] for m in body["matches"]] == [
        m["filename"] for m in all_matches[:2]
    ]
    assert body["notebook_count"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0

    resp = client.get(
        "/api/functions",
        params={"search": "search_functions_page_marker", "offset": 1},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [m["filename"] for m in body["matches"]] == [
        m["filename"] for m in all_matches[1:]
    ]
    assert body["notebook_count"] == 3
    assert body["offset"] == 1


def test_search_functions_rejects_a_negative_offset():

    resp = client.get(
        "/api/functions", params={"search": "anything", "offset": -1}
    )

    assert resp.status_code == 400


def test_search_functions_rejects_a_non_positive_limit():

    resp = client.get(
        "/api/functions", params={"search": "anything", "limit": 0}
    )

    assert resp.status_code == 400


def test_search_functions_regex_matches_a_pattern_not_a_literal_substring():

    content_a = _notebook_bytes(
        "def train_model_v1(epochs: int) -> str:\n    return 'done'\n"
    )
    content_b = _notebook_bytes(
        "def train_model(epochs: int) -> str:\n    return 'done'\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("search_functions_regex_a.ipynb", io.BytesIO(content_a), "application/json")},
    )
    client.post(
        "/api/upload",
        files={"file": ("search_functions_regex_b.ipynb", io.BytesIO(content_b), "application/json")},
    )

    # A plain substring for "train_model" alone matches both notebooks;
    # the pattern below only matches the one whose name also ends in a
    # version suffix.
    resp = client.get(
        "/api/functions", params={"search": r"_v\d+$", "regex": "true"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["regex"] is True
    assert body["notebook_count"] == 1
    assert body["matches"][0]["filename"] == "search_functions_regex_a.ipynb"


def test_search_functions_regex_is_case_insensitive():

    content = _notebook_bytes(
        "def TrainModel(epochs: int) -> str:\n    return 'done'\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("search_functions_regex_case.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.get(
        "/api/functions", params={"search": "^trainmodel$", "regex": "true"}
    )

    assert resp.status_code == 200
    assert resp.json()["notebook_count"] == 1


def test_search_functions_regex_false_treats_search_as_a_plain_substring():

    content = _notebook_bytes("def abc() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": ("search_functions_no_regex.ipynb", io.BytesIO(content), "application/json")},
    )

    # "a.c" would match "abc" as a *regex* (any-character "."), but not
    # as a literal substring the function name doesn't actually contain.
    resp = client.get(
        "/api/functions", params={"search": "a.c", "regex": "false"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["regex"] is False
    assert body["notebook_count"] == 0


def test_search_functions_regex_rejects_an_invalid_pattern():

    resp = client.get(
        "/api/functions", params={"search": "(unclosed", "regex": "true"}
    )

    assert resp.status_code == 400
    assert "regular expression" in resp.json()["detail"]


def test_search_functions_regex_rejects_a_catastrophically_backtracking_pattern():

    resp = client.get(
        "/api/functions", params={"search": "(a+)+", "regex": "true"}
    )

    assert resp.status_code == 400
    assert "nested" in resp.json()["detail"].lower()


def test_list_notebooks_reports_tags_for_each_entry():

    _upload_sample_notebook("tags_in_list.ipynb")

    client.put("/api/notebooks/tags_in_list.ipynb/tags", json={"tags": ["demo"]})

    notebooks = {
        nb["filename"]: nb for nb in client.get("/api/notebooks").json()["notebooks"]
    }

    assert notebooks["tags_in_list.ipynb"]["tags"] == ["demo"]


def test_list_notebooks_filters_by_tag():

    _upload_sample_notebook("tags_filter_a.ipynb")
    _upload_sample_notebook("tags_filter_b.ipynb")

    client.put("/api/notebooks/tags_filter_a.ipynb/tags", json={"tags": ["keepme"]})

    notebooks = client.get(
        "/api/notebooks?search=tags_filter_&tag=keepme"
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == ["tags_filter_a.ipynb"]


def test_list_notebooks_filters_by_description_search():

    _upload_sample_notebook("desc_search_a.ipynb")
    _upload_sample_notebook("desc_search_b.ipynb")

    client.put(
        "/api/notebooks/desc_search_a.ipynb/description",
        json={"description": "The quarterly churn model."},
    )
    client.put(
        "/api/notebooks/desc_search_b.ipynb/description",
        json={"description": "Unrelated pricing notebook."},
    )

    notebooks = client.get(
        "/api/notebooks?search=desc_search_&description_search=churn"
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == ["desc_search_a.ipynb"]


def test_list_notebooks_filters_by_sha256():

    import hashlib

    content_a = _notebook_bytes("def sha_filter_a() -> int:\n    return 1\n")
    content_b = _notebook_bytes("def sha_filter_b() -> int:\n    return 2\n")

    for filename, content in (
        ("sha_filter_a.ipynb", content_a),
        ("sha_filter_b.ipynb", content_b),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    target_sha256 = hashlib.sha256(content_a).hexdigest()

    notebooks = client.get(
        "/api/notebooks", params={"search": "sha_filter_", "sha256": target_sha256}
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == ["sha_filter_a.ipynb"]


def test_list_notebooks_sha256_matches_every_notebook_with_that_content():

    import hashlib

    content = _notebook_bytes("def sha_filter_shared() -> int:\n    return 1\n")

    for filename in ("sha_filter_dup_a.ipynb", "sha_filter_dup_b.ipynb"):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    notebooks = client.get(
        "/api/notebooks",
        params={"search": "sha_filter_dup_", "sha256": hashlib.sha256(content).hexdigest()},
    ).json()["notebooks"]

    assert sorted(nb["filename"] for nb in notebooks) == [
        "sha_filter_dup_a.ipynb", "sha_filter_dup_b.ipynb",
    ]


def test_list_notebooks_unknown_sha256_yields_no_notebooks():

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "sha_filter_none.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert resp.status_code == 200

    notebooks = client.get(
        "/api/notebooks",
        params={"search": "sha_filter_none", "sha256": "no-such-hash"},
    ).json()["notebooks"]

    assert notebooks == []


def test_list_notebooks_filters_by_modified_after_and_before():

    resp_older = client.post(
        "/api/upload",
        files={
            "file": (
                "modified_filter_older.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert resp_older.status_code == 200

    older_path = Path(UPLOAD_DIR) / "modified_filter_older.ipynb"
    older_stat = older_path.stat()
    os.utime(older_path, (older_stat.st_atime, older_stat.st_mtime - 7200))

    resp_newer = client.post(
        "/api/upload",
        files={
            "file": (
                "modified_filter_newer.ipynb",
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    assert resp_newer.status_code == 200

    older_modified_at = client.get(
        "/api/notebooks/modified_filter_older.ipynb/info"
    ).json()["modified_at"]
    newer_modified_at = client.get(
        "/api/notebooks/modified_filter_newer.ipynb/info"
    ).json()["modified_at"]

    # modified_after excludes the older entry, keeps the newer one.
    notebooks = client.get(
        "/api/notebooks",
        params={"search": "modified_filter_", "modified_after": newer_modified_at},
    ).json()["notebooks"]
    assert [nb["filename"] for nb in notebooks] == ["modified_filter_newer.ipynb"]

    # modified_before excludes the newer entry, keeps the older one.
    notebooks = client.get(
        "/api/notebooks",
        params={"search": "modified_filter_", "modified_before": older_modified_at},
    ).json()["notebooks"]
    assert [nb["filename"] for nb in notebooks] == ["modified_filter_older.ipynb"]

    # Both bounds together, wide enough to include both.
    notebooks = client.get(
        "/api/notebooks",
        params={
            "search": "modified_filter_",
            "modified_after": older_modified_at,
            "modified_before": newer_modified_at,
        },
    ).json()["notebooks"]
    assert sorted(nb["filename"] for nb in notebooks) == [
        "modified_filter_newer.ipynb", "modified_filter_older.ipynb",
    ]


def test_list_notebooks_modified_after_is_inclusive():

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "modified_inclusive.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert resp.status_code == 200

    modified_at = client.get(
        "/api/notebooks/modified_inclusive.ipynb/info"
    ).json()["modified_at"]

    notebooks = client.get(
        "/api/notebooks",
        params={
            "search": "modified_inclusive",
            "modified_after": modified_at,
            "modified_before": modified_at,
        },
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == ["modified_inclusive.ipynb"]


def test_list_notebooks_rejects_modified_after_later_than_modified_before():

    resp = client.get(
        "/api/notebooks",
        params={
            "modified_after": "2026-06-01T00:00:00+00:00",
            "modified_before": "2026-01-01T00:00:00+00:00",
        },
    )

    assert resp.status_code == 400
    assert "modified_after" in resp.json()["detail"]


def test_list_notebooks_rejects_a_malformed_modified_after():

    resp = client.get("/api/notebooks", params={"modified_after": "not-a-date"})

    assert resp.status_code == 400
    assert "modified_after" in resp.json()["detail"]


def test_list_notebooks_rejects_a_malformed_modified_before():

    resp = client.get("/api/notebooks", params={"modified_before": "not-a-date"})

    assert resp.status_code == 400
    assert "modified_before" in resp.json()["detail"]


def test_list_notebooks_modified_after_naive_value_is_treated_as_utc():

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "modified_naive.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert resp.status_code == 200

    modified_at = client.get(
        "/api/notebooks/modified_naive.ipynb/info"
    ).json()["modified_at"]
    naive_modified_at = modified_at.split("+")[0]

    notebooks = client.get(
        "/api/notebooks",
        params={"search": "modified_naive", "modified_after": naive_modified_at},
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == ["modified_naive.ipynb"]


def test_list_notebooks_description_search_is_case_insensitive():

    _upload_sample_notebook("desc_search_case.ipynb")
    client.put(
        "/api/notebooks/desc_search_case.ipynb/description",
        json={"description": "The Quarterly Churn Model."},
    )

    notebooks = client.get(
        "/api/notebooks?search=desc_search_case&description_search=quarterly churn"
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == ["desc_search_case.ipynb"]


def test_list_notebooks_description_search_excludes_notebooks_without_a_description():

    _upload_sample_notebook("desc_search_none.ipynb")

    notebooks = client.get(
        "/api/notebooks?search=desc_search_none&description_search=anything"
    ).json()["notebooks"]

    assert notebooks == []


def test_list_notebooks_regex_matches_a_pattern_not_a_literal_substring():

    _upload_sample_notebook("list_regex_report_2024.ipynb")
    _upload_sample_notebook("list_regex_report_2025.ipynb")

    # A plain substring for "list_regex_report_" alone matches both
    # notebooks; the pattern below only matches the one ending in
    # "2024.ipynb".
    notebooks = client.get(
        "/api/notebooks",
        params={"search": r"2024\.ipynb$", "regex": "true"},
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == ["list_regex_report_2024.ipynb"]


def test_list_notebooks_regex_is_case_insensitive():

    _upload_sample_notebook("list_regex_CaseTest.ipynb")

    notebooks = client.get(
        "/api/notebooks",
        params={"search": "^list_regex_casetest", "regex": "true"},
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == ["list_regex_CaseTest.ipynb"]


def test_list_notebooks_regex_false_treats_search_as_a_plain_substring():

    _upload_sample_notebook("list_regex_abc.ipynb")

    # "a.c" would match "abc" as a *regex* (any-character "."), but not
    # as a literal substring the filename doesn't actually contain.
    notebooks = client.get(
        "/api/notebooks",
        params={"search": "regex_a.c", "regex": "false"},
    ).json()["notebooks"]

    assert notebooks == []


def test_list_notebooks_regex_applies_to_description_search_too():

    _upload_sample_notebook("list_regex_desc.ipynb")
    client.put(
        "/api/notebooks/list_regex_desc.ipynb/description",
        json={"description": "quarterly churn model v2"},
    )

    notebooks = client.get(
        "/api/notebooks",
        params={
            "search": "list_regex_desc",
            "description_search": r"v\d+",
            "regex": "true",
        },
    ).json()["notebooks"]

    assert [nb["filename"] for nb in notebooks] == ["list_regex_desc.ipynb"]

    no_match = client.get(
        "/api/notebooks",
        params={
            "search": "list_regex_desc",
            "description_search": r"v\d{3,}",
            "regex": "true",
        },
    ).json()["notebooks"]

    assert no_match == []


def test_list_notebooks_regex_rejects_an_invalid_search_pattern():

    resp = client.get(
        "/api/notebooks", params={"search": "(unclosed", "regex": "true"}
    )

    assert resp.status_code == 400
    assert "search is not a valid regular expression" in resp.json()["detail"]


def test_list_notebooks_regex_rejects_an_invalid_description_search_pattern():

    resp = client.get(
        "/api/notebooks",
        params={"description_search": "(unclosed", "regex": "true"},
    )

    assert resp.status_code == 400
    assert "description_search is not a valid regular expression" in resp.json()["detail"]


def test_list_notebooks_regex_rejects_a_catastrophically_backtracking_search_pattern():
    """GET /api/notebooks?search=&regex=true is one of three endpoints
    sharing _compile_search_regex's own nested-unbounded-repetition
    check (see its own docstring) -- confirms the wiring actually reaches
    it, not just _compile_search_regex/_regex_has_nested_unbounded_repetition
    in isolation (covered directly below).
    """

    resp = client.get(
        "/api/notebooks", params={"search": "(a+)+", "regex": "true"}
    )

    assert resp.status_code == 400
    assert "search" in resp.json()["detail"]
    assert "nested" in resp.json()["detail"].lower()


def test_list_notebooks_regex_rejects_an_overlong_search_pattern():

    resp = client.get(
        "/api/notebooks",
        params={"search": "a" * (MAX_SEARCH_REGEX_LENGTH + 1), "regex": "true"},
    )

    assert resp.status_code == 400
    assert "maximum regular expression length" in resp.json()["detail"]


# _compile_search_regex/_regex_has_nested_unbounded_repetition are shared
# by GET /api/functions, GET /api/notebooks (its own "search"/
# "description_search"), and GET /api/notebooks/search-content -- each of
# which already has its own "rejects an invalid pattern"-style test
# exercising this through the real HTTP route (see
# test_list_notebooks_regex_rejects_a_catastrophically_backtracking_search_pattern
# above, test_search_functions_regex_rejects_a_catastrophically_backtracking_pattern,
# and test_search_notebook_content_regex_rejects_a_catastrophically_backtracking_pattern
# below) -- this block instead tests the shared helpers directly and
# exhaustively, once, rather than repeating the same battery of
# dangerous/safe patterns at each of the three call sites.
def test_regex_has_nested_unbounded_repetition_flags_classic_redos_shapes():

    dangerous_patterns = [
        "(a+)+",
        "(a*)*",
        "(a+)*",
        "(a*)+",
        "(.*)*",
        "(.+)+",
        "(?:a+)+",
        r"(\d+)+",
        "(a+)+b",
    ]

    for pattern_text in dangerous_patterns:

        parsed = _sre_parser.parse(pattern_text)

        assert _regex_has_nested_unbounded_repetition(parsed), pattern_text


def test_regex_has_nested_unbounded_repetition_does_not_flag_ordinary_patterns():

    safe_patterns = [
        "train_model",
        r"\w+",
        r"def \w+\(",
        "a+b+",
        "(a|b)+",
        "foo.*bar",
        "^https?://",
        r"\d{3}-\d{4}",
        "(abc)+",
        r"[a-z]+_[a-z]+",
        r"(?=a+)b+",
        "(foo|bar){2,5}",
        r"(a{2,5})+",
    ]

    for pattern_text in safe_patterns:

        parsed = _sre_parser.parse(pattern_text)

        assert not _regex_has_nested_unbounded_repetition(parsed), pattern_text


def test_compile_search_regex_rejects_a_nested_unbounded_repetition():

    with pytest.raises(HTTPException) as exc_info:
        _compile_search_regex("(a+)+")

    assert exc_info.value.status_code == 400
    assert "nested" in exc_info.value.detail.lower()


def test_compile_search_regex_rejects_an_overlong_pattern():

    with pytest.raises(HTTPException) as exc_info:
        _compile_search_regex("a" * (MAX_SEARCH_REGEX_LENGTH + 1))

    assert exc_info.value.status_code == 400
    assert "maximum regular expression length" in exc_info.value.detail


def test_compile_search_regex_rejects_a_syntactically_invalid_pattern():

    with pytest.raises(HTTPException) as exc_info:
        _compile_search_regex("(unclosed")

    assert exc_info.value.status_code == 400
    assert "not a valid regular expression" in exc_info.value.detail


def test_compile_search_regex_uses_field_name_in_every_error_message():

    for bad_pattern, expected_fragment in (
        ("(a+)+", "nested"),
        ("a" * (MAX_SEARCH_REGEX_LENGTH + 1), "maximum regular expression length"),
        ("(unclosed", "not a valid regular expression"),
    ):

        with pytest.raises(HTTPException) as exc_info:
            _compile_search_regex(bad_pattern, field_name="description_search")

        assert exc_info.value.detail.startswith("description_search ")
        assert expected_fragment in exc_info.value.detail.lower() or (
            expected_fragment in exc_info.value.detail
        )


def test_compile_search_regex_returns_a_working_pattern_for_a_safe_regex():

    pattern = _compile_search_regex(r"train_\w+")

    assert pattern.search("train_model") is not None
    assert pattern.search("TRAIN_MODEL") is not None  # case-insensitive
    assert pattern.search("predict") is None


def test_delete_notebook_removes_its_tags_sidecar_file():

    _upload_sample_notebook("tags_delete_single.ipynb")
    client.put("/api/notebooks/tags_delete_single.ipynb/tags", json={"tags": ["bug"]})
    assert _tags_sidecar_path("tags_delete_single.ipynb").is_file()

    delete_resp = client.delete("/api/notebooks/tags_delete_single.ipynb")
    assert delete_resp.status_code == 200

    assert not _tags_sidecar_path("tags_delete_single.ipynb").is_file()

    # A notebook re-uploaded under the same name afterward must not
    # silently inherit the deleted notebook's old tags.
    _upload_sample_notebook("tags_delete_single.ipynb")
    assert client.get(
        "/api/notebooks/tags_delete_single.ipynb/tags"
    ).json()["tags"] == []


def test_delete_all_notebooks_removes_tags_sidecar_files():

    _upload_sample_notebook("tags_delete_all.ipynb")
    client.put("/api/notebooks/tags_delete_all.ipynb/tags", json={"tags": ["bug"]})
    assert _tags_sidecar_path("tags_delete_all.ipynb").is_file()

    resp = client.delete("/api/notebooks?confirm=true")
    assert resp.status_code == 200

    assert not _tags_sidecar_path("tags_delete_all.ipynb").is_file()


def test_rename_notebook_moves_its_tags_to_the_new_name():

    _upload_sample_notebook("tags_rename_source.ipynb")
    client.put(
        "/api/notebooks/tags_rename_source.ipynb/tags", json={"tags": ["bug"]}
    )

    rename_resp = client.patch(
        "/api/notebooks/tags_rename_source.ipynb",
        json={"new_filename": "tags_rename_target.ipynb"},
    )
    assert rename_resp.status_code == 200

    assert not _tags_sidecar_path("tags_rename_source.ipynb").is_file()
    assert client.get(
        "/api/notebooks/tags_rename_target.ipynb/tags"
    ).json()["tags"] == ["bug"]

    os.remove(Path(UPLOAD_DIR) / "tags_rename_target.ipynb")
    _tags_sidecar_path("tags_rename_target.ipynb").unlink(missing_ok=True)


def test_rename_notebook_moves_its_source_url_to_the_new_name(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/source_url_rename_source.ipynb", "filename": "source_url_rename_source.ipynb"},
    )

    rename_resp = client.patch(
        "/api/notebooks/source_url_rename_source.ipynb",
        json={"new_filename": "source_url_rename_target.ipynb"},
    )
    assert rename_resp.status_code == 200

    assert not _source_url_sidecar_path("source_url_rename_source.ipynb").is_file()
    assert client.get(
        "/api/notebooks/source_url_rename_target.ipynb/info"
    ).json()["source_url"] == f"{base_url}/source_url_rename_source.ipynb"

    os.remove(Path(UPLOAD_DIR) / "source_url_rename_target.ipynb")
    _source_url_sidecar_path("source_url_rename_target.ipynb").unlink(missing_ok=True)


def test_copy_notebook_inherits_source_url_from_the_source(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f(): return 1\n")

    client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/copy_source_url_source.ipynb", "filename": "copy_source_url_source.ipynb"},
    )

    copy_resp = client.post(
        "/api/notebooks/copy_source_url_source.ipynb/copy",
        json={"new_filename": "copy_source_url_target.ipynb"},
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/copy_source_url_target.ipynb/info"
    ).json()["source_url"] == f"{base_url}/copy_source_url_source.ipynb"

    os.remove(Path(UPLOAD_DIR) / "copy_source_url_target.ipynb")
    _source_url_sidecar_path("copy_source_url_target.ipynb").unlink(missing_ok=True)


def test_copy_notebook_source_url_is_null_when_the_source_has_none():

    _upload_sample_notebook("copy_source_url_none_source.ipynb")

    copy_resp = client.post(
        "/api/notebooks/copy_source_url_none_source.ipynb/copy",
        json={"new_filename": "copy_source_url_none_target.ipynb"},
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/copy_source_url_none_target.ipynb/info"
    ).json()["source_url"] is None

    os.remove(Path(UPLOAD_DIR) / "copy_source_url_none_target.ipynb")


def test_rename_notebook_overwrite_discards_the_destinations_previous_tags():

    _upload_sample_notebook("tags_rename_overwrite_source.ipynb")
    _upload_sample_notebook("tags_rename_overwrite_target.ipynb")
    client.put(
        "/api/notebooks/tags_rename_overwrite_target.ipynb/tags",
        json={"tags": ["stale"]},
    )

    rename_resp = client.patch(
        "/api/notebooks/tags_rename_overwrite_source.ipynb",
        json={
            "new_filename": "tags_rename_overwrite_target.ipynb",
            "overwrite": True,
        },
    )
    assert rename_resp.status_code == 200

    assert client.get(
        "/api/notebooks/tags_rename_overwrite_target.ipynb/tags"
    ).json()["tags"] == []

    os.remove(Path(UPLOAD_DIR) / "tags_rename_overwrite_target.ipynb")
    _tags_sidecar_path("tags_rename_overwrite_target.ipynb").unlink(missing_ok=True)


def test_list_notebook_versions_is_empty_for_a_notebook_never_overwritten():

    _upload_sample_notebook("versions_never_overwritten.ipynb")

    resp = client.get("/api/notebooks/versions_never_overwritten.ipynb/versions")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "filename": "versions_never_overwritten.ipynb",
        "versions": [],
        "total_count": 0,
        "limit": None,
        "offset": 0,
    }


def test_list_notebook_versions_returns_404_for_missing_notebook():

    resp = client.get("/api/notebooks/versions_does_not_exist.ipynb/versions")

    assert resp.status_code == 404


def test_list_notebook_versions_limit_caps_the_returned_versions():

    filename = "versions_list_limit.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    for i in range(3):
        client.post(
            "/api/upload?overwrite=true",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes(f"def g{i}() -> int:\n    return {i}\n")),
                    "application/json",
                )
            },
        )

    all_versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(all_versions) == 3

    resp = client.get(f"/api/notebooks/{filename}/versions", params={"limit": 2})

    assert resp.status_code == 200
    body = resp.json()
    assert [v["version_id"] for v in body["versions"]] == [
        v["version_id"] for v in all_versions[:2]
    ]
    assert body["total_count"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0


def test_list_notebook_versions_offset_skips_the_newest_first_entries():

    filename = "versions_list_offset.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    for i in range(3):
        client.post(
            "/api/upload?overwrite=true",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes(f"def g{i}() -> int:\n    return {i}\n")),
                    "application/json",
                )
            },
        )

    all_versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]

    resp = client.get(f"/api/notebooks/{filename}/versions", params={"offset": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert [v["version_id"] for v in body["versions"]] == [
        v["version_id"] for v in all_versions[1:]
    ]
    assert body["total_count"] == 3
    assert body["offset"] == 1


def test_list_notebook_versions_rejects_a_negative_offset():

    _upload_sample_notebook("versions_list_bad_offset.ipynb")

    resp = client.get(
        "/api/notebooks/versions_list_bad_offset.ipynb/versions",
        params={"offset": -1},
    )

    assert resp.status_code == 400


def test_list_notebook_versions_rejects_a_non_positive_limit():

    _upload_sample_notebook("versions_list_bad_limit.ipynb")

    resp = client.get(
        "/api/notebooks/versions_list_bad_limit.ipynb/versions",
        params={"limit": 0},
    )

    assert resp.status_code == 400


def test_list_notebook_versions_csv_format_returns_a_csv_response():

    filename = "versions_csv.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    json_versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(json_versions) == 1

    resp = client.get(f"/api/notebooks/{filename}/versions", params={"format": "csv"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert (
        f'attachment; filename="{filename}_versions.csv"'
        in resp.headers["content-disposition"]
    )

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == "version_id,size_bytes,saved_at"
    assert len(rows) == 2
    version = json_versions[0]
    assert rows[1] == f"{version['version_id']},{version['size_bytes']},{version['saved_at']}"


def test_list_notebook_versions_csv_format_returns_404_for_missing_notebook():

    resp = client.get(
        "/api/notebooks/does_not_exist_versions_csv.ipynb/versions",
        params={"format": "csv"},
    )

    assert resp.status_code == 404


def test_list_notebook_versions_rejects_an_unknown_format():

    _upload_sample_notebook("versions_list_bad_format.ipynb")

    resp = client.get(
        "/api/notebooks/versions_list_bad_format.ipynb/versions",
        params={"format": "xml"},
    )

    assert resp.status_code == 400
    assert "format" in resp.json()["detail"]


def test_list_notebook_versions_checksums_reports_each_versions_own_sha256():

    filename = "versions_checksums.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    resp = client.get(
        f"/api/notebooks/{filename}/versions", params={"checksums": "true"}
    )

    assert resp.status_code == 200
    version = resp.json()["versions"][0]

    downloaded = client.get(
        f"/api/notebooks/{filename}/versions/{version['version_id']}"
    )
    assert version["sha256"] == hashlib.sha256(downloaded.content).hexdigest()


def test_list_notebook_versions_omits_sha256_by_default():

    filename = "versions_no_checksums.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    resp = client.get(f"/api/notebooks/{filename}/versions")

    assert "sha256" not in resp.json()["versions"][0]


def test_list_notebook_versions_checksums_csv_format_adds_a_sha256_column():

    filename = "versions_checksums_csv.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    json_versions = client.get(
        f"/api/notebooks/{filename}/versions", params={"checksums": "true"}
    ).json()["versions"]

    resp = client.get(
        f"/api/notebooks/{filename}/versions",
        params={"checksums": "true", "format": "csv"},
    )

    assert resp.status_code == 200
    rows = resp.text.strip().split("\r\n")
    assert rows[0] == "version_id,size_bytes,saved_at,sha256"

    version = json_versions[0]
    assert rows[1] == (
        f"{version['version_id']},{version['size_bytes']},"
        f"{version['saved_at']},{version['sha256']}"
    )


def _age_notebook_version(filename, version_id, seconds_ago):
    version_path = _notebook_versions_dir(filename) / version_id
    version_stat = version_path.stat()
    os.utime(
        version_path, (version_stat.st_atime, version_stat.st_mtime - seconds_ago)
    )


def test_list_notebook_versions_filters_by_saved_after_and_before():

    filename = "versions_saved_filter.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def h() -> int:\n    return 3\n")),
                "application/json",
            )
        },
    )

    all_versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(all_versions) == 2

    older_version, newer_version = all_versions[1], all_versions[0]
    _age_notebook_version(filename, older_version["version_id"], 7200)

    older_saved_at = client.get(f"/api/notebooks/{filename}/versions").json()[
        "versions"
    ][1]["saved_at"]
    newer_saved_at = newer_version["saved_at"]

    # saved_after excludes the older entry, keeps the newer one.
    versions = client.get(
        f"/api/notebooks/{filename}/versions",
        params={"saved_after": newer_saved_at},
    ).json()["versions"]
    assert [v["version_id"] for v in versions] == [newer_version["version_id"]]

    # saved_before excludes the newer entry, keeps the older one.
    versions = client.get(
        f"/api/notebooks/{filename}/versions",
        params={"saved_before": older_saved_at},
    ).json()["versions"]
    assert [v["version_id"] for v in versions] == [older_version["version_id"]]

    # Both bounds together, wide enough to include both.
    resp = client.get(
        f"/api/notebooks/{filename}/versions",
        params={"saved_after": older_saved_at, "saved_before": newer_saved_at},
    )
    body = resp.json()
    assert sorted(v["version_id"] for v in body["versions"]) == sorted(
        [older_version["version_id"], newer_version["version_id"]]
    )
    assert body["total_count"] == 2


def test_list_notebook_versions_saved_after_is_inclusive():

    filename = "versions_saved_inclusive.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    saved_at = client.get(f"/api/notebooks/{filename}/versions").json()["versions"][0][
        "saved_at"
    ]

    versions = client.get(
        f"/api/notebooks/{filename}/versions",
        params={"saved_after": saved_at, "saved_before": saved_at},
    ).json()["versions"]

    assert len(versions) == 1
    assert versions[0]["saved_at"] == saved_at


def test_list_notebook_versions_rejects_saved_after_later_than_saved_before():

    _upload_sample_notebook("versions_saved_bad_range.ipynb")

    resp = client.get(
        "/api/notebooks/versions_saved_bad_range.ipynb/versions",
        params={
            "saved_after": "2026-06-01T00:00:00+00:00",
            "saved_before": "2026-01-01T00:00:00+00:00",
        },
    )

    assert resp.status_code == 400
    assert "saved_after" in resp.json()["detail"]


def test_list_notebook_versions_rejects_a_malformed_saved_after():

    _upload_sample_notebook("versions_saved_bad_after.ipynb")

    resp = client.get(
        "/api/notebooks/versions_saved_bad_after.ipynb/versions",
        params={"saved_after": "not-a-date"},
    )

    assert resp.status_code == 400
    assert "saved_after" in resp.json()["detail"]


def test_list_notebook_versions_rejects_a_malformed_saved_before():

    _upload_sample_notebook("versions_saved_bad_before.ipynb")

    resp = client.get(
        "/api/notebooks/versions_saved_bad_before.ipynb/versions",
        params={"saved_before": "not-a-date"},
    )

    assert resp.status_code == 400
    assert "saved_before" in resp.json()["detail"]


def test_overwriting_a_notebook_snapshots_the_previous_content():

    filename = "versions_overwrite_snapshots.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    resp = client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    assert resp.status_code == 200

    resp = client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    assert resp.status_code == 200
    assert resp.json()["overwritten"] is True

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]

    assert len(versions) == 1
    assert versions[0]["size_bytes"] == len(original_content)
    assert "saved_at" in versions[0]

    downloaded = client.get(
        f"/api/notebooks/{filename}/versions/{versions[0]['version_id']}"
    )
    assert downloaded.status_code == 200
    assert downloaded.content == original_content


def test_get_notebook_version_reports_the_content_sha256_header():

    filename = "versions_get_sha256.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.get(f"/api/notebooks/{filename}/versions/{version_id}")

    assert resp.status_code == 200
    assert resp.headers["x-content-sha256"] == hashlib.sha256(resp.content).hexdigest()


def test_get_notebook_version_content_sha256_header_differs_from_the_current_content():

    filename = "versions_get_sha256_differs.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    version_sha256 = client.get(
        f"/api/notebooks/{filename}/versions/{version_id}"
    ).headers["x-content-sha256"]
    current_sha256 = client.get(
        f"/api/notebooks/{filename}"
    ).headers["x-content-sha256"]

    assert version_sha256 != current_sha256


def test_get_notebook_version_returns_304_when_if_none_match_matches_the_current_etag():

    filename = "versions_get_conditional.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    first = client.get(f"/api/notebooks/{filename}/versions/{version_id}")
    etag = first.headers["etag"]
    assert etag == f'"{first.headers["x-content-sha256"]}"'

    second = client.get(
        f"/api/notebooks/{filename}/versions/{version_id}",
        headers={"If-None-Match": etag},
    )

    assert second.status_code == 304
    assert second.content == b""
    assert first.headers["cache-control"] == "no-cache"
    assert second.headers["cache-control"] == "no-cache"


def test_get_notebook_version_returns_200_when_if_none_match_does_not_match():

    filename = "versions_get_conditional_mismatch.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.get(
        f"/api/notebooks/{filename}/versions/{version_id}",
        headers={"If-None-Match": '"not-a-real-etag"'},
    )

    assert resp.status_code == 200
    assert resp.content


def test_uploading_a_brand_new_notebook_does_not_snapshot_anything():

    filename = "versions_brand_new.ipynb"
    _upload_sample_notebook(filename)

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]

    assert versions == []


def test_get_notebook_version_returns_404_for_an_unknown_version_id():

    _upload_sample_notebook("versions_unknown_id.ipynb")

    resp = client.get(
        "/api/notebooks/versions_unknown_id.ipynb/versions/not_a_real_version.ipynb"
    )

    assert resp.status_code == 404


def test_get_notebook_version_rejects_an_absolute_version_id():
    """Same protection resolve_upload_path/resolve_generated_path already
    apply to their own respective root directories (see
    test_get_notebook_rejects_absolute_filename), applied here to
    version_id's own root -- this notebook's version directory.
    """

    _upload_sample_notebook("versions_traversal.ipynb")

    resp = client.get("/api/notebooks/versions_traversal.ipynb/versions/%2Fetc%2Fpasswd")

    assert resp.status_code in (400, 404)
    assert "root:" not in resp.text


def test_export_notebook_versions_bundles_current_content_and_every_version():

    filename = "versions_export_bundle.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")
    middle_content = _notebook_bytes("def g() -> int:\n    return 2\n")
    current_content = _notebook_bytes("def h() -> int:\n    return 3\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(middle_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(current_content), "application/json")},
    )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(versions) == 2
    version_ids = {v["version_id"] for v in versions}

    export_resp = client.get(f"/api/notebooks/{filename}/versions/export")

    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:

        names = set(archive.namelist())
        assert filename in names
        assert names - {filename} == {f"versions/{vid}" for vid in version_ids}

        assert archive.read(filename) == current_content

        version_contents = {archive.read(f"versions/{vid}") for vid in version_ids}
        assert version_contents == {original_content, middle_content}


def test_export_notebook_versions_reports_a_bundle_sha256():

    filename = "versions_export_bundle_sha.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")
    current_content = _notebook_bytes("def h() -> int:\n    return 3\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(current_content), "application/json")},
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    export_resp = client.get(f"/api/notebooks/{filename}/versions/export")

    assert export_resp.status_code == 200
    bundle_sha256 = export_resp.headers["x-bundle-sha256"]

    expected = _bundle_sha256([
        {"filename": filename, "sha256": hashlib.sha256(current_content).hexdigest()},
        {
            "filename": f"versions/{version_id}",
            "sha256": hashlib.sha256(original_content).hexdigest(),
        },
    ])
    assert bundle_sha256 == expected


def test_export_notebook_versions_succeeds_with_no_version_history():

    _upload_sample_notebook("versions_export_no_history.ipynb")

    export_resp = client.get(
        "/api/notebooks/versions_export_no_history.ipynb/versions/export"
    )

    assert export_resp.status_code == 200

    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:
        assert archive.namelist() == ["versions_export_no_history.ipynb"]


def test_export_notebook_versions_version_ids_exports_only_the_chosen_subset():

    filename = "versions_export_subset.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")
    middle_content = _notebook_bytes("def g() -> int:\n    return 2\n")
    current_content = _notebook_bytes("def h() -> int:\n    return 3\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(middle_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(current_content), "application/json")},
    )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(versions) == 2
    chosen_version_id = versions[0]["version_id"]
    other_version_id = versions[1]["version_id"]

    export_resp = client.get(
        f"/api/notebooks/{filename}/versions/export",
        params={"version_ids": chosen_version_id},
    )

    assert export_resp.status_code == 200

    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:

        names = set(archive.namelist())
        # Current content is always included, regardless of version_ids.
        assert filename in names
        assert f"versions/{chosen_version_id}" in names
        assert f"versions/{other_version_id}" not in names
        assert archive.read(filename) == current_content


def test_export_notebook_versions_version_ids_accepts_a_comma_separated_list():

    filename = "versions_export_subset_multi.ipynb"

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(_notebook_bytes("def h() -> int:\n    return 3\n")), "application/json")},
    )

    version_ids = [
        v["version_id"]
        for v in client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    ]
    assert len(version_ids) == 2

    export_resp = client.get(
        f"/api/notebooks/{filename}/versions/export",
        params={"version_ids": ",".join(version_ids)},
    )

    assert export_resp.status_code == 200

    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:
        names = set(archive.namelist())
        assert names - {filename} == {f"versions/{vid}" for vid in version_ids}


def test_export_notebook_versions_version_ids_returns_404_naming_every_unknown_id():

    filename = "versions_export_subset_unknown.ipynb"
    _upload_sample_notebook(filename)

    export_resp = client.get(
        f"/api/notebooks/{filename}/versions/export",
        params={"version_ids": "does-not-exist-a.ipynb,does-not-exist-b.ipynb"},
    )

    assert export_resp.status_code == 404
    detail = export_resp.json()["detail"]
    assert "does-not-exist-a.ipynb" in detail
    assert "does-not-exist-b.ipynb" in detail


def test_export_notebook_versions_returns_404_for_missing_notebook():

    resp = client.get(
        "/api/notebooks/versions_export_does_not_exist.ipynb/versions/export"
    )

    assert resp.status_code == 404


def test_import_notebook_versions_round_trips_an_export_archive():

    filename = "versions_import_round_trip.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")
    middle_content = _notebook_bytes("def g() -> int:\n    return 2\n")
    current_content = _notebook_bytes("def h() -> int:\n    return 3\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(middle_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(current_content), "application/json")},
    )

    original_versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(original_versions) == 2

    export_bytes = client.get(f"/api/notebooks/{filename}/versions/export").content

    # Delete the notebook (and its history) entirely, then restore it
    # under a brand-new filename from the exported archive alone.
    client.delete(f"/api/notebooks/{filename}")

    new_filename = "versions_import_round_trip_restored.ipynb"

    import_resp = client.post(
        f"/api/notebooks/{new_filename}/versions/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert import_resp.status_code == 200
    body = import_resp.json()
    assert body["status"] == "success"
    assert body["filename"] == new_filename
    assert body["overwritten"] is False
    assert body["imported_version_count"] == 2
    assert set(body["imported_version_ids"]) == {v["version_id"] for v in original_versions}

    assert client.get(f"/api/notebooks/{new_filename}").content == current_content

    restored_versions = client.get(f"/api/notebooks/{new_filename}/versions").json()["versions"]
    assert {v["version_id"] for v in restored_versions} == {
        v["version_id"] for v in original_versions
    }

    # "saved_at" is reconstructed from each version_id's own encoded
    # timestamp, not left at import time -- close to (within filesystem
    # mtime rounding of) the original save time, not "just now".
    original_saved_at = {v["version_id"]: v["saved_at"] for v in original_versions}
    for restored in restored_versions:
        original_dt = datetime.fromisoformat(original_saved_at[restored["version_id"]])
        restored_dt = datetime.fromisoformat(restored["saved_at"])
        assert abs((restored_dt - original_dt).total_seconds()) < 1

    restored_contents = {
        client.get(f"/api/notebooks/{new_filename}/versions/{v['version_id']}").content
        for v in restored_versions
    }
    assert restored_contents == {original_content, middle_content}


def test_export_notebook_versions_round_trips_tags_and_description():

    filename = "versions_meta_round_trip.ipynb"
    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(content), "application/json")},
    )
    client.put(f"/api/notebooks/{filename}/tags", json={"tags": ["prod", "bug"]})
    client.put(
        f"/api/notebooks/{filename}/description",
        json={"description": "a described notebook"},
    )

    export_bytes = client.get(f"/api/notebooks/{filename}/versions/export").content

    with zipfile.ZipFile(io.BytesIO(export_bytes)) as archive:
        assert json.loads(archive.read("tags.json")) == ["bug", "prod"]
        assert (
            archive.read("description.txt").decode("utf-8") == "a described notebook"
        )

    new_filename = "versions_meta_round_trip_restored.ipynb"

    import_resp = client.post(
        f"/api/notebooks/{new_filename}/versions/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert import_resp.status_code == 200
    body = import_resp.json()
    assert body["restored_tags"] == ["bug", "prod"]
    assert body["restored_description"] == "a described notebook"

    assert client.get(
        f"/api/notebooks/{new_filename}/tags"
    ).json()["tags"] == ["bug", "prod"]
    assert client.get(
        f"/api/notebooks/{new_filename}/description"
    ).json()["description"] == "a described notebook"


def test_export_notebook_versions_omits_tags_and_description_entries_when_unset():

    filename = "versions_meta_unset.ipynb"
    content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(content), "application/json")},
    )

    export_bytes = client.get(f"/api/notebooks/{filename}/versions/export").content

    with zipfile.ZipFile(io.BytesIO(export_bytes)) as archive:
        assert "tags.json" not in archive.namelist()
        assert "description.txt" not in archive.namelist()

    new_filename = "versions_meta_unset_restored.ipynb"

    import_resp = client.post(
        f"/api/notebooks/{new_filename}/versions/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert import_resp.status_code == 200
    body = import_resp.json()
    assert body["restored_tags"] is None
    assert body["restored_description"] is None
    assert client.get(f"/api/notebooks/{new_filename}/tags").json()["tags"] == []


def test_export_notebook_versions_round_trips_source_url(
    notebook_url_server, _bypass_import_url_ssrf_guard
):

    filename = "versions_source_url_round_trip.ipynb"
    base_url, handler = notebook_url_server
    handler.content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/notebooks/import-url",
        json={"url": f"{base_url}/{filename}", "filename": filename},
    )

    export_bytes = client.get(f"/api/notebooks/{filename}/versions/export").content

    with zipfile.ZipFile(io.BytesIO(export_bytes)) as archive:
        assert json.loads(archive.read("source_url.json")) == {
            "source_url": f"{base_url}/{filename}"
        }

    new_filename = "versions_source_url_round_trip_restored.ipynb"

    import_resp = client.post(
        f"/api/notebooks/{new_filename}/versions/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert import_resp.status_code == 200
    body = import_resp.json()
    assert body["restored_source_url"] == f"{base_url}/{filename}"

    assert client.get(
        f"/api/notebooks/{new_filename}/info"
    ).json()["source_url"] == f"{base_url}/{filename}"


def test_export_notebook_versions_omits_source_url_entry_when_unset():

    filename = "versions_source_url_unset.ipynb"
    _upload_sample_notebook(filename)

    export_bytes = client.get(f"/api/notebooks/{filename}/versions/export").content

    with zipfile.ZipFile(io.BytesIO(export_bytes)) as archive:
        assert "source_url.json" not in archive.namelist()

    new_filename = "versions_source_url_unset_restored.ipynb"

    import_resp = client.post(
        f"/api/notebooks/{new_filename}/versions/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert import_resp.status_code == 200
    assert import_resp.json()["restored_source_url"] is None
    assert client.get(
        f"/api/notebooks/{new_filename}/info"
    ).json()["source_url"] is None


def test_import_notebook_versions_succeeds_with_no_version_history():

    filename = "versions_import_no_history.ipynb"
    _upload_sample_notebook(filename)

    export_bytes = client.get(f"/api/notebooks/{filename}/versions/export").content

    new_filename = "versions_import_no_history_restored.ipynb"

    resp = client.post(
        f"/api/notebooks/{new_filename}/versions/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["imported_version_count"] == 0
    assert body["imported_version_ids"] == []
    assert client.get(f"/api/notebooks/{new_filename}/versions").json()["versions"] == []


def test_import_notebook_versions_reports_a_collision_error_without_overwrite():

    filename = "versions_import_collision_source.ipynb"
    _upload_sample_notebook(filename)
    export_bytes = client.get(f"/api/notebooks/{filename}/versions/export").content

    existing_filename = "versions_import_collision_target.ipynb"
    _upload_sample_notebook(existing_filename)

    resp = client.post(
        f"/api/notebooks/{existing_filename}/versions/import",
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert resp.status_code == 409


def test_import_notebook_versions_overwrite_snapshots_the_pre_import_content():

    filename = "versions_import_overwrite_source.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")
    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    export_bytes = client.get(f"/api/notebooks/{filename}/versions/export").content

    existing_filename = "versions_import_overwrite_target.ipynb"
    pre_import_content = _notebook_bytes("def pre() -> int:\n    return 99\n")
    client.post(
        "/api/upload",
        files={
            "file": (
                existing_filename, io.BytesIO(pre_import_content), "application/json"
            )
        },
    )

    resp = client.post(
        f"/api/notebooks/{existing_filename}/versions/import",
        params={"overwrite": "true"},
        files={"file": ("backup.zip", io.BytesIO(export_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["overwritten"] is True

    assert client.get(f"/api/notebooks/{existing_filename}").content == original_content

    versions = client.get(f"/api/notebooks/{existing_filename}/versions").json()["versions"]
    assert len(versions) == 1
    assert (
        client.get(
            f"/api/notebooks/{existing_filename}/versions/{versions[0]['version_id']}"
        ).content
        == pre_import_content
    )


def test_import_notebook_versions_rejects_an_archive_with_no_current_content_entry():

    archive_bytes = _zip_bytes({"README.md": b"not a notebook"})

    resp = client.post(
        "/api/notebooks/versions_import_no_content.ipynb/versions/import",
        files={"file": ("backup.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 400


def test_import_notebook_versions_rejects_an_archive_with_multiple_current_content_entries():

    archive_bytes = _zip_bytes({
        "a.ipynb": _notebook_bytes("def f() -> int:\n    return 1\n"),
        "b.ipynb": _notebook_bytes("def g() -> int:\n    return 2\n"),
    })

    resp = client.post(
        "/api/notebooks/versions_import_multi_content.ipynb/versions/import",
        files={"file": ("backup.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 400


def test_import_notebook_versions_rejects_a_non_zip_file():

    resp = client.post(
        "/api/notebooks/versions_import_bad_file.ipynb/versions/import",
        files={"file": ("backup.txt", io.BytesIO(b"not a zip"), "text/plain")},
    )

    assert resp.status_code == 400


def test_import_notebook_versions_rejects_a_corrupt_zip_file():

    resp = client.post(
        "/api/notebooks/versions_import_corrupt.ipynb/versions/import",
        files={"file": ("backup.zip", io.BytesIO(b"not actually a zip"), "application/zip")},
    )

    assert resp.status_code == 400


def test_import_notebook_versions_succeeds_with_a_matching_expected_sha256():

    filename = "versions_import_expected_sha256_ok.ipynb"
    content = _notebook_bytes("def f() -> int:\n    return 1\n")
    archive_bytes = _zip_bytes({filename: content})
    expected = hashlib.sha256(archive_bytes).hexdigest()

    resp = client.post(
        f"/api/notebooks/{filename}/versions/import",
        params={"expected_sha256": expected},
        files={"file": ("backup.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 200
    assert (Path(UPLOAD_DIR) / filename).is_file()


def test_import_notebook_versions_rejects_a_mismatched_expected_sha256_before_writing_anything():

    filename = "versions_import_expected_sha256_bad.ipynb"
    content = _notebook_bytes("def f() -> int:\n    return 1\n")
    archive_bytes = _zip_bytes({filename: content})

    resp = client.post(
        f"/api/notebooks/{filename}/versions/import",
        params={"expected_sha256": "0" * 64},
        files={"file": ("backup.zip", io.BytesIO(archive_bytes), "application/zip")},
    )

    assert resp.status_code == 400
    assert "does not match expected_sha256" in resp.json()["detail"]
    assert not (Path(UPLOAD_DIR) / filename).is_file()


def test_inspect_notebook_version_reports_functions_and_dependencies_for_that_snapshot():

    filename = "versions_inspect_snapshot.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "import pandas\n\n"
                    "def add(a: int, b: int) -> int:\n    return a + b\n"
                )),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.get(f"/api/notebooks/{filename}/versions/{version_id}/inspect")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["filename"] == filename
    assert body["version_id"] == version_id
    assert [f["name"] for f in body["functions"]] == ["add"]
    assert any(dep.startswith("pandas") for dep in body["dependencies"])
    assert body["endpoints"] == [
        {"path": "/add", "method": "POST", "is_async": False}
    ]
    assert body["reserved_name_conflicts"] == []
    assert body["skipped_functions"] == []


def test_inspect_notebook_version_reflects_the_old_snapshot_not_current_content():

    filename = "versions_inspect_not_current.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def old_fn() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def new_fn() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.get(f"/api/notebooks/{filename}/versions/{version_id}/inspect")

    assert resp.status_code == 200
    function_names = [f["name"] for f in resp.json()["functions"]]
    assert function_names == ["old_fn"]
    assert "new_fn" not in function_names


def test_inspect_notebook_version_reports_reserved_name_conflicts():

    filename = "versions_inspect_reserved_name.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def health_check() -> dict:\n    return {}\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def fine() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.get(f"/api/notebooks/{filename}/versions/{version_id}/inspect")

    assert resp.status_code == 200
    assert resp.json()["reserved_name_conflicts"] == ["health_check"]


def test_inspect_notebook_version_returns_404_for_missing_notebook():

    resp = client.get(
        "/api/notebooks/versions_inspect_missing_notebook.ipynb/versions/x.ipynb/inspect"
    )

    assert resp.status_code == 404


def test_inspect_notebook_version_returns_404_for_an_unknown_version_id():

    _upload_sample_notebook("versions_inspect_unknown_id.ipynb")

    resp = client.get(
        "/api/notebooks/versions_inspect_unknown_id.ipynb/versions/not_real.ipynb/inspect"
    )

    assert resp.status_code == 404


def test_diff_notebook_version_against_current_live_content():

    filename = "versions_diff_against_live.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def add(a: int, b: int) -> int:\n    return a + b\n\n"
                    "def remove_me() -> int:\n    return 0\n\n"
                    "def unchanged_fn() -> int:\n    return 1\n"
                )),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def add(a: int, b: int, c: int) -> int:\n    return a + b + c\n\n"
                    "def add_me() -> int:\n    return 2\n\n"
                    "def unchanged_fn() -> int:\n    return 1\n"
                )),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.get(f"/api/notebooks/{filename}/versions/{version_id}/diff")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["filename"] == filename
    assert body["version_id"] == version_id
    assert body["against"] is None
    assert [f["name"] for f in body["added"]] == ["add_me"]
    assert [f["name"] for f in body["removed"]] == ["remove_me"]
    assert [c["name"] for c in body["changed"]] == ["add"]
    assert body["unchanged"] == ["unchanged_fn"]
    assert "content_diff" not in body
    assert body["compatible"] is False
    breaking_types = {c["type"] for c in body["breaking_changes"]}
    assert "removed_endpoint" in breaking_types
    assert "required_parameter_added" in breaking_types


def test_diff_notebook_version_reports_compatible_when_nothing_would_break_callers():

    filename = "versions_diff_compatible.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def add(a: int, b: int) -> int:\n    return a + b\n"
                )),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def add(a: int, b: int, c: int = 0) -> int:\n    return a + b + c\n"
                )),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.get(f"/api/notebooks/{filename}/versions/{version_id}/diff")

    assert resp.status_code == 200
    body = resp.json()
    assert body["compatible"] is True
    assert body["breaking_changes"] == []


def test_diff_notebook_version_content_true_reports_a_line_level_diff_using_friendly_labels():

    filename = "versions_diff_content.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def add(a: int, b: int) -> int:\n    return a + b\n"
                )),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def add(a: int, b: int, c: int) -> int:\n    return a + b + c\n"
                )),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.get(
        f"/api/notebooks/{filename}/versions/{version_id}/diff",
        params={"content": "true"},
    )

    assert resp.status_code == 200
    body = resp.json()
    content_diff = body["content_diff"]
    assert any("def add(a: int, b: int, c: int)" in line for line in content_diff)

    # Friendly labels, not the version snapshot's own on-disk path under
    # UPLOAD_DIR/.versions/.
    header = "\n".join(content_diff[:2])
    assert f"version '{version_id}'" in header
    assert f"the current live content of '{filename}'" in header
    assert ".versions" not in header


def test_diff_notebook_version_against_another_version():

    filename = "versions_diff_against_version.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def f() -> int:\n    return 1\n\ndef g() -> int:\n    return 2\n"
                )),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )

    # Newest first (see list_notebook_versions), so index 0 is the
    # second-uploaded content (with `g`) and index 1 is the very first.
    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    newer_version_id = versions[0]["version_id"]
    older_version_id = versions[1]["version_id"]

    resp = client.get(
        f"/api/notebooks/{filename}/versions/{older_version_id}/diff",
        params={"against": newer_version_id},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["against"] == newer_version_id
    assert [f["name"] for f in body["added"]] == ["g"]
    assert body["removed"] == []
    assert body["unchanged"] == ["f"]


def test_diff_notebook_version_returns_404_for_missing_notebook():

    resp = client.get(
        "/api/notebooks/versions_diff_missing_notebook.ipynb/versions/x.ipynb/diff"
    )

    assert resp.status_code == 404


def test_diff_notebook_version_returns_404_for_an_unknown_version_id():

    _upload_sample_notebook("versions_diff_unknown_id.ipynb")

    resp = client.get(
        "/api/notebooks/versions_diff_unknown_id.ipynb/versions/not_real.ipynb/diff"
    )

    assert resp.status_code == 404


def test_diff_notebook_version_returns_404_for_an_unknown_against_version_id():

    filename = "versions_diff_unknown_against.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.get(
        f"/api/notebooks/{filename}/versions/{version_id}/diff",
        params={"against": "not_real.ipynb"},
    )

    assert resp.status_code == 404


def test_copy_notebook_version_duplicates_a_past_version_into_a_new_notebook():

    filename = "versions_copy_source.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    copy_resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={"new_filename": "versions_copy_target.ipynb"},
    )

    assert copy_resp.status_code == 200
    assert copy_resp.json() == {
        "status": "success",
        "dry_run": False,
        "filename": filename,
        "version_id": version_id,
        "new_filename": "versions_copy_target.ipynb",
    }

    assert (
        Path(UPLOAD_DIR) / "versions_copy_target.ipynb"
    ).read_bytes() == original_content

    # The source notebook's own current content is untouched -- still the
    # *second* upload's content, not the version that was just copied.
    assert (Path(UPLOAD_DIR) / filename).read_bytes() != original_content

    os.remove(Path(UPLOAD_DIR) / "versions_copy_target.ipynb")


def test_copy_notebook_version_does_not_inherit_tags():

    filename = "versions_copy_tags_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    client.put(f"/api/notebooks/{filename}/tags", json={"tags": ["production"]})

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    copy_resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={"new_filename": "versions_copy_tags_target.ipynb"},
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/versions_copy_tags_target.ipynb/tags"
    ).json()["tags"] == []

    os.remove(Path(UPLOAD_DIR) / "versions_copy_tags_target.ipynb")


def test_copy_notebook_version_accepts_explicit_tags_and_description():

    filename = "versions_copy_override_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    copy_resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={
            "new_filename": "versions_copy_override_target.ipynb",
            "tags": ["recovered"],
            "description": "recovered snapshot",
        },
    )
    assert copy_resp.status_code == 200

    target_info = client.get(
        "/api/notebooks/versions_copy_override_target.ipynb/info"
    ).json()
    assert target_info["tags"] == ["recovered"]
    assert target_info["description"] == "recovered snapshot"

    os.remove(Path(UPLOAD_DIR) / "versions_copy_override_target.ipynb")
    _tags_sidecar_path("versions_copy_override_target.ipynb").unlink(missing_ok=True)
    _description_sidecar_path("versions_copy_override_target.ipynb").unlink(missing_ok=True)


def test_copy_notebook_version_rejects_an_invalid_tags_override():

    filename = "versions_copy_bad_tags_source.ipynb"
    _upload_sample_notebook(filename)
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={
            "new_filename": "versions_copy_bad_tags_target.ipynb",
            "tags": "not-a-list",
        },
    )

    assert resp.status_code == 400
    assert not (Path(UPLOAD_DIR) / "versions_copy_bad_tags_target.ipynb").exists()


def test_copy_notebook_version_does_not_copy_source_version_history():

    filename = "versions_copy_history_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    copy_resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={"new_filename": "versions_copy_history_target.ipynb"},
    )
    assert copy_resp.status_code == 200

    assert client.get(
        "/api/notebooks/versions_copy_history_target.ipynb/versions"
    ).json()["versions"] == []

    os.remove(Path(UPLOAD_DIR) / "versions_copy_history_target.ipynb")


def test_copy_notebook_version_returns_404_for_missing_notebook():

    resp = client.post(
        "/api/notebooks/versions_copy_missing_notebook.ipynb/versions/x.ipynb/copy",
        json={"new_filename": "whatever.ipynb"},
    )

    assert resp.status_code == 404


def test_copy_notebook_version_returns_404_for_an_unknown_version_id():

    _upload_sample_notebook("versions_copy_unknown_id.ipynb")

    resp = client.post(
        "/api/notebooks/versions_copy_unknown_id.ipynb/versions/not_real.ipynb/copy",
        json={"new_filename": "whatever.ipynb"},
    )

    assert resp.status_code == 404


def test_copy_notebook_version_requires_new_filename():

    filename = "versions_copy_missing_target.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={},
    )

    assert resp.status_code == 400


def test_copy_notebook_version_rejects_a_non_ipynb_target_name():

    filename = "versions_copy_bad_ext.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={"new_filename": "versions_copy_bad_ext.txt"},
    )

    assert resp.status_code == 400


def test_copy_notebook_version_rejects_copying_onto_its_own_source_filename():

    filename = "versions_copy_onto_self.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={"new_filename": filename},
    )

    assert resp.status_code == 400
    assert "restore" in resp.json()["detail"]


def test_copy_notebook_version_rejects_collision_without_overwrite():

    filename = "versions_copy_collision_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    _upload_sample_notebook("versions_copy_collision_target.ipynb")

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={"new_filename": "versions_copy_collision_target.ipynb"},
    )

    assert resp.status_code == 409
    os.remove(Path(UPLOAD_DIR) / "versions_copy_collision_target.ipynb")


def test_copy_notebook_version_dry_run_reports_the_new_filename_without_copying():

    filename = "versions_copy_dry_run_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={"new_filename": "versions_copy_dry_run_target.ipynb", "dry_run": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["new_filename"] == "versions_copy_dry_run_target.ipynb"

    # Nothing was actually copied.
    assert not (Path(UPLOAD_DIR) / "versions_copy_dry_run_target.ipynb").exists()


def test_copy_notebook_version_overwrite_discards_the_destinations_previous_tags_and_history():

    filename = "versions_copy_overwrite_source.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    target = "versions_copy_overwrite_target.ipynb"
    _upload_sample_notebook(target)
    client.put(f"/api/notebooks/{target}/tags", json={"tags": ["stale"]})
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                target,
                io.BytesIO(_notebook_bytes("def stale_history() -> int:\n    return 0\n")),
                "application/json",
            )
        },
    )
    assert client.get(f"/api/notebooks/{target}/versions").json()["versions"]

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/copy",
        json={"new_filename": target, "overwrite": True},
    )

    assert resp.status_code == 200
    assert (Path(UPLOAD_DIR) / target).read_bytes() == original_content
    assert client.get(f"/api/notebooks/{target}/tags").json()["tags"] == []
    assert client.get(f"/api/notebooks/{target}/versions").json()["versions"] == []

    os.remove(Path(UPLOAD_DIR) / target)


def test_copy_notebook_versions_batch_duplicates_each_version_to_its_own_new_notebook():

    filename = "versions_copy_batch_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def h() -> int:\n    return 3\n")),
                "application/json",
            )
        },
    )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(versions) == 2
    version_a, version_b = versions[0]["version_id"], versions[1]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/copy-batch",
        json={
            "entries": [
                {"version_id": version_a, "new_filename": "versions_copy_batch_a.ipynb"},
                {"version_id": version_b, "new_filename": "versions_copy_batch_b.ipynb"},
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["dry_run"] is False
    assert body["filename"] == filename
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_version = {r["version_id"]: r for r in body["results"]}
    assert results_by_version[version_a]["status"] == "success"
    assert results_by_version[version_a]["new_filename"] == "versions_copy_batch_a.ipynb"
    assert results_by_version[version_b]["status"] == "success"
    assert results_by_version[version_b]["new_filename"] == "versions_copy_batch_b.ipynb"

    assert client.get("/api/notebooks/versions_copy_batch_a.ipynb").status_code == 200
    assert client.get("/api/notebooks/versions_copy_batch_b.ipynb").status_code == 200
    # Source's own version history is untouched.
    assert len(client.get(f"/api/notebooks/{filename}/versions").json()["versions"]) == 2


def test_copy_notebook_versions_batch_per_entry_tags_and_description():

    filename = "versions_copy_batch_override_source.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/copy-batch",
        json={
            "entries": [
                {
                    "version_id": version_id,
                    "new_filename": "versions_copy_batch_override_tagged.ipynb",
                    "tags": ["recovered"],
                    "description": "recovered snapshot",
                },
                {
                    "version_id": version_id,
                    "new_filename": "versions_copy_batch_override_plain.ipynb",
                },
            ]
        },
    )

    assert resp.status_code == 200
    assert resp.json()["succeeded_count"] == 2

    tagged_info = client.get(
        "/api/notebooks/versions_copy_batch_override_tagged.ipynb/info"
    ).json()
    assert tagged_info["tags"] == ["recovered"]
    assert tagged_info["description"] == "recovered snapshot"

    plain_info = client.get(
        "/api/notebooks/versions_copy_batch_override_plain.ipynb/info"
    ).json()
    assert plain_info["tags"] == []
    assert plain_info["description"] == ""


def test_copy_notebook_versions_batch_reports_a_bad_entry_without_aborting_the_rest():

    filename = "versions_copy_batch_partial.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/copy-batch",
        json={
            "entries": [
                {"version_id": version_id, "new_filename": "versions_copy_batch_good.ipynb"},
                {"version_id": "does_not_exist.ipynb", "new_filename": "versions_copy_batch_bad.ipynb"},
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_version = {r["version_id"]: r for r in body["results"]}
    assert results_by_version[version_id]["status"] == "success"
    assert results_by_version["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_version["does_not_exist.ipynb"]["detail"]

    assert client.get("/api/notebooks/versions_copy_batch_good.ipynb").status_code == 200
    assert client.get("/api/notebooks/versions_copy_batch_bad.ipynb").status_code == 404


def test_copy_notebook_versions_batch_dry_run_reports_the_plan_without_copying():

    filename = "versions_copy_batch_dry_run.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/copy-batch",
        json={
            "entries": [
                {"version_id": version_id, "new_filename": "versions_copy_batch_dry_run_target.ipynb"},
            ],
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1

    # Nothing was actually copied.
    assert client.get("/api/notebooks/versions_copy_batch_dry_run_target.ipynb").status_code == 404


def test_copy_notebook_versions_batch_returns_404_for_missing_notebook():

    resp = client.post(
        "/api/notebooks/versions_copy_batch_missing_notebook.ipynb/versions/copy-batch",
        json={"entries": [{"version_id": "whatever.ipynb", "new_filename": "x.ipynb"}]},
    )

    assert resp.status_code == 404


def test_copy_notebook_versions_batch_rejects_a_non_list_entries_value():

    _upload_sample_notebook("versions_copy_batch_bad_value.ipynb")

    resp = client.post(
        "/api/notebooks/versions_copy_batch_bad_value.ipynb/versions/copy-batch",
        json={"entries": "not-a-list"},
    )

    assert resp.status_code == 400


def test_copy_notebook_versions_batch_rejects_an_empty_entries_list():

    _upload_sample_notebook("versions_copy_batch_empty.ipynb")

    resp = client.post(
        "/api/notebooks/versions_copy_batch_empty.ipynb/versions/copy-batch",
        json={"entries": []},
    )

    assert resp.status_code == 400


def test_copy_notebook_versions_batch_rejects_an_entry_missing_new_filename():

    _upload_sample_notebook("versions_copy_batch_missing_field.ipynb")

    resp = client.post(
        "/api/notebooks/versions_copy_batch_missing_field.ipynb/versions/copy-batch",
        json={"entries": [{"version_id": "whatever.ipynb"}]},
    )

    assert resp.status_code == 400


def test_copy_notebook_versions_batch_rejects_more_entries_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    _upload_sample_notebook("versions_copy_batch_too_many.ipynb")

    resp = client.post(
        "/api/notebooks/versions_copy_batch_too_many.ipynb/versions/copy-batch",
        json={
            "entries": [
                {"version_id": "a.ipynb", "new_filename": "x.ipynb"},
                {"version_id": "b.ipynb", "new_filename": "y.ipynb"},
            ]
        },
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_restore_notebook_version_makes_it_the_current_content_again():

    filename = "versions_restore.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    version_id = versions[0]["version_id"]

    restore_resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/restore"
    )

    assert restore_resp.status_code == 200
    assert restore_resp.json() == {
        "status": "success",
        "filename": filename,
        "restored_version_id": version_id,
        "dry_run": False,
        "was_currently_compiled": False,
    }

    assert (Path(UPLOAD_DIR) / filename).read_bytes() == original_content


def test_restore_notebook_version_reports_was_currently_compiled_true_for_the_compiled_source():
    """A restore overwrites `filename`'s own current content exactly like
    POST /api/upload?overwrite=true already does, with the identical
    staleness effect if it's the notebook currently backing
    GENERATED_DIR -- the same "was_currently_compiled" signal that
    endpoint's own overwrite path now reports.
    """

    filename = "versions_restore_was_currently_compiled.ipynb"
    _compile_a_notebook(filename)

    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    version_id = versions[0]["version_id"]

    restore_resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/restore"
    )

    assert restore_resp.status_code == 200
    assert restore_resp.json()["was_currently_compiled"] is True


def test_restore_notebook_version_dry_run_reports_the_plan_without_restoring():
    """"dry_run" confirms the version exists without actually snapshotting
    the current content or copying the version over it -- the identical
    preview POST /api/notebooks/versions/restore-batch's own "dry_run"
    already provides for restoring several different notebooks at once.
    """

    filename = "versions_restore_dry_run.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    current_content = _notebook_bytes("def g() -> int:\n    return 2\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(current_content), "application/json")},
    )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    version_id = versions[0]["version_id"]

    restore_resp = client.post(
        f"/api/notebooks/{filename}/versions/{version_id}/restore",
        params={"dry_run": "true"},
    )

    assert restore_resp.status_code == 200
    assert restore_resp.json() == {
        "status": "success",
        "filename": filename,
        "restored_version_id": version_id,
        "dry_run": True,
        "was_currently_compiled": False,
    }

    # Nothing was actually restored: the current content is unchanged,
    # and no new snapshot was taken of it.
    assert (Path(UPLOAD_DIR) / filename).read_bytes() == current_content
    assert client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"] == versions


def test_restore_notebook_version_itself_snapshots_the_content_it_replaces():
    """Restoring must be undoable too -- otherwise picking the wrong
    version_id would be exactly as destructive as the plain overwrite this
    whole feature exists to make recoverable.
    """

    filename = "versions_restore_is_undoable.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    second_content = _notebook_bytes("def g() -> int:\n    return 2\n")
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(second_content), "application/json")},
    )

    first_version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    client.post(f"/api/notebooks/{filename}/versions/{first_version_id}/restore")

    versions_after_restore = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"]

    assert len(versions_after_restore) == 2

    saved_second_content = next(
        v for v in versions_after_restore if v["version_id"] != first_version_id
    )
    downloaded = client.get(
        f"/api/notebooks/{filename}/versions/{saved_second_content['version_id']}"
    )
    assert downloaded.content == second_content


def test_restore_notebook_version_returns_404_for_missing_notebook():

    resp = client.post(
        "/api/notebooks/versions_missing_notebook.ipynb/versions/whatever.ipynb/restore"
    )

    assert resp.status_code == 404


def test_restore_notebook_version_returns_404_for_an_unknown_version_id():

    _upload_sample_notebook("versions_restore_unknown_id.ipynb")

    resp = client.post(
        "/api/notebooks/versions_restore_unknown_id.ipynb/versions/nope.ipynb/restore"
    )

    assert resp.status_code == 404


def test_restore_notebook_versions_batch_restores_each_notebook_to_its_own_version():

    filename_a = "versions_restore_batch_a.ipynb"
    filename_b = "versions_restore_batch_b.ipynb"

    original_a = _notebook_bytes("def a1() -> int:\n    return 1\n")
    original_b = _notebook_bytes("def b1() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": (filename_a, io.BytesIO(original_a), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename_a,
                io.BytesIO(_notebook_bytes("def a2() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload",
        files={"file": (filename_b, io.BytesIO(original_b), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename_b,
                io.BytesIO(_notebook_bytes("def b2() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_a = client.get(f"/api/notebooks/{filename_a}/versions").json()["versions"][0]["version_id"]
    version_b = client.get(f"/api/notebooks/{filename_b}/versions").json()["versions"][0]["version_id"]

    resp = client.post(
        "/api/notebooks/versions/restore-batch",
        json={
            "entries": [
                {"filename": filename_a, "version_id": version_a},
                {"filename": filename_b, "version_id": version_b},
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["dry_run"] is False
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename[filename_a]["status"] == "success"
    assert results_by_filename[filename_a]["restored_version_id"] == version_a
    assert results_by_filename[filename_b]["status"] == "success"
    assert results_by_filename[filename_b]["restored_version_id"] == version_b

    assert (Path(UPLOAD_DIR) / filename_a).read_bytes() == original_a
    assert (Path(UPLOAD_DIR) / filename_b).read_bytes() == original_b


def test_restore_notebook_versions_batch_reports_was_currently_compiled_per_entry():
    """Mirrors the singular POST .../restore's own identical field --
    only the entry naming the notebook currently backing GENERATED_DIR
    should report True, regardless of how many other entries the same
    batch also restores.
    """

    compiled_filename = "versions_restore_batch_compiled.ipynb"
    _compile_a_notebook(compiled_filename)
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                compiled_filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    other_filename = "versions_restore_batch_uncompiled.ipynb"
    client.post(
        "/api/upload",
        files={"file": (other_filename, io.BytesIO(_notebook_bytes("def h() -> int:\n    return 3\n")), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                other_filename,
                io.BytesIO(_notebook_bytes("def h2() -> int:\n    return 4\n")),
                "application/json",
            )
        },
    )

    compiled_version = client.get(
        f"/api/notebooks/{compiled_filename}/versions"
    ).json()["versions"][0]["version_id"]
    other_version = client.get(
        f"/api/notebooks/{other_filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        "/api/notebooks/versions/restore-batch",
        json={
            "entries": [
                {"filename": compiled_filename, "version_id": compiled_version},
                {"filename": other_filename, "version_id": other_version},
            ]
        },
    )

    assert resp.status_code == 200
    results_by_filename = {r["filename"]: r for r in resp.json()["results"]}
    assert results_by_filename[compiled_filename]["was_currently_compiled"] is True
    assert results_by_filename[other_filename]["was_currently_compiled"] is False

    os.remove(Path(UPLOAD_DIR) / other_filename)


def test_restore_notebook_versions_batch_itself_snapshots_the_content_it_replaces():

    filename = "versions_restore_batch_undoable.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    second_content = _notebook_bytes("def g() -> int:\n    return 2\n")
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(second_content), "application/json")},
    )

    first_version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    client.post(
        "/api/notebooks/versions/restore-batch",
        json={"entries": [{"filename": filename, "version_id": first_version_id}]},
    )

    versions_after_restore = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"]

    assert len(versions_after_restore) == 2


def test_restore_notebook_versions_batch_reports_a_bad_entry_without_aborting_the_rest():

    filename = "versions_restore_batch_partial.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        "/api/notebooks/versions/restore-batch",
        json={
            "entries": [
                {"filename": filename, "version_id": version_id},
                {"filename": "does_not_exist.ipynb", "version_id": "whatever.ipynb"},
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename[filename]["status"] == "success"
    assert results_by_filename["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_filename["does_not_exist.ipynb"]["detail"]


def test_restore_notebook_versions_batch_dry_run_reports_the_plan_without_restoring():

    filename = "versions_restore_batch_dry_run.ipynb"

    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")
    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        "/api/notebooks/versions/restore-batch",
        json={
            "entries": [{"filename": filename, "version_id": version_id}],
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1

    # Nothing was actually restored.
    current_content = (Path(UPLOAD_DIR) / filename).read_bytes()
    assert current_content != original_content


def test_restore_notebook_versions_batch_rejects_a_non_list_entries_value():

    resp = client.post(
        "/api/notebooks/versions/restore-batch",
        json={"entries": "not-a-list"},
    )

    assert resp.status_code == 400


def test_restore_notebook_versions_batch_rejects_an_empty_entries_list():

    resp = client.post(
        "/api/notebooks/versions/restore-batch",
        json={"entries": []},
    )

    assert resp.status_code == 400


def test_restore_notebook_versions_batch_rejects_more_entries_than_the_configured_maximum(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    resp = client.post(
        "/api/notebooks/versions/restore-batch",
        json={
            "entries": [
                {"filename": "a.ipynb", "version_id": "v1.ipynb"},
                {"filename": "b.ipynb", "version_id": "v2.ipynb"},
            ]
        },
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_restore_notebook_versions_batch_rejects_an_entry_missing_version_id():

    _upload_sample_notebook("versions_restore_batch_missing_field.ipynb")

    resp = client.post(
        "/api/notebooks/versions/restore-batch",
        json={"entries": [{"filename": "versions_restore_batch_missing_field.ipynb"}]},
    )

    assert resp.status_code == 400


def test_delete_notebook_version_removes_only_that_snapshot():

    filename = "versions_delete.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def h() -> int:\n    return 3\n")),
                "application/json",
            )
        },
    )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(versions) == 2
    version_id_to_delete = versions[0]["version_id"]
    version_id_to_keep = versions[1]["version_id"]

    delete_resp = client.delete(
        f"/api/notebooks/{filename}/versions/{version_id_to_delete}"
    )

    assert delete_resp.status_code == 200
    assert delete_resp.json() == {
        "status": "success",
        "dry_run": False,
        "filename": filename,
        "deleted_version_id": version_id_to_delete,
    }

    remaining = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert [v["version_id"] for v in remaining] == [version_id_to_keep]

    assert client.get(
        f"/api/notebooks/{filename}/versions/{version_id_to_delete}"
    ).status_code == 404


def test_delete_notebook_version_dry_run_reports_success_without_deleting():

    filename = "versions_delete_dry_run.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.delete(
        f"/api/notebooks/{filename}/versions/{version_id}",
        params={"dry_run": "true"},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "dry_run": True,
        "filename": filename,
        "deleted_version_id": version_id,
    }

    # Nothing was actually deleted.
    remaining = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert [v["version_id"] for v in remaining] == [version_id]


def test_delete_notebook_version_dry_run_still_returns_404_for_an_unknown_version_id():

    _upload_sample_notebook("versions_delete_dry_run_missing.ipynb")

    resp = client.delete(
        "/api/notebooks/versions_delete_dry_run_missing.ipynb/versions/does-not-exist.ipynb",
        params={"dry_run": "true"},
    )

    assert resp.status_code == 404


def test_delete_notebook_versions_batch_removes_only_the_named_versions():

    filename = "versions_delete_batch.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    for i in range(3):
        client.post(
            "/api/upload?overwrite=true",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes(f"def g{i}() -> int:\n    return {i}\n")),
                    "application/json",
                )
            },
        )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(versions) == 3
    to_delete = [versions[0]["version_id"], versions[1]["version_id"]]
    to_keep = versions[2]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/delete-batch",
        json={"version_ids": to_delete},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["filename"] == filename
    assert body["succeeded_count"] == 2
    assert body["failed_count"] == 0

    results_by_id = {r["version_id"]: r for r in body["results"]}
    assert results_by_id[to_delete[0]]["status"] == "success"
    assert results_by_id[to_delete[1]]["status"] == "success"

    remaining = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert [v["version_id"] for v in remaining] == [to_keep]


def test_delete_notebook_versions_batch_reports_a_missing_version_id_without_aborting_the_rest():

    filename = "versions_delete_batch_partial.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/delete-batch",
        json={"version_ids": [version_id, "does_not_exist.ipynb"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_id = {r["version_id"]: r for r in body["results"]}
    assert results_by_id[version_id]["status"] == "success"
    assert results_by_id["does_not_exist.ipynb"]["status"] == "error"
    assert "not found" in results_by_id["does_not_exist.ipynb"]["detail"]

    assert client.get(f"/api/notebooks/{filename}/versions").json()["versions"] == []


def test_delete_notebook_versions_batch_dry_run_reports_the_plan_without_deleting():

    filename = "versions_delete_batch_dry_run.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/delete-batch",
        json={"version_ids": [version_id, "does_not_exist.ipynb"], "dry_run": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1

    results_by_id = {r["version_id"]: r for r in body["results"]}
    assert results_by_id[version_id]["status"] == "success"
    assert results_by_id["does_not_exist.ipynb"]["status"] == "error"

    # Nothing was actually deleted.
    remaining = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert [v["version_id"] for v in remaining] == [version_id]


def test_delete_notebook_versions_batch_non_dry_run_reports_dry_run_false():

    filename = "versions_delete_batch_real.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    resp = client.post(
        f"/api/notebooks/{filename}/versions/delete-batch",
        json={"version_ids": [version_id]},
    )

    assert resp.status_code == 200
    assert resp.json()["dry_run"] is False
    assert client.get(f"/api/notebooks/{filename}/versions").json()["versions"] == []


def test_delete_notebook_versions_batch_returns_404_for_missing_notebook():

    resp = client.post(
        "/api/notebooks/versions_delete_batch_missing_notebook.ipynb/versions/delete-batch",
        json={"version_ids": ["whatever.ipynb"]},
    )

    assert resp.status_code == 404


def test_delete_notebook_versions_batch_rejects_a_non_list_version_ids_value():

    _upload_sample_notebook("versions_delete_batch_bad_value.ipynb")

    resp = client.post(
        "/api/notebooks/versions_delete_batch_bad_value.ipynb/versions/delete-batch",
        json={"version_ids": "not-a-list"},
    )

    assert resp.status_code == 400


def test_delete_notebook_versions_batch_rejects_an_empty_version_ids_list():

    _upload_sample_notebook("versions_delete_batch_empty.ipynb")

    resp = client.post(
        "/api/notebooks/versions_delete_batch_empty.ipynb/versions/delete-batch",
        json={"version_ids": []},
    )

    assert resp.status_code == 400


def test_delete_notebook_versions_batch_rejects_more_version_ids_than_the_configured_maximum(
    monkeypatch,
):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 1)

    _upload_sample_notebook("versions_delete_batch_too_many.ipynb")

    resp = client.post(
        "/api/notebooks/versions_delete_batch_too_many.ipynb/versions/delete-batch",
        json={"version_ids": ["v1.ipynb", "v2.ipynb"]},
    )

    assert resp.status_code == 400
    assert "at most 1" in resp.json()["detail"]


def test_delete_notebook_version_returns_404_for_missing_notebook():

    resp = client.delete(
        "/api/notebooks/versions_delete_missing_notebook.ipynb/versions/whatever.ipynb"
    )

    assert resp.status_code == 404


def test_delete_notebook_version_returns_404_for_an_unknown_version_id():

    _upload_sample_notebook("versions_delete_unknown_id.ipynb")

    resp = client.delete(
        "/api/notebooks/versions_delete_unknown_id.ipynb/versions/nope.ipynb"
    )

    assert resp.status_code == 404


def test_delete_notebook_version_rejects_an_absolute_version_id():

    _upload_sample_notebook("versions_delete_traversal.ipynb")

    resp = client.delete(
        "/api/notebooks/versions_delete_traversal.ipynb/versions/%2Fetc%2Fpasswd"
    )

    assert resp.status_code in (400, 404)
    assert "root:" not in resp.text


def test_clear_notebook_versions_removes_every_snapshot():

    filename = "versions_clear.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    current_content = _notebook_bytes("def h() -> int:\n    return 3\n")
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(current_content),
                "application/json",
            )
        },
    )

    versions_before = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(versions_before) == 2

    clear_resp = client.delete(f"/api/notebooks/{filename}/versions")

    assert clear_resp.status_code == 200
    body = clear_resp.json()
    assert body["status"] == "success"
    assert body["filename"] == filename
    assert body["deleted_count"] == 2
    assert sorted(body["deleted_version_ids"]) == sorted(
        v["version_id"] for v in versions_before
    )

    assert client.get(f"/api/notebooks/{filename}/versions").json()["versions"] == []

    # The notebook's own current content is completely untouched.
    get_resp = client.get(f"/api/notebooks/{filename}")
    assert get_resp.status_code == 200
    assert get_resp.content == current_content


def test_clear_notebook_versions_dry_run_reports_the_plan_without_deleting():

    filename = "versions_clear_dry_run.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    versions_before = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(versions_before) == 1

    resp = client.delete(
        f"/api/notebooks/{filename}/versions", params={"dry_run": "true"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["deleted_count"] == 1
    assert body["deleted_version_ids"] == [versions_before[0]["version_id"]]

    # Nothing was actually deleted.
    remaining = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert [v["version_id"] for v in remaining] == [versions_before[0]["version_id"]]


def test_clear_notebook_versions_non_dry_run_reports_dry_run_false():

    filename = "versions_clear_real_run.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    resp = client.delete(f"/api/notebooks/{filename}/versions")

    assert resp.status_code == 200
    assert resp.json()["dry_run"] is False
    assert client.get(f"/api/notebooks/{filename}/versions").json()["versions"] == []


def test_clear_notebook_versions_is_a_no_op_success_for_a_notebook_with_no_history():

    _upload_sample_notebook("versions_clear_none.ipynb")

    resp = client.delete("/api/notebooks/versions_clear_none.ipynb/versions")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "dry_run": False,
        "filename": "versions_clear_none.ipynb",
        "older_than_days": None,
        "deleted_version_ids": [],
        "deleted_count": 0,
    }


def test_clear_notebook_versions_returns_404_for_missing_notebook():

    resp = client.delete("/api/notebooks/versions_clear_missing.ipynb/versions")

    assert resp.status_code == 404


def test_clear_notebook_versions_does_not_affect_a_different_notebooks_history():

    filename_a = "versions_clear_a.ipynb"
    filename_b = "versions_clear_b.ipynb"

    for filename in (filename_a, filename_b):
        client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                )
            },
        )
        client.post(
            "/api/upload?overwrite=true",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                    "application/json",
                )
            },
        )

    clear_resp = client.delete(f"/api/notebooks/{filename_a}/versions")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["deleted_count"] == 1

    assert client.get(f"/api/notebooks/{filename_a}/versions").json()["versions"] == []
    assert len(client.get(f"/api/notebooks/{filename_b}/versions").json()["versions"]) == 1


def test_clear_notebook_versions_older_than_days_keeps_recent_snapshots():

    filename = "versions_clear_older_than_days.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def h() -> int:\n    return 3\n")),
                "application/json",
            )
        },
    )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(versions) == 2

    old_version_id = versions[1]["version_id"]
    recent_version_id = versions[0]["version_id"]
    _backdate_notebook_version(filename, old_version_id, days_ago=40)

    resp = client.delete(
        f"/api/notebooks/{filename}/versions", params={"older_than_days": 30}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["older_than_days"] == 30
    assert body["deleted_count"] == 1
    assert body["deleted_version_ids"] == [old_version_id]

    remaining = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert [v["version_id"] for v in remaining] == [recent_version_id]


def test_clear_notebook_versions_older_than_days_dry_run_reports_the_plan_without_deleting():

    filename = "versions_clear_older_than_days_dry_run.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]
    _backdate_notebook_version(filename, version_id, days_ago=40)

    resp = client.delete(
        f"/api/notebooks/{filename}/versions",
        params={"older_than_days": 30, "dry_run": "true"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["deleted_version_ids"] == [version_id]

    # Nothing was actually deleted.
    remaining = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert [v["version_id"] for v in remaining] == [version_id]


def test_clear_notebook_versions_older_than_days_rejects_a_non_positive_value():

    _upload_sample_notebook("versions_clear_older_than_days_invalid.ipynb")

    resp = client.delete(
        "/api/notebooks/versions_clear_older_than_days_invalid.ipynb/versions",
        params={"older_than_days": 0},
    )

    assert resp.status_code == 400


def test_notebook_versions_are_pruned_beyond_the_configured_maximum():

    filename = "versions_pruned.ipynb"
    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f0() -> int:\n    return 0\n")),
                "application/json",
            )
        },
    )

    for i in range(1, MAX_NOTEBOOK_VERSIONS + 3):
        client.post(
            "/api/upload?overwrite=true",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes(f"def f{i}() -> int:\n    return {i}\n")),
                    "application/json",
                )
            },
        )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]

    assert len(versions) == MAX_NOTEBOOK_VERSIONS


def _backdate_notebook_version(filename, version_id, days_ago):
    version_path = Path(UPLOAD_DIR) / ".versions" / filename / version_id
    old_time = version_path.stat().st_mtime - (days_ago * 86400)
    os.utime(version_path, (old_time, old_time))


def test_prune_all_notebook_versions_deletes_only_versions_older_than_cutoff():

    filename = "prune_versions_a.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def h() -> int:\n    return 3\n")),
                "application/json",
            )
        },
    )

    versions = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert len(versions) == 2

    old_version_id = versions[1]["version_id"]
    recent_version_id = versions[0]["version_id"]
    _backdate_notebook_version(filename, old_version_id, days_ago=40)

    resp = client.delete("/api/notebooks/versions", params={"older_than_days": 30})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["older_than_days"] == 30
    assert body["notebook_count_affected"] == 1
    assert body["total_deleted_count"] == 1
    assert body["results"] == [{
        "filename": filename,
        "deleted_version_ids": [old_version_id],
        "deleted_count": 1,
    }]

    remaining = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert [v["version_id"] for v in remaining] == [recent_version_id]


def test_prune_all_notebook_versions_spans_multiple_notebooks():

    for filename in ("prune_versions_multi_a.ipynb", "prune_versions_multi_b.ipynb"):

        client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                )
            },
        )
        client.post(
            "/api/upload?overwrite=true",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                    "application/json",
                )
            },
        )
        version_id = client.get(
            f"/api/notebooks/{filename}/versions"
        ).json()["versions"][0]["version_id"]
        _backdate_notebook_version(filename, version_id, days_ago=40)

    resp = client.delete("/api/notebooks/versions", params={"older_than_days": 30})

    assert resp.status_code == 200
    body = resp.json()
    assert body["notebook_count_affected"] == 2
    assert body["total_deleted_count"] == 2

    for filename in ("prune_versions_multi_a.ipynb", "prune_versions_multi_b.ipynb"):
        assert client.get(f"/api/notebooks/{filename}/versions").json()["versions"] == []


def test_prune_all_notebook_versions_filters_by_tag():

    for filename in ("prune_versions_tag_prod.ipynb", "prune_versions_tag_other.ipynb"):

        client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                )
            },
        )
        client.post(
            "/api/upload?overwrite=true",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                    "application/json",
                )
            },
        )
        version_id = client.get(
            f"/api/notebooks/{filename}/versions"
        ).json()["versions"][0]["version_id"]
        _backdate_notebook_version(filename, version_id, days_ago=40)

    client.put(
        "/api/notebooks/prune_versions_tag_prod.ipynb/tags", json={"tags": ["prod"]}
    )

    resp = client.delete(
        "/api/notebooks/versions", params={"older_than_days": 30, "tag": "prod"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["notebook_count_affected"] == 1
    assert body["total_deleted_count"] == 1
    assert [r["filename"] for r in body["results"]] == ["prune_versions_tag_prod.ipynb"]

    assert client.get(
        "/api/notebooks/prune_versions_tag_prod.ipynb/versions"
    ).json()["versions"] == []
    assert len(client.get(
        "/api/notebooks/prune_versions_tag_other.ipynb/versions"
    ).json()["versions"]) == 1


def test_prune_all_notebook_versions_dry_run_reports_the_plan_without_deleting():

    filename = "prune_versions_dry_run.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]
    _backdate_notebook_version(filename, version_id, days_ago=40)

    resp = client.delete(
        "/api/notebooks/versions",
        params={"older_than_days": 30, "dry_run": "true"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["notebook_count_affected"] == 1
    assert body["total_deleted_count"] == 1
    assert body["results"][0]["deleted_version_ids"] == [version_id]

    # Nothing was actually deleted.
    remaining = client.get(f"/api/notebooks/{filename}/versions").json()["versions"]
    assert [v["version_id"] for v in remaining] == [version_id]


def test_prune_all_notebook_versions_non_dry_run_reports_dry_run_false():

    filename = "prune_versions_real_run.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )

    resp = client.delete("/api/notebooks/versions", params={"older_than_days": 30})

    assert resp.status_code == 200
    assert resp.json()["dry_run"] is False


def test_prune_all_notebook_versions_is_a_no_op_when_nothing_is_old_enough():

    filename = "prune_versions_recent.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    resp = client.delete("/api/notebooks/versions", params={"older_than_days": 30})

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"] == []
    assert body["notebook_count_affected"] == 0
    assert body["total_deleted_count"] == 0

    assert len(client.get(f"/api/notebooks/{filename}/versions").json()["versions"]) == 1


def test_prune_all_notebook_versions_leaves_current_content_and_tags_untouched():

    filename = "prune_versions_untouched.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.put(f"/api/notebooks/{filename}/tags", json={"tags": ["production"]})
    current_content = _notebook_bytes("def g() -> int:\n    return 2\n")
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(current_content), "application/json")},
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]
    _backdate_notebook_version(filename, version_id, days_ago=40)

    resp = client.delete("/api/notebooks/versions", params={"older_than_days": 30})
    assert resp.status_code == 200
    assert resp.json()["total_deleted_count"] == 1

    assert client.get(f"/api/notebooks/{filename}").content == current_content
    assert client.get(f"/api/notebooks/{filename}/tags").json()["tags"] == ["production"]


def test_prune_all_notebook_versions_requires_a_positive_older_than_days():

    resp = client.delete("/api/notebooks/versions")
    assert resp.status_code == 400

    resp = client.delete("/api/notebooks/versions", params={"older_than_days": 0})
    assert resp.status_code == 400

    resp = client.delete("/api/notebooks/versions", params={"older_than_days": -5})
    assert resp.status_code == 400


def test_prune_temp_files_deletes_only_part_files_older_than_the_default_threshold(
    monkeypatch,
):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "STALE_UPLOAD_TEMP_FILE_SECONDS", 3600)

    stale_path = Path(UPLOAD_DIR) / ".prune_temp_stale.ipynb.deadbeef.part"
    stale_path.write_text("leftover from a crashed upload", encoding="utf-8")
    stale_size_bytes = stale_path.stat().st_size
    old_time = datetime.now(timezone.utc).timestamp() - 7200
    os.utime(stale_path, (old_time, old_time))

    recent_path = Path(UPLOAD_DIR) / ".prune_temp_recent.ipynb.deadbeef.part"
    recent_path.write_text("still streaming", encoding="utf-8")

    try:
        resp = client.delete("/api/upload/temp-files")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["dry_run"] is False
        assert body["older_than_seconds"] == 3600
        assert body["deleted_count"] == 1
        assert body["deleted_files"][0]["filename"] == stale_path.name
        assert body["deleted_files"][0]["size_bytes"] == stale_size_bytes
        assert body["deleted_files"][0]["age_seconds"] >= 7200
        assert body["reclaimed_bytes"] == stale_size_bytes

        assert not stale_path.exists()
        assert recent_path.exists()
    finally:
        recent_path.unlink(missing_ok=True)
        stale_path.unlink(missing_ok=True)


def test_prune_temp_files_dry_run_reports_the_plan_without_deleting(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "STALE_UPLOAD_TEMP_FILE_SECONDS", 1)

    stale_path = Path(UPLOAD_DIR) / ".prune_temp_dry_run.ipynb.deadbeef.part"
    stale_path.write_text("leftover", encoding="utf-8")
    old_time = datetime.now(timezone.utc).timestamp() - 100
    os.utime(stale_path, (old_time, old_time))

    try:
        resp = client.delete("/api/upload/temp-files", params={"dry_run": "true"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["dry_run"] is True
        assert body["deleted_count"] == 1
        assert body["deleted_files"][0]["filename"] == stale_path.name

        # Still there -- dry_run must never actually delete anything.
        assert stale_path.exists()
    finally:
        stale_path.unlink(missing_ok=True)


def test_prune_temp_files_supports_a_custom_older_than_seconds_override():

    fresh_path = Path(UPLOAD_DIR) / ".prune_temp_override.ipynb.deadbeef.part"
    fresh_path.write_text("just created", encoding="utf-8")
    old_time = datetime.now(timezone.utc).timestamp() - 5
    os.utime(fresh_path, (old_time, old_time))

    try:
        # Default threshold (a full hour) would never catch a 5-second-old
        # file -- but an explicit, much shorter override should.
        resp = client.delete(
            "/api/upload/temp-files", params={"older_than_seconds": 1}
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["older_than_seconds"] == 1
        assert body["deleted_count"] == 1
        assert body["deleted_files"][0]["filename"] == fresh_path.name

        assert not fresh_path.exists()
    finally:
        fresh_path.unlink(missing_ok=True)


def test_prune_temp_files_rejects_a_negative_older_than_seconds():

    resp = client.delete(
        "/api/upload/temp-files", params={"older_than_seconds": -1}
    )

    assert resp.status_code == 400


def test_prune_temp_files_is_a_no_op_when_nothing_is_stale():

    resp = client.delete(
        "/api/upload/temp-files", params={"older_than_seconds": 999999}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_files"] == []
    assert body["deleted_count"] == 0
    assert body["reclaimed_bytes"] == 0


def test_prune_temp_files_ignores_notebooks_and_other_non_part_files(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "STALE_UPLOAD_TEMP_FILE_SECONDS", 1)

    unrelated_path = Path(UPLOAD_DIR) / "not_a_temp_file.txt"
    unrelated_path.write_text("just a stray file", encoding="utf-8")
    old_time = datetime.now(timezone.utc).timestamp() - 100
    os.utime(unrelated_path, (old_time, old_time))

    notebook_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "prune_temp_ignores_notebooks.ipynb",
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    assert notebook_resp.status_code == 200
    notebook_path = Path(UPLOAD_DIR) / "prune_temp_ignores_notebooks.ipynb"
    old_time = datetime.now(timezone.utc).timestamp() - 100
    os.utime(notebook_path, (old_time, old_time))

    try:
        resp = client.delete(
            "/api/upload/temp-files", params={"older_than_seconds": 1}
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted_files"] == []
        assert unrelated_path.exists()
        assert notebook_path.exists()
    finally:
        unrelated_path.unlink(missing_ok=True)


def test_delete_notebook_removes_its_version_history():

    filename = "versions_delete_single.ipynb"
    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    assert _notebook_versions_dir(filename).is_dir()

    delete_resp = client.delete(f"/api/notebooks/{filename}")
    assert delete_resp.status_code == 200

    assert not _notebook_versions_dir(filename).is_dir()


def test_delete_all_notebooks_removes_version_history():

    filename = "versions_delete_all.ipynb"
    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    assert _notebook_versions_dir(filename).is_dir()

    resp = client.delete("/api/notebooks?confirm=true")
    assert resp.status_code == 200

    assert not _notebook_versions_dir(filename).is_dir()


def test_rename_notebook_moves_its_version_history_to_the_new_name():

    source = "versions_rename_source.ipynb"
    target = "versions_rename_target.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                source,
                io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                source,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    rename_resp = client.patch(
        f"/api/notebooks/{source}", json={"new_filename": target}
    )
    assert rename_resp.status_code == 200

    assert not _notebook_versions_dir(source).is_dir()

    versions = client.get(f"/api/notebooks/{target}/versions").json()["versions"]
    assert len(versions) == 1

    os.remove(Path(UPLOAD_DIR) / target)
    shutil.rmtree(_notebook_versions_dir(target), ignore_errors=True)


def test_rename_notebook_overwrite_discards_the_destinations_previous_version_history():

    source = "versions_rename_overwrite_source.ipynb"
    target = "versions_rename_overwrite_target.ipynb"

    for filename in (source, target):
        client.post(
            "/api/upload",
            files={
                "file": (
                    filename,
                    io.BytesIO(_notebook_bytes("def f() -> int:\n    return 1\n")),
                    "application/json",
                )
            },
        )
    client.post(
        f"/api/upload?overwrite=true",
        files={
            "file": (
                target,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )
    assert client.get(f"/api/notebooks/{target}/versions").json()["versions"]

    rename_resp = client.patch(
        f"/api/notebooks/{source}",
        json={"new_filename": target, "overwrite": True},
    )
    assert rename_resp.status_code == 200

    assert client.get(f"/api/notebooks/{target}/versions").json()["versions"] == []

    os.remove(Path(UPLOAD_DIR) / target)
    shutil.rmtree(_notebook_versions_dir(target), ignore_errors=True)


def test_inspect_rejects_absolute_notebook_path():
    """Confirmed exploitable before this fix: passing an absolute path
    like /etc/passwd caused the server to read that file and leak its
    contents back in the HTTP error response.
    """

    resp = client.post("/api/inspect", json={"notebook_path": "/etc/passwd"})

    assert resp.status_code == 400
    assert "passwd" not in resp.text.lower() or "root:" not in resp.text


def test_inspect_rejects_relative_traversal_notebook_path():

    resp = client.post(
        "/api/inspect", json={"notebook_path": "../../../../etc/passwd"}
    )

    assert resp.status_code == 400


def test_compile_rejects_absolute_notebook_path():

    resp = client.post("/api/compile", json={"notebook_path": "/etc/passwd"})

    assert resp.status_code == 400


def test_compile_rejects_relative_traversal_notebook_path():

    resp = client.post(
        "/api/compile", json={"notebook_path": "../../../../etc/passwd"}
    )

    assert resp.status_code == 400


def test_inspect_rejects_a_notebook_path_with_an_embedded_null_byte():
    """Confirmed exploitable before this fix: a null byte in
    "notebook_path" sailed past resolve_upload_path's absolute-path guard
    clause (a null byte isn't special to pathlib's own parsing), but the
    later .resolve() call raised a bare ValueError from the underlying
    os.path.realpath/lstat syscalls, an unhandled 500 instead of the same
    clean 400 an absolute or traversal path already gets above.
    """

    resp = client.post("/api/inspect", json={"notebook_path": "nb\x00.ipynb"})

    assert resp.status_code == 400


def test_compile_rejects_a_notebook_path_with_an_embedded_null_byte():

    resp = client.post("/api/compile", json={"notebook_path": "nb\x00.ipynb"})

    assert resp.status_code == 400


@pytest.mark.parametrize(
    "bad_notebook_path", [123, 1.5, True, ["a.ipynb"], {"path": "a.ipynb"}]
)
def test_inspect_rejects_a_non_string_notebook_path(bad_notebook_path):
    """Confirmed exploitable before this fix: "notebook_path" arrives as a
    raw JSON body field, not a Pydantic-validated string, so a caller can
    send any JSON type there. Path(123) raises a bare TypeError nothing
    here caught, crashing the request with an unhandled 500 instead of
    the same clean 400 a malformed *string* path already got (see
    test_inspect_rejects_absolute_notebook_path above).
    """

    resp = client.post(
        "/api/inspect", json={"notebook_path": bad_notebook_path}
    )

    assert resp.status_code == 400


@pytest.mark.parametrize(
    "bad_notebook_path", [123, 1.5, True, ["a.ipynb"], {"path": "a.ipynb"}]
)
def test_compile_rejects_a_non_string_notebook_path(bad_notebook_path):

    resp = client.post(
        "/api/compile", json={"notebook_path": bad_notebook_path}
    )

    assert resp.status_code == 400


def test_upload_inspect_compile_still_works_for_a_legitimate_notebook():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("legit_test.ipynb", io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200
    assert upload_resp.json()["filename"] == "legit_test.ipynb"

    inspect_resp = client.post(
        "/api/inspect", json={"notebook_path": "legit_test.ipynb"}
    )
    assert inspect_resp.status_code == 200
    functions = inspect_resp.json()["functions"]
    assert functions[0]["name"] == "add"

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "legit_test.ipynb"}
    )
    assert compile_resp.status_code == 200
    assert compile_resp.json()["endpoints"] == [
        {"path": "/add", "method": "POST", "is_async": False}
    ]


def test_compile_endpoints_flag_background_functions_as_async():
    """A dashboard frontend building a UI from /api/compile's response
    previously had no way to tell a background/task_id-based endpoint
    (see LONG_RUNNING_KEYWORDS in generator/api_generator.py) apart from
    a synchronous one -- only the separately-fetched OpenAPI schema
    marked these with x-notebook-to-api-async (see
    test_background_endpoint_documents_the_task_response_it_actually_sends
    in test_generator.py).
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def train_model(epochs: int) -> str:\n    return 'done'\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "async_endpoints_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "async_endpoints_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    endpoints = {e["path"]: e for e in compile_resp.json()["endpoints"]}

    assert endpoints["/add"] == {"path": "/add", "method": "POST", "is_async": False}
    assert endpoints["/train_model"] == {
        "path": "/train_model", "method": "POST", "is_async": True
    }


def test_compile_only_restricts_endpoints_to_the_named_functions():
    """POST /api/compile's "only" field mirrors the CLI's own local
    --only: only the named function(s) should become endpoints, and the
    response's "functions"/"endpoints" should reflect that restriction --
    not just the compiled app on disk.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "compile_only_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile",
        json={"notebook_path": "compile_only_test.ipynb", "only": ["add"]},
    )
    assert compile_resp.status_code == 200

    body = compile_resp.json()
    assert [f["name"] for f in body["functions"]] == ["add"]
    assert [e["path"] for e in body["endpoints"]] == ["/add"]

    generated_app_source = (Path("generated") / "app.py").read_text()
    assert "def add(" in generated_app_source
    assert "def subtract(" not in generated_app_source


def test_compile_exclude_removes_the_named_functions_endpoints():
    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "compile_exclude_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile",
        json={"notebook_path": "compile_exclude_test.ipynb", "exclude": ["subtract"]},
    )
    assert compile_resp.status_code == 200

    body = compile_resp.json()
    assert [f["name"] for f in body["functions"]] == ["add"]
    assert [e["path"] for e in body["endpoints"]] == ["/add"]


def test_compile_rejects_both_only_and_exclude():
    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "compile_only_and_exclude_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile",
        json={
            "notebook_path": "compile_only_and_exclude_test.ipynb",
            "only": ["add"],
            "exclude": ["add"],
        },
    )
    assert compile_resp.status_code == 400
    assert "only and exclude" in compile_resp.json()["detail"]


def test_compile_only_names_an_unknown_function():
    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "compile_only_unknown_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile",
        json={
            "notebook_path": "compile_only_unknown_test.ipynb",
            "only": ["does_not_exist"],
        },
    )
    assert compile_resp.status_code == 400
    assert "does_not_exist" in compile_resp.json()["detail"]


def test_compile_rejects_a_non_list_only_field():
    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "compile_bad_only_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile",
        json={"notebook_path": "compile_bad_only_test.ipynb", "only": "add"},
    )
    assert compile_resp.status_code == 400
    assert "only" in compile_resp.json()["detail"]


def test_compile_response_reports_the_dependencies_actually_pinned_in_requirements_txt():
    """Before this, /api/compile's response had no "dependencies" field
    at all -- a dashboard frontend showing "here's what your notebook
    compiled into" had no way to say what would actually get installed
    into the Docker image (`deploy`/`docker build`'s `pip install -r
    requirements.txt`) without a separate, redundant POST /api/inspect
    call right after compiling.
    """

    content = _notebook_bytes(
        "import pandas as pd\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "compile_dependencies_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "compile_dependencies_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    assert "pandas" in compile_resp.json()["dependencies"]


def test_compile_response_lists_the_generated_files_it_just_wrote():
    """Same gap as "dependencies" above, for the files this compile
    actually produced (app.py, requirements.txt, Dockerfile, ...) -- the
    same "generated_files" field GET /api/download's zip and
    /api/inspect's preview already expose, now also available from the
    compile response itself instead of requiring a follow-up call.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "compile_generated_files_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "compile_generated_files_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    generated_files = compile_resp.json()["generated_files"]

    assert "app.py" in generated_files
    assert "requirements.txt" in generated_files
    assert "Dockerfile" in generated_files


def test_compile_reports_skipped_functions():
    """Before this, a function that couldn't be turned into an endpoint
    (e.g. one taking **kwargs) just silently had no corresponding route in
    /api/compile's response, with nothing to explain why -- the same gap
    /api/inspect's "skipped_functions" field closes for the pre-compile
    preview.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def unsupported(a, **kwargs):\n    return a\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "compile_skipped_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "compile_skipped_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    body = compile_resp.json()

    assert body["skipped_functions"] == [
        {
            "name": "unsupported",
            "reason": (
                "uses *args/**kwargs, which can't be represented as a "
                "fixed set of request fields"
            ),
        }
    ]
    assert {f["name"] for f in body["functions"]} == {"add"}


def test_compile_with_version_id_compiles_the_snapshotted_content():

    filename = "compile_version_id.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def add(a: int, b: int) -> int:\n    return a + b\n"
                )),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def multiply(a: int, b: int) -> int:\n    return a * b\n"
                )),
                "application/json",
            )
        },
    )

    version_id = client.get(f"/api/notebooks/{filename}/versions").json()["versions"][0]["version_id"]

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": filename, "version_id": version_id}
    )

    assert compile_resp.status_code == 200
    body = compile_resp.json()
    assert body["version_id"] == version_id
    assert {f["name"] for f in body["functions"]} == {"add"}
    assert any(ep["path"] == "/add" for ep in body["endpoints"])

    # The notebook's own current content on disk is untouched -- still
    # "multiply", never overwritten by compiling an old version.
    current_content = client.get(f"/api/notebooks/{filename}").text
    assert "def multiply" in current_content
    assert "def add" not in current_content


def test_compile_with_version_id_keeps_the_real_notebook_as_currently_compiled():

    filename = "compile_version_id_metadata.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def add(a: int, b: int) -> int:\n    return a + b\n"
                )),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def multiply(a: int, b: int) -> int:\n    return a * b\n"
                )),
                "application/json",
            )
        },
    )

    version_id = client.get(f"/api/notebooks/{filename}/versions").json()["versions"][0]["version_id"]

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": filename, "version_id": version_id}
    )
    assert compile_resp.status_code == 200

    notebooks = client.get("/api/notebooks").json()["notebooks"]
    entry = next(nb for nb in notebooks if nb["filename"] == filename)

    # source_notebook still names the real notebook (not the version
    # snapshot's own internal path), so it's still recognized as the
    # currently-compiled one -- but its current content ("multiply")
    # genuinely differs from what got compiled ("add"), so this correctly
    # reports it as changed since that compile.
    assert entry["currently_compiled"] is True
    assert entry["notebook_changed_since_compile"] is True


def test_compile_with_version_id_is_persisted_as_compiled_version_id_everywhere():
    """.compile_metadata.json's own "compiled_version_id" (written by
    write_compile_metadata, backend/compiler.py) must be visible from
    every endpoint that already surfaces the currently-compiled entry's
    other compile-time fields ("compiled_at", "notebook_changed_since_
    compile") -- GET /api/notebooks, GET /api/notebooks/{filename}/info,
    and GET /api/generated -- not just POST /api/compile's own one-time
    response.
    """

    filename = "compile_version_id_persisted.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def add(a: int, b: int) -> int:\n    return a + b\n"
                )),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "def multiply(a: int, b: int) -> int:\n    return a * b\n"
                )),
                "application/json",
            )
        },
    )

    version_id = client.get(f"/api/notebooks/{filename}/versions").json()["versions"][0]["version_id"]

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": filename, "version_id": version_id}
    )
    assert compile_resp.status_code == 200

    notebooks = client.get("/api/notebooks").json()["notebooks"]
    list_entry = next(nb for nb in notebooks if nb["filename"] == filename)
    assert list_entry["compiled_version_id"] == version_id

    info_entry = client.get(f"/api/notebooks/{filename}/info").json()
    assert info_entry["compiled_version_id"] == version_id

    generated = client.get("/api/generated").json()
    assert generated["compiled_version_id"] == version_id


def test_list_notebooks_reports_a_null_compiled_version_id_for_an_ordinary_compile():

    filename = "compile_ordinary_version_id.ipynb"
    _compile_a_notebook(filename)

    notebooks = client.get("/api/notebooks").json()["notebooks"]
    entry = next(nb for nb in notebooks if nb["filename"] == filename)

    assert entry["compiled_version_id"] is None

    os.remove(Path(UPLOAD_DIR) / filename)


def test_compile_returns_404_for_an_unknown_version_id():

    filename = "compile_version_id_unknown.ipynb"
    _upload_sample_notebook(filename)

    resp = client.post(
        "/api/compile",
        json={"notebook_path": filename, "version_id": "does-not-exist.ipynb"},
    )

    assert resp.status_code == 404


def test_compile_rejects_a_non_string_version_id():

    filename = "compile_version_id_bad_type.ipynb"
    _upload_sample_notebook(filename)

    resp = client.post(
        "/api/compile", json={"notebook_path": filename, "version_id": 123}
    )

    assert resp.status_code == 400


def test_inspect_reports_skipped_functions_before_compiling():

    content = _notebook_bytes(
        "class Model:\n    def predict(self, x):\n        return x\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "inspect_skipped_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    inspect_resp = client.post(
        "/api/inspect", json={"notebook_path": "inspect_skipped_test.ipynb"}
    )
    assert inspect_resp.status_code == 200

    assert inspect_resp.json()["skipped_functions"] == [
        {
            "name": "predict",
            "reason": (
                "defined inside a class or nested function, so it isn't "
                "callable as a standalone endpoint"
            ),
        }
    ]


def test_inspect_missing_notebook_still_returns_404_not_400():
    """A well-formed, in-bounds filename that simply doesn't exist should
    still 404 (existing behaviour), not be confused with a rejected path.
    """

    resp = client.post(
        "/api/inspect", json={"notebook_path": "does_not_exist_at_all.ipynb"}
    )

    assert resp.status_code == 404


def test_inspect_returns_404_not_500_when_notebook_path_is_a_directory():
    """Confirmed exploitable before this fix: the existence check here
    used to be full_path.exists(), which is also true for a directory --
    and UPLOAD_DIR itself is a valid, in-bounds resolution target for
    notebook_path ("." resolves right back to it via resolve_upload_path,
    the same way it would for any other relative path staying within
    UPLOAD_DIR). The load_notebook call just after it raises
    IsADirectoryError for a directory -- an OSError subclass, not one of
    MALFORMED_NOTEBOOK_ERRORS -- so it propagated completely unhandled,
    past both of this endpoint's own try blocks, into FastAPI's generic,
    detail-free 500.
    """

    resp = client.post("/api/inspect", json={"notebook_path": "."})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Notebook file not found"


def test_compile_returns_404_not_500_when_notebook_path_is_a_directory():
    """Same underlying gap as /api/inspect's identical fix just above,
    for /api/compile's own identical full_path.exists() check -- this
    endpoint doesn't crash unhandled (its own broad `except Exception`
    catches the IsADirectoryError load_notebook raises), but it still
    surfaced as an unhelpful `500 {"detail": "Compilation error: [Errno
    21] Is a directory: ..."}` instead of the same clean 404 a missing
    notebook_path already gets.
    """

    resp = client.post("/api/compile", json={"notebook_path": "."})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Notebook file not found"


def test_inspect_returns_400_not_500_for_a_malformed_notebook_file():
    """Confirmed exploitable before this fix: a notebook file that fails
    nbformat's own load/validation (invalid JSON, or valid JSON missing
    required notebook keys) is a problem with the file's content, not
    this server -- but /api/inspect reported it as a bare 500, the same
    misdiagnosis ReservedFunctionNameError's dedicated 400 already fixed
    for a different failure mode in /api/compile. /api/upload itself
    would never accept content like this, so this writes straight into
    UPLOAD_DIR to reach the endpoint with it, e.g. a file placed there
    outside the API.
    """

    filename = "malformed_inspect_test.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.post("/api/inspect", json={"notebook_path": filename})

    assert resp.status_code == 400
    assert "not a valid Jupyter notebook" in resp.json()["detail"]


def test_compile_returns_400_not_500_for_a_malformed_notebook_file():

    filename = "malformed_compile_test.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.post("/api/compile", json={"notebook_path": filename})

    assert resp.status_code == 400
    assert "not a valid Jupyter notebook" in resp.json()["detail"]


def test_compile_without_smoke_test_omits_the_field():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={
            "file": (
                "smoke_test_omitted.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/compile", json={"notebook_path": "smoke_test_omitted.ipynb"}
    )

    assert resp.status_code == 200
    assert "smoke_test" not in resp.json()


def test_compile_smoke_test_passes_for_a_healthy_compile():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={
            "file": (
                "smoke_test_pass.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/compile",
        json={"notebook_path": "smoke_test_pass.ipynb", "smoke_test": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["smoke_test"] == {
        "passed": True,
        "status_code": 200,
        "detail": None,
    }


def test_compile_smoke_test_reflects_a_recompile_within_the_same_process():
    """The compiled app is imported fresh every time -- a second compile
    exposing a different function must be reflected by a second smoke
    test too, not a module-cache-stale import of the first compile (the
    identical staleness _evict_compiled_app_from_module_cache's own
    docstring already documents for export-openapi).
    """

    first = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={
            "file": (
                "smoke_test_recompile.ipynb",
                io.BytesIO(first),
                "application/json",
            )
        },
    )

    first_resp = client.post(
        "/api/compile",
        json={"notebook_path": "smoke_test_recompile.ipynb", "smoke_test": True},
    )
    assert first_resp.status_code == 200
    assert first_resp.json()["smoke_test"]["passed"] is True

    second = _notebook_bytes("def multiply(a: int, b: int) -> int:\n    return a * b\n")

    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                "smoke_test_recompile.ipynb",
                io.BytesIO(second),
                "application/json",
            )
        },
    )

    second_resp = client.post(
        "/api/compile",
        json={"notebook_path": "smoke_test_recompile.ipynb", "smoke_test": True},
    )
    assert second_resp.status_code == 200
    assert second_resp.json()["smoke_test"]["passed"] is True


def test_compile_smoke_test_fails_cleanly_when_the_compiled_app_cannot_import(
    monkeypatch,
):
    """A codegen bug that writes a syntactically-broken app.py must be
    reported back as a failed smoke test, not crash the whole endpoint --
    the compile itself already succeeded (every file is really on disk),
    so this is a diagnostic, not a fatal error.
    """
    from backend.routes import upload as upload_module

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={
            "file": (
                "smoke_test_broken.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "smoke_test_broken.ipynb"}
    )
    assert compile_resp.status_code == 200

    app_path = Path(GENERATED_DIR) / "app.py"
    original = app_path.read_text(encoding="utf-8")
    app_path.write_text(original + "\nthis is not valid python(((\n", encoding="utf-8")

    try:

        package_name = package_name_for_output_dir(GENERATED_DIR)
        upload_module._evict_compiled_app_from_module_cache(package_name)

        result = upload_module._run_compile_smoke_test(package_name)

        assert result["passed"] is False
        assert result["status_code"] is None
        assert "failed to import" in result["detail"]

    finally:
        app_path.write_text(original, encoding="utf-8")
        upload_module._evict_compiled_app_from_module_cache(
            package_name_for_output_dir(GENERATED_DIR)
        )


def test_inspect_returns_400_not_500_for_valid_json_missing_required_notebook_keys():
    """Distinct failure mode from the invalid-JSON case above: valid JSON
    that isn't a valid notebook (e.g. no "cells" key) raises nbformat's
    ValidationError, not NotJSONError -- both must be treated the same
    way.
    """

    filename = "missing_keys_inspect_test.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"not_a_notebook": True}))

    resp = client.post("/api/inspect", json={"notebook_path": filename})

    assert resp.status_code == 400
    assert "not a valid Jupyter notebook" in resp.json()["detail"]


def test_inspect_reports_dependencies_and_generated_files_after_a_compile():
    """/api/inspect previously only ever returned "functions", even though
    inspect_notebook_data (backend/inspector.py) already computed
    dependencies and generated_files -- it just wasn't wired to this
    route.

    Includes a standard-library import ("math") alongside the real
    third-party one specifically to also confirm "dependencies" only ever
    reports what actually gets pinned into requirements.txt -- "math"
    never does (see _third_party_dependencies in backend/inspector.py),
    so it must not appear here either.
    """

    content = _notebook_bytes(
        "import math\n"
        "import pandas as pd\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "inspect_flow_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "inspect_flow_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    inspect_resp = client.post(
        "/api/inspect", json={"notebook_path": "inspect_flow_test.ipynb"}
    )
    assert inspect_resp.status_code == 200
    body = inspect_resp.json()

    assert body["functions"][0]["name"] == "add"
    assert body["dependencies"] == ["pandas"]
    assert "app.py" in body["generated_files"]
    assert "requirements.txt" in body["generated_files"]


def test_inspect_reports_a_private_directive_marked_function_separately():
    """A function the notebook itself marks "# notebook-to-api: private"
    must be reported in its own "private_functions" field, and must never
    show up in "functions"/"endpoints" -- an actual compile of the same
    notebook would never generate an endpoint for it either.
    """

    content = _notebook_bytes(
        "# notebook-to-api: private\n"
        "def helper(x: int) -> int:\n    return x\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "inspect_private_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/inspect", json={"notebook_path": "inspect_private_test.ipynb"}
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["private_functions"] == ["helper"]
    assert [f["name"] for f in body["functions"]] == ["add"]
    assert [e["path"] for e in body["endpoints"]] == ["/add"]


def test_inspect_reports_an_exclude_directive_marked_import_separately():
    """An import a notebook itself opts out of requirements.txt via
    "# notebook-to-api: exclude <import-name>" must be reported in its own
    "excluded_imports" field, and must never show up in "dependencies" --
    the same "silently dropped, but surfaced separately" precedent
    "private_functions" already sets for "# notebook-to-api: private".
    """

    content = _notebook_bytes(
        "# notebook-to-api: exclude pytest\n"
        "import pytest\n"
        "import requests\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "inspect_excluded_import_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/inspect", json={"notebook_path": "inspect_excluded_import_test.ipynb"}
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["excluded_imports"] == ["pytest"]
    assert "pytest" not in body["dependencies"]
    assert "requests" in body["dependencies"]


def test_inspect_reports_a_function_without_a_docstring_separately():
    """generate_fastapi_code already falls back to a generic, auto-
    generated OpenAPI description for a function with no docstring --
    this must name which function(s) will get that fallback, the same
    "silently missing signal" precedent "private_functions"/
    "excluded_imports" already close for their own directives.
    """

    content = _notebook_bytes(
        "def documented(a: int) -> int:\n"
        "    \"\"\"Doubles a.\"\"\"\n"
        "    return a * 2\n\n"
        "def undocumented(a: int) -> int:\n    return a + 1\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "inspect_no_docstring_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/inspect", json={"notebook_path": "inspect_no_docstring_test.ipynb"}
    )

    assert resp.status_code == 200
    assert resp.json()["functions_without_docstrings"] == ["undocumented"]


def test_inspect_reports_a_redefined_function_as_duplicate():
    """A function name defined more than once in a notebook is silently
    collapsed to its last definition by deduplicate_functions_by_name
    (backend/parser/ast_parser.py) -- but before this, nothing reported
    that a redefinition even happened, indistinguishable from a notebook
    that only ever defined that name once.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def add(a: int, b: int) -> int:\n    return a * b\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "inspect_duplicate_function_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/inspect", json={"notebook_path": "inspect_duplicate_function_test.ipynb"}
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["duplicate_functions"] == ["add"]
    assert len(body["functions"]) == 1


def test_inspect_reports_reserved_name_conflicts_for_a_colliding_function():
    """/api/inspect is the tool's own "preview what compiling this
    notebook will do" step, but had no idea a function named
    "health_check" collides with an identifier the generated app itself
    defines (see RESERVED_INFRASTRUCTURE_NAMES in
    generator/api_generator.py) until /api/compile actually failed on it.
    """

    content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "reserved_name_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    inspect_resp = client.post(
        "/api/inspect", json={"notebook_path": "reserved_name_test.ipynb"}
    )
    assert inspect_resp.status_code == 200

    body = inspect_resp.json()
    assert body["reserved_name_conflicts"] == ["health_check"]


def test_inspect_reports_no_reserved_name_conflicts_for_a_clean_notebook():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "no_conflicts_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    inspect_resp = client.post(
        "/api/inspect", json={"notebook_path": "no_conflicts_test.ipynb"}
    )
    assert inspect_resp.status_code == 200
    assert inspect_resp.json()["reserved_name_conflicts"] == []


def test_validate_reports_pass_for_a_clean_notebook():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "validate_clean.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post("/api/validate", json={"notebook_path": "validate_clean.ipynb"})

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "pass",
        "notebook": "validate_clean.ipynb",
        "version_id": None,
        "reserved_name_conflicts": [],
        "skipped_functions": [],
        "duplicate_functions": [],
    }


def test_validate_reports_warn_for_skipped_functions_without_strict():

    content = _notebook_bytes(
        "def unsupported(a, **kwargs):\n    return a\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "validate_warn.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post("/api/validate", json={"notebook_path": "validate_warn.ipynb"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "warn"
    assert body["reserved_name_conflicts"] == []
    assert [f["name"] for f in body["skipped_functions"]] == ["unsupported"]


def test_validate_reports_fail_for_skipped_functions_with_strict():

    content = _notebook_bytes(
        "def unsupported(a, **kwargs):\n    return a\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "validate_strict_fail.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/validate",
        json={"notebook_path": "validate_strict_fail.ipynb", "strict": True},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "fail"


def test_validate_reports_warn_for_duplicate_functions_without_strict():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def add(a: int, b: int) -> int:\n    return a * b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "validate_duplicate_warn.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/validate", json={"notebook_path": "validate_duplicate_warn.ipynb"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "warn"
    assert body["duplicate_functions"] == ["add"]


def test_validate_reports_fail_for_duplicate_functions_with_strict():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def add(a: int, b: int) -> int:\n    return a * b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "validate_duplicate_strict_fail.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/validate",
        json={"notebook_path": "validate_duplicate_strict_fail.ipynb", "strict": True},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "fail"


def test_validate_reports_fail_for_a_reserved_name_conflict_even_without_strict():

    content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "validate_reserved.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post("/api/validate", json={"notebook_path": "validate_reserved.ipynb"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "fail"
    assert body["reserved_name_conflicts"] == ["health_check"]


def test_validate_exclude_of_the_conflicting_function_reports_pass():

    content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "validate_exclude_reserved.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/validate",
        json={
            "notebook_path": "validate_exclude_reserved.ipynb",
            "exclude": ["health_check"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pass"
    assert body["reserved_name_conflicts"] == []


def test_validate_only_without_the_conflicting_function_reports_pass():

    content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "validate_only_reserved.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/validate",
        json={
            "notebook_path": "validate_only_reserved.ipynb",
            "only": ["add"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pass"
    assert body["reserved_name_conflicts"] == []


def test_validate_only_including_the_conflicting_function_still_reports_fail():

    content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "validate_only_still_conflicts.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/validate",
        json={
            "notebook_path": "validate_only_still_conflicts.ipynb",
            "only": ["health_check", "add"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "fail"
    assert body["reserved_name_conflicts"] == ["health_check"]


def test_validate_rejects_only_and_exclude_together():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("validate_both.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/validate",
        json={
            "notebook_path": "validate_both.ipynb",
            "only": ["add"], "exclude": ["add"],
        },
    )

    assert resp.status_code == 400


def test_validate_rejects_an_unknown_only_name():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("validate_unknown.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/validate",
        json={"notebook_path": "validate_unknown.ipynb", "only": ["does_not_exist"]},
    )

    assert resp.status_code == 400


def test_validate_does_not_touch_generated_dir(monkeypatch, tmp_path):

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "validate_no_side_effects.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    generated_dir = tmp_path / "validate_generated"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.post(
        "/api/validate", json={"notebook_path": "validate_no_side_effects.ipynb"}
    )

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_validate_returns_404_for_a_missing_notebook():

    resp = client.post(
        "/api/validate", json={"notebook_path": "does_not_exist.ipynb"}
    )

    assert resp.status_code == 404


def test_validate_returns_400_for_a_malformed_notebook_file():

    filename = "validate_malformed.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.post("/api/validate", json={"notebook_path": filename})

    assert resp.status_code == 400


def test_validate_with_version_id_validates_that_snapshot_not_current_content():

    filename = "validate_version_id.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def health_check() -> dict:\n    return {}\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(f"/api/notebooks/{filename}/versions").json()["versions"][0]["version_id"]

    current_resp = client.post("/api/validate", json={"notebook_path": filename})
    assert current_resp.json()["status"] == "pass"

    version_resp = client.post(
        "/api/validate", json={"notebook_path": filename, "version_id": version_id}
    )

    assert version_resp.status_code == 200
    body = version_resp.json()
    assert body["status"] == "fail"
    assert body["version_id"] == version_id
    assert body["reserved_name_conflicts"] == ["health_check"]


def test_validate_returns_404_for_an_unknown_version_id():

    filename = "validate_version_id_unknown.ipynb"
    _upload_sample_notebook(filename)

    resp = client.post(
        "/api/validate",
        json={"notebook_path": filename, "version_id": "does-not-exist.ipynb"},
    )

    assert resp.status_code == 404


def test_validate_all_reports_pass_warn_and_fail_across_the_catalog():

    client.delete("/api/notebooks?confirm=true")

    clean_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    warn_content = _notebook_bytes(
        "def unsupported(a, **kwargs):\n    return a\n\n"
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )
    fail_content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n"
    )

    for filename, content in (
        ("validate_all_pass.ipynb", clean_content),
        ("validate_all_warn.ipynb", warn_content),
        ("validate_all_fail.ipynb", fail_content),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    resp = client.get("/api/validate-all")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["pass_count"] == 1
    assert body["warn_count"] == 1
    assert body["fail_count"] == 1

    results_by_filename = {r["filename"]: r for r in body["results"]}
    assert results_by_filename["validate_all_pass.ipynb"]["status"] == "pass"
    assert results_by_filename["validate_all_warn.ipynb"]["status"] == "warn"
    assert results_by_filename["validate_all_fail.ipynb"]["status"] == "fail"
    assert results_by_filename["validate_all_fail.ipynb"]["reserved_name_conflicts"] == [
        "health_check"
    ]


def test_validate_all_csv_format_returns_a_csv_response():

    client.delete("/api/notebooks?confirm=true")

    clean_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    warn_content = _notebook_bytes(
        "def unsupported(a, **kwargs):\n    return a\n\n"
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )
    fail_content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n"
    )

    for filename, content in (
        ("validate_all_csv_pass.ipynb", clean_content),
        ("validate_all_csv_warn.ipynb", warn_content),
        ("validate_all_csv_fail.ipynb", fail_content),
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get("/api/validate-all", params={"format": "csv"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="validate_all.csv"' in resp.headers["content-disposition"]

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == (
        "filename,status,reserved_name_conflicts,skipped_functions,"
        "duplicate_functions,detail"
    )

    by_filename = {row.split(",", 1)[0]: row for row in rows[1:]}

    assert by_filename["validate_all_csv_pass.ipynb"] == "validate_all_csv_pass.ipynb,pass,,,,"
    assert by_filename["validate_all_csv_fail.ipynb"] == (
        "validate_all_csv_fail.ipynb,fail,health_check,,,"
    )
    assert "unsupported: " in by_filename["validate_all_csv_warn.ipynb"]
    assert by_filename["validate_all_csv_warn.ipynb"].startswith(
        "validate_all_csv_warn.ipynb,warn,,"
    )


def test_validate_all_csv_format_composes_with_tag_and_strict():

    client.delete("/api/notebooks?confirm=true")

    warn_content = _notebook_bytes(
        "def unsupported(a, **kwargs):\n    return a\n\n"
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("validate_all_csv_tagged.ipynb", io.BytesIO(warn_content), "application/json")},
    )
    client.put(
        "/api/notebooks/validate_all_csv_tagged.ipynb/tags", json={"tags": ["prod"]}
    )

    resp = client.get(
        "/api/validate-all",
        params={"format": "csv", "tag": "prod", "strict": "true"},
    )

    assert resp.status_code == 200
    rows = resp.text.strip().split("\r\n")
    assert len(rows) == 2
    assert rows[1].startswith("validate_all_csv_tagged.ipynb,fail,")


def test_validate_all_rejects_an_unknown_format():

    resp = client.get("/api/validate-all", params={"format": "xml"})

    assert resp.status_code == 400
    assert "format" in resp.json()["detail"]


def test_validate_all_filters_by_tag():

    client.delete("/api/notebooks?confirm=true")

    clean_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    fail_content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n"
    )

    for filename, content in (
        ("validate_all_tag_prod.ipynb", fail_content),
        ("validate_all_tag_other.ipynb", clean_content),
    ):
        resp = client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    client.put(
        "/api/notebooks/validate_all_tag_prod.ipynb/tags", json={"tags": ["prod"]}
    )

    resp = client.get("/api/validate-all", params={"tag": "prod"})

    assert resp.status_code == 200
    body = resp.json()
    assert [r["filename"] for r in body["results"]] == ["validate_all_tag_prod.ipynb"]
    assert body["fail_count"] == 1
    assert body["pass_count"] == 0


def test_validate_all_unknown_tag_yields_no_results():

    client.delete("/api/notebooks?confirm=true")

    clean_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    resp = client.post(
        "/api/upload",
        files={"file": ("validate_all_no_tag.ipynb", io.BytesIO(clean_content), "application/json")},
    )
    assert resp.status_code == 200

    resp = client.get("/api/validate-all", params={"tag": "no-such-tag"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"] == []
    assert body["pass_count"] == 0
    assert body["warn_count"] == 0
    assert body["fail_count"] == 0


def test_validate_all_strict_turns_skipped_functions_into_a_failure():

    client.delete("/api/notebooks?confirm=true")

    warn_content = _notebook_bytes(
        "def unsupported(a, **kwargs):\n    return a\n"
    )

    resp = client.post(
        "/api/upload",
        files={"file": ("validate_all_strict.ipynb", io.BytesIO(warn_content), "application/json")},
    )
    assert resp.status_code == 200

    resp = client.get("/api/validate-all", params={"strict": "true"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["fail_count"] == 1
    assert body["warn_count"] == 0
    assert body["results"][0]["status"] == "fail"


def test_validate_all_strict_turns_duplicate_functions_into_a_failure():

    client.delete("/api/notebooks?confirm=true")

    duplicate_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def add(a: int, b: int) -> int:\n    return a * b\n"
    )

    resp = client.post(
        "/api/upload",
        files={
            "file": (
                "validate_all_duplicate_strict.ipynb",
                io.BytesIO(duplicate_content),
                "application/json",
            )
        },
    )
    assert resp.status_code == 200

    resp = client.get(
        "/api/validate-all",
        params={"strict": "true"},
    )

    assert resp.status_code == 200
    body = resp.json()
    result = next(
        r for r in body["results"]
        if r["filename"] == "validate_all_duplicate_strict.ipynb"
    )
    assert result["status"] == "fail"
    assert result["duplicate_functions"] == ["add"]


def test_validate_all_reports_a_malformed_notebook_as_fail_instead_of_skipping_it():
    """Deliberately different from GET /api/functions' own bulk search,
    which silently skips a notebook it can't parse -- here, a parse
    failure is exactly the kind of problem this endpoint exists to
    surface, not incidental to a different question.
    """

    client.delete("/api/notebooks?confirm=true")

    filename = "validate_all_malformed.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.get("/api/validate-all")

    assert resp.status_code == 200
    body = resp.json()
    assert body["fail_count"] == 1
    assert body["results"][0]["filename"] == filename
    assert body["results"][0]["status"] == "fail"
    assert "not a valid Jupyter notebook" in body["results"][0]["detail"]


def test_validate_all_limit_caps_the_returned_results():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    for filename in (
        "validate_all_limit_a.ipynb",
        "validate_all_limit_b.ipynb",
        "validate_all_limit_c.ipynb",
    ):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get("/api/validate-all", params={"limit": 2})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert body["result_count"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0
    # Counts still cover every notebook, not just the returned page.
    assert body["pass_count"] == 3


def test_validate_all_offset_skips_the_already_validated_notebooks():

    client.delete("/api/notebooks?confirm=true")

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    for filename in ("validate_all_offset_a.ipynb", "validate_all_offset_b.ipynb"):
        client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), "application/json")},
        )

    resp = client.get("/api/validate-all", params={"offset": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["result_count"] == 2
    assert body["offset"] == 1


def test_validate_all_rejects_a_negative_offset():

    resp = client.get("/api/validate-all", params={"offset": -1})

    assert resp.status_code == 400


def test_validate_all_rejects_a_non_positive_limit():

    resp = client.get("/api/validate-all", params={"limit": 0})

    assert resp.status_code == 400


def test_validate_all_reports_zero_when_nothing_uploaded():

    client.delete("/api/notebooks?confirm=true")

    resp = client.get("/api/validate-all")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "results": [],
        "result_count": 0,
        "limit": None,
        "offset": 0,
        "pass_count": 0,
        "warn_count": 0,
        "fail_count": 0,
    }


def test_requirements_preview_matches_what_an_actual_compile_writes():

    content = _notebook_bytes(
        "import pandas\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "requirements_preview_match.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    preview_resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": "requirements_preview_match.ipynb"},
    )
    assert preview_resp.status_code == 200
    preview_body = preview_resp.json()
    assert preview_body["status"] == "success"
    assert preview_body["notebook"] == "requirements_preview_match.ipynb"

    compile_resp = client.post(
        "/api/compile",
        json={"notebook_path": "requirements_preview_match.ipynb"},
    )
    assert compile_resp.status_code == 200

    actual_requirements = client.get(
        "/api/generated/requirements.txt"
    ).json()["content"].split()

    assert sorted(preview_body["requirements"]) == sorted(actual_requirements)
    assert any(dep.startswith("fastapi") for dep in preview_body["requirements"])
    assert any(dep.startswith("pandas") for dep in preview_body["requirements"])


def test_requirements_preview_explicit_directive_overrides_a_conflicting_auto_detected_import():
    """A notebook importing a package directly while also declaring a
    "# notebook-to-api: requires <same-package>==<version>" directive for
    it must never preview *both* a version-pinned auto-detected line and
    the explicit one -- two pins for the same distribution is a
    requirement pip refuses outright.
    """

    content = _notebook_bytes(
        "# notebook-to-api: requires python-multipart==999.0.0\n"
        "import multipart\n\n"
        "def noop() -> int:\n    return 1\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "requirements_preview_conflict.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": "requirements_preview_conflict.ipynb"},
    )

    assert resp.status_code == 200
    requirements = resp.json()["requirements"]
    multipart_lines = [
        line for line in requirements if line.split("==")[0] == "python-multipart"
    ]
    assert multipart_lines == ["python-multipart==999.0.0"]


def test_requirements_preview_includes_an_explicit_requirement_directive():

    content = _notebook_bytes(
        "# notebook-to-api: requires definitely-not-a-real-pkg==1.2.3\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "requirements_preview_directive.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": "requirements_preview_directive.ipynb"},
    )

    assert resp.status_code == 200
    assert "definitely-not-a-real-pkg==1.2.3" in resp.json()["requirements"]


def test_requirements_preview_explicit_requirements_field_lists_only_directive_lines():

    content = _notebook_bytes(
        "# notebook-to-api: requires definitely-not-a-real-pkg==1.2.3\n"
        "import json\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "requirements_preview_explicit_field.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": "requirements_preview_explicit_field.ipynb"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["explicit_requirements"] == ["definitely-not-a-real-pkg==1.2.3"]
    # Every explicit entry is also present in the merged "requirements" list.
    assert set(body["explicit_requirements"]) <= set(body["requirements"])
    # Auto-detected core dependencies are NOT reported as explicit.
    assert "fastapi" not in body["explicit_requirements"]


def test_requirements_preview_explicit_requirements_field_is_empty_without_a_directive():

    _upload_sample_notebook("requirements_preview_no_directive.ipynb")

    resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": "requirements_preview_no_directive.ipynb"},
    )

    assert resp.status_code == 200
    assert resp.json()["explicit_requirements"] == []


def test_requirements_preview_omits_an_excluded_import():

    content = _notebook_bytes(
        "# notebook-to-api: exclude nbformat\n"
        "import nbformat\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "requirements_preview_exclude.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": "requirements_preview_exclude.ipynb"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert not any(
        dep.startswith("nbformat") for dep in body["requirements"]
    )
    assert body["excluded_imports"] == ["nbformat"]


def test_requirements_preview_excluded_imports_field_is_empty_without_a_directive():

    _upload_sample_notebook("requirements_preview_no_exclude_directive.ipynb")

    resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": "requirements_preview_no_exclude_directive.ipynb"},
    )

    assert resp.status_code == 200
    assert resp.json()["excluded_imports"] == []


def test_requirements_preview_falls_back_to_a_bare_name_for_an_uninstalled_dependency():

    content = _notebook_bytes(
        "import definitely_not_installed_pkg_hopefully\n\n"
        "def noop() -> int:\n    return 1\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "requirements_preview_uninstalled.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": "requirements_preview_uninstalled.ipynb"},
    )

    assert resp.status_code == 200
    assert "definitely_not_installed_pkg_hopefully" in resp.json()["requirements"]


def test_requirements_preview_does_not_touch_generated_dir(monkeypatch, tmp_path):

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "requirements_preview_no_side_effects.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    generated_dir = tmp_path / "requirements_preview_generated"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": "requirements_preview_no_side_effects.ipynb"},
    )

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_requirements_preview_returns_404_for_a_missing_notebook():

    resp = client.post(
        "/api/requirements-preview", json={"notebook_path": "does_not_exist.ipynb"}
    )

    assert resp.status_code == 404


def test_requirements_preview_returns_400_for_a_malformed_notebook_file():

    filename = "requirements_preview_malformed.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.post("/api/requirements-preview", json={"notebook_path": filename})

    assert resp.status_code == 400


def test_requirements_preview_returns_400_for_conflicting_explicit_requirement_directives():
    """_extract_explicit_requirements (backend/compiler.py) raises a
    ValueError for two "# notebook-to-api: requires" directives naming
    the same package with different specs -- this is the notebook's own
    problem, not this server's, so it must surface as a 400 the caller
    can act on, not an unhandled 500.
    """

    filename = "requirements_preview_conflicting_directives.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "# notebook-to-api: requires numpy==1.24.0\n"
                    "# notebook-to-api: requires numpy==1.26.0\n"
                    "def noop() -> int:\n    return 1\n"
                )),
                "application/json",
            )
        },
    )

    resp = client.post("/api/requirements-preview", json={"notebook_path": filename})

    assert resp.status_code == 400
    assert "numpy" in resp.json()["detail"]


def test_requirements_preview_requires_a_notebook_path():

    resp = client.post("/api/requirements-preview", json={})

    assert resp.status_code == 400


def test_requirements_preview_with_version_id_previews_that_snapshots_requirements():

    filename = "requirements_preview_version_id.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes(
                    "import pandas\n\ndef add(a: int, b: int) -> int:\n    return a + b\n"
                )),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(f"/api/notebooks/{filename}/versions").json()["versions"][0]["version_id"]

    current_resp = client.post("/api/requirements-preview", json={"notebook_path": filename})
    assert not any(
        dep.startswith("pandas") for dep in current_resp.json()["requirements"]
    )

    version_resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": filename, "version_id": version_id},
    )

    assert version_resp.status_code == 200
    body = version_resp.json()
    assert body["version_id"] == version_id
    assert any(dep.startswith("pandas") for dep in body["requirements"])


def test_requirements_preview_returns_404_for_an_unknown_version_id():

    filename = "requirements_preview_version_id_unknown.ipynb"
    _upload_sample_notebook(filename)

    resp = client.post(
        "/api/requirements-preview",
        json={"notebook_path": filename, "version_id": "does-not-exist.ipynb"},
    )

    assert resp.status_code == 404


def test_app_preview_matches_what_an_actual_compile_writes():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "app_preview_match.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    preview_resp = client.post(
        "/api/app-preview",
        json={"notebook_path": "app_preview_match.ipynb"},
    )
    assert preview_resp.status_code == 200
    preview_body = preview_resp.json()
    assert preview_body["status"] == "success"
    assert preview_body["notebook"] == "app_preview_match.ipynb"
    assert preview_body["package_name"] == "generated"
    assert "def add(" in preview_body["app_code"]

    compile_resp = client.post(
        "/api/compile",
        json={"notebook_path": "app_preview_match.ipynb"},
    )
    assert compile_resp.status_code == 200

    actual_app_code = client.get("/api/generated/app.py").json()["content"]

    assert preview_body["app_code"] == actual_app_code


def test_app_preview_bakes_in_the_notebooks_own_sha256():

    import hashlib

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "app_preview_sha256.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/app-preview",
        json={"notebook_path": "app_preview_sha256.ipynb"},
    )

    assert resp.status_code == 200
    expected_sha256 = hashlib.sha256(content).hexdigest()
    assert (
        f"SOURCE_NOTEBOOK_SHA256 = '{expected_sha256}'" in resp.json()["app_code"]
    )


def test_app_preview_bakes_in_the_real_tool_version():
    """Must bake in this dashboard's own NOTEBOOK_TO_API_VERSION, the
    same version a real POST /api/compile of the identical notebook
    would -- otherwise this preview's own "matches what a real compile
    would produce" guarantee silently breaks the moment the two drift.
    """

    from backend.compiler import NOTEBOOK_TO_API_VERSION

    _upload_sample_notebook("app_preview_version.ipynb")

    resp = client.post(
        "/api/app-preview",
        json={"notebook_path": "app_preview_version.ipynb"},
    )

    assert resp.status_code == 200
    app_code = resp.json()["app_code"]
    assert f"NOTEBOOK_TO_API_VERSION = {NOTEBOOK_TO_API_VERSION!r}" in app_code
    # The FastAPI(...) app object's own `version=` kwarg (which feeds
    # its OpenAPI "info.version", user-visible in /docs and baked into
    # POST /api/export-openapi's own output) must match too -- a third,
    # separately-hardcoded "1.0.0" literal missed the first time this
    # was fixed.
    assert f"version={NOTEBOOK_TO_API_VERSION!r}" in app_code


def test_app_preview_respects_only_and_exclude():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "app_preview_only.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/app-preview",
        json={"notebook_path": "app_preview_only.ipynb", "only": ["add"]},
    )

    assert resp.status_code == 200
    app_code = resp.json()["app_code"]
    assert "def add(" in app_code
    assert "def subtract(" not in app_code


def test_app_preview_never_exposes_a_private_directive_marked_function():

    content = _notebook_bytes(
        "# notebook-to-api: private\n"
        "def helper(x: int) -> int:\n    return x\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "app_preview_private.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/app-preview",
        json={"notebook_path": "app_preview_private.ipynb"},
    )

    assert resp.status_code == 200
    app_code = resp.json()["app_code"]
    assert '"/add"' in app_code
    assert '"/helper"' not in app_code


def test_app_preview_returns_400_for_only_naming_a_private_function():

    content = _notebook_bytes(
        "# notebook-to-api: private\n"
        "def helper(x: int) -> int:\n    return x\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "app_preview_private_only.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/app-preview",
        json={
            "notebook_path": "app_preview_private_only.ipynb",
            "only": ["helper"],
        },
    )

    assert resp.status_code == 400
    assert "notebook-to-api: private" in resp.json()["detail"]


def test_app_preview_does_not_touch_generated_dir(monkeypatch, tmp_path):

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "app_preview_no_side_effects.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    generated_dir = tmp_path / "app_preview_generated"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.post(
        "/api/app-preview",
        json={"notebook_path": "app_preview_no_side_effects.ipynb"},
    )

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_app_preview_returns_404_for_a_missing_notebook():

    resp = client.post(
        "/api/app-preview", json={"notebook_path": "does_not_exist.ipynb"}
    )

    assert resp.status_code == 404


def test_app_preview_returns_400_for_a_malformed_notebook_file():

    filename = "app_preview_malformed.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.post("/api/app-preview", json={"notebook_path": filename})

    assert resp.status_code == 400


def test_app_preview_requires_a_notebook_path():

    resp = client.post("/api/app-preview", json={})

    assert resp.status_code == 400


def test_app_preview_rejects_both_only_and_exclude():

    resp = client.post(
        "/api/app-preview",
        json={
            "notebook_path": "anything.ipynb",
            "only": ["a"],
            "exclude": ["b"],
        },
    )

    assert resp.status_code == 400


def test_app_preview_returns_400_for_an_unknown_only_name():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "app_preview_unknown_only.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/app-preview",
        json={
            "notebook_path": "app_preview_unknown_only.ipynb",
            "only": ["does_not_exist_fn"],
        },
    )

    assert resp.status_code == 400
    assert "does_not_exist_fn" in resp.json()["detail"]


def test_app_preview_returns_400_for_a_reserved_function_name():

    content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "app_preview_reserved_name.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/app-preview",
        json={"notebook_path": "app_preview_reserved_name.ipynb"},
    )

    assert resp.status_code == 400
    assert "health_check" in resp.json()["detail"]


def test_app_preview_with_version_id_previews_that_snapshots_source():

    filename = "app_preview_version_id.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def old_func() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def new_func() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(f"/api/notebooks/{filename}/versions").json()["versions"][0]["version_id"]

    version_resp = client.post(
        "/api/app-preview", json={"notebook_path": filename, "version_id": version_id}
    )

    assert version_resp.status_code == 200
    body = version_resp.json()
    assert body["version_id"] == version_id
    assert "def old_func" in body["app_code"]
    assert "def new_func" not in body["app_code"]

    current_resp = client.post("/api/app-preview", json={"notebook_path": filename})
    assert "def new_func" in current_resp.json()["app_code"]
    assert "def old_func" not in current_resp.json()["app_code"]


def test_app_preview_returns_404_for_an_unknown_version_id():

    filename = "app_preview_version_id_unknown.ipynb"
    _upload_sample_notebook(filename)

    resp = client.post(
        "/api/app-preview",
        json={"notebook_path": filename, "version_id": "does-not-exist.ipynb"},
    )

    assert resp.status_code == 404


def test_curl_preview_returns_one_command_per_function():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "curl_preview_basic.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/curl-preview",
        json={"notebook_path": "curl_preview_basic.ipynb"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["notebook"] == "curl_preview_basic.ipynb"
    assert len(body["commands"]) == 2
    assert "curl -X POST http://localhost:8000/add" in body["commands"][0]
    assert "X-API-Key: notebook-to-api-dev-key" in body["commands"][0]
    assert "curl -X POST http://localhost:8000/subtract" in body["commands"][1]


def test_curl_preview_respects_custom_host_port_and_api_key():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "curl_preview_custom.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/curl-preview",
        json={
            "notebook_path": "curl_preview_custom.ipynb",
            "host": "api.example.com",
            "port": 9000,
            "api_key": "mykey123",
        },
    )

    assert resp.status_code == 200
    command = resp.json()["commands"][0]
    assert "curl -X POST http://api.example.com:9000/add" in command
    assert "X-API-Key: mykey123" in command


def test_curl_preview_excludes_a_reserved_name_conflict():

    content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "curl_preview_reserved.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/curl-preview",
        json={"notebook_path": "curl_preview_reserved.ipynb"},
    )

    assert resp.status_code == 200
    assert resp.json()["commands"] == []


def test_curl_preview_only_restricts_to_the_named_functions():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("curl_preview_only.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/curl-preview",
        json={"notebook_path": "curl_preview_only.ipynb", "only": ["add"]},
    )

    assert resp.status_code == 200
    commands = resp.json()["commands"]
    assert len(commands) == 1
    assert "curl -X POST http://localhost:8000/add" in commands[0]


def test_curl_preview_exclude_omits_the_named_functions():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("curl_preview_exclude.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/curl-preview",
        json={"notebook_path": "curl_preview_exclude.ipynb", "exclude": ["subtract"]},
    )

    assert resp.status_code == 200
    commands = resp.json()["commands"]
    assert len(commands) == 1
    assert "curl -X POST http://localhost:8000/add" in commands[0]


def test_curl_preview_rejects_only_and_exclude_together():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("curl_preview_both.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/curl-preview",
        json={
            "notebook_path": "curl_preview_both.ipynb",
            "only": ["add"], "exclude": ["add"],
        },
    )

    assert resp.status_code == 400


def test_curl_preview_rejects_an_unknown_only_name():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("curl_preview_unknown.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/curl-preview",
        json={"notebook_path": "curl_preview_unknown.ipynb", "only": ["does_not_exist"]},
    )

    assert resp.status_code == 400


def test_curl_preview_rejects_a_non_list_only_value():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("curl_preview_bad_only.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/curl-preview",
        json={"notebook_path": "curl_preview_bad_only.ipynb", "only": "add"},
    )

    assert resp.status_code == 400


def test_curl_preview_does_not_touch_generated_dir(monkeypatch, tmp_path):

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "curl_preview_no_side_effects.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    generated_dir = tmp_path / "curl_preview_generated"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.post(
        "/api/curl-preview",
        json={"notebook_path": "curl_preview_no_side_effects.ipynb"},
    )

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_curl_preview_returns_404_for_a_missing_notebook():

    resp = client.post(
        "/api/curl-preview", json={"notebook_path": "does_not_exist.ipynb"}
    )

    assert resp.status_code == 404


def test_curl_preview_returns_400_for_a_malformed_notebook_file():

    filename = "curl_preview_malformed.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.post("/api/curl-preview", json={"notebook_path": filename})

    assert resp.status_code == 400


def test_curl_preview_requires_a_notebook_path():

    resp = client.post("/api/curl-preview", json={})

    assert resp.status_code == 400


def test_curl_preview_rejects_a_non_integer_port():

    _upload_sample_notebook("curl_preview_bad_port.ipynb")

    resp = client.post(
        "/api/curl-preview",
        json={"notebook_path": "curl_preview_bad_port.ipynb", "port": "not-a-number"},
    )

    assert resp.status_code == 400


def test_postman_preview_returns_one_item_per_function():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "postman_preview_basic.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/postman-preview",
        json={"notebook_path": "postman_preview_basic.ipynb"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["notebook"] == "postman_preview_basic.ipynb"
    collection = body["collection"]
    assert [item["name"] for item in collection["item"]] == ["add", "subtract"]
    assert collection["info"]["name"] == "postman_preview_basic"


def test_postman_preview_respects_custom_host_port_api_key_and_collection_name():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "postman_preview_custom.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    resp = client.post(
        "/api/postman-preview",
        json={
            "notebook_path": "postman_preview_custom.ipynb",
            "host": "api.example.com",
            "port": 9000,
            "api_key": "mykey123",
            "collection_name": "My API",
        },
    )

    assert resp.status_code == 200
    collection = resp.json()["collection"]
    assert collection["info"]["name"] == "My API"
    variables = {v["key"]: v["value"] for v in collection["variable"]}
    assert variables["base_url"] == "http://api.example.com:9000"
    assert variables["api_key"] == "mykey123"


def test_postman_preview_adds_a_task_status_item_for_a_background_function():

    content = _notebook_bytes(
        "def train_model(epochs: int) -> str:\n    return 'done'\n"
    )

    client.post(
        "/api/upload",
        files={
            "file": (
                "postman_preview_background.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/postman-preview",
        json={"notebook_path": "postman_preview_background.ipynb"},
    )

    assert resp.status_code == 200
    items = resp.json()["collection"]["item"]
    assert [item["name"] for item in items] == [
        "train_model", "train_model - Task Status",
    ]


def test_postman_preview_excludes_a_reserved_name_conflict():

    content = _notebook_bytes("def health_check() -> dict:\n    return {}\n")

    client.post(
        "/api/upload",
        files={
            "file": (
                "postman_preview_reserved.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/postman-preview",
        json={"notebook_path": "postman_preview_reserved.ipynb"},
    )

    assert resp.status_code == 200
    assert resp.json()["collection"]["item"] == []


def test_postman_preview_only_restricts_to_the_named_functions():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )

    client.post(
        "/api/upload",
        files={"file": ("postman_preview_only.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/postman-preview",
        json={"notebook_path": "postman_preview_only.ipynb", "only": ["add"]},
    )

    assert resp.status_code == 200
    items = resp.json()["collection"]["item"]
    assert [item["name"] for item in items] == ["add"]


def test_postman_preview_rejects_only_and_exclude_together():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("postman_preview_both.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/postman-preview",
        json={
            "notebook_path": "postman_preview_both.ipynb",
            "only": ["add"], "exclude": ["add"],
        },
    )

    assert resp.status_code == 400


def test_postman_preview_rejects_an_unknown_only_name():

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    client.post(
        "/api/upload",
        files={"file": ("postman_preview_unknown.ipynb", io.BytesIO(content), "application/json")},
    )

    resp = client.post(
        "/api/postman-preview",
        json={"notebook_path": "postman_preview_unknown.ipynb", "only": ["does_not_exist"]},
    )

    assert resp.status_code == 400


def test_postman_preview_rejects_a_non_string_collection_name():

    _upload_sample_notebook("postman_preview_bad_name.ipynb")

    resp = client.post(
        "/api/postman-preview",
        json={
            "notebook_path": "postman_preview_bad_name.ipynb",
            "collection_name": 123,
        },
    )

    assert resp.status_code == 400


def test_postman_preview_does_not_touch_generated_dir(monkeypatch, tmp_path):

    content = _notebook_bytes("def add(a: int, b: int) -> int:\n    return a + b\n")

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "postman_preview_no_side_effects.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    generated_dir = tmp_path / "postman_preview_generated"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.post(
        "/api/postman-preview",
        json={"notebook_path": "postman_preview_no_side_effects.ipynb"},
    )

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_postman_preview_returns_404_for_a_missing_notebook():

    resp = client.post(
        "/api/postman-preview", json={"notebook_path": "does_not_exist.ipynb"}
    )

    assert resp.status_code == 404


def test_postman_preview_returns_400_for_a_malformed_notebook_file():

    filename = "postman_preview_malformed.ipynb"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("not valid json at all")

    resp = client.post("/api/postman-preview", json={"notebook_path": filename})

    assert resp.status_code == 400


def test_postman_preview_requires_a_notebook_path():

    resp = client.post("/api/postman-preview", json={})

    assert resp.status_code == 400


def test_postman_preview_rejects_a_non_integer_port():

    _upload_sample_notebook("postman_preview_bad_port.ipynb")

    resp = client.post(
        "/api/postman-preview",
        json={"notebook_path": "postman_preview_bad_port.ipynb", "port": "not-a-number"},
    )

    assert resp.status_code == 400


def test_curl_preview_with_version_id_previews_that_snapshots_commands():

    filename = "curl_preview_version_id.ipynb"

    client.post(
        "/api/upload",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def old_func() -> int:\n    return 1\n")),
                "application/json",
            )
        },
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def new_func() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(f"/api/notebooks/{filename}/versions").json()["versions"][0]["version_id"]

    version_resp = client.post(
        "/api/curl-preview", json={"notebook_path": filename, "version_id": version_id}
    )

    assert version_resp.status_code == 200
    body = version_resp.json()
    assert body["version_id"] == version_id
    assert any("old_func" in cmd for cmd in body["commands"])
    assert not any("new_func" in cmd for cmd in body["commands"])


def test_curl_preview_returns_404_for_an_unknown_version_id():

    filename = "curl_preview_version_id_unknown.ipynb"
    _upload_sample_notebook(filename)

    resp = client.post(
        "/api/curl-preview",
        json={"notebook_path": filename, "version_id": "does-not-exist.ipynb"},
    )

    assert resp.status_code == 404


def test_dockerfile_preview_requires_no_notebook_and_needs_no_body():

    resp = client.get("/api/dockerfile-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "FROM python:" in body["dockerfile"]
    assert ".git/" in body["dockerignore"]


def test_dockerfile_preview_reports_the_actual_compiling_python_version():

    resp = client.get("/api/dockerfile-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["compiling_python_version"] == compiling_python_version()
    assert f"FROM python:{compiling_python_version()}-slim" in body["dockerfile"]


def test_dockerfile_preview_matches_what_an_actual_compile_writes():

    filename = "dockerfile_preview_match.ipynb"
    _upload_sample_notebook(filename)

    preview_resp = client.get("/api/dockerfile-preview")
    assert preview_resp.status_code == 200
    preview_body = preview_resp.json()

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": filename}
    )
    assert compile_resp.status_code == 200

    actual_dockerfile = client.get(
        "/api/generated/Dockerfile"
    ).json()["content"]
    actual_dockerignore = client.get(
        "/api/generated/.dockerignore"
    ).json()["content"]

    assert preview_body["dockerfile"] == actual_dockerfile
    assert preview_body["dockerignore"] == actual_dockerignore
    assert preview_body["package_name"] == package_name_for_output_dir(GENERATED_DIR)


def test_dockerfile_preview_reflects_a_configured_package_name(monkeypatch, tmp_path):

    generated_dir = tmp_path / "my_custom_pkg"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.get("/api/dockerfile-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["package_name"] == "my_custom_pkg"
    assert "COPY . my_custom_pkg/" in body["dockerfile"]
    assert "uvicorn my_custom_pkg.app:app" in body["dockerfile"]


def test_dockerfile_preview_does_not_touch_generated_dir(monkeypatch, tmp_path):

    generated_dir = tmp_path / "dockerfile_preview_no_side_effects"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.get("/api/dockerfile-preview")

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_docker_compose_preview_requires_no_notebook_and_needs_no_body():

    resp = client.get("/api/docker-compose-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "services:" in body["docker_compose"]
    assert "NOTEBOOK_API_KEY=${NOTEBOOK_API_KEY:-notebook-to-api-dev-key}" in body["docker_compose"]


def test_docker_compose_preview_matches_what_an_actual_compile_writes():

    filename = "docker_compose_preview_match.ipynb"
    _upload_sample_notebook(filename)

    preview_resp = client.get("/api/docker-compose-preview")
    assert preview_resp.status_code == 200
    preview_body = preview_resp.json()

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": filename}
    )
    assert compile_resp.status_code == 200

    actual_docker_compose = client.get(
        "/api/generated/docker-compose.yml"
    ).json()["content"]

    assert preview_body["docker_compose"] == actual_docker_compose
    assert preview_body["package_name"] == package_name_for_output_dir(GENERATED_DIR)


def test_docker_compose_preview_reflects_a_configured_package_name(monkeypatch, tmp_path):

    generated_dir = tmp_path / "my_custom_pkg"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.get("/api/docker-compose-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["package_name"] == "my_custom_pkg"
    assert "  my_custom_pkg:\n" in body["docker_compose"]


def test_docker_compose_preview_does_not_touch_generated_dir(monkeypatch, tmp_path):

    generated_dir = tmp_path / "docker_compose_preview_no_side_effects"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.get("/api/docker-compose-preview")

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_k8s_preview_requires_no_notebook_and_needs_no_body():

    resp = client.get("/api/k8s-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "kind: Deployment" in body["kubernetes_manifest"]
    assert "kind: Service" in body["kubernetes_manifest"]
    assert 'value: "notebook-to-api-dev-key"' in body["kubernetes_manifest"]


def test_k8s_preview_matches_what_an_actual_compile_writes():

    filename = "k8s_preview_match.ipynb"
    _upload_sample_notebook(filename)

    preview_resp = client.get("/api/k8s-preview")
    assert preview_resp.status_code == 200
    preview_body = preview_resp.json()

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": filename}
    )
    assert compile_resp.status_code == 200

    actual_manifest = client.get(
        "/api/generated/kubernetes.yaml"
    ).json()["content"]

    assert preview_body["kubernetes_manifest"] == actual_manifest
    assert preview_body["package_name"] == package_name_for_output_dir(GENERATED_DIR)


def test_k8s_preview_reflects_a_configured_package_name(monkeypatch, tmp_path):

    generated_dir = tmp_path / "my_custom_pkg"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.get("/api/k8s-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["package_name"] == "my_custom_pkg"
    assert "  name: my_custom_pkg\n" in body["kubernetes_manifest"]
    assert "image: my_custom_pkg:latest\n" in body["kubernetes_manifest"]


def test_k8s_preview_does_not_touch_generated_dir(monkeypatch, tmp_path):

    generated_dir = tmp_path / "k8s_preview_no_side_effects"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.get("/api/k8s-preview")

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_env_example_preview_requires_no_notebook_and_needs_no_body():

    resp = client.get("/api/env-example-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "PORT=8000" in body["env_example"]
    assert "NOTEBOOK_API_KEY=notebook-to-api-dev-key" in body["env_example"]


def test_env_example_preview_matches_what_an_actual_compile_writes():

    filename = "env_example_preview_match.ipynb"
    _upload_sample_notebook(filename)

    preview_resp = client.get("/api/env-example-preview")
    assert preview_resp.status_code == 200
    preview_body = preview_resp.json()

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": filename}
    )
    assert compile_resp.status_code == 200

    actual_env_example = client.get(
        "/api/generated/.env.example"
    ).json()["content"]

    assert preview_body["env_example"] == actual_env_example


def test_env_example_preview_does_not_touch_generated_dir(monkeypatch, tmp_path):

    generated_dir = tmp_path / "env_example_preview_no_side_effects"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.get("/api/env-example-preview")

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_env_vars_preview_requires_no_notebook_and_needs_no_body():

    resp = client.get("/api/env-vars-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"

    env_vars = {entry["name"]: entry for entry in body["environment_variables"]}
    assert set(env_vars) == {
        "NOTEBOOK_API_KEY",
        "NOTEBOOK_API_ALLOWED_ORIGINS",
        "NOTEBOOK_API_MAX_REQUEST_BYTES",
        "NOTEBOOK_API_TASK_TTL_SECONDS",
        "NOTEBOOK_API_MAX_TASKS",
        "NOTEBOOK_API_RATE_LIMIT_PER_MINUTE",
        "NOTEBOOK_API_WEBHOOK_TIMEOUT_SECONDS",
        "NOTEBOOK_API_WEBHOOK_SECRET",
        "NOTEBOOK_API_PUBLIC_URL",
        "NOTEBOOK_API_DISABLE_DOCS",
    }
    assert env_vars["NOTEBOOK_API_KEY"]["default"] == "notebook-to-api-dev-key"
    assert env_vars["NOTEBOOK_API_ALLOWED_ORIGINS"]["default"] == "*"
    assert env_vars["NOTEBOOK_API_MAX_REQUEST_BYTES"]["default"] == str(10 * 1024 * 1024)
    assert env_vars["NOTEBOOK_API_TASK_TTL_SECONDS"]["default"] == "3600"
    assert env_vars["NOTEBOOK_API_MAX_TASKS"]["default"] == "10000"
    assert env_vars["NOTEBOOK_API_RATE_LIMIT_PER_MINUTE"]["default"] == "0"
    assert env_vars["NOTEBOOK_API_WEBHOOK_SECRET"]["default"] == ""
    assert env_vars["NOTEBOOK_API_PUBLIC_URL"]["default"] == "http://localhost:8000"
    assert env_vars["NOTEBOOK_API_DISABLE_DOCS"]["default"] == "false"

    for entry in body["environment_variables"]:
        assert entry["description"]


def test_env_vars_preview_matches_what_an_actual_compiled_app_reads():
    """Every default GET /api/env-vars-preview reports for a
    NOTEBOOK_API_* variable must be the exact same default the actually
    compiled app.py falls back to when that variable is unset -- both
    read from generate_fastapi_code's own GENERATED_APP_ENV_VARS, so
    they can never drift apart.
    """

    filename = "env_vars_preview_match.ipynb"
    _upload_sample_notebook(filename)

    preview_body = client.get("/api/env-vars-preview").json()
    env_vars = {entry["name"]: entry for entry in preview_body["environment_variables"]}

    compile_resp = client.post("/api/compile", json={"notebook_path": filename})
    assert compile_resp.status_code == 200

    generated_app = client.get("/api/generated/app.py").json()["content"]

    assert (
        f'os.getenv("NOTEBOOK_API_KEY", "{env_vars["NOTEBOOK_API_KEY"]["default"]}")'
        in generated_app
    )
    assert (
        'os.getenv("NOTEBOOK_API_ALLOWED_ORIGINS", '
        f'"{env_vars["NOTEBOOK_API_ALLOWED_ORIGINS"]["default"]}")'
        in generated_app
    )
    assert (
        'os.getenv("NOTEBOOK_API_MAX_REQUEST_BYTES", '
        f'"{env_vars["NOTEBOOK_API_MAX_REQUEST_BYTES"]["default"]}")'
        in generated_app
    )
    assert (
        'os.getenv("NOTEBOOK_API_TASK_TTL_SECONDS", '
        f'"{env_vars["NOTEBOOK_API_TASK_TTL_SECONDS"]["default"]}")'
        in generated_app
    )
    assert (
        'os.getenv("NOTEBOOK_API_MAX_TASKS", '
        f'"{env_vars["NOTEBOOK_API_MAX_TASKS"]["default"]}")'
        in generated_app
    )
    assert (
        'os.getenv("NOTEBOOK_API_PUBLIC_URL", '
        f'"{env_vars["NOTEBOOK_API_PUBLIC_URL"]["default"]}")'
        in generated_app
    )
    assert (
        'os.getenv("NOTEBOOK_API_DISABLE_DOCS", '
        f'"{env_vars["NOTEBOOK_API_DISABLE_DOCS"]["default"]}")'
        in generated_app
    )


def test_env_vars_preview_does_not_touch_generated_dir(monkeypatch, tmp_path):

    generated_dir = tmp_path / "env_vars_preview_no_side_effects"
    monkeypatch.setattr("backend.routes.upload.GENERATED_DIR", str(generated_dir))

    resp = client.get("/api/env-vars-preview")

    assert resp.status_code == 200
    assert not generated_dir.exists()


def test_inspect_reports_endpoints_and_flags_background_ones_before_compiling():
    """Mirrors test_compile_endpoints_flag_background_functions_as_async
    above, but for /api/inspect: before this fix, that classification was
    only visible in /api/compile's response, after compiling.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def train_model(epochs: int) -> str:\n    return 'done'\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "inspect_async_endpoints_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    inspect_resp = client.post(
        "/api/inspect", json={"notebook_path": "inspect_async_endpoints_test.ipynb"}
    )
    assert inspect_resp.status_code == 200

    endpoints = {e["path"]: e for e in inspect_resp.json()["endpoints"]}

    assert endpoints["/add"] == {"path": "/add", "method": "POST", "is_async": False}
    assert endpoints["/train_model"] == {
        "path": "/train_model", "method": "POST", "is_async": True
    }


def test_inspect_generated_files_is_empty_before_any_compile(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_inspect_test_missing_dir"
    )

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "inspect_no_compile_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    inspect_resp = client.post(
        "/api/inspect", json={"notebook_path": "inspect_no_compile_test.ipynb"}
    )
    assert inspect_resp.status_code == 200
    body = inspect_resp.json()

    assert body["functions"][0]["name"] == "add"
    assert body["generated_files"] == []


def test_compile_returns_400_for_a_reserved_function_name():
    """generate_fastapi_code (backend/generator/api_generator.py) refuses
    to compile a function named "health_check" -- it collides with an
    identifier the generated app itself defines. Before this, that
    ReservedFunctionNameError (the notebook's own fault, not this
    server's) fell through the endpoint's generic `except Exception` and
    came back as a misleading 500, identical to what a genuine server-side
    bug would produce.
    """

    content = _notebook_bytes(
        "def health_check() -> dict:\n    return {}\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "reserved_name_compile_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "reserved_name_compile_test.ipynb"}
    )

    assert compile_resp.status_code == 400
    assert "health_check" in compile_resp.json()["detail"]


def test_compile_respects_a_configured_generated_dir(tmp_path, monkeypatch):
    """POST /api/compile previously always wrote to a hardcoded "generated"
    string, ignoring GENERATED_DIR entirely -- every other endpoint that
    reads compiled output (list_notebooks' currently_compiled check,
    /api/export-openapi, /api/export-sdk, /api/deploy, /api/download)
    already honored GENERATED_DIR (now configurable via
    NOTEBOOK_API_GENERATED_DIR), so pointing it elsewhere would have
    silently only taken effect for those, while /api/compile kept writing
    to "generated/" regardless -- the two would disagree about where the
    compiled app actually lives.
    """

    from backend.routes import upload as upload_module

    custom_dir = tmp_path / "custom_generated"
    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(custom_dir))

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "custom_generated_dir_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "custom_generated_dir_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    assert (custom_dir / "app.py").is_file()
    assert (custom_dir / "requirements.txt").is_file()
    assert (custom_dir / COMPILE_METADATA_FILENAME).is_file()


def test_compile_into_a_custom_generated_dir_is_visible_to_other_endpoints(
    tmp_path, monkeypatch
):
    """End-to-end proof that /api/compile and the rest of the dashboard API
    agree on where the compiled app lives once GENERATED_DIR is
    configured: list_notebooks' currently_compiled check and /api/download
    both read from GENERATED_DIR, so if /api/compile had still written to
    the hardcoded "generated/" instead, neither would ever find it.
    """

    from backend.routes import upload as upload_module

    custom_dir = tmp_path / "custom_generated_2"
    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(custom_dir))

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    filename = "custom_generated_dir_consistency_test.ipynb"

    upload_resp = client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post("/api/compile", json={"notebook_path": filename})
    assert compile_resp.status_code == 200

    notebooks_resp = client.get("/api/notebooks")
    assert notebooks_resp.status_code == 200
    entry = next(
        n for n in notebooks_resp.json()["notebooks"] if n["filename"] == filename
    )
    assert entry["currently_compiled"] is True

    download_resp = client.get("/api/download")
    assert download_resp.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(download_resp.content))
    assert "app.py" in archive.namelist()


def test_export_openapi_and_export_sdk_full_flow():
    """The dashboard frontend can compile a notebook via /api/compile but,
    before this, had no way to fetch the OpenAPI schema or an SDK client
    without shelling out to the `export-openapi`/`export-sdk` CLI commands.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "export_flow_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "export_flow_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    openapi_resp = client.post("/api/export-openapi", json={"format": "json"})
    assert openapi_resp.status_code == 200
    openapi_body = openapi_resp.json()
    assert openapi_body["format"] == "json"
    assert "/add" in openapi_body["schema"]["paths"]

    yaml_resp = client.post("/api/export-openapi", json={"format": "yaml"})
    assert yaml_resp.status_code == 200
    yaml_body = yaml_resp.json()
    assert yaml_body["format"] == "yaml"
    assert "content" in yaml_body

    sdk_resp = client.post("/api/export-sdk", json={"language": "python"})
    assert sdk_resp.status_code == 200
    sdk_body = sdk_resp.json()
    assert sdk_body["language"] == "python"
    assert "class NotebookAPIClient" in sdk_body["code"]
    assert "def add" in sdk_body["code"]

    ts_resp = client.post("/api/export-sdk", json={"language": "typescript"})
    assert ts_resp.status_code == 200
    ts_body = ts_resp.json()
    assert ts_body["language"] == "typescript"
    assert "class NotebookAPIClient" in ts_body["code"]


def test_export_openapi_reflects_a_recompile_within_the_same_process():
    """Confirmed exploitable before this fix: export_openapi_schema
    (backend/exporters/openapi_exporter.py) imports "<package_name>.app"
    with plain importlib.import_module, which Python resolves from
    sys.modules -- not from disk -- once that name has already been
    imported in this process. The dashboard is exactly that kind of
    long-running process, so the second /api/compile -> /api/export-openapi
    round trip silently returned the *first* compile's schema: compiling a
    notebook exposing `add`, exporting it, then recompiling the same
    upload to expose `multiply` instead and exporting again still returned
    `add` in the schema's paths, with the freshly-written app.py on disk
    never actually read.
    """

    filename = "reimport_staleness_test.ipynb"

    first_content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    upload_resp = client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(first_content), "application/json")},
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post("/api/compile", json={"notebook_path": filename})
    assert compile_resp.status_code == 200

    first_export = client.post("/api/export-openapi", json={"format": "json"})
    assert first_export.status_code == 200
    assert "/add" in first_export.json()["schema"]["paths"]

    second_content = _notebook_bytes(
        "def multiply(a: int, b: int) -> int:\n    return a * b\n"
    )
    overwrite_resp = client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(second_content), "application/json")},
    )
    assert overwrite_resp.status_code == 200

    recompile_resp = client.post("/api/compile", json={"notebook_path": filename})
    assert recompile_resp.status_code == 200

    second_export = client.post("/api/export-openapi", json={"format": "json"})
    assert second_export.status_code == 200
    second_paths = second_export.json()["schema"]["paths"]

    assert "/multiply" in second_paths
    assert "/add" not in second_paths


def test_export_openapi_rejects_invalid_format():

    resp = client.post("/api/export-openapi", json={"format": "xml"})

    assert resp.status_code == 400


def test_export_sdk_rejects_invalid_language():

    resp = client.post("/api/export-sdk", json={"language": "rust"})

    assert resp.status_code == 400


def test_export_openapi_returns_404_when_nothing_compiled_yet(monkeypatch):
    """Uses a package name that has never been compiled anywhere on
    sys.path, so the dynamic `importlib.import_module` in
    export_openapi_schema is guaranteed to raise ModuleNotFoundError
    rather than risk resolving a stale cached "generated" module from an
    earlier test in this same process.
    """

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_export_test_missing_dir"
    )

    resp = client.post("/api/export-openapi", json={"format": "json"})

    assert resp.status_code == 404


def test_export_sdk_returns_404_without_prior_openapi_export(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_export_test_missing_dir_2"
    )

    resp = client.post("/api/export-sdk", json={"language": "python"})

    assert resp.status_code == 404


def test_export_sdk_reports_a_clean_400_for_a_corrupt_openapi_schema(monkeypatch, tmp_path):
    """Confirmed exploitable before this fix: _load_openapi_schema
    (exporters/sdk_generator.py) raises ValueError for content that isn't
    valid JSON, but this endpoint's except clause only ever caught bare
    Exception and wrapped it in a 500 -- indistinguishable from an actual
    server-side bug, unlike every other malformed-client-input case in
    this codebase (a bad language, a missing export, ...). "openapi.json"
    existing but being corrupt (e.g. a truncated write) is squarely the
    client-visible state's own problem, not this server's.
    """

    from backend.routes import upload as upload_module

    isolated_dir = tmp_path / "generated_export_sdk_corrupt_test"
    isolated_dir.mkdir()
    (isolated_dir / "openapi.json").write_text("not json at all", encoding="utf-8")

    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(isolated_dir))

    resp = client.post("/api/export-sdk", json={"language": "python"})

    assert resp.status_code == 400
    assert "not a valid OpenAPI JSON schema" in resp.json()["detail"]


def test_export_sdk_gives_a_clean_400_not_a_misleading_404_for_a_yaml_only_export(
    monkeypatch, tmp_path
):
    """Confirmed exploitable before this fix: POST /api/export-sdk only
    ever checked for "openapi.json", hardcoded -- but POST
    /api/export-openapi is just as capable of writing "openapi.yaml"
    instead, via {"format": "yaml"}. A caller who did exactly that, then
    called export-sdk, got a 404 saying "No exported OpenAPI schema
    found. Run /api/export-openapi first" -- flatly wrong, since they
    just had, in the only other format this same API offers -- instead of
    the clean 400 with a specific "re-export with format=json" hint
    _load_openapi_schema (exporters/sdk_generator.py) already writes for
    exactly this situation, and which the CLI's own `export-sdk --openapi
    generated/openapi.yaml` could already reach.
    """

    from backend.routes import upload as upload_module

    isolated_dir = tmp_path / "generated_export_sdk_yaml_only_test"
    isolated_dir.mkdir()
    (isolated_dir / "openapi.yaml").write_text(
        "openapi: 3.0.0\ninfo:\n  title: Test\n  version: '1.0'\npaths: {}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(isolated_dir))

    resp = client.post("/api/export-sdk", json={"language": "python"})

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "YAML export" in detail
    assert "format json" in detail or "--format json" in detail


def test_export_sdk_prefers_a_json_export_over_a_yaml_one_when_both_exist(
    monkeypatch, tmp_path
):
    """If a caller exported both formats (e.g. json first, then yaml for
    a human-readable copy), export-sdk must still read the json export --
    the one it actually knows how to parse -- rather than picking
    whichever file the directory listing happens to prefer.
    """

    from backend.routes import upload as upload_module

    isolated_dir = tmp_path / "generated_export_sdk_prefers_json_test"
    isolated_dir.mkdir()
    (isolated_dir / "openapi.json").write_text("{}", encoding="utf-8")
    # Deliberately not valid JSON -- if export-sdk picked this file
    # instead, _load_openapi_schema would raise and this test would see a
    # 400, not the 200 a real "both exports present" caller expects.
    (isolated_dir / "openapi.yaml").write_text(
        "openapi: 3.0.0\ninfo:\n  title: Test\n  version: '1.0'\npaths: {}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(isolated_dir))

    resp = client.post("/api/export-sdk", json={"language": "python"})

    assert resp.status_code == 200
    assert "class NotebookAPIClient" in resp.json()["code"]


def _install_fake_docker(bin_dir, log_path):
    """A fake `docker` executable that records how it was invoked instead
    of actually building/pushing an image (mirrors the technique used in
    tests/test_cli_deploy.py for the CLI's own `deploy` command). Appends
    a record per invocation so build and push calls can each be
    inspected independently, in order.
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


def _compile_a_notebook(filename):

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(content), "application/json")},
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post("/api/compile", json={"notebook_path": filename})
    assert compile_resp.status_code == 200


def test_deploy_endpoint_returns_404_when_nothing_compiled_yet(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_deploy_test_missing_dir"
    )

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 404


def test_deploy_endpoint_returns_409_when_the_compiled_notebook_is_stale(
    tmp_path, monkeypatch
):
    """Unlike the CLI's `deploy` command -- which always recompiles from
    the notebook as its own first step, so it can never build a stale
    image -- /api/deploy builds whatever is already sitting in
    GENERATED_DIR from an earlier, separate /api/compile call. Before
    this, editing the notebook after that compile (e.g. via
    /api/upload?overwrite=true) without recompiling went completely
    unchecked: this could silently build (and, with "push": true,
    publish) a Docker image reflecting outdated code -- the exact
    staleness list_notebooks' notebook_changed_since_compile field
    already exists to warn about, just never enforced here.
    """

    filename = "deploy_stale_test.ipynb"
    _compile_a_notebook(filename)

    changed_content = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )
    overwrite_resp = client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(changed_content), "application/json")},
    )
    assert overwrite_resp.status_code == 200

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 409
    assert "edited since the last compile" in resp.json()["detail"]
    # Docker must never have been invoked at all.
    assert not log_path.exists()


def test_deploy_endpoint_force_true_deploys_a_stale_build_anyway(tmp_path, monkeypatch):

    filename = "deploy_stale_force_test.ipynb"
    _compile_a_notebook(filename)

    changed_content = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )
    overwrite_resp = client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(changed_content), "application/json")},
    )
    assert overwrite_resp.status_code == 200

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={"force": True})

    assert resp.status_code == 200
    assert log_path.exists()


def test_deploy_endpoint_does_not_require_force_when_the_notebook_is_unchanged(
    tmp_path, monkeypatch
):

    _compile_a_notebook("deploy_not_stale_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 200
    assert log_path.exists()


def test_deploy_endpoint_builds_image_with_default_tag(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_flow_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["tag"] == "generated:latest"
    assert body["pushed"] is False

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    assert len(calls) == 1
    build_call = calls[0].splitlines()
    assert build_call[:-1] == ["build", "-t", "generated:latest", "."]
    assert build_call[-1] == os.path.abspath("generated")


def test_deploy_endpoint_respects_custom_tag(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_tag_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={"tag": "myregistry.example.com/myapp:v2"})

    assert resp.status_code == 200
    assert resp.json()["tag"] == "myregistry.example.com/myapp:v2"

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    build_call = calls[0].splitlines()
    assert build_call[:-1] == ["build", "-t", "myregistry.example.com/myapp:v2", "."]


def test_deploy_endpoint_respects_custom_platform(tmp_path, monkeypatch):
    """`docker build`'s own default target platform is the local Docker
    daemon's host architecture -- not necessarily the deploy target's
    (almost every cloud PaaS runs linux/amd64). Before "platform" existed
    here, the dashboard's /api/deploy had no way to override it at all.
    """

    _compile_a_notebook("deploy_platform_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={"platform": "linux/amd64"})

    assert resp.status_code == 200

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    build_call = calls[0].splitlines()
    assert build_call[:-1] == [
        "build", "-t", "generated:latest", "--platform", "linux/amd64", ".",
    ]


def test_deploy_endpoint_omits_platform_flag_by_default(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_no_platform_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 200

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    build_call = calls[0].splitlines()
    assert "--platform" not in build_call


def test_deploy_endpoint_respects_no_cache(tmp_path, monkeypatch):
    """Docker's own layer cache can silently reuse a stale `pip install`
    layer even after requirements.txt changed -- before "no_cache"
    existed, an operator ruling that out had no way to force a clean
    rebuild through this endpoint at all.
    """

    _compile_a_notebook("deploy_no_cache_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={"no_cache": True})

    assert resp.status_code == 200

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    build_call = calls[0].splitlines()
    assert build_call[:-1] == [
        "build", "-t", "generated:latest", "--no-cache", ".",
    ]


def test_deploy_endpoint_omits_no_cache_flag_by_default(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_no_no_cache_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 200

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    build_call = calls[0].splitlines()
    assert "--no-cache" not in build_call


def test_deploy_endpoint_combines_no_cache_with_platform(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_no_cache_platform_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post(
        "/api/deploy", json={"no_cache": True, "platform": "linux/amd64"}
    )

    assert resp.status_code == 200

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    build_call = calls[0].splitlines()
    assert build_call[:-1] == [
        "build", "-t", "generated:latest",
        "--platform", "linux/amd64", "--no-cache", ".",
    ]


@pytest.mark.parametrize("bad_platform", [123, 1.5, ["linux/amd64"], {"platform": "linux/amd64"}])
def test_deploy_endpoint_rejects_a_non_string_platform(bad_platform):
    """Mirrors test_deploy_endpoint_rejects_a_non_string_tag: "platform"
    flows into the same subprocess argument list "tag" does.
    """

    _compile_a_notebook("deploy_bad_platform_test.ipynb")

    resp = client.post("/api/deploy", json={"platform": bad_platform})

    assert resp.status_code == 400


@pytest.mark.parametrize("bad_tag", [123, 1.5, ["myapp:v1"], {"tag": "myapp:v1"}])
def test_deploy_endpoint_rejects_a_non_string_tag(bad_tag):
    """Confirmed exploitable before this fix: "tag" flows straight into a
    `docker build`/`docker push` subprocess argument list -- subprocess.run
    requires every element to be str/bytes/PathLike, so a non-string tag
    crashed with an unhandled TypeError from deep inside subprocess
    internals instead of the same clean 400 a legitimate string tag
    already validates fine with.
    """

    _compile_a_notebook("deploy_bad_tag_test.ipynb")

    resp = client.post("/api/deploy", json={"tag": bad_tag})

    assert resp.status_code == 400


def test_deploy_endpoint_pushes_when_requested(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_push_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post(
        "/api/deploy", json={"tag": "myregistry.example.com/myapp:v3", "push": True}
    )

    assert resp.status_code == 200
    assert resp.json()["pushed"] is True

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    assert len(calls) == 2
    assert calls[0].splitlines()[:-1] == ["build", "-t", "myregistry.example.com/myapp:v3", "."]
    assert calls[1].splitlines()[:-1] == ["push", "myregistry.example.com/myapp:v3"]


def test_deploy_endpoint_does_not_push_by_default(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_no_push_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 200
    assert resp.json()["pushed"] is False

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    assert len(calls) == 1


def test_deploy_endpoint_dry_run_does_not_invoke_docker(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_dry_run_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={"dry_run": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["tag"] == "generated:latest"
    assert body["pushed"] is False

    # Docker must never have been invoked at all.
    assert not log_path.exists()


def test_deploy_endpoint_dry_run_reports_pushed_true_when_push_was_requested(
    tmp_path, monkeypatch
):

    _compile_a_notebook("deploy_dry_run_push_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={"dry_run": True, "push": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    # Reports what a *real* deploy would do with "push": true -- even
    # though this dry run never actually pushed anything.
    assert body["pushed"] is True
    assert not log_path.exists()


def test_deploy_endpoint_dry_run_still_returns_404_when_nothing_compiled_yet(
    monkeypatch,
):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_deploy_dry_run_missing_dir"
    )

    resp = client.post("/api/deploy", json={"dry_run": True})

    assert resp.status_code == 404


def test_deploy_endpoint_dry_run_still_returns_409_when_the_compiled_notebook_is_stale(
    tmp_path, monkeypatch
):

    filename = "deploy_dry_run_stale_test.ipynb"
    _compile_a_notebook(filename)

    changed_content = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )
    overwrite_resp = client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(changed_content), "application/json")},
    )
    assert overwrite_resp.status_code == 200

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={"dry_run": True})

    assert resp.status_code == 409
    assert "edited since the last compile" in resp.json()["detail"]
    assert not log_path.exists()


def test_deploy_endpoint_dry_run_does_not_record_a_history_entry(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_dry_run_history_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    entry_count_before = client.get("/api/deploy/history").json()["entry_count"]

    resp = client.post(
        "/api/deploy", json={"dry_run": True, "tag": "deploy-dry-run-history:latest"}
    )
    assert resp.status_code == 200

    history_resp = client.get("/api/deploy/history")
    assert history_resp.json()["entry_count"] == entry_count_before
    assert all(
        entry["tag"] != "deploy-dry-run-history:latest"
        for entry in history_resp.json()["entries"]
    )


def _install_fake_docker_with_smoke_test_support(bin_dir, log_path):
    """A fake `docker` executable supporting `run`/`port`/`stop`/`logs`
    (for POST /api/deploy's own "smoke_test") on top of the same
    invocation-logging `_install_fake_docker` above already provides for
    `build`/`push` -- each subcommand's own behavior is driven by
    environment variables a test sets via monkeypatch.setenv, read at
    invocation time by this same script (inherited by the real `docker
    run`/`docker port`/... subprocess.run calls _run_deploy_smoke_test
    makes), so one stub can stand in for a healthy, unhealthy, or
    outright failing container across different tests without rewriting
    the script itself:

      FAKE_DOCKER_RUN_EXIT_CODE (default 0), FAKE_DOCKER_RUN_STDOUT
      (default "fake-container-id"), FAKE_DOCKER_RUN_STDERR (default "")
      FAKE_DOCKER_PORT_EXIT_CODE (default 0), FAKE_DOCKER_PORT_STDOUT
      (default "") -- a real test points this at a real local HTTP
      server's own "127.0.0.1:<port>" (see fake_container_health_server
      below) so _run_deploy_smoke_test's own GET /health polling has
      something real to actually reach.
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
    "container" _run_deploy_smoke_test's own `docker run` would otherwise
    have started for real -- fake_container_health_server below points
    the fake docker's own "docker port" output at this real local server
    instead, so the smoke test's actual polling/timeout/retry logic runs
    against a genuine HTTP server, not a mock of one.
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


@pytest.fixture
def _fast_deploy_smoke_test_timing(monkeypatch):
    """Every failure-path smoke-test test below needs the real
    poll-until-timeout loop to actually run to completion -- without
    this, each one would otherwise take DEPLOY_SMOKE_TEST_TIMEOUT_SECONDS's
    own real 30s default.
    """
    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "DEPLOY_SMOKE_TEST_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(
        upload_module, "_DEPLOY_SMOKE_TEST_POLL_INTERVAL_SECONDS", 0.05
    )


def test_deploy_endpoint_omits_smoke_test_by_default(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_no_smoke_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 200
    assert "smoke_test" not in resp.json()

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    assert all(call.splitlines()[0] != "run" for call in calls)


def test_deploy_endpoint_smoke_test_passes_when_health_responds_200(
    tmp_path, monkeypatch, fake_container_health_server
):

    port, _handler = fake_container_health_server

    _compile_a_notebook("deploy_smoke_test_pass.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_PORT_STDOUT", f"127.0.0.1:{port}")

    resp = client.post("/api/deploy", json={"smoke_test": True})

    assert resp.status_code == 200
    smoke_test = resp.json()["smoke_test"]
    assert smoke_test == {"passed": True, "status_code": 200, "detail": None}

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    subcommands = [call.splitlines()[0] for call in calls]
    assert "run" in subcommands
    assert "stop" in subcommands


def test_deploy_endpoint_smoke_test_fails_when_docker_run_fails(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_smoke_test_run_fails.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_RUN_EXIT_CODE", "1")
    monkeypatch.setenv("FAKE_DOCKER_RUN_STDERR", "no such image")

    resp = client.post("/api/deploy", json={"smoke_test": True})

    assert resp.status_code == 200
    smoke_test = resp.json()["smoke_test"]
    assert smoke_test["passed"] is False
    assert smoke_test["status_code"] is None
    assert "no such image" in smoke_test["detail"]

    # docker run failed before ever producing a container id -- there is
    # nothing to stop.
    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    subcommands = [call.splitlines()[0] for call in calls]
    assert "stop" not in subcommands


def test_deploy_endpoint_smoke_test_fails_when_docker_port_fails(
    tmp_path, monkeypatch
):

    _compile_a_notebook("deploy_smoke_test_port_fails.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_PORT_EXIT_CODE", "1")

    resp = client.post("/api/deploy", json={"smoke_test": True})

    assert resp.status_code == 200
    smoke_test = resp.json()["smoke_test"]
    assert smoke_test["passed"] is False
    assert "Could not determine the container's own port" in smoke_test["detail"]

    # The container was still successfully started -- it must still be
    # stopped even though the smoke test itself failed.
    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    subcommands = [call.splitlines()[0] for call in calls]
    assert "stop" in subcommands


def test_deploy_endpoint_smoke_test_times_out_and_reports_container_logs(
    tmp_path, monkeypatch, _fast_deploy_smoke_test_timing
):
    """Points "docker port" at a closed local port -- nothing is
    listening there, so every poll attempt fails to even connect -- and
    confirms the failure detail actually surfaces the container's own
    logs, not just a bare timeout message with no way to diagnose why.
    """

    _compile_a_notebook("deploy_smoke_test_timeout.ipynb")

    closed_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    closed_socket.bind(("127.0.0.1", 0))
    closed_port = closed_socket.getsockname()[1]
    closed_socket.close()

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_PORT_STDOUT", f"127.0.0.1:{closed_port}")
    monkeypatch.setenv("FAKE_DOCKER_LOGS_STDOUT", "Traceback: ImportError: no module named foo")

    resp = client.post("/api/deploy", json={"smoke_test": True})

    assert resp.status_code == 200
    smoke_test = resp.json()["smoke_test"]
    assert smoke_test["passed"] is False
    assert smoke_test["status_code"] is None
    assert "ImportError: no module named foo" in smoke_test["detail"]

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    subcommands = [call.splitlines()[0] for call in calls]
    assert "stop" in subcommands


def test_deploy_endpoint_smoke_test_fails_when_health_never_returns_200(
    tmp_path, monkeypatch, fake_container_health_server, _fast_deploy_smoke_test_timing
):

    port, handler = fake_container_health_server
    handler.status_code = 503

    _compile_a_notebook("deploy_smoke_test_unhealthy.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_PORT_STDOUT", f"127.0.0.1:{port}")

    resp = client.post("/api/deploy", json={"smoke_test": True})

    assert resp.status_code == 200
    smoke_test = resp.json()["smoke_test"]
    assert smoke_test["passed"] is False
    assert smoke_test["status_code"] == 503
    assert "last responded 503" in smoke_test["detail"]


def test_deploy_endpoint_smoke_test_never_blocks_a_requested_push(
    tmp_path, monkeypatch
):
    """A failed smoke test is purely diagnostic -- it must never prevent
    an already-requested push from actually happening, the same
    "diagnostic, not gating" relationship POST /api/compile's own
    "smoke_test" already has with a successful compile.
    """

    _compile_a_notebook("deploy_smoke_test_does_not_block_push.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_RUN_EXIT_CODE", "1")

    resp = client.post("/api/deploy", json={"smoke_test": True, "push": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["smoke_test"]["passed"] is False
    assert body["pushed"] is True

    calls = [block for block in log_path.read_text(encoding="utf-8").split("==CALL==\n") if block]
    subcommands = [call.splitlines()[0] for call in calls]
    assert "push" in subcommands


def test_deploy_endpoint_dry_run_skips_smoke_test_entirely(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_smoke_test_dry_run.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker_with_smoke_test_support(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={"smoke_test": True, "dry_run": True})

    assert resp.status_code == 200
    assert "smoke_test" not in resp.json()
    assert not log_path.exists()


def test_deploy_history_is_empty_before_any_deploy(monkeypatch, tmp_path):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "deploy_history_empty_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    resp = client.get("/api/deploy/history")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "entries": [],
        "entry_count": 0,
    }


def test_deploy_records_a_history_entry_on_successful_build(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_history_record_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={"tag": "myapp:v1"})
    assert resp.status_code == 200

    history_resp = client.get("/api/deploy/history")
    assert history_resp.status_code == 200
    body = history_resp.json()
    assert body["entry_count"] >= 1

    entry = body["entries"][0]
    assert entry["tag"] == "myapp:v1"
    assert entry["platform"] is None
    assert entry["pushed"] is False
    assert entry["source_notebook_filename"] == "deploy_history_record_test.ipynb"
    assert entry["source_notebook_sha256"]
    assert "deployed_at" in entry


def test_deploy_records_pushed_true_when_push_requested(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_history_pushed_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post(
        "/api/deploy",
        json={"tag": "myapp:pushed", "push": True, "platform": "linux/amd64"},
    )
    assert resp.status_code == 200

    entry = client.get("/api/deploy/history").json()["entries"][0]
    assert entry["tag"] == "myapp:pushed"
    assert entry["pushed"] is True
    assert entry["platform"] == "linux/amd64"


def test_deploy_history_lists_most_recent_first(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_history_order_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    log_path = tmp_path / "docker_invocation.log"
    _install_fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    client.post("/api/deploy", json={"tag": "order:first"})
    client.post("/api/deploy", json={"tag": "order:second"})

    entries = client.get("/api/deploy/history").json()["entries"]
    tags_in_order = [e["tag"] for e in entries if e["tag"] in ("order:first", "order:second")]
    assert tags_in_order == ["order:second", "order:first"]


def test_deploy_history_is_capped_at_the_configured_maximum(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "deploy_history_cap_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))
    monkeypatch.setattr(upload_module, "MAX_DEPLOY_HISTORY_ENTRIES", 3)

    for i in range(5):
        upload_module._append_deploy_history_entry({
            "deployed_at": f"2024-01-0{i + 1}T00:00:00+00:00",
            "tag": f"cap:{i}",
            "platform": None,
            "pushed": False,
            "source_notebook_filename": None,
            "source_notebook_sha256": None,
        })

    body = client.get("/api/deploy/history").json()
    assert body["entry_count"] == 3
    assert [e["tag"] for e in body["entries"]] == ["cap:4", "cap:3", "cap:2"]


def _seed_deploy_history_for_filtering(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "deploy_history_filter_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    entries = [
        {
            "deployed_at": "2024-01-01T00:00:00+00:00", "tag": "filter:a",
            "platform": "linux/amd64", "pushed": True,
            "source_notebook_filename": "one.ipynb", "source_notebook_sha256": "aaa",
        },
        {
            "deployed_at": "2024-01-02T00:00:00+00:00", "tag": "filter:b",
            "platform": "linux/arm64", "pushed": False,
            "source_notebook_filename": "two.ipynb", "source_notebook_sha256": "bbb",
        },
        {
            "deployed_at": "2024-01-03T00:00:00+00:00", "tag": "filter:c",
            "platform": "linux/amd64", "pushed": False,
            "source_notebook_filename": "one.ipynb", "source_notebook_sha256": "ccc",
        },
    ]

    for entry in entries:
        upload_module._append_deploy_history_entry(entry)


def test_deploy_history_filters_by_source_notebook_filename(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/deploy/history", params={"source_notebook_filename": "one.ipynb"}
    ).json()

    assert [e["tag"] for e in body["entries"]] == ["filter:c", "filter:a"]
    assert body["entry_count"] == 2


def test_deploy_history_filters_by_source_notebook_sha256(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/deploy/history", params={"source_notebook_sha256": "bbb"}
    ).json()

    assert [e["tag"] for e in body["entries"]] == ["filter:b"]
    assert body["entry_count"] == 1


def test_deploy_history_unknown_source_notebook_sha256_yields_no_entries(
    tmp_path, monkeypatch
):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/deploy/history", params={"source_notebook_sha256": "no-such-hash"}
    ).json()

    assert body["entries"] == []
    assert body["entry_count"] == 0


def test_deploy_history_csv_format_returns_a_csv_response(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    resp = client.get("/api/deploy/history", params={"format": "csv"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="deploy_history.csv"' in resp.headers["content-disposition"]

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == (
        "deployed_at,tag,platform,pushed,source_notebook_filename,"
        "source_notebook_sha256"
    )
    # Most-recent-first, exactly like the "json" response.
    assert rows[1] == "2024-01-03T00:00:00+00:00,filter:c,linux/amd64,False,one.ipynb,ccc"
    assert rows[2] == "2024-01-02T00:00:00+00:00,filter:b,linux/arm64,False,two.ipynb,bbb"
    assert rows[3] == "2024-01-01T00:00:00+00:00,filter:a,linux/amd64,True,one.ipynb,aaa"
    assert len(rows) == 4


def test_deploy_history_csv_format_composes_with_filters_and_pagination(
    tmp_path, monkeypatch
):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    resp = client.get(
        "/api/deploy/history",
        params={"format": "csv", "source_notebook_filename": "one.ipynb", "limit": 1},
    )

    assert resp.status_code == 200
    rows = resp.text.strip().split("\r\n")
    assert len(rows) == 2
    assert rows[1].startswith("2024-01-03T00:00:00+00:00,filter:c,")


def test_deploy_history_csv_format_on_an_empty_history_is_just_the_header(monkeypatch, tmp_path):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "deploy_history_csv_empty_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    resp = client.get("/api/deploy/history", params={"format": "csv"})

    assert resp.status_code == 200
    rows = resp.text.strip().split("\r\n")
    assert rows == [
        "deployed_at,tag,platform,pushed,source_notebook_filename,"
        "source_notebook_sha256"
    ]


def test_deploy_history_rejects_an_unknown_format():

    resp = client.get("/api/deploy/history", params={"format": "xml"})

    assert resp.status_code == 400
    assert "format" in resp.json()["detail"]


def test_deploy_history_filters_by_platform(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get("/api/deploy/history", params={"platform": "linux/arm64"}).json()

    assert [e["tag"] for e in body["entries"]] == ["filter:b"]


def test_deploy_history_filters_by_tag(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get("/api/deploy/history", params={"tag": "filter:b"}).json()

    assert [e["tag"] for e in body["entries"]] == ["filter:b"]
    assert body["entry_count"] == 1


def test_deploy_history_filters_by_tag_and_platform_combined(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/deploy/history",
        params={"tag": "filter:a", "platform": "linux/arm64"},
    ).json()

    assert body["entries"] == []
    assert body["entry_count"] == 0


def test_deploy_history_unknown_tag_yields_no_entries(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get("/api/deploy/history", params={"tag": "no-such-tag:latest"}).json()

    assert body["entries"] == []
    assert body["entry_count"] == 0


def test_deploy_history_filters_by_pushed(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    pushed_body = client.get("/api/deploy/history", params={"pushed": "true"}).json()
    assert [e["tag"] for e in pushed_body["entries"]] == ["filter:a"]

    not_pushed_body = client.get("/api/deploy/history", params={"pushed": "false"}).json()
    assert [e["tag"] for e in not_pushed_body["entries"]] == ["filter:c", "filter:b"]


def test_deploy_history_filters_by_deployed_after_and_before(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    after_body = client.get(
        "/api/deploy/history", params={"deployed_after": "2024-01-02T00:00:00+00:00"}
    ).json()
    assert [e["tag"] for e in after_body["entries"]] == ["filter:c", "filter:b"]

    before_body = client.get(
        "/api/deploy/history", params={"deployed_before": "2024-01-02T00:00:00+00:00"}
    ).json()
    assert [e["tag"] for e in before_body["entries"]] == ["filter:b", "filter:a"]

    window_body = client.get(
        "/api/deploy/history",
        params={
            "deployed_after": "2024-01-02T00:00:00+00:00",
            "deployed_before": "2024-01-02T00:00:00+00:00",
        },
    ).json()
    assert [e["tag"] for e in window_body["entries"]] == ["filter:b"]


def test_deploy_history_rejects_deployed_after_later_than_deployed_before():

    resp = client.get(
        "/api/deploy/history",
        params={
            "deployed_after": "2026-06-01T00:00:00+00:00",
            "deployed_before": "2026-01-01T00:00:00+00:00",
        },
    )

    assert resp.status_code == 400
    assert "deployed_after" in resp.json()["detail"]


def test_deploy_history_rejects_a_malformed_deployed_after():

    resp = client.get("/api/deploy/history", params={"deployed_after": "not-a-date"})

    assert resp.status_code == 400
    assert "deployed_after" in resp.json()["detail"]


def test_deploy_history_respects_limit(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get("/api/deploy/history", params={"limit": 2}).json()

    assert [e["tag"] for e in body["entries"]] == ["filter:c", "filter:b"]
    assert body["entry_count"] == 2


def test_deploy_history_combines_filters_and_limit(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/deploy/history",
        params={"source_notebook_filename": "one.ipynb", "limit": 1},
    ).json()

    assert [e["tag"] for e in body["entries"]] == ["filter:c"]


def test_deploy_history_rejects_a_negative_limit(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    resp = client.get("/api/deploy/history", params={"limit": -1})

    assert resp.status_code == 400


def test_deploy_history_respects_offset(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get("/api/deploy/history", params={"offset": 1}).json()

    assert [e["tag"] for e in body["entries"]] == ["filter:b", "filter:a"]
    assert body["entry_count"] == 2


def test_deploy_history_combines_offset_and_limit_for_paging(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/deploy/history", params={"offset": 1, "limit": 1}
    ).json()

    assert [e["tag"] for e in body["entries"]] == ["filter:b"]
    assert body["entry_count"] == 1


def test_deploy_history_offset_past_the_end_yields_no_entries(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    body = client.get("/api/deploy/history", params={"offset": 100}).json()

    assert body["entries"] == []
    assert body["entry_count"] == 0


def test_deploy_history_rejects_a_negative_offset(tmp_path, monkeypatch):

    _seed_deploy_history_for_filtering(tmp_path, monkeypatch)

    resp = client.get("/api/deploy/history", params={"offset": -1})

    assert resp.status_code == 400


def test_deploy_does_not_record_a_history_entry_on_build_failure(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "deploy_history_build_failure_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    _compile_a_notebook("deploy_history_build_failure_test.ipynb")

    bin_dir = tmp_path / "fakebin"
    docker_stub = bin_dir / "docker"
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker_stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    docker_stub.chmod(docker_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    resp = client.post("/api/deploy", json={})
    assert resp.status_code == 500

    assert client.get("/api/deploy/history").json()["entries"] == []


def test_clear_deploy_history_removes_every_entry(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    for i in range(3):
        upload_module._append_deploy_history_entry({
            "deployed_at": f"2024-01-0{i + 1}T00:00:00+00:00",
            "tag": f"clear:{i}",
            "platform": None,
            "pushed": False,
            "source_notebook_filename": None,
            "source_notebook_sha256": None,
        })

    assert client.get("/api/deploy/history").json()["entry_count"] == 3

    clear_resp = client.delete("/api/deploy/history")

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 3}

    assert client.get("/api/deploy/history").json() == {
        "status": "success",
        "entries": [],
        "entry_count": 0,
    }


def test_clear_deploy_history_filters_by_source_notebook_filename(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_filter_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    for i in range(3):
        upload_module._append_deploy_history_entry({
            "deployed_at": f"2024-01-0{i + 1}T00:00:00+00:00",
            "tag": f"clear:{i}",
            "platform": None,
            "pushed": False,
            "source_notebook_filename": "keep.ipynb" if i != 1 else "drop.ipynb",
            "source_notebook_sha256": None,
        })

    clear_resp = client.delete(
        "/api/deploy/history", params={"source_notebook_filename": "drop.ipynb"}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 1}

    remaining = client.get("/api/deploy/history").json()
    assert remaining["entry_count"] == 2
    assert all(
        e["source_notebook_filename"] == "keep.ipynb" for e in remaining["entries"]
    )


def test_clear_deploy_history_unknown_source_notebook_filename_deletes_nothing(
    tmp_path, monkeypatch
):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_unknown_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_deploy_history_entry({
        "deployed_at": "2024-01-01T00:00:00+00:00",
        "tag": "keep:0",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": "keep.ipynb",
        "source_notebook_sha256": None,
    })

    clear_resp = client.delete(
        "/api/deploy/history", params={"source_notebook_filename": "no-such.ipynb"}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 0}
    assert client.get("/api/deploy/history").json()["entry_count"] == 1


def test_clear_deploy_history_filters_by_source_notebook_sha256(tmp_path, monkeypatch):
    """Confirmed missing before this fix: GET /api/deploy/history's own
    "source_notebook_sha256" filter, matching a notebook's exact content
    rather than whichever filename it happened to be deployed under, had
    no DELETE counterpart -- an operator wanting to drop just one
    (possibly since-renamed) notebook's stale deploy history by content
    had no choice but to wipe the whole log, or fall back to
    "source_notebook_filename" (which a rename since deploy time already
    makes unreachable).
    """

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_sha256_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    for i in range(3):
        upload_module._append_deploy_history_entry({
            "deployed_at": f"2024-01-0{i + 1}T00:00:00+00:00",
            "tag": f"clear:{i}",
            "platform": None,
            "pushed": False,
            "source_notebook_filename": f"renamed-{i}.ipynb",
            "source_notebook_sha256": "abc" if i != 1 else "def",
        })

    clear_resp = client.delete(
        "/api/deploy/history", params={"source_notebook_sha256": "def"}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 1}

    remaining = client.get("/api/deploy/history").json()
    assert remaining["entry_count"] == 2
    assert all(e["source_notebook_sha256"] == "abc" for e in remaining["entries"])


def test_clear_deploy_history_unknown_source_notebook_sha256_deletes_nothing(
    tmp_path, monkeypatch
):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_unknown_sha256_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_deploy_history_entry({
        "deployed_at": "2024-01-01T00:00:00+00:00",
        "tag": "keep:0",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": "keep.ipynb",
        "source_notebook_sha256": "abc",
    })

    clear_resp = client.delete(
        "/api/deploy/history", params={"source_notebook_sha256": "no-such-hash"}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 0}
    assert client.get("/api/deploy/history").json()["entry_count"] == 1


def test_clear_deploy_history_composes_source_notebook_sha256_with_filename(
    tmp_path, monkeypatch
):
    """Both given must act as an AND -- matching only one of the two
    isn't enough, the same composition "source_notebook_filename" and
    "older_than_days" already give each other.
    """

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_sha256_and_filename_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_deploy_history_entry({
        "deployed_at": "2024-01-01T00:00:00+00:00",
        "tag": "match",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": "match.ipynb",
        "source_notebook_sha256": "abc",
    })
    # Matches the sha256 but not the filename -- must survive.
    upload_module._append_deploy_history_entry({
        "deployed_at": "2024-01-02T00:00:00+00:00",
        "tag": "sha-only",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": "other.ipynb",
        "source_notebook_sha256": "abc",
    })

    clear_resp = client.delete(
        "/api/deploy/history",
        params={
            "source_notebook_sha256": "abc",
            "source_notebook_filename": "match.ipynb",
        },
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json()["deleted_count"] == 1

    remaining = client.get("/api/deploy/history").json()
    assert remaining["entry_count"] == 1
    assert remaining["entries"][0]["source_notebook_filename"] == "other.ipynb"


def test_clear_deploy_history_is_a_no_op_success_when_nothing_was_ever_deployed(
    tmp_path, monkeypatch
):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_empty_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    resp = client.delete("/api/deploy/history")

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "dry_run": False, "deleted_count": 0}


def test_clear_deploy_history_does_not_touch_generated_dir_or_notebooks(
    tmp_path, monkeypatch
):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_isolation_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_deploy_history_entry({
        "deployed_at": "2024-01-01T00:00:00+00:00",
        "tag": "isolation:0",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": None,
        "source_notebook_sha256": None,
    })

    generated_dir_before = Path(upload_module.GENERATED_DIR)
    dockerfile_existed_before = (generated_dir_before / "Dockerfile").is_file()

    client.delete("/api/deploy/history")

    assert (generated_dir_before / "Dockerfile").is_file() == dockerfile_existed_before


def test_clear_deploy_history_older_than_days_keeps_recent_entries(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_age_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    now = datetime.now(timezone.utc)
    old_deployed_at = (now - timedelta(days=40)).isoformat()
    recent_deployed_at = (now - timedelta(days=1)).isoformat()

    upload_module._append_deploy_history_entry({
        "deployed_at": old_deployed_at,
        "tag": "old:0",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": None,
        "source_notebook_sha256": None,
    })
    upload_module._append_deploy_history_entry({
        "deployed_at": recent_deployed_at,
        "tag": "recent:0",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": None,
        "source_notebook_sha256": None,
    })

    clear_resp = client.delete(
        "/api/deploy/history", params={"older_than_days": 30}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 1}

    remaining = client.get("/api/deploy/history").json()
    assert remaining["entry_count"] == 1
    assert remaining["entries"][0]["tag"] == "recent:0"


def test_clear_deploy_history_composes_older_than_days_with_source_notebook_filename(
    tmp_path, monkeypatch
):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_age_and_source_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    now = datetime.now(timezone.utc)
    old_deployed_at = (now - timedelta(days=40)).isoformat()

    upload_module._append_deploy_history_entry({
        "deployed_at": old_deployed_at,
        "tag": "old-other:0",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": "other.ipynb",
        "source_notebook_sha256": None,
    })
    upload_module._append_deploy_history_entry({
        "deployed_at": old_deployed_at,
        "tag": "old-target:0",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": "target.ipynb",
        "source_notebook_sha256": None,
    })

    clear_resp = client.delete(
        "/api/deploy/history",
        params={"older_than_days": 30, "source_notebook_filename": "target.ipynb"},
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 1}

    remaining = client.get("/api/deploy/history").json()
    assert remaining["entry_count"] == 1
    assert remaining["entries"][0]["source_notebook_filename"] == "other.ipynb"


def test_clear_deploy_history_dry_run_reports_the_plan_without_deleting(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_dry_run_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_deploy_history_entry({
        "deployed_at": "2024-01-01T00:00:00+00:00",
        "tag": "dry-run:0",
        "platform": None,
        "pushed": False,
        "source_notebook_filename": None,
        "source_notebook_sha256": None,
    })

    clear_resp = client.delete("/api/deploy/history", params={"dry_run": True})

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": True, "deleted_count": 1}

    # Nothing was actually deleted.
    assert client.get("/api/deploy/history").json()["entry_count"] == 1


def test_clear_deploy_history_rejects_a_non_positive_older_than_days(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_deploy_history_bad_age_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    resp = client.delete("/api/deploy/history", params={"older_than_days": 0})

    assert resp.status_code == 400


def test_clear_compile_history_older_than_days_keeps_recent_entries(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_age_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    now = datetime.now(timezone.utc)
    old_compiled_at = (now - timedelta(days=40)).isoformat()
    recent_compiled_at = (now - timedelta(days=1)).isoformat()

    upload_module._append_compile_history_entry({
        "compiled_at": old_compiled_at,
        "notebook_filename": "old.ipynb",
        "source_notebook_sha256": "abc",
        "only": None,
        "exclude": None,
        "endpoint_count": 1,
        "dependency_count": 0,
        "skipped_function_count": 0,
    })
    upload_module._append_compile_history_entry({
        "compiled_at": recent_compiled_at,
        "notebook_filename": "recent.ipynb",
        "source_notebook_sha256": "def",
        "only": None,
        "exclude": None,
        "endpoint_count": 1,
        "dependency_count": 0,
        "skipped_function_count": 0,
    })

    clear_resp = client.delete(
        "/api/compile/history", params={"older_than_days": 30}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 1}

    remaining = client.get("/api/compile/history").json()
    assert remaining["entry_count"] == 1
    assert remaining["entries"][0]["notebook_filename"] == "recent.ipynb"


def test_clear_compile_history_dry_run_reports_the_plan_without_deleting(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_dry_run_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_compile_history_entry({
        "compiled_at": "2024-01-01T00:00:00+00:00",
        "notebook_filename": "dry_run.ipynb",
        "source_notebook_sha256": "abc",
        "only": None,
        "exclude": None,
        "endpoint_count": 1,
        "dependency_count": 0,
        "skipped_function_count": 0,
    })

    clear_resp = client.delete("/api/compile/history", params={"dry_run": True})

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": True, "deleted_count": 1}

    # Nothing was actually deleted.
    assert client.get("/api/compile/history").json()["entry_count"] == 1


def test_clear_compile_history_rejects_a_non_positive_older_than_days(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_bad_age_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    resp = client.delete("/api/compile/history", params={"older_than_days": -1})

    assert resp.status_code == 400


def test_compile_history_is_empty_before_any_compile(monkeypatch, tmp_path):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "compile_history_empty_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    resp = client.get("/api/compile/history")

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "entries": [], "entry_count": 0}


def test_compile_records_a_history_entry(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "compile_history_record_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    _compile_a_notebook("compile_history_record_test.ipynb")

    history_resp = client.get("/api/compile/history")
    assert history_resp.status_code == 200

    body = history_resp.json()
    assert body["entry_count"] == 1

    entry = body["entries"][0]
    assert entry["notebook_filename"] == "compile_history_record_test.ipynb"
    assert entry["endpoint_count"] == 1
    assert entry["only"] is None
    assert entry["exclude"] is None
    assert entry["source_notebook_sha256"]
    assert entry["compiled_at"]


def test_compile_history_records_only_and_exclude(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "compile_history_only_exclude_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
        "def sub(a: int, b: int) -> int:\n    return a - b\n"
    )
    client.post(
        "/api/upload",
        files={
            "file": (
                "compile_history_only_test.ipynb", io.BytesIO(content),
                "application/json",
            )
        },
    )

    resp = client.post(
        "/api/compile",
        json={"notebook_path": "compile_history_only_test.ipynb", "only": ["add"]},
    )
    assert resp.status_code == 200

    entry = client.get("/api/compile/history").json()["entries"][0]
    assert entry["only"] == ["add"]
    assert entry["exclude"] is None
    assert entry["endpoint_count"] == 1


def test_compile_history_lists_most_recent_first(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "compile_history_order_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    _compile_a_notebook("compile_history_order_a.ipynb")
    _compile_a_notebook("compile_history_order_b.ipynb")

    entries = client.get("/api/compile/history").json()["entries"]
    filenames_in_order = [e["notebook_filename"] for e in entries]
    assert filenames_in_order == [
        "compile_history_order_b.ipynb", "compile_history_order_a.ipynb",
    ]


def test_compile_history_is_capped_at_the_configured_maximum(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "compile_history_cap_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))
    monkeypatch.setattr(upload_module, "MAX_COMPILE_HISTORY_ENTRIES", 3)

    for i in range(5):
        upload_module._append_compile_history_entry({
            "compiled_at": f"2024-01-0{i + 1}T00:00:00+00:00",
            "notebook_filename": f"cap_{i}.ipynb",
            "source_notebook_sha256": "abc",
            "only": None,
            "exclude": None,
            "endpoint_count": 1,
            "dependency_count": 0,
            "skipped_function_count": 0,
        })

    body = client.get("/api/compile/history").json()
    assert body["entry_count"] == 3
    assert [e["notebook_filename"] for e in body["entries"]] == [
        "cap_4.ipynb", "cap_3.ipynb", "cap_2.ipynb",
    ]


def _seed_compile_history_for_filtering(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "compile_history_filter_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    entries = [
        {
            "compiled_at": "2024-01-01T00:00:00+00:00", "notebook_filename": "one.ipynb",
            "source_notebook_sha256": "aaa", "only": None, "exclude": None,
            "endpoint_count": 1, "dependency_count": 0, "skipped_function_count": 0,
        },
        {
            "compiled_at": "2024-01-02T00:00:00+00:00", "notebook_filename": "two.ipynb",
            "source_notebook_sha256": "bbb", "only": None, "exclude": None,
            "endpoint_count": 2, "dependency_count": 1, "skipped_function_count": 0,
        },
        {
            "compiled_at": "2024-01-03T00:00:00+00:00", "notebook_filename": "one.ipynb",
            "source_notebook_sha256": "ccc", "only": None, "exclude": None,
            "endpoint_count": 3, "dependency_count": 1, "skipped_function_count": 1,
        },
    ]

    for entry in entries:
        upload_module._append_compile_history_entry(entry)


def test_compile_history_filters_by_notebook_filename(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/compile/history", params={"notebook_filename": "one.ipynb"}
    ).json()

    assert [e["source_notebook_sha256"] for e in body["entries"]] == ["ccc", "aaa"]
    assert body["entry_count"] == 2


def test_compile_history_filters_by_compiled_after_and_before(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    after_body = client.get(
        "/api/compile/history", params={"compiled_after": "2024-01-02T00:00:00+00:00"}
    ).json()
    assert [e["notebook_filename"] for e in after_body["entries"]] == [
        "one.ipynb", "two.ipynb",
    ]

    before_body = client.get(
        "/api/compile/history", params={"compiled_before": "2024-01-02T00:00:00+00:00"}
    ).json()
    assert [e["notebook_filename"] for e in before_body["entries"]] == [
        "two.ipynb", "one.ipynb",
    ]

    window_body = client.get(
        "/api/compile/history",
        params={
            "compiled_after": "2024-01-02T00:00:00+00:00",
            "compiled_before": "2024-01-02T00:00:00+00:00",
        },
    ).json()
    assert [e["notebook_filename"] for e in window_body["entries"]] == ["two.ipynb"]


def test_compile_history_rejects_compiled_after_later_than_compiled_before():

    resp = client.get(
        "/api/compile/history",
        params={
            "compiled_after": "2026-06-01T00:00:00+00:00",
            "compiled_before": "2026-01-01T00:00:00+00:00",
        },
    )

    assert resp.status_code == 400
    assert "compiled_after" in resp.json()["detail"]


def test_compile_history_csv_format_returns_a_csv_response(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "compile_history_csv_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_compile_history_entry({
        "compiled_at": "2024-01-01T00:00:00+00:00", "notebook_filename": "nb.ipynb",
        "source_notebook_sha256": "aaa", "only": ["add", "subtract"], "exclude": None,
        "endpoint_count": 2, "dependency_count": 0, "skipped_function_count": 0,
    })

    resp = client.get("/api/compile/history", params={"format": "csv"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="compile_history.csv"' in resp.headers["content-disposition"]

    rows = resp.text.strip().split("\r\n")
    assert rows[0] == (
        "compiled_at,notebook_filename,source_notebook_sha256,only,exclude,"
        "endpoint_count,dependency_count,skipped_function_count"
    )
    # "only" is a semicolon-joined cell, not one CSV column per function.
    assert rows[1] == "2024-01-01T00:00:00+00:00,nb.ipynb,aaa,add;subtract,,2,0,0"
    assert len(rows) == 2


def test_compile_history_csv_format_composes_with_filters(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    resp = client.get(
        "/api/compile/history",
        params={"format": "csv", "notebook_filename": "one.ipynb"},
    )

    assert resp.status_code == 200
    rows = resp.text.strip().split("\r\n")
    assert len(rows) == 3
    assert rows[1].startswith("2024-01-03T00:00:00+00:00,one.ipynb,ccc,")
    assert rows[2].startswith("2024-01-01T00:00:00+00:00,one.ipynb,aaa,")


def test_compile_history_rejects_an_unknown_format():

    resp = client.get("/api/compile/history", params={"format": "xml"})

    assert resp.status_code == 400
    assert "format" in resp.json()["detail"]


def test_compile_history_filters_by_source_notebook_sha256(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/compile/history", params={"source_notebook_sha256": "bbb"}
    ).json()

    assert [e["notebook_filename"] for e in body["entries"]] == ["two.ipynb"]
    assert body["entry_count"] == 1


def test_compile_history_combines_source_notebook_sha256_and_notebook_filename(
    tmp_path, monkeypatch
):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/compile/history",
        params={"source_notebook_sha256": "ccc", "notebook_filename": "one.ipynb"},
    ).json()

    assert [e["source_notebook_sha256"] for e in body["entries"]] == ["ccc"]

    # A mismatched combination (a hash that was never compiled under this
    # filename) narrows to nothing, exactly like every other pair of
    # filters here.
    body = client.get(
        "/api/compile/history",
        params={"source_notebook_sha256": "ccc", "notebook_filename": "two.ipynb"},
    ).json()

    assert body["entries"] == []


def test_compile_history_unknown_source_notebook_sha256_yields_no_entries(
    tmp_path, monkeypatch
):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/compile/history", params={"source_notebook_sha256": "no-such-hash"}
    ).json()

    assert body["entries"] == []
    assert body["entry_count"] == 0


def test_compile_history_filters_by_unknown_notebook_filename_yields_no_entries(
    tmp_path, monkeypatch
):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/compile/history", params={"notebook_filename": "does_not_exist.ipynb"}
    ).json()

    assert body == {"status": "success", "entries": [], "entry_count": 0}


def test_compile_history_respects_limit(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get("/api/compile/history", params={"limit": 2}).json()

    assert [e["source_notebook_sha256"] for e in body["entries"]] == ["ccc", "bbb"]
    assert body["entry_count"] == 2


def test_compile_history_combines_notebook_filter_and_limit(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/compile/history",
        params={"notebook_filename": "one.ipynb", "limit": 1},
    ).json()

    assert [e["source_notebook_sha256"] for e in body["entries"]] == ["ccc"]


def test_compile_history_rejects_a_negative_limit(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    resp = client.get("/api/compile/history", params={"limit": -1})

    assert resp.status_code == 400


def test_compile_history_respects_offset(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get("/api/compile/history", params={"offset": 1}).json()

    assert [e["source_notebook_sha256"] for e in body["entries"]] == ["bbb", "aaa"]
    assert body["entry_count"] == 2


def test_compile_history_combines_offset_and_limit_for_paging(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get(
        "/api/compile/history", params={"offset": 1, "limit": 1}
    ).json()

    assert [e["source_notebook_sha256"] for e in body["entries"]] == ["bbb"]
    assert body["entry_count"] == 1


def test_compile_history_offset_past_the_end_yields_no_entries(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    body = client.get("/api/compile/history", params={"offset": 100}).json()

    assert body["entries"] == []
    assert body["entry_count"] == 0


def test_compile_history_rejects_a_negative_offset(tmp_path, monkeypatch):

    _seed_compile_history_for_filtering(tmp_path, monkeypatch)

    resp = client.get("/api/compile/history", params={"offset": -1})

    assert resp.status_code == 400


def test_clear_compile_history_removes_every_entry(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    for i in range(3):
        upload_module._append_compile_history_entry({
            "compiled_at": f"2024-01-0{i + 1}T00:00:00+00:00",
            "notebook_filename": f"clear_{i}.ipynb",
            "source_notebook_sha256": "abc",
            "only": None,
            "exclude": None,
            "endpoint_count": 1,
            "dependency_count": 0,
            "skipped_function_count": 0,
        })

    assert client.get("/api/compile/history").json()["entry_count"] == 3

    clear_resp = client.delete("/api/compile/history")

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 3}
    assert client.get("/api/compile/history").json() == {
        "status": "success", "entries": [], "entry_count": 0,
    }


def test_clear_compile_history_filters_by_notebook_filename(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_filter_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    for i in range(3):
        upload_module._append_compile_history_entry({
            "compiled_at": f"2024-01-0{i + 1}T00:00:00+00:00",
            "notebook_filename": "keep.ipynb" if i != 1 else "drop.ipynb",
            "source_notebook_sha256": "abc",
            "only": None,
            "exclude": None,
            "endpoint_count": 1,
            "dependency_count": 0,
            "skipped_function_count": 0,
        })

    clear_resp = client.delete(
        "/api/compile/history", params={"notebook_filename": "drop.ipynb"}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 1}

    remaining = client.get("/api/compile/history").json()
    assert remaining["entry_count"] == 2
    assert all(e["notebook_filename"] == "keep.ipynb" for e in remaining["entries"])


def test_clear_compile_history_filters_by_source_notebook_sha256(tmp_path, monkeypatch):
    """Mirrors test_clear_deploy_history_filters_by_source_notebook_sha256:
    GET /api/compile/history's own "source_notebook_sha256" filter,
    matching a notebook's exact content rather than whichever filename it
    happened to be compiled under, had no DELETE counterpart either.
    """

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_sha256_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    for i in range(3):
        upload_module._append_compile_history_entry({
            "compiled_at": f"2024-01-0{i + 1}T00:00:00+00:00",
            "notebook_filename": f"renamed-{i}.ipynb",
            "source_notebook_sha256": "abc" if i != 1 else "def",
            "only": None,
            "exclude": None,
            "endpoint_count": 1,
            "dependency_count": 0,
            "skipped_function_count": 0,
        })

    clear_resp = client.delete(
        "/api/compile/history", params={"source_notebook_sha256": "def"}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 1}

    remaining = client.get("/api/compile/history").json()
    assert remaining["entry_count"] == 2
    assert all(e["source_notebook_sha256"] == "abc" for e in remaining["entries"])


def test_clear_compile_history_unknown_source_notebook_sha256_deletes_nothing(
    tmp_path, monkeypatch
):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_unknown_sha256_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_compile_history_entry({
        "compiled_at": "2024-01-01T00:00:00+00:00",
        "notebook_filename": "keep.ipynb",
        "source_notebook_sha256": "abc",
        "only": None,
        "exclude": None,
        "endpoint_count": 1,
        "dependency_count": 0,
        "skipped_function_count": 0,
    })

    clear_resp = client.delete(
        "/api/compile/history", params={"source_notebook_sha256": "no-such-hash"}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 0}
    assert client.get("/api/compile/history").json()["entry_count"] == 1


def test_clear_compile_history_unknown_notebook_filename_deletes_nothing(
    tmp_path, monkeypatch
):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_unknown_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_compile_history_entry({
        "compiled_at": "2024-01-01T00:00:00+00:00",
        "notebook_filename": "keep.ipynb",
        "source_notebook_sha256": "abc",
        "only": None,
        "exclude": None,
        "endpoint_count": 1,
        "dependency_count": 0,
        "skipped_function_count": 0,
    })

    clear_resp = client.delete(
        "/api/compile/history", params={"notebook_filename": "no-such.ipynb"}
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json() == {"status": "success", "dry_run": False, "deleted_count": 0}
    assert client.get("/api/compile/history").json()["entry_count"] == 1


def test_clear_compile_history_is_a_no_op_success_when_nothing_was_ever_compiled(
    tmp_path, monkeypatch
):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_empty_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    resp = client.delete("/api/compile/history")

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "dry_run": False, "deleted_count": 0}


def test_clear_compile_history_does_not_touch_generated_dir_or_notebooks(
    tmp_path, monkeypatch
):

    from backend.routes import upload as upload_module

    isolated_upload_dir = tmp_path / "clear_compile_history_isolation_upload_dir"
    isolated_upload_dir.mkdir()
    monkeypatch.setattr(upload_module, "UPLOAD_DIR", str(isolated_upload_dir))

    upload_module._append_compile_history_entry({
        "compiled_at": "2024-01-01T00:00:00+00:00",
        "notebook_filename": "isolation.ipynb",
        "source_notebook_sha256": "abc",
        "only": None,
        "exclude": None,
        "endpoint_count": 1,
        "dependency_count": 0,
        "skipped_function_count": 0,
    })

    generated_dir_before = Path(upload_module.GENERATED_DIR)
    dockerfile_existed_before = (generated_dir_before / "Dockerfile").is_file()

    client.delete("/api/compile/history")

    assert (generated_dir_before / "Dockerfile").is_file() == dockerfile_existed_before


def test_deploy_endpoint_returns_500_when_docker_is_missing(tmp_path, monkeypatch):

    _compile_a_notebook("deploy_missing_docker_test.ipynb")

    empty_bin_dir = tmp_path / "empty_bin"
    empty_bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin_dir))

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 500
    assert "Docker CLI not found" in resp.json()["detail"]


def test_deploy_endpoint_returns_500_when_docker_is_missing_for_push(monkeypatch):
    """Before this fix, only the `docker build` call handled Docker being
    missing at all -- `docker push` had no FileNotFoundError handling
    whatsoever, so losing Docker between a successful build and the push
    step crashed the request instead of returning a clean error.
    """

    from backend.routes import upload as upload_module

    _compile_a_notebook("deploy_missing_docker_for_push_test.ipynb")

    def fake_run(args, **kwargs):
        if args[1] == "build":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(upload_module.subprocess, "run", fake_run)

    resp = client.post("/api/deploy", json={"push": True})

    assert resp.status_code == 500
    assert "Docker CLI not found" in resp.json()["detail"]


def test_deploy_endpoint_returns_504_when_docker_build_times_out(monkeypatch):
    """subprocess.run(..., timeout=...) raises TimeoutExpired, not
    FileNotFoundError -- before this fix, that exception type wasn't
    caught anywhere in /api/deploy at all, so a `docker build` that ran
    past the timeout crashed the request with FastAPI's generic
    unhandled-exception 500 instead of an actionable error.
    """

    from backend.routes import upload as upload_module

    _compile_a_notebook("deploy_build_timeout_test.ipynb")

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(upload_module.subprocess, "run", fake_run)

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 504
    assert "did not finish within" in resp.json()["detail"]


def test_deploy_endpoint_returns_504_when_docker_push_times_out(monkeypatch):

    from backend.routes import upload as upload_module

    _compile_a_notebook("deploy_push_timeout_test.ipynb")

    def fake_run(args, **kwargs):
        if args[1] == "build":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(upload_module.subprocess, "run", fake_run)

    resp = client.post("/api/deploy", json={"push": True})

    assert resp.status_code == 504
    assert "did not finish within" in resp.json()["detail"]


def test_deploy_subprocess_timeout_is_configurable(monkeypatch):
    """DEPLOY_SUBPROCESS_TIMEOUT_SECONDS matches the existing
    NOTEBOOK_API_* env-var convention (see MAX_UPLOAD_BYTES) instead of
    the fixed 600s previously hardcoded directly into each subprocess.run
    call, so a deploy environment that needs longer (a slow/cold layer
    cache) or wants it clamped shorter (fail fast in CI) can configure it.
    """

    from backend.routes import upload as upload_module

    _compile_a_notebook("deploy_custom_timeout_test.ipynb")

    monkeypatch.setattr(upload_module, "DEPLOY_SUBPROCESS_TIMEOUT_SECONDS", 5)

    captured_kwargs = {}

    def fake_run(args, **kwargs):
        captured_kwargs.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(upload_module.subprocess, "run", fake_run)

    resp = client.post("/api/deploy", json={})

    assert resp.status_code == 200
    assert captured_kwargs["timeout"] == 5


def test_download_returns_404_when_nothing_compiled_yet(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_download_test_missing_dir"
    )

    resp = client.get("/api/download")

    assert resp.status_code == 404


def test_download_returns_a_zip_of_the_compiled_app():

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "download_flow_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": "download_flow_test.ipynb"}
    )
    assert compile_resp.status_code == 200

    download_resp = client.get("/api/download")

    assert download_resp.status_code == 200
    assert download_resp.headers["content-type"] == "application/zip"
    assert (
        'attachment; filename="generated.zip"'
        == download_resp.headers["content-disposition"]
    )

    archive = zipfile.ZipFile(io.BytesIO(download_resp.content))
    names = set(archive.namelist())

    assert "app.py" in names
    assert "requirements.txt" in names
    assert "Dockerfile" in names
    assert "runtime/notebook_module.py" in names

    app_source = archive.read("app.py").decode("utf-8")
    assert "def add(" in app_source


def test_download_reports_a_bundle_sha256_matching_generated_checksums():
    """"X-Bundle-SHA256" summarizes this exact zip's own file set with the
    same _bundle_sha256 GET /api/generated?checksums=true's own
    "bundle_sha256" already uses for the identical GENERATED_DIR content
    -- so a caller who downloaded this zip can confirm it matches what
    this dashboard currently has compiled without re-fetching or
    re-hashing anything.
    """

    filename = "download_bundle_sha256_test.ipynb"
    _compile_a_notebook(filename)

    download_resp = client.get("/api/download")
    assert download_resp.status_code == 200
    bundle_sha256 = download_resp.headers["x-bundle-sha256"]
    assert bundle_sha256

    generated_resp = client.get("/api/generated", params={"checksums": "true"})
    assert generated_resp.status_code == 200
    assert generated_resp.json()["bundle_sha256"] == bundle_sha256


def test_download_reports_a_quoted_etag_matching_the_bundle_sha256():

    filename = "download_etag_test.ipynb"
    _compile_a_notebook(filename)

    resp = client.get("/api/download")

    assert resp.status_code == 200
    assert resp.headers["etag"] == f'"{resp.headers["x-bundle-sha256"]}"'


def test_download_returns_304_when_if_none_match_matches_the_current_etag():

    filename = "download_conditional_test.ipynb"
    _compile_a_notebook(filename)

    first = client.get("/api/download")
    etag = first.headers["etag"]

    second = client.get("/api/download", headers={"If-None-Match": etag})

    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["etag"] == etag
    assert second.headers["x-bundle-sha256"] == first.headers["x-bundle-sha256"]
    assert (
        second.headers["x-notebook-changed-since-compile"]
        == first.headers["x-notebook-changed-since-compile"]
    )
    assert first.headers["cache-control"] == "no-cache"
    assert second.headers["cache-control"] == "no-cache"


def test_download_returns_200_after_a_recompile_changes_the_bundle():

    filename = "download_conditional_recompile_test.ipynb"
    _compile_a_notebook(filename)

    stale_etag = client.get("/api/download").headers["etag"]

    changed_content = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(changed_content), "application/json")},
    )
    client.post("/api/compile", json={"notebook_path": filename})

    resp = client.get("/api/download", headers={"If-None-Match": stale_etag})

    assert resp.status_code == 200
    assert resp.headers["etag"] != stale_etag
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    assert "app.py" in archive.namelist()


def test_download_reports_not_stale_right_after_compile():

    filename = "download_not_stale_test.ipynb"
    _compile_a_notebook(filename)

    resp = client.get("/api/download")

    assert resp.status_code == 200
    assert resp.headers["x-notebook-changed-since-compile"] == "false"


def test_download_reports_stale_after_the_source_notebook_changes():
    """Unlike POST /api/deploy, GET /api/download never refuses a stale
    build outright -- it has no "force" escape hatch, and downloading a
    zip doesn't ship it anywhere the way a Docker build/push would -- but
    a caller who does care about staleness (e.g. this CLI's own
    `remote-build`, which warns on it) needs a way to find out without a
    separate GET /api/notebooks call.
    """

    filename = "download_stale_test.ipynb"
    _compile_a_notebook(filename)

    changed_content = _notebook_bytes(
        "def subtract(a: int, b: int) -> int:\n    return a - b\n"
    )
    overwrite_resp = client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(changed_content), "application/json")},
    )
    assert overwrite_resp.status_code == 200

    resp = client.get("/api/download")

    assert resp.status_code == 200
    assert resp.headers["x-notebook-changed-since-compile"] == "true"

    # Still returns the (now-stale) zip itself -- unlike /api/deploy, this
    # never turns into a 409.
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    assert "app.py" in archive.namelist()


def test_download_excludes_pycache_from_the_zip():
    """__pycache__ is created by Python itself the first time the compiled
    app or its runtime module gets imported (e.g. by a prior
    /api/export-openapi call) -- it is not part of what the compiler
    actually wrote, and its .pyc filenames are tied to whichever Python
    version happened to import it. Before this fix, the downloaded
    "compiled app" bundle could ship this non-portable bytecode cache
    alongside the actual deliverable.
    """

    _compile_a_notebook("download_pycache_test.ipynb")

    generated_dir = Path("generated")
    pycache_dir = generated_dir / "__pycache__"
    nested_pycache_dir = generated_dir / "runtime" / "__pycache__"

    try:
        pycache_dir.mkdir(exist_ok=True)
        (pycache_dir / "app.cpython-314.pyc").write_bytes(b"\x00")

        nested_pycache_dir.mkdir(parents=True, exist_ok=True)
        (nested_pycache_dir / "notebook_module.cpython-314.pyc").write_bytes(b"\x00")

        download_resp = client.get("/api/download")

        assert download_resp.status_code == 200

        archive = zipfile.ZipFile(io.BytesIO(download_resp.content))
        names = archive.namelist()

        assert "app.py" in names
        assert not any("__pycache__" in name for name in names)
        assert not any(name.endswith(".pyc") for name in names)
    finally:
        shutil.rmtree(pycache_dir, ignore_errors=True)
        shutil.rmtree(nested_pycache_dir, ignore_errors=True)


def test_download_excludes_compile_metadata_from_the_zip():
    """.compile_metadata.json (write_compile_metadata, backend/compiler.py)
    is dashboard-internal bookkeeping, not a compiled deliverable -- and
    its "source_notebook" field is the source notebook's absolute
    filesystem path on the compiling server. Before this fix, it was
    written into GENERATED_DIR by every compile like any other file, so it
    was zipped up and handed back by this endpoint too, alongside the
    actual deliverable.
    """

    _compile_a_notebook("download_compile_metadata_test.ipynb")

    assert (Path("generated") / ".compile_metadata.json").is_file()

    download_resp = client.get("/api/download")

    assert download_resp.status_code == 200

    archive = zipfile.ZipFile(io.BytesIO(download_resp.content))
    names = archive.namelist()

    assert "app.py" in names
    assert ".compile_metadata.json" not in names


def test_download_waits_for_an_in_flight_compile_to_release_compile_lock():
    """GET /api/download walks GENERATED_DIR to build its zip -- without
    holding COMPILE_LOCK (see backend/compiler.py) for that walk, a
    concurrent POST /api/compile racing it on another thread (both run in
    FastAPI's worker threadpool -- see the plain `def` routes in this
    module) could rewrite files out from under it mid-zip, downloading a
    torn mix of the old and new compile. Verified by holding the lock
    from a background thread and confirming this request doesn't return
    until it's released.
    """

    _compile_a_notebook("download_lock_test.ipynb")

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        with COMPILE_LOCK:
            lock_acquired.set()
            assert release_lock.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_acquired.wait(timeout=5)

    result = {}

    def do_download():
        result["resp"] = client.get("/api/download")

    request_thread = threading.Thread(target=do_download)
    request_thread.start()
    request_thread.join(timeout=0.3)

    assert request_thread.is_alive(), (
        "GET /api/download should still be blocked on COMPILE_LOCK"
    )

    release_lock.set()
    request_thread.join(timeout=5)
    holder.join(timeout=5)

    assert not request_thread.is_alive()
    assert result["resp"].status_code == 200


def test_list_generated_files_returns_empty_before_any_compile(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_list_files_test_missing_dir"
    )

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["generated_files"] == []
    assert body["file_details"] == []
    assert body["compiled_at"] is None
    assert body["source_notebook_filename"] is None
    assert body["source_notebook_exists"] is False
    assert body["generated_files_modified_since_compile"] is None


def test_list_generated_files_reports_false_immediately_after_a_compile():

    _compile_a_notebook("generated_files_modified_false_test.ipynb")

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    assert resp.json()["generated_files_modified_since_compile"] is False

    os.remove(Path(UPLOAD_DIR) / "generated_files_modified_false_test.ipynb")


def test_list_generated_files_reports_true_after_a_generated_file_is_hand_edited():
    """The output-side mirror of notebook_changed_since_compile: catches
    the *compiled output itself* (not the source notebook) having been
    hand-edited on the server since the last compile.
    """

    _compile_a_notebook("generated_files_modified_true_test.ipynb")

    (Path(GENERATED_DIR) / "requirements.txt").write_text(
        "fastapi==0.0.0\n", encoding="utf-8"
    )

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    assert resp.json()["generated_files_modified_since_compile"] is True

    os.remove(Path(UPLOAD_DIR) / "generated_files_modified_true_test.ipynb")


def test_list_generated_files_modified_since_compile_ignores_an_unrelated_export():
    """A later POST /api/export-openapi/export-sdk writing openapi.json/
    sdk/ into GENERATED_DIR must never itself be mistaken for the
    compiled output having been tampered with -- neither is part of what
    a compile itself actually produces.
    """

    _compile_a_notebook("generated_files_modified_export_test.ipynb")

    (Path(GENERATED_DIR) / "openapi.json").write_text("{}", encoding="utf-8")

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    assert resp.json()["generated_files_modified_since_compile"] is False

    os.remove(Path(UPLOAD_DIR) / "generated_files_modified_export_test.ipynb")
    os.remove(Path(GENERATED_DIR) / "openapi.json")


def test_list_generated_files_lists_the_compiled_output():

    _compile_a_notebook("list_generated_files_test.ipynb")

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "app.py" in body["generated_files"]
    assert "requirements.txt" in body["generated_files"]
    assert body["compiled_at"] is not None
    assert body["source_notebook_filename"] == "list_generated_files_test.ipynb"
    assert body["source_notebook_exists"] is True

    os.remove(Path(UPLOAD_DIR) / "list_generated_files_test.ipynb")


def test_list_generated_files_file_details_reports_size_and_modified_at():
    """"file_details" closes a gap "generated_files" (a bare list of
    names) always had: a dashboard frontend wanting to show a real file
    browser for what's compiled (file sizes, most-recently-touched-first,
    ...) had to issue a separate GET /api/generated/{filename} call per
    file just to learn how big each one is -- exactly the level of detail
    GET /api/notebooks already reports per uploaded notebook.
    """

    from backend.routes import upload as upload_module

    filename = "list_generated_files_file_details_test.ipynb"
    _compile_a_notebook(filename)

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    body = resp.json()

    file_details_by_name = {
        entry["filename"]: entry for entry in body["file_details"]
    }

    assert set(file_details_by_name) == set(body["generated_files"])

    app_py_details = file_details_by_name["app.py"]
    expected_size = (Path(upload_module.GENERATED_DIR) / "app.py").stat().st_size

    assert app_py_details["size_bytes"] == expected_size
    assert app_py_details["size_bytes"] > 0
    assert "modified_at" in app_py_details

    os.remove(Path(UPLOAD_DIR) / filename)


def test_list_generated_files_omits_checksums_by_default():

    filename = "list_generated_files_no_checksums_test.ipynb"
    _compile_a_notebook(filename)

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    body = resp.json()
    assert "bundle_sha256" not in body
    for entry in body["file_details"]:
        assert "sha256" not in entry

    os.remove(Path(UPLOAD_DIR) / filename)


def test_list_generated_files_checksums_true_reports_a_sha256_per_file_and_a_bundle_hash():

    import hashlib

    from backend.routes import upload as upload_module

    filename = "list_generated_files_checksums_test.ipynb"
    _compile_a_notebook(filename)

    resp = client.get("/api/generated", params={"checksums": "true"})

    assert resp.status_code == 200
    body = resp.json()

    file_details_by_name = {
        entry["filename"]: entry for entry in body["file_details"]
    }

    app_py_path = Path(upload_module.GENERATED_DIR) / "app.py"
    expected_sha256 = hashlib.sha256(app_py_path.read_bytes()).hexdigest()

    assert file_details_by_name["app.py"]["sha256"] == expected_sha256
    assert all("sha256" in entry for entry in body["file_details"])
    assert isinstance(body["bundle_sha256"], str)
    assert len(body["bundle_sha256"]) == 64

    os.remove(Path(UPLOAD_DIR) / filename)


def test_list_generated_files_bundle_sha256_changes_when_a_file_changes():

    filename = "list_generated_files_bundle_hash_changes_test.ipynb"
    _compile_a_notebook(filename)

    first = client.get("/api/generated", params={"checksums": "true"}).json()

    # Recompile with different content -- app.py's own bytes change.
    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n    return a * b\n"
    )
    client.post(
        "/api/upload?overwrite=true",
        files={"file": (filename, io.BytesIO(content), "application/json")},
    )
    client.post("/api/compile", json={"notebook_path": filename})

    second = client.get("/api/generated", params={"checksums": "true"}).json()

    assert first["bundle_sha256"] != second["bundle_sha256"]

    os.remove(Path(UPLOAD_DIR) / filename)


def test_bundle_sha256_is_independent_of_input_order():

    from backend.routes.upload import _bundle_sha256

    entries = [
        {"filename": "app.py", "sha256": "aaa"},
        {"filename": "requirements.txt", "sha256": "bbb"},
    ]

    assert _bundle_sha256(entries) == _bundle_sha256(list(reversed(entries)))


def test_bundle_sha256_changes_when_a_files_hash_changes():

    from backend.routes.upload import _bundle_sha256

    entries = [{"filename": "app.py", "sha256": "aaa"}]
    changed = [{"filename": "app.py", "sha256": "different"}]

    assert _bundle_sha256(entries) != _bundle_sha256(changed)


def test_list_generated_files_file_details_excludes_pycache_and_compile_metadata():

    from backend.routes import upload as upload_module

    filename = "list_generated_files_file_details_exclusions_test.ipynb"
    _compile_a_notebook(filename)

    pycache_dir = Path(upload_module.GENERATED_DIR) / "__pycache__"
    pycache_dir.mkdir(exist_ok=True)
    (pycache_dir / "app.cpython-000.pyc").write_bytes(b"")

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    body = resp.json()

    detail_filenames = {entry["filename"] for entry in body["file_details"]}

    assert not any("__pycache__" in name for name in detail_filenames)
    assert COMPILE_METADATA_FILENAME not in detail_filenames

    shutil.rmtree(pycache_dir)
    os.remove(Path(UPLOAD_DIR) / filename)


def test_list_generated_files_excludes_pycache_and_compile_metadata():
    """Same exclusions GET /api/download's zip and GET
    /api/generated/{filename} already apply -- __pycache__ is a Python-
    created implementation artifact never written by the compiler, and
    .compile_metadata.json is dashboard-internal bookkeeping whose
    "source_notebook" field is an absolute filesystem path on the
    compiling server, not a compiled deliverable this listing should
    expose.
    """

    from backend.routes import upload as upload_module

    filename = "list_generated_files_exclusions_test.ipynb"
    _compile_a_notebook(filename)

    pycache_dir = Path(upload_module.GENERATED_DIR) / "__pycache__"
    pycache_dir.mkdir(exist_ok=True)
    (pycache_dir / "app.cpython-000.pyc").write_bytes(b"")

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    body = resp.json()
    assert not any("__pycache__" in f for f in body["generated_files"])
    assert COMPILE_METADATA_FILENAME not in body["generated_files"]

    shutil.rmtree(pycache_dir)
    os.remove(Path(UPLOAD_DIR) / filename)


def test_list_generated_files_reports_the_notebook_still_exists_after_deleting_an_unrelated_one():

    filename = "list_generated_files_survives_test.ipynb"
    _compile_a_notebook(filename)

    other_content = _notebook_bytes("def sub(a, b):\n    return a - b\n")
    client.post(
        "/api/upload",
        files={
            "file": (
                "list_generated_files_unrelated.ipynb",
                io.BytesIO(other_content),
                "application/json",
            )
        },
    )

    delete_resp = client.delete("/api/notebooks/list_generated_files_unrelated.ipynb")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["was_currently_compiled"] is False

    resp = client.get("/api/generated")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_notebook_filename"] == filename
    assert body["source_notebook_exists"] is True

    os.remove(Path(UPLOAD_DIR) / filename)


def test_list_generated_files_reports_source_notebook_gone_after_it_is_deleted():
    """The gap this endpoint closes: deleting the notebook that produced
    GENERATED_DIR's current contents (DELETE /api/notebooks/{filename},
    "was_currently_compiled": true) doesn't touch GENERATED_DIR at all --
    the compiled app keeps running exactly as before -- but previously
    left no way to even list what's still in it, short of GET
    /api/download's opaque zip bytes or already knowing an exact filename
    to pass GET /api/generated/{filename}.
    """

    filename = "list_generated_files_orphan_test.ipynb"
    _compile_a_notebook(filename)

    delete_resp = client.delete(f"/api/notebooks/{filename}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["was_currently_compiled"] is True

    resp = client.get("/api/generated")

    assert resp.status_code == 200
    body = resp.json()
    # The generated app itself is untouched by deleting its source
    # notebook -- it's still fully listable.
    assert "app.py" in body["generated_files"]
    assert body["source_notebook_filename"] == filename
    assert body["source_notebook_exists"] is False


def test_delete_generated_app_returns_404_when_nothing_compiled_yet(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_delete_test_missing_dir"
    )

    resp = client.delete("/api/generated")

    assert resp.status_code == 404


def test_delete_generated_app_removes_the_generated_directory(tmp_path, monkeypatch):
    """Before this, the only ways to make GENERATED_DIR empty again were
    to delete it by hand on the server's filesystem, or to recompile some
    other notebook over it -- which still leaves *a* compiled app sitting
    there, just a different one. An operator who wants to actually
    reclaim the disk space or reset the dashboard to a clean slate had no
    endpoint to call for it.
    """

    from backend.routes import upload as upload_module

    custom_dir = tmp_path / "generated_delete_test"
    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(custom_dir))

    filename = "delete_generated_app_test.ipynb"
    _compile_a_notebook(filename)

    assert custom_dir.is_dir()

    resp = client.delete("/api/generated")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["generated_dir"] == str(custom_dir)
    assert not custom_dir.exists()

    os.remove(Path(UPLOAD_DIR) / filename)


def test_delete_generated_app_resets_list_generated_files_to_empty(tmp_path, monkeypatch):

    from backend.routes import upload as upload_module

    custom_dir = tmp_path / "generated_delete_reset_test"
    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(custom_dir))

    filename = "delete_generated_app_reset_test.ipynb"
    _compile_a_notebook(filename)

    delete_resp = client.delete("/api/generated")
    assert delete_resp.status_code == 200

    list_resp = client.get("/api/generated")
    assert list_resp.status_code == 200
    body = list_resp.json()
    assert body["generated_files"] == []
    assert body["compiled_at"] is None
    assert body["source_notebook_filename"] is None
    assert body["source_notebook_exists"] is False

    os.remove(Path(UPLOAD_DIR) / filename)


def test_delete_generated_app_evicts_the_compiled_app_from_the_module_cache():
    """Confirmed exploitable for a different endpoint before
    _evict_compiled_app_from_module_cache existed (see its own docstring
    on POST /api/export-openapi): plain importlib.import_module resolves
    an already-imported package straight from sys.modules, not from disk.
    Once GENERATED_DIR is deleted entirely, a cached import of it in this
    long-running dashboard process no longer corresponds to anything on
    disk at all -- the same staleness class /api/export-openapi's own
    eviction already guards against, just total instead of partial.

    Deliberately exercises the real, default GENERATED_DIR ("generated")
    rather than an isolated tmp_path one: export_openapi_schema imports
    "<package_name>.app" via plain importlib.import_module, which only
    resolves at all when GENERATED_DIR's parent is already on sys.path --
    true for the project root every other export-openapi test in this file
    already relies on, not for an arbitrary tmp_path directory.
    """

    import sys

    from backend.routes import upload as upload_module

    filename = "delete_generated_app_evict_test.ipynb"
    _compile_a_notebook(filename)

    # POST /api/export-openapi already imports "<package_name>.app" into
    # sys.modules as part of exporting the schema.
    export_resp = client.post("/api/export-openapi", json={"format": "json"})
    assert export_resp.status_code == 200

    package_name = upload_module.package_name_for_output_dir(
        upload_module.GENERATED_DIR
    )
    assert package_name in sys.modules

    delete_resp = client.delete("/api/generated")
    assert delete_resp.status_code == 200

    assert package_name not in sys.modules
    assert not any(
        name == package_name or name.startswith(f"{package_name}.")
        for name in sys.modules
    )

    os.remove(Path(UPLOAD_DIR) / filename)


def test_get_generated_file_returns_404_when_nothing_compiled_yet(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(
        upload_module, "GENERATED_DIR", "generated_get_file_test_missing_dir"
    )

    resp = client.get("/api/generated/app.py")

    assert resp.status_code == 404


def test_get_generated_file_returns_app_py_content():
    """GET /api/download already lets a caller retrieve the whole compiled
    output as a zip, and inspect_notebook_data's "generated_files" field
    already lists what's in it by name -- but before this, there was no
    way to read any *one* of those files' actual content back through the
    API: a dashboard wanting to preview "here's the app.py you're about
    to deploy" had no choice but to download and unzip the entire bundle
    client-side just to show a single file.
    """

    _compile_a_notebook("get_file_app_py_test.ipynb")

    resp = client.get("/api/generated/app.py")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["filename"] == "app.py"
    assert "def add(" in body["content"]


def test_get_generated_file_reports_content_sha256():
    """The "sha256" response field lets a caller verify the file it just
    fetched, the same content-integrity check GET /api/notebooks/{filename}
    (via its "X-Content-SHA256" header) and GET /api/generated's own
    "checksums" query param already provide, applied here to a single
    compiled file preview.
    """

    _compile_a_notebook("get_file_sha256_test.ipynb")

    resp = client.get("/api/generated/app.py")

    assert resp.status_code == 200
    body = resp.json()
    expected_sha256 = hashlib.sha256(body["content"].encode("utf-8")).hexdigest()
    assert body["sha256"] == expected_sha256


def test_get_generated_file_returns_requirements_txt_content():

    _compile_a_notebook("get_file_requirements_test.ipynb")

    resp = client.get("/api/generated/requirements.txt")

    assert resp.status_code == 200
    body = resp.json()
    assert "fastapi" in body["content"]


def test_get_generated_file_supports_nested_paths():
    """The runtime module lives under a subdirectory ("runtime/
    notebook_module.py"), not directly in GENERATED_DIR -- the route must
    accept a nested path as a single parameter, not just a bare filename,
    the way GET /api/notebooks/{filename} does.
    """

    _compile_a_notebook("get_file_nested_test.ipynb")

    resp = client.get("/api/generated/runtime/notebook_module.py")

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "runtime/notebook_module.py"
    assert "def add(" in body["content"]


def test_get_generated_file_returns_404_for_a_file_that_does_not_exist():

    _compile_a_notebook("get_file_missing_test.ipynb")

    resp = client.get("/api/generated/does_not_exist.txt")

    assert resp.status_code == 404


def test_get_generated_file_rejects_absolute_path():

    _compile_a_notebook("get_file_absolute_test.ipynb")

    resp = client.get("/api/generated//etc/passwd")

    assert resp.status_code in (400, 404)


def test_get_generated_file_rejects_relative_traversal():
    """Confirmed exploitable before resolve_generated_path existed: a
    filename like "../../etc/passwd" would resolve outside GENERATED_DIR
    entirely, the same traversal hazard resolve_upload_path already
    guards against for UPLOAD_DIR.
    """

    _compile_a_notebook("get_file_traversal_test.ipynb")

    resp = client.get("/api/generated/../../../../etc/passwd")

    assert resp.status_code in (400, 404)
    assert "root:" not in resp.text


def test_get_generated_file_rejects_a_filename_with_an_embedded_null_byte():
    """Confirmed exploitable before this fix: a null byte in the filename
    sailed past resolve_generated_path's absolute-path guard clause, but
    the later .resolve() call raised a bare ValueError from the
    underlying os.path.realpath/lstat syscalls, an unhandled 500 instead
    of the same clean 400/404 an absolute or traversal path already gets
    above.
    """

    _compile_a_notebook("get_file_null_byte_test.ipynb")

    resp = client.get("/api/generated/app%00.py")

    assert resp.status_code == 400


def test_get_generated_file_excludes_pycache():
    """Same __pycache__ exclusion GET /api/download and
    inspect_notebook_data's "generated_files" field already apply (see
    EXCLUDED_GENERATED_DIR_NAMES) -- it's a Python-created implementation
    artifact never actually written by the compiler, not a real
    deliverable this endpoint should serve back.
    """

    _compile_a_notebook("get_file_pycache_test.ipynb")

    generated_dir = Path("generated")
    pycache_dir = generated_dir / "__pycache__"

    try:
        pycache_dir.mkdir(exist_ok=True)
        (pycache_dir / "app.cpython-314.pyc").write_bytes(b"\x00")

        resp = client.get("/api/generated/__pycache__/app.cpython-314.pyc")

        assert resp.status_code == 404
    finally:
        shutil.rmtree(pycache_dir, ignore_errors=True)


def test_get_generated_file_excludes_compile_metadata():
    """.compile_metadata.json (write_compile_metadata, backend/compiler.py)
    is dashboard-internal bookkeeping, never a compiled deliverable this
    endpoint should serve back -- and its "source_notebook" field is the
    source notebook's absolute filesystem path on the compiling server, so
    serving it back would leak server-side filesystem layout to any caller
    who guesses the filename.
    """

    _compile_a_notebook("get_file_compile_metadata_test.ipynb")

    assert (Path("generated") / ".compile_metadata.json").is_file()

    resp = client.get("/api/generated/.compile_metadata.json")

    assert resp.status_code == 404


def test_get_generated_file_waits_for_an_in_flight_compile_to_release_compile_lock():
    """GET /api/generated/{filename} reads a file out of GENERATED_DIR --
    without holding COMPILE_LOCK (see backend/compiler.py) for that read,
    a concurrent POST /api/compile racing it on another thread could
    rewrite that exact file out from under it mid-read, the same hazard
    already guarded against for GET /api/download's zip walk.
    """

    _compile_a_notebook("get_file_lock_test.ipynb")

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        with COMPILE_LOCK:
            lock_acquired.set()
            assert release_lock.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_acquired.wait(timeout=5)

    result = {}

    def do_get():
        result["resp"] = client.get("/api/generated/app.py")

    request_thread = threading.Thread(target=do_get)
    request_thread.start()
    request_thread.join(timeout=0.3)

    assert request_thread.is_alive(), (
        "GET /api/generated/{filename} should still be blocked on COMPILE_LOCK"
    )

    release_lock.set()
    request_thread.join(timeout=5)
    holder.join(timeout=5)

    assert not request_thread.is_alive()
    assert result["resp"].status_code == 200


def test_export_openapi_waits_for_an_in_flight_compile_to_release_compile_lock():
    """POST /api/export-openapi dynamically imports "<package_name>.app"
    -- without holding COMPILE_LOCK for that import, a concurrent POST
    /api/compile racing it on another thread could rewrite app.py (and
    the runtime module it imports) mid-import, importing a torn mix of
    the old and new compile instead of a consistent one.
    """

    _compile_a_notebook("export_openapi_lock_test.ipynb")

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        with COMPILE_LOCK:
            lock_acquired.set()
            assert release_lock.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_acquired.wait(timeout=5)

    result = {}

    def do_export():
        result["resp"] = client.post("/api/export-openapi", json={})

    request_thread = threading.Thread(target=do_export)
    request_thread.start()
    request_thread.join(timeout=0.3)

    assert request_thread.is_alive(), (
        "POST /api/export-openapi should still be blocked on COMPILE_LOCK"
    )

    release_lock.set()
    request_thread.join(timeout=5)
    holder.join(timeout=5)

    assert not request_thread.is_alive()
    assert result["resp"].status_code == 200


def test_health_check_reports_no_compiled_app_before_anything_has_been_compiled(
    monkeypatch, tmp_path
):
    """Before this, GET /api/health returned the exact same static body
    whether or not a notebook had ever been compiled -- a readinessProbe
    pointed at it could only ever confirm the process itself was up, not
    that it actually had a compiled app ready to serve traffic for.
    """

    from backend.routes import upload as upload_module

    empty_dir = tmp_path / "generated_health_empty_test"
    empty_dir.mkdir()

    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(empty_dir))

    resp = client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["compiled_app_present"] is False
    assert body["compiled_at"] is None
    assert body["compiled_version_id"] is None
    assert body["generated_files_modified_since_compile"] is None


def test_health_check_reports_a_compiled_app_and_its_compiled_at_timestamp():

    _compile_a_notebook("health_check_compiled_test.ipynb")

    resp = client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["compiled_app_present"] is True
    assert body["compiled_at"] is not None
    assert body["compiled_version_id"] is None
    assert body["generated_files_modified_since_compile"] is False


def test_health_check_reports_generated_files_modified_since_compile_true_after_a_hand_edit():
    """The output-side integrity check GET /api/generated already
    exposes (see its own "generated_files_modified_since_compile"), now
    reachable from this one liveness/readiness probe too -- a compiled
    app hand-tampered with directly on the server previously kept
    reporting "healthy" here indefinitely, with no way for a Kubernetes
    probe or load balancer polling only this one endpoint to catch it
    short of a second, separate GET /api/generated call.
    """

    _compile_a_notebook("health_check_tampered_test.ipynb")

    (Path(GENERATED_DIR) / "requirements.txt").write_text(
        "fastapi==0.0.0\n", encoding="utf-8"
    )

    resp = client.get("/api/health")

    assert resp.status_code == 200
    assert resp.json()["generated_files_modified_since_compile"] is True


def test_health_check_reports_the_compiled_version_id_for_a_version_pinned_compile():
    """GET /api/notebooks and GET /api/generated already report
    "compiled_version_id" for the currently-compiled entry -- this
    endpoint's own _currently_compiled_notebook_metadata() call already
    resolves the identical value, but discarded it before this, leaving
    a caller polling only this one liveness/readiness endpoint no way to
    tell a version-pinned compile apart from an ordinary one.
    """

    filename = "health_check_version_pinned_test.ipynb"
    original_content = _notebook_bytes("def f() -> int:\n    return 1\n")

    client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(original_content), "application/json")},
    )
    client.post(
        "/api/upload?overwrite=true",
        files={
            "file": (
                filename,
                io.BytesIO(_notebook_bytes("def g() -> int:\n    return 2\n")),
                "application/json",
            )
        },
    )

    version_id = client.get(
        f"/api/notebooks/{filename}/versions"
    ).json()["versions"][0]["version_id"]

    compile_resp = client.post(
        "/api/compile", json={"notebook_path": filename, "version_id": version_id}
    )
    assert compile_resp.status_code == 200

    resp = client.get("/api/health")

    assert resp.status_code == 200
    assert resp.json()["compiled_version_id"] == version_id


def test_health_check_reports_this_tools_own_version():

    resp = client.get("/api/health")

    assert resp.status_code == 200
    assert resp.json()["version"] == NOTEBOOK_TO_API_VERSION


def test_health_check_omits_writable_fields_by_default():

    resp = client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert "upload_dir_writable" not in body
    assert "generated_dir_writable" not in body


def test_health_check_with_check_writable_reports_both_directories_writable():

    resp = client.get("/api/health", params={"check_writable": "true"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["upload_dir_writable"] is True
    assert body["generated_dir_writable"] is True


def test_health_check_check_writable_does_not_leave_a_probe_file_behind():

    resp = client.get("/api/health", params={"check_writable": "true"})

    assert resp.status_code == 200

    assert not any(
        entry.name.startswith(".health_write_probe_")
        for entry in Path(UPLOAD_DIR).iterdir()
    )


def test_health_check_check_writable_reports_true_for_a_generated_dir_that_does_not_exist_yet(
    monkeypatch, tmp_path
):
    """GENERATED_DIR (unlike UPLOAD_DIR) isn't created until the first
    successful compile -- the probe should fall back to the nearest
    existing ancestor rather than failing outright just because nothing
    has been compiled yet.
    """

    from backend.routes import upload as upload_module

    not_yet_created = tmp_path / "generated_health_writable_missing_test"
    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(not_yet_created))

    resp = client.get("/api/health", params={"check_writable": "true"})

    assert resp.status_code == 200
    assert resp.json()["generated_dir_writable"] is True
    assert not not_yet_created.exists()


def test_health_check_check_writable_reports_false_for_an_unwritable_directory(
    monkeypatch, tmp_path
):

    from backend.routes import upload as upload_module

    readonly_dir = tmp_path / "generated_health_readonly_test"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o500)

    monkeypatch.setattr(upload_module, "GENERATED_DIR", str(readonly_dir))

    try:

        resp = client.get("/api/health", params={"check_writable": "true"})

        assert resp.status_code == 200
        assert resp.json()["generated_dir_writable"] is False

    finally:
        readonly_dir.chmod(0o700)


def test_health_check_never_leaks_the_source_notebooks_server_side_filesystem_path():
    """.compile_metadata.json's "source_notebook" field is the source
    notebook's absolute filesystem path on the compiling server -- the
    same field EXCLUDED_GENERATED_FILE_NAMES already keeps out of GET
    /api/download and GET /api/generated/{filename}. A health probe,
    polled by infrastructure outside this tool's own trust boundary, has
    even less business exposing that than an authenticated dashboard
    caller does.
    """

    _compile_a_notebook("health_check_no_leak_test.ipynb")

    resp = client.get("/api/health")

    assert resp.status_code == 200
    assert "source_notebook" not in resp.json()
    assert "uploads" not in json.dumps(resp.json())


def test_export_sdk_waits_for_an_in_flight_compile_to_release_compile_lock():
    """Confirmed missing before this fix: unlike export-openapi, deploy,
    download, and get_generated_file (see the identical tests above),
    export-sdk held COMPILE_LOCK nowhere at all -- a concurrent POST
    /api/compile racing it on another thread runs
    clear_stale_export_artifacts (backend/compiler.py) as part of every
    recompile, which unlinks openapi.json/.yaml and rmtree's the sdk/
    directory. Without the lock, this could read a half-deleted openapi
    export or write its generated client into a sdk/ directory a
    concurrent recompile is simultaneously removing out from under it.
    """

    _compile_a_notebook("export_sdk_lock_test.ipynb")

    export_resp = client.post("/api/export-openapi", json={"format": "json"})
    assert export_resp.status_code == 200

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        with COMPILE_LOCK:
            lock_acquired.set()
            assert release_lock.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_acquired.wait(timeout=5)

    result = {}

    def do_export():
        result["resp"] = client.post("/api/export-sdk", json={"language": "python"})

    request_thread = threading.Thread(target=do_export)
    request_thread.start()
    request_thread.join(timeout=0.3)

    assert request_thread.is_alive(), (
        "POST /api/export-sdk should still be blocked on COMPILE_LOCK"
    )

    release_lock.set()
    request_thread.join(timeout=5)
    holder.join(timeout=5)

    assert not request_thread.is_alive()
    assert result["resp"].status_code == 200


def test_inspect_waits_for_an_in_flight_compile_to_release_compile_lock():
    """Confirmed missing before this fix: unlike export-openapi,
    export-sdk, deploy, download, and get_generated_file (see the
    identical tests above), /api/inspect held COMPILE_LOCK nowhere at
    all, even though its response's "generated_files" field walks
    GENERATED_DIR the exact same way those other routes read it. A
    concurrent POST /api/compile racing it on another thread runs
    clear_stale_export_artifacts (backend/compiler.py) as part of every
    recompile, which rmtree's the sdk/ subdirectory -- without the lock,
    that walk could raise FileNotFoundError if the subdirectory
    disappeared out from under it mid-walk.
    """

    _compile_a_notebook("inspect_lock_test.ipynb")

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        with COMPILE_LOCK:
            lock_acquired.set()
            assert release_lock.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_acquired.wait(timeout=5)

    result = {}

    def do_inspect():
        result["resp"] = client.post(
            "/api/inspect", json={"notebook_path": "inspect_lock_test.ipynb"}
        )

    request_thread = threading.Thread(target=do_inspect)
    request_thread.start()
    request_thread.join(timeout=0.3)

    assert request_thread.is_alive(), (
        "POST /api/inspect should still be blocked on COMPILE_LOCK"
    )

    release_lock.set()
    request_thread.join(timeout=5)
    holder.join(timeout=5)

    assert not request_thread.is_alive()
    assert result["resp"].status_code == 200


def test_compile_response_waits_for_a_release_compile_lock_before_reading_generated_files():
    """Confirmed missing before this fix: compile_notebook (called first,
    inside compile_notebook_endpoint) only holds COMPILE_LOCK for its own
    write phase, releasing it before returning -- the endpoint's
    subsequent inspect_notebook_data call (which builds the
    "dependencies"/"generated_files" fields of the response, see
    test_compile_response_reports_the_dependencies_actually_pinned_in_requirements_txt
    above) read GENERATED_DIR with no lock held at all. A concurrent
    POST /api/compile for a *different* notebook racing in that exact
    window runs clear_stale_export_artifacts as part of its own
    recompile, which rmtree's the sdk/ subdirectory -- the os.walk inside
    _list_generated_files (backend/inspector.py) can raise
    FileNotFoundError if that subdirectory disappears out from under it
    mid-walk. Held externally here, this proves the endpoint's entire
    compile-then-read lifecycle is now serialized against a concurrent
    lock holder, not just its write phase.
    """

    content = _notebook_bytes(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    upload_resp = client.post(
        "/api/upload",
        files={
            "file": (
                "compile_lock_test.ipynb",
                io.BytesIO(content),
                "application/json",
            )
        },
    )
    assert upload_resp.status_code == 200

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        with COMPILE_LOCK:
            lock_acquired.set()
            assert release_lock.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_acquired.wait(timeout=5)

    result = {}

    def do_compile():
        result["resp"] = client.post(
            "/api/compile", json={"notebook_path": "compile_lock_test.ipynb"}
        )

    request_thread = threading.Thread(target=do_compile)
    request_thread.start()
    request_thread.join(timeout=0.3)

    assert request_thread.is_alive(), (
        "POST /api/compile should still be blocked on COMPILE_LOCK"
    )

    release_lock.set()
    request_thread.join(timeout=5)
    holder.join(timeout=5)

    assert not request_thread.is_alive()
    assert result["resp"].status_code == 200
    assert "generated_files" in result["resp"].json()


def test_export_openapi_read_back_does_not_race_a_concurrent_recompiles_cleanup():
    """Confirmed exploitable before this fix: export_openapi_endpoint
    released COMPILE_LOCK right after export_openapi_schema wrote
    output_path, then read that same file back completely unlocked. A
    concurrent POST /api/compile racing in during that exact window runs
    clear_stale_export_artifacts (backend/compiler.py) as part of its own
    recompile, which unlinks openapi.json/.yaml unconditionally --
    reproduced directly against clear_stale_export_artifacts: a file
    written successfully one moment raised a bare FileNotFoundError on
    the very next read, immediately after. The write and the read-back
    are now both inside the same COMPILE_LOCK section, so a thread
    racing to acquire that same lock (simulating clear_stale_export_
    artifacts, which itself only ever runs from inside a compile that
    already holds COMPILE_LOCK -- see compile_notebook_to_api) can't run
    between them anymore -- it can only run before the write starts or
    after the read has already finished.
    """

    _compile_a_notebook("export_openapi_read_race_test.ipynb")

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        with COMPILE_LOCK:
            lock_acquired.set()
            assert release_lock.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_acquired.wait(timeout=5)

    result = {}

    def do_export():
        result["resp"] = client.post("/api/export-openapi", json={"format": "json"})

    request_thread = threading.Thread(target=do_export)
    request_thread.start()
    request_thread.join(timeout=0.3)

    assert request_thread.is_alive(), (
        "POST /api/export-openapi should still be blocked on COMPILE_LOCK "
        "for its whole write-then-read-back sequence"
    )

    release_lock.set()
    request_thread.join(timeout=5)
    holder.join(timeout=5)

    assert not request_thread.is_alive()
    assert result["resp"].status_code == 200
    assert "paths" in result["resp"].json()["schema"]


def test_get_config_reports_the_configured_limits():

    resp = client.get("/api/config")

    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "success"
    assert isinstance(body["max_upload_bytes"], int)
    assert isinstance(body["max_batch_upload_files"], int)
    assert isinstance(body["max_notebook_versions"], int)
    assert isinstance(body["max_tag_length"], int)
    assert isinstance(body["max_tags_per_notebook"], int)
    assert isinstance(body["max_deploy_history_entries"], int)
    assert isinstance(body["max_compile_history_entries"], int)
    assert isinstance(body["deploy_subprocess_timeout_seconds"], int)
    assert isinstance(body["max_source_url_length"], int)
    assert isinstance(body["max_search_regex_length"], int)


def test_get_config_reports_the_max_source_url_length():

    from backend.routes.upload import _MAX_SOURCE_URL_LENGTH

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["max_source_url_length"] == _MAX_SOURCE_URL_LENGTH


def test_get_config_reports_the_max_search_regex_length():

    from backend.routes.upload import MAX_SEARCH_REGEX_LENGTH

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["max_search_regex_length"] == MAX_SEARCH_REGEX_LENGTH


def test_get_config_reflects_a_configured_max_search_regex_length(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_SEARCH_REGEX_LENGTH", 42)

    resp = client.get("/api/config")

    assert resp.json()["max_search_regex_length"] == 42


def test_get_config_reports_the_allowed_origins():

    from backend.dashboard import DEFAULT_ALLOWED_ORIGINS

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["allowed_origins"] == DEFAULT_ALLOWED_ORIGINS


def test_get_config_reflects_a_configured_allowed_origins(monkeypatch):

    monkeypatch.setenv(
        "NOTEBOOK_API_ALLOWED_ORIGINS", "https://a.example.com,https://b.example.com"
    )

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["allowed_origins"] == [
        "https://a.example.com", "https://b.example.com",
    ]


def test_get_config_reports_a_null_dashboard_rate_limit_when_disabled(monkeypatch):

    monkeypatch.delenv("NOTEBOOK_API_DASHBOARD_RATE_LIMIT_PER_MINUTE", raising=False)

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["dashboard_rate_limit_per_minute"] is None


def test_get_config_reflects_a_configured_dashboard_rate_limit(monkeypatch):

    monkeypatch.setenv("NOTEBOOK_API_DASHBOARD_RATE_LIMIT_PER_MINUTE", "90")

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["dashboard_rate_limit_per_minute"] == 90


def test_get_config_reports_the_compiling_python_version():

    resp = client.get("/api/config")

    assert resp.status_code == 200
    body = resp.json()

    assert body["compiling_python_version"] == compiling_python_version()


def test_get_config_reflects_a_configured_max_upload_bytes(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_UPLOAD_BYTES", 12345)

    resp = client.get("/api/config")

    assert resp.json()["max_upload_bytes"] == 12345


def test_get_config_reflects_a_configured_max_batch_upload_files(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_BATCH_UPLOAD_FILES", 7)

    resp = client.get("/api/config")

    assert resp.json()["max_batch_upload_files"] == 7


def test_get_config_reflects_a_configured_max_deploy_history_entries(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_DEPLOY_HISTORY_ENTRIES", 9)

    resp = client.get("/api/config")

    assert resp.json()["max_deploy_history_entries"] == 9


def test_get_config_reports_the_url_import_timeout():

    from backend.routes.upload import URL_IMPORT_TIMEOUT_SECONDS

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["url_import_timeout_seconds"] == URL_IMPORT_TIMEOUT_SECONDS


def test_get_config_reflects_a_configured_url_import_timeout(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "URL_IMPORT_TIMEOUT_SECONDS", 12.5)

    resp = client.get("/api/config")

    assert resp.json()["url_import_timeout_seconds"] == 12.5


def test_get_config_reports_the_stale_upload_temp_file_seconds():

    from backend.routes.upload import STALE_UPLOAD_TEMP_FILE_SECONDS

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["stale_upload_temp_file_seconds"] == STALE_UPLOAD_TEMP_FILE_SECONDS


def test_get_config_reflects_a_configured_stale_upload_temp_file_seconds(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "STALE_UPLOAD_TEMP_FILE_SECONDS", 42)

    resp = client.get("/api/config")

    assert resp.json()["stale_upload_temp_file_seconds"] == 42


def test_get_config_reflects_a_configured_max_compile_history_entries(monkeypatch):

    from backend.routes import upload as upload_module

    monkeypatch.setattr(upload_module, "MAX_COMPILE_HISTORY_ENTRIES", 11)

    resp = client.get("/api/config")

    assert resp.json()["max_compile_history_entries"] == 11


def test_get_config_reports_notebook_sort_keys_and_orders_matching_list_notebooks():
    """GET /api/notebooks' own "sort"/"order" query parameters accept
    exactly _NOTEBOOK_SORT_KEYS/_NOTEBOOK_SORT_ORDERS -- this must report
    the same values, not a second, independently-drifting copy of them.
    """

    from backend.routes.upload import _NOTEBOOK_SORT_KEYS, _NOTEBOOK_SORT_ORDERS

    resp = client.get("/api/config")
    body = resp.json()

    assert body["notebook_sort_keys"] == sorted(_NOTEBOOK_SORT_KEYS)
    assert body["notebook_sort_orders"] == sorted(_NOTEBOOK_SORT_ORDERS)

    for sort_key in body["notebook_sort_keys"]:
        assert client.get(f"/api/notebooks?sort={sort_key}").status_code == 200

    for order in body["notebook_sort_orders"]:
        assert client.get(f"/api/notebooks?order={order}").status_code == 200


def test_get_config_never_leaks_the_upload_or_generated_directory_path():
    """UPLOAD_DIR/GENERATED_DIR are filesystem paths on the compiling
    server -- the same category of information GET /api/health's own
    docstring already explains has no business leaking out of a
    dashboard API response.
    """

    resp = client.get("/api/config")

    assert resp.status_code == 200
    body_text = json.dumps(resp.json())

    assert "upload_dir" not in resp.json()
    assert "generated_dir" not in resp.json()
    assert UPLOAD_DIR not in body_text
