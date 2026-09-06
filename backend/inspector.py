import difflib
import json
import os
from collections import Counter
from pathlib import Path

from backend.compiler import (
    COMPILE_METADATA_FILENAME,
    _extract_excluded_imports,
    _extract_private_function_names,
    _filter_functions_by_name,
    distribution_name_for_import,
    STANDARD_LIBS,
)

from backend.parser.notebook_parser import (
    load_notebook,
    extract_code_cells,
)

from backend.parser.ast_parser import (
    extract_functions_from_code,
    extract_imports_from_code,
    extract_skipped_functions_from_code,
    deduplicate_functions_by_name,
)

from backend.generator.api_generator import (
    LONG_RUNNING_KEYWORDS,
    RESERVED_INFRASTRUCTURE_NAMES,
)

# Directory names to exclude when walking an output directory for
# generated_files (below) and, via the same set, GET /api/download's zip
# archive (see backend/routes/upload.py). __pycache__ is created by Python
# itself the first time the compiled app or its runtime module gets
# imported (e.g. by `serve`'s uvicorn subprocess, `export-openapi`, or a
# test suite) -- it is not part of what compile_notebook_to_api actually
# wrote, and its .pyc filenames are tied to the interpreter that happened
# to import it (e.g. "app.cpython-314.pyc"), so they vary across machines
# and Python versions for the exact same compiled output. Left unfiltered,
# this polluted both `inspect`'s generated-files listing and the
# downloadable app bundle with a non-deliverable, non-deterministic
# implementation artifact that has nothing to do with what was actually
# compiled (confirmed against this repo's own generated/ directory, which
# already had a __pycache__ picked up by both).
EXCLUDED_GENERATED_DIR_NAMES = frozenset({"__pycache__"})

# File names to exclude when walking an output directory for generated_files
# (below) and, via the same set, GET /api/download's zip archive and GET
# /api/generated/{filename}'s preview (see backend/routes/upload.py), and
# the generated Dockerfile's .dockerignore (see
# generator/docker_generator.py, which excludes this same filename by its
# own literal copy of it -- it can't import COMPILE_METADATA_FILENAME
# without introducing a circular import, since compiler.py already imports
# docker_generator.py).
#
# .compile_metadata.json (write_compile_metadata, backend/compiler.py) is
# pure dashboard-internal bookkeeping -- read only by
# list_notebooks/_currently_compiled_notebook_metadata in routes/upload.py
# to answer "which uploaded notebook produced what's currently compiled" --
# never by the compiled app itself at runtime, exactly like the
# openapi.json/openapi.yaml/sdk/ exports .dockerignore already excludes for
# that same reason (see generate_dockerignore's own docstring). Left
# unfiltered here, it was previously baked into every `deploy`/`docker
# build`, downloadable via GET /api/download, and readable via GET
# /api/generated/.compile_metadata.json -- leaking the source notebook's
# *absolute filesystem path on the compiling server* (its "source_notebook"
# field) into a deployable image and a client-facing API response, neither
# of which has any business exposing server-side filesystem layout.
EXCLUDED_GENERATED_FILE_NAMES = frozenset({COMPILE_METADATA_FILENAME})


def _list_generated_files(output_dir):
    """Files under `output_dir`, relative to it, excluding
    EXCLUDED_GENERATED_DIR_NAMES subtrees and EXCLUDED_GENERATED_FILE_NAMES
    files. Shared by inspect_notebook and inspect_notebook_data so their
    "Generated Files" listings can't drift from each other.
    """
    generated_path = Path(output_dir)

    generated_files = []

    if generated_path.is_dir():

        for root, dirs, files in os.walk(generated_path):

            dirs[:] = [
                d for d in dirs if d not in EXCLUDED_GENERATED_DIR_NAMES
            ]

            for f in files:

                if f in EXCLUDED_GENERATED_FILE_NAMES:
                    continue

                rel = (
                    Path(root)
                    .relative_to(generated_path)
                    / f
                )

                generated_files.append(str(rel))

    return generated_files


def list_generated_files(output_dir):
    """Public, sorted wrapper around _list_generated_files.

    Used directly by GET /api/generated (backend/routes/upload.py) to list
    what's currently sitting in an output directory without needing a
    notebook_path to inspect alongside it -- unlike inspect_notebook and
    inspect_notebook_data above, both of which require a notebook to
    inspect and only report generated_files as one field of that larger
    report. GET /api/generated has no notebook to parse at all (the whole
    point is listing what's compiled even after its source notebook is
    gone -- see DELETE /api/notebooks/{filename}'s "was_currently_compiled"
    flag), so it calls straight through to this instead.
    """
    return sorted(_list_generated_files(output_dir))


def _third_party_dependencies(all_imports):
    """`all_imports` (raw import names collected via
    extract_imports_from_code) filtered down to the ones that will
    actually end up pinned in requirements.txt by write_requirements
    (backend/compiler.py), and resolved to the exact same distribution
    names write_requirements itself pins -- both apply this identical
    STANDARD_LIBS filter and distribution_name_for_import resolution to
    what write_requirements writes, so this can't drift from it.

    Before the STANDARD_LIBS filter, inspect_notebook/inspect_notebook_data's
    "dependencies" listed every import a notebook made, standard-library
    ones included (e.g. "os", "json", "sys") -- confirmed misleading: a
    notebook doing `import os` and `import pandas` had both `inspect` and,
    after an actual compile, `compile`/`serve`/`deploy`'s own
    print_compile_summary (below) report "Dependencies: json, os, pandas",
    even though requirements.txt -- and from there the Docker image
    `deploy`/`docker build` would actually ship -- only ever contained
    "pandas".

    Before the distribution_name_for_import resolution, added later, the
    identical drift reopened for a different reason: a notebook's `import`
    statement names a *module*, not necessarily the PyPI *distribution*
    that provides it (`import multipart` is provided by "python-multipart",
    `import cv2` by "opencv-python", ...) -- write_requirements was fixed
    to resolve and pin the real distribution name, but this function kept
    returning the raw import name unchanged. Confirmed: compiling a
    notebook importing "multipart" wrote "python-multipart==<version>" to
    requirements.txt, while `compile --json`'s own "dependencies" field
    (built from this function) still reported "multipart" -- a name that
    appears nowhere in the requirements.txt that same compile just
    produced, and isn't installable via pip on its own. A notebook author
    reading either "preview what compiling will do" or "here's what this
    compile just produced" had no way to tell which of their imports
    would actually be installed, or under what name, without separately
    reading requirements.txt themselves.
    """
    return sorted({
        distribution_name_for_import(imp)
        for imp in all_imports
        if imp not in STANDARD_LIBS
    })


def _is_background_function(name):
    """Whether generate_fastapi_code (generator/api_generator.py) will
    compile `name` into a background/task_id-based endpoint rather than a
    synchronous one, per LONG_RUNNING_KEYWORDS.

    Shared by every surface that classifies a function this way --
    inspect_notebook, _endpoint_metadata below, and print_compile_summary
    -- so they can't drift from each other or from POST /api/compile's own
    identical check in routes/upload.py.
    """
    return any(kw in name.lower() for kw in LONG_RUNNING_KEYWORDS)


def _endpoint_metadata(functions):
    """The {"path", "method", "is_async"} shape POST /api/compile's
    "endpoints" field already returns (see routes/upload.py), computed
    here from a notebook that hasn't been compiled yet.

    Before this, whether a function would become a background/task_id
    endpoint or a synchronous one -- a real difference in what the
    generated app does, not just a formatting detail -- was only ever
    visible *after* compiling. inspect_notebook_data already exists
    specifically to preview compile-time outcomes (see
    reserved_name_conflicts above, added for the same reason), so it had
    the same gap for this classification that reserved_name_conflicts
    closed for name collisions.
    """
    return [
        {
            "path": f"/{func['name']}",
            "method": "POST",
            "is_async": _is_background_function(func["name"]),
        }
        for func in functions
    ]


def _reserved_name_conflicts(functions):
    """Names in `functions` that generate_fastapi_code will refuse to
    compile (see ReservedFunctionNameError in generator/api_generator.py),
    because they collide with an identifier the generated app itself
    defines (auth, task management, or infrastructure routes).

    Before this, /api/inspect and the `inspect` CLI command -- the tool's
    own "preview what compiling this notebook will do" step -- had no
    idea this check existed. A notebook with a function named
    "health_check" or "verify_api_key" inspected cleanly with no warning
    at all, and the very first time its author learned about the
    conflict was a compile failure (or, from the dashboard, a bare 500)
    at the *next* step, for a problem the tool already had the exact
    answer to.
    """
    return sorted(
        {func["name"] for func in functions} & RESERVED_INFRASTRUCTURE_NAMES
    )


def _duplicate_function_names(functions):
    """Names `functions` (the raw, pre-deduplicate_functions_by_name list
    extracted straight from every code cell) defines more than once.

    Notebooks are edited iteratively -- a cell defining `def add(...)` is
    commonly re-run later with a fixed/changed body under the same name
    -- and deduplicate_functions_by_name (backend/parser/ast_parser.py)
    already handles that correctly by keeping only the last definition,
    matching what running the whole notebook top to bottom in a single
    kernel would actually do. But that resolution happened silently: a
    notebook author who *didn't* intend a redefinition (a copy-pasted
    cell, a typo'd name reused by accident) got no signal at all that one
    of their functions had simply vanished, silently replaced by an
    unrelated later definition sharing its name -- inspect_notebook_data's
    "functions" (and "endpoints") already only ever reflected the
    winning definition, with nothing anywhere naming what got shadowed.

    Computed from `functions` *before* deduplication -- a name appearing
    exactly once is never "duplicate" regardless of how many other names
    exist alongside it, and a name appearing three or more times is still
    reported once, not once per extra repetition.
    """
    counts = Counter(func["name"] for func in functions)

    return sorted(name for name, count in counts.items() if count > 1)


def _functions_without_docstrings(functions):
    """Names in `functions` (the final, post-deduplication/post-private-
    filter list -- the same set inspect_notebook_data's own "functions"
    field already reports, each of which becomes a real compiled
    endpoint) that have no docstring at all, or one that's empty/all-
    whitespace (ast.get_docstring(clean=True) already normalizes either
    down to a falsy "docstring" field -- see extract_functions_from_code,
    backend/parser/ast_parser.py).

    generate_fastapi_code (backend/generator/api_generator.py) already
    falls back to a generic, auto-generated OpenAPI "description" (e.g.
    "Auto-generated endpoint for train_model. Operation ID: ...") for
    exactly this case -- functional, but far less useful to anyone
    reading /docs, or a client generated from openapi.json, than the
    description a notebook author would have written themselves. Before
    this, neither `inspect` nor `compile`'s own summary said anything
    about which endpoints would get that fallback instead of real
    documentation -- a notebook author only ever discovered it by
    opening the compiled app's own /docs (or openapi.json) after the
    fact, the exact "silently missing signal" pattern
    "private_functions"/"excluded_imports" already close for their own
    respective directives.

    A private function (already filtered out of `functions` by the time
    this is ever called -- see inspect_notebook_data/inspect_notebook
    below) never becomes an endpoint at all, so its own docstring (or
    lack of one) is irrelevant here, unlike every function actually
    reported.
    """
    return sorted(
        func["name"] for func in functions if not func.get("docstring")
    )


def _aggregate_skipped_functions(code_cells, exposed_function_names):
    """Skipped-function reports (see extract_skipped_functions_from_code)
    across every cell in `code_cells`, deduplicated by name and with any
    name that ultimately made it into `exposed_function_names` filtered
    back out.

    That last part matters because deduplicate_functions_by_name lets a
    later cell's clean redefinition of a name override an earlier cell's
    unsupported one (e.g. a `**kwargs` version fixed up and re-run under
    the same name) -- exactly what running the whole notebook top to
    bottom in a single kernel would do. Without this filter, a name that
    ended up perfectly fine would still be reported as skipped because of
    what an earlier, superseded cell once looked like.
    """
    skipped_by_name = {}

    for cell in code_cells:
        for skipped in extract_skipped_functions_from_code(cell):
            skipped_by_name[skipped["name"]] = skipped

    return [
        skipped
        for name, skipped in sorted(skipped_by_name.items())
        if name not in exposed_function_names
    ]


def inspect_notebook(notebook_path, output_dir="generated"):
    """
    Print a detailed analysis report for a notebook.
    """

    notebook = load_notebook(notebook_path)

    code_cells = extract_code_cells(notebook)

    all_functions = []
    all_imports = set()

    for cell in code_cells:

        funcs = extract_functions_from_code(cell)
        imports = extract_imports_from_code(cell)

        all_functions.extend(funcs)
        all_imports.update(imports)

    duplicate_function_names = _duplicate_function_names(all_functions)

    all_functions = deduplicate_functions_by_name(all_functions)

    # Same "# notebook-to-api: private" directive compile_notebook_to_api
    # (backend/compiler.py, via _drop_private_functions) already applies
    # before generating an app.py -- without this, a function the
    # notebook itself declares should never become an endpoint would
    # still be reported here as if compiling would expose it.
    private_function_names = _extract_private_function_names(code_cells)

    if private_function_names:
        all_functions = [
            func for func in all_functions
            if func["name"] not in private_function_names
        ]

    # Same "# notebook-to-api: exclude <import-name>" directive
    # extract_third_party_imports (backend/compiler.py) already applies
    # before write_requirements pins anything -- without this, an import
    # a notebook author explicitly opted out of requirements.txt would
    # still show up here as a "dependency", even though it's never
    # actually pinned by an ensuing compile.
    excluded_imports = _extract_excluded_imports(code_cells)
    all_imports -= excluded_imports

    reserved_name_conflicts = _reserved_name_conflicts(all_functions)

    skipped_functions = _aggregate_skipped_functions(
        code_cells, {func["name"] for func in all_functions}
    )

    print("\nNotebook Analysis Report")
    print("=" * 30)

    if reserved_name_conflicts:
        print("\n⚠ Reserved Name Conflicts (compilation will fail):")
        print("-" * 20)
        for name in reserved_name_conflicts:
            print(f"- {name}")

    if duplicate_function_names:
        print("\n⚠ Duplicate Functions (redefined; only the last definition is compiled):")
        print("-" * 20)
        for name in duplicate_function_names:
            print(f"- {name}")

    if private_function_names:
        print("\nPrivate Functions (never exposed as an endpoint):")
        print("-" * 20)
        for name in sorted(private_function_names):
            print(f"- {name}")

    if skipped_functions:
        print("\n⚠ Skipped Functions (no endpoint will be generated):")
        print("-" * 20)
        for skipped in skipped_functions:
            print(f"- {skipped['name']}: {skipped['reason']}")

    print("\nFunctions Found:")
    print("-" * 20)

    for idx, func in enumerate(all_functions, start=1):

        args_str = []

        for arg in func.get("args", []):

            if arg.get("type"):
                args_str.append(
                    f"{arg['name']}: {arg['type']}"
                )
            else:
                args_str.append(arg["name"])

        args_formatted = ", ".join(args_str)

        ret = (
            f" -> {func['return_type']}"
            if func.get("return_type")
            else ""
        )

        print(
            f"\n{idx}. {func['name']}({args_formatted}){ret}"
        )

        route_suffix = (
            "  [background]" if _is_background_function(func["name"]) else ""
        )

        print(
            f"   Route: POST /{func['name']}{route_suffix}"
        )

        # A notebook function's own docstring already becomes its
        # compiled endpoint's OpenAPI description (see api_generator.py)
        # -- but `inspect` is this tool's own "preview what compiling
        # this notebook will do" report, and never showed it at all, even
        # though inspect_notebook_data's "functions" field (and `inspect
        # --json`) already carries it. A notebook author checking
        # `inspect` to confirm their documentation looks right before
        # compiling had no way to see it here.
        docstring = func.get("docstring")

        if docstring:
            for line in docstring.split("\n"):
                print(f"   {line}" if line else "")

        print(
            f"   Example Payload: {func.get('example_payload', {})}"
        )

        print(
            f"   Example Response: {func.get('example_response', {})}"
        )

    if excluded_imports:
        print("\nExcluded Imports (opted out of requirements.txt):")
        print("-" * 20)
        for name in sorted(excluded_imports):
            print(f"- {name}")

    functions_without_docstrings = _functions_without_docstrings(all_functions)

    if functions_without_docstrings:
        print("\nFunctions Without Docstrings (will get a generic OpenAPI description):")
        print("-" * 20)
        for name in functions_without_docstrings:
            print(f"- {name}")

    print("\nDependencies:")
    print("-" * 20)

    for dep in _third_party_dependencies(all_imports):
        print(f"- {dep}")

    generated_files = _list_generated_files(output_dir)

    print("\nGenerated Files:")
    print("-" * 20)

    for gf in sorted(generated_files):
        print(f"- {gf}")


def inspect_notebook_data(
    notebook_path,
    output_dir="generated"
):
    """
    Return notebook metadata as JSON-serializable data.
    Perfect for FastAPI endpoints and frontend dashboards.
    """

    notebook = load_notebook(notebook_path)

    code_cells = extract_code_cells(notebook)

    all_functions = []
    all_imports = set()

    for cell in code_cells:

        funcs = extract_functions_from_code(cell)
        imports = extract_imports_from_code(cell)

        all_functions.extend(funcs)
        all_imports.update(imports)

    duplicate_function_names = _duplicate_function_names(all_functions)

    all_functions = deduplicate_functions_by_name(all_functions)

    # See the identical comment in inspect_notebook above -- keeps
    # "functions"/"endpoints"/"reserved_name_conflicts" below (POST
    # /api/inspect, POST /api/validate, POST /api/compile's own response,
    # print_compile_summary, and generate_curl_commands below, which all
    # read them through this one function) from drifting out of sync
    # with what compile_notebook_to_api (backend/compiler.py, via
    # _drop_private_functions) actually exposes.
    private_function_names = _extract_private_function_names(code_cells)

    if private_function_names:
        all_functions = [
            func for func in all_functions
            if func["name"] not in private_function_names
        ]

    # See the identical comment in inspect_notebook above -- keeps this
    # "dependencies" field (POST /api/inspect, POST /api/validate, POST
    # /api/compile's own response, and print_compile_summary below, which
    # all read it through this one function) from drifting out of sync
    # with what write_requirements/extract_third_party_imports
    # (backend/compiler.py) actually excludes.
    excluded_imports = _extract_excluded_imports(code_cells)
    all_imports -= excluded_imports

    return {
        "functions": all_functions,
        "dependencies": _third_party_dependencies(all_imports),
        "generated_files": list_generated_files(output_dir),
        "reserved_name_conflicts": _reserved_name_conflicts(all_functions),
        "endpoints": _endpoint_metadata(all_functions),
        "skipped_functions": _aggregate_skipped_functions(
            code_cells, {func["name"] for func in all_functions}
        ),
        "private_functions": sorted(private_function_names),
        # The complementary "# notebook-to-api: exclude <import-name>"
        # directive's own effect, surfaced the same way "private_functions"
        # already surfaces "# notebook-to-api: private"'s -- before this,
        # an import a notebook author explicitly opted out of
        # requirements.txt was silently absorbed into the subtraction
        # above, with "dependencies" giving no way to tell "never
        # imported" apart from "imported, but deliberately excluded".
        "excluded_imports": sorted(excluded_imports),
        # Names defined more than once in the notebook's own raw
        # extraction (before deduplicate_functions_by_name, backend/
        # parser/ast_parser.py, silently collapses each down to its last
        # definition) -- see _duplicate_function_names' own docstring for
        # exactly what this catches that "functions"/"endpoints" alone
        # can't: an accidental redefinition that shadowed an earlier
        # function of the same name with no signal anywhere that it
        # happened.
        "duplicate_functions": duplicate_function_names,
        # See _functions_without_docstrings' own docstring -- computed
        # from `all_functions` here specifically (post-dedup, post-
        # private-filter), the exact same final set this same return's
        # own "functions"/"endpoints" already report, so this can never
        # name a function that isn't actually a real compiled endpoint.
        "functions_without_docstrings": _functions_without_docstrings(all_functions),
    }


def print_compile_summary(notebook_path, output_dir="generated", only=None, exclude=None):
    """Print what compiling `notebook_path` into `output_dir` actually
    produced: its endpoints (flagging background/task_id-based ones the
    same way POST /api/compile's "endpoints" field does) and third-party
    dependencies.

    Shared by the CLI's `compile` command and `serve`'s initial compile
    and every hot-recompile it triggers. Before this existed for `serve`,
    a live dev session gave no feedback at all about what a recompile had
    actually changed -- just "Recompilation complete." -- even though a
    fast, informative feedback loop after every save is the entire point
    of running a live server in the first place. `compile` had the same
    gap until this was first added there and later reused here.

    `only`/`exclude`, when given, restrict which functions are reported
    here to whichever ones _filter_functions_by_name (backend/compiler.py)
    actually left compiled -- inspect_notebook_data below re-parses the
    notebook fresh and has no idea a compile that just ran was restricted
    at all, so without this, `compile --only add` would still report an
    endpoint for every *other* function the notebook defines, claiming
    endpoints exist for functions the compiled app doesn't actually have.
    """
    data = inspect_notebook_data(
        notebook_path=notebook_path, output_dir=str(output_dir)
    )

    functions = data["functions"]

    if only or exclude:
        functions = _filter_functions_by_name(functions, only, exclude)

    print(f"\nGenerated {len(functions)} endpoint(s):")

    for func in functions:
        name = func["name"]
        suffix = "  [background]" if _is_background_function(name) else ""
        print(f"  POST /{name}{suffix}")

    if data["dependencies"]:
        print(f"\nDependencies: {', '.join(data['dependencies'])}")

    if data["skipped_functions"]:
        print(
            f"\nSkipped {len(data['skipped_functions'])} function(s) "
            "(no endpoint generated):"
        )
        for skipped in data["skipped_functions"]:
            print(f"  {skipped['name']}: {skipped['reason']}")

    if data["private_functions"]:
        print(
            f"\nPrivate {len(data['private_functions'])} function(s) "
            "(never exposed as an endpoint):"
        )
        for name in data["private_functions"]:
            print(f"  {name}")

    if data["excluded_imports"]:
        print(
            f"\nExcluded {len(data['excluded_imports'])} import(s) "
            "(opted out of requirements.txt):"
        )
        for name in data["excluded_imports"]:
            print(f"  {name}")

    if data["duplicate_functions"]:
        print(
            f"\n⚠ {len(data['duplicate_functions'])} duplicate function(s) "
            "(redefined; only the last definition is compiled):"
        )
        for name in data["duplicate_functions"]:
            print(f"  {name}")

    if data["functions_without_docstrings"]:
        print(
            f"\n{len(data['functions_without_docstrings'])} function(s) "
            "without a docstring (will get a generic OpenAPI description):"
        )
        for name in data["functions_without_docstrings"]:
            print(f"  {name}")


def _extract_notebook_functions(notebook_path):
    """The same extract-every-cell/deduplicate pipeline inspect_notebook
    and inspect_notebook_data (above) already run, on their way to their
    own "functions" field -- factored out here for diff_notebook_functions
    below, which only ever needs this half of what those two compute (no
    dependencies, generated_files, or reserved_name_conflicts to diff two
    notebooks' function signatures against each other).
    """
    notebook = load_notebook(notebook_path)

    code_cells = extract_code_cells(notebook)

    all_functions = []

    for cell in code_cells:
        all_functions.extend(extract_functions_from_code(cell))

    return deduplicate_functions_by_name(all_functions)


def _function_signature_key(func):
    """Comparable signature summary for a function dict (as produced by
    extract_functions_from_code) -- used by diff_notebook_functions below
    to decide whether a function present in both notebooks actually
    changed, or is merely present in both untouched.

    Deliberately excludes "docstring", "example_payload", and
    "example_response": those affect the endpoint's *documentation*, not
    its actual request/response contract, so an edit to just a
    function's docstring shouldn't be reported as a breaking change to
    its signature.

    Each arg's own "description" (extract_functions_from_code's own
    per-parameter docstring extraction -- see
    _parse_docstring_arg_descriptions, backend/parser/ast_parser.py) is
    excluded the identical way and for the identical reason: it's
    derived entirely from the function's own docstring, not from
    anything that changes what a caller must actually send. Confirmed
    exploitable before this: rewording just one parameter's own Args:
    entry (with every type/default/kind byte-for-byte identical)
    produced a different `args` list -- since each arg dict now carries
    that description inline -- which this function's own tuple
    comparison then reported as a genuine signature change, the exact
    "docstring edit reported as a breaking change" bug this function
    otherwise already exists to prevent for the whole-function case.
    """
    args_without_docs = tuple(
        tuple(
            (key, value) for key, value in sorted(arg.items())
            if key != "description"
        )
        for arg in func.get("args", [])
    )

    return (
        args_without_docs,
        func.get("return_type"),
        func.get("is_async", False),
    )


def diff_notebook_functions(old_notebook_path, new_notebook_path):
    """Compare the top-level functions two notebooks would each compile
    into endpoints, without actually compiling either one.

    Before this, the only way to tell whether editing a notebook changed
    its compiled API's surface -- a new endpoint, a removed one, or an
    existing one's request/response contract changing in a way that could
    break an existing caller -- was to compile both versions and diff
    app.py by eye. Generated code changes shape for plenty of reasons that
    have nothing to do with the actual API contract (formatting, internal
    ordering, ...), making a genuine "did this endpoint's request shape
    change" question much harder to answer than it needs to be. This
    compares just the extracted function signatures instead -- the same
    per-function shape inspect_notebook_data's own "functions" field
    already exposes for a single notebook -- so a CI check (or a
    developer reviewing a notebook change before merging or deploying it)
    can answer that directly, against two arbitrary notebook paths (an
    old revision checked out at another path, a previous upload's
    downloaded copy, ...), with no compile step involved at all.

    Returns {"added", "removed", "changed", "unchanged"}:
      - "added": full function entries present only in new_notebook_path.
      - "removed": full function entries present only in
        old_notebook_path.
      - "changed": functions present in both, sorted by name, each as
        {"name", "old", "new"} (that function's own full entry from each
        notebook) wherever its args, return type, or sync/async-ness
        differ (see _function_signature_key) -- a docstring-only edit
        does not count as "changed".
      - "unchanged": names present in both with an identical signature.
    """
    old_functions = {
        func["name"]: func
        for func in _extract_notebook_functions(old_notebook_path)
    }
    new_functions = {
        func["name"]: func
        for func in _extract_notebook_functions(new_notebook_path)
    }

    old_names = set(old_functions)
    new_names = set(new_functions)

    added_names = sorted(new_names - old_names)
    removed_names = sorted(old_names - new_names)

    changed = []
    unchanged = []

    for name in sorted(old_names & new_names):

        old_func = old_functions[name]
        new_func = new_functions[name]

        if _function_signature_key(old_func) != _function_signature_key(new_func):
            changed.append({"name": name, "old": old_func, "new": new_func})
        else:
            unchanged.append(name)

    return {
        "added": [new_functions[name] for name in added_names],
        "removed": [old_functions[name] for name in removed_names],
        "changed": changed,
        "unchanged": unchanged,
    }


def classify_notebook_diff(diff):
    """Classify a diff_notebook_functions report by whether it would
    break an existing caller of the compiled API, on top of the raw
    "added"/"removed"/"changed"/"unchanged" report itself.

    Before this, every diff endpoint/command (GET /api/notebooks/diff,
    GET /api/notebooks/{filename}/versions/{version_id}/diff, `diff`,
    `remote-diff`, `diff-notebooks`, `versions diff`, `versions compare`)
    already answered "what changed", but a caller wanting to answer "is
    this change safe to deploy" had to interpret "removed"/"changed"
    itself -- `diff --json`'s own help text even suggests doing exactly
    that by hand ("failing a CI check when 'removed' or 'changed' is
    non-empty"), even though "changed" also includes purely additive,
    non-breaking edits (e.g. a new parameter with a default) that never
    should fail such a check.

    Returns {"compatible", "breaking_changes"}, meant to be merged into
    diff's own dict (`diff.update(classify_notebook_diff(diff))`), not
    replace it -- "added"/"removed"/"changed"/"unchanged" remain the
    source of truth for what changed; this only adds a verdict on top,
    the same "extra fields alongside the existing report" approach GET
    /api/notebooks/diff's own "content_diff" already takes.

    A function present in "removed" is always breaking: an existing
    caller's request to that endpoint now 404s.

    A function present in "changed" is broken down parameter-by-parameter
    (matched by name, not position -- the generated endpoint's request
    body is a Pydantic model keyed by parameter name, so reordering two
    parameters, or moving one from positional to keyword-only, changes
    neither the JSON shape callers send nor what this reports):
      - a parameter present in the old signature but missing from the new
        one ("removed_parameter") -- an existing caller's request, valid
        before, now includes a field the endpoint no longer accepts.
      - a parameter that had a default and no longer does
        ("parameter_became_required") -- an existing caller's request that
        omitted it, valid before, is now missing a required field.
      - a new parameter with no default ("required_parameter_added") --
        the same "existing request now misses a required field" break,
        just for a parameter that didn't exist at all before.
      - a parameter whose type annotation changed on both sides
        ("parameter_type_changed") -- a value valid for the old type may
        no longer validate against the new one.
      - a changed return type ("return_type_changed") -- an existing
        caller's own response parsing may no longer match what the
        endpoint actually returns.

    Deliberately NOT breaking, and so never added to "breaking_changes":
      - a function present in "added": a new endpoint no existing caller
        could already have been depending on.
      - a new parameter that has a default: an existing caller's request,
        missing that field entirely, still validates exactly as before.
      - a "changed" entry whose only difference is sync/async-ness
        (already reported under "changed" by diff_notebook_functions --
        see _function_signature_key -- but is an internal implementation
        detail invisible to an HTTP caller).
    """
    breaking_changes = []

    for func in diff["removed"]:
        breaking_changes.append({
            "type": "removed_endpoint",
            "name": func["name"],
            "detail": f"Endpoint '{func['name']}' was removed.",
        })

    for entry in diff["changed"]:

        name = entry["name"]
        old_args = {arg["name"]: arg for arg in entry["old"].get("args", [])}
        new_args = {arg["name"]: arg for arg in entry["new"].get("args", [])}

        for arg_name, old_arg in old_args.items():

            new_arg = new_args.get(arg_name)

            if new_arg is None:
                breaking_changes.append({
                    "type": "removed_parameter",
                    "name": name,
                    "detail": f"Parameter '{arg_name}' was removed from '{name}'.",
                })
                continue

            if old_arg.get("has_default") and not new_arg.get("has_default"):
                breaking_changes.append({
                    "type": "parameter_became_required",
                    "name": name,
                    "detail": (
                        f"Parameter '{arg_name}' of '{name}' lost its "
                        "default and is now required."
                    ),
                })

            old_type = old_arg.get("type")
            new_type = new_arg.get("type")

            if old_type is not None and new_type is not None and old_type != new_type:
                breaking_changes.append({
                    "type": "parameter_type_changed",
                    "name": name,
                    "detail": (
                        f"Parameter '{arg_name}' of '{name}' changed type "
                        f"from '{old_type}' to '{new_type}'."
                    ),
                })

        for arg_name, new_arg in new_args.items():

            if arg_name not in old_args and not new_arg.get("has_default"):
                breaking_changes.append({
                    "type": "required_parameter_added",
                    "name": name,
                    "detail": (
                        f"New required parameter '{arg_name}' was added "
                        f"to '{name}'."
                    ),
                })

        old_return = entry["old"].get("return_type")
        new_return = entry["new"].get("return_type")

        if old_return != new_return:
            breaking_changes.append({
                "type": "return_type_changed",
                "name": name,
                "detail": (
                    f"'{name}' return type changed from '{old_return}' "
                    f"to '{new_return}'."
                ),
            })

    return {
        "compatible": not breaking_changes,
        "breaking_changes": breaking_changes,
    }


def _notebook_source_lines(notebook_path):
    """Every code cell's own raw source, in on-disk order (the same cells
    extract_functions_from_code and every other per-cell walk in this
    file already use), as one flat list of lines -- one "# Cell <n>"
    marker line ahead of each cell's own lines, so a line in a resulting
    diff can still be traced back to which cell it came from.

    Used only by diff_notebook_source (below) to build the two sides it
    diffs; not itself a per-notebook report of any kind.
    """
    notebook = load_notebook(notebook_path)

    lines = []

    for i, cell in enumerate(extract_code_cells(notebook)):

        lines.append(f"# Cell {i}")
        lines.extend(cell.splitlines())
        lines.append("")

    return lines


def diff_notebook_source(
    old_notebook_path, new_notebook_path, old_label=None, new_label=None
):
    """Line-level unified diff of every code cell's own raw source
    between two notebooks -- the literal text a reviewer would actually
    want to see, distinct from diff_notebook_functions' own structural
    added/removed/changed-signature report.

    diff_notebook_functions (above) deliberately only reports whether a
    function's *compiled API contract* changed -- by its own docstring,
    "a docstring-only edit does not count as changed", and a plain
    non-function code cell (a stray `print()`, a constant, a `load_data()`
    call with no `def`) never shows up there at all, since it was never a
    function to begin with. A reviewer or CI check that wants to actually
    see *what changed in the notebook's own code* -- not just whether its
    compiled endpoints did -- had no way to get that from this project's
    own API at all: the only route was downloading both notebooks' raw
    content and diffing them locally, entirely outside it.

    Built with Python's own difflib.unified_diff -- the same three-line-
    of-context unified diff format `git diff`/`diff -u` produce, so it
    reads exactly like either and can be piped straight into any tool
    that already understands that format. Returns a plain list of lines,
    each with its own trailing newline already stripped (unified_diff's
    own lineterm="" -- the join below is bridging JSON's own
    array-of-lines convention, not writing to a text stream) -- an empty
    list when the two notebooks' code cells are identical line-for-line.

    "old_label"/"new_label" (each optional) are used as unified_diff's
    own "fromfile"/"tofile" header line, defaulting to
    "old_notebook_path"/"new_notebook_path" themselves when omitted --
    callers comparing two of a notebook's own snapshotted versions (see
    GET /api/notebooks/{filename}/versions/{version_id}/diff,
    routes/upload.py) pass a human-readable label ("version '<id>'")
    instead, the same "never leak a raw server-side path" precedent
    GET /api/config's own docstring already sets for UPLOAD_DIR/
    GENERATED_DIR: the paths passed in here can be an absolute path
    under UPLOAD_DIR/.versions/, dashboard-internal layout no API
    response elsewhere in this project exposes either.
    """
    old_lines = _notebook_source_lines(old_notebook_path)
    new_lines = _notebook_source_lines(new_notebook_path)

    return list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_label or old_notebook_path,
            tofile=new_label or new_notebook_path,
            lineterm="",
        )
    )


def print_notebook_diff(diff):
    """Print diff_notebook_functions' own report in the same
    human-readable style print_compile_summary/inspect_notebook already
    use elsewhere in this file.

    If `diff` also carries classify_notebook_diff's own "compatible"/
    "breaking_changes" fields (every diff-producing endpoint/command now
    merges those in), also prints a compatibility verdict after the
    added/removed/changed report above. A plain diff_notebook_functions
    dict (no such keys) prints exactly as before this existed.
    """
    if diff["added"]:
        print(f"\n+ Added {len(diff['added'])} endpoint(s):")
        for func in diff["added"]:
            print(f"  + POST /{func['name']}")

    if diff["removed"]:
        print(f"\n- Removed {len(diff['removed'])} endpoint(s):")
        for func in diff["removed"]:
            print(f"  - POST /{func['name']}")

    if diff["changed"]:
        print(f"\n~ Changed {len(diff['changed'])} endpoint(s):")
        for entry in diff["changed"]:
            print(f"  ~ POST /{entry['name']}")

    if not (diff["added"] or diff["removed"] or diff["changed"]):
        print("\nNo changes to the compiled API surface.")

    if "compatible" in diff:
        if diff["compatible"]:
            print("\nNo breaking changes to the compiled API's contract.")
        else:
            print(
                f"\n! {len(diff['breaking_changes'])} breaking change(s) "
                "to the compiled API's contract:"
            )
            for change in diff["breaking_changes"]:
                print(f"  ! {change['detail']}")

# The generated app's own default API key (see write_app_config /
# verify_api_key in generator/api_generator.py: `API_KEYS` defaults to
# os.getenv("NOTEBOOK_API_KEY", "notebook-to-api-dev-key")) -- duplicated
# here as a literal, not imported, since api_generator.py only ever
# embeds it as a string inside generated source lines, never as an
# importable constant of its own. The same soft-coupling
# EXCLUDED_GENERATED_FILE_NAMES's own docstring above already accepts for
# generator/docker_generator.py's identical inability to import
# COMPILE_METADATA_FILENAME without a circular import.
DEFAULT_DEV_API_KEY = "notebook-to-api-dev-key"


def generate_curl_commands(
    notebook_path, host="localhost", port=8000, api_key=None,
    only=None, exclude=None,
):
    """A ready-to-run `curl` command for every function `notebook_path`
    would compile into an endpoint, using each function's own
    "example_payload" (see inspect_notebook_data's "functions" field) as
    the request body.

    Before this, trying out a notebook's compiled API meant either
    hand-writing a curl invocation (or opening a separate HTTP client)
    per endpoint -- correctly guessing its JSON body shape and, easy to
    miss, that every generated endpoint requires an `X-API-Key` header at
    all (see verify_api_key, generator/api_generator.py) -- or compiling
    first and reading /docs. This needs no compile step at all: it works
    directly from the same notebook analysis `inspect`/POST /api/inspect
    already do, so a caller can preview exactly what they'll be able to
    call before ever running `compile`.

    Functions with a reserved-name collision (see
    _reserved_name_conflicts above) are skipped -- generate_fastapi_code
    refuses to compile a notebook containing one at all, so a curl
    command hitting that path would never actually resolve to anything.

    A background/task_id-based function's own command is commented to
    note that its POST only returns {"task_id": ...}, not the actual
    result -- unlike a synchronous endpoint's command, there's no way to
    know the task_id ahead of actually running it, so this can't also
    generate the GET /tasks/{task_id} follow-up command the way it can
    for everything else.

    `api_key` defaults to DEFAULT_DEV_API_KEY -- the generated app's own
    default when NOTEBOOK_API_KEY isn't set -- so the emitted commands
    work out of the box against a locally `serve`d app with no
    configuration; a caller who has set NOTEBOOK_API_KEY to something
    else needs to pass the matching value here instead.

    `only`/`exclude` restrict which functions get a command at all, via
    the exact same _filter_functions_by_name (backend/compiler.py) every
    other function-selecting entry point here already goes through
    (`compile`, `deploy`, `serve`, `watch`, POST /api/compile, POST
    /api/app-preview). Before this, `export-curl`/`remote-curl`/POST
    /api/curl-preview always emitted a command for *every* function a
    notebook defines, even one a caller's own --only/--exclude compile
    would never actually expose as an endpoint -- silently wrong output
    for anyone previewing curl commands for a notebook they intend to
    compile with either flag, the exact "can't drift from what an actual
    compile would produce" guarantee every sibling preview endpoint in
    this file already gives. Raises the identical ValueError
    _filter_functions_by_name itself raises for both-given or an
    unrecognized function name.

    Returns a list of ready-to-paste (or execute) multi-line command
    strings, one per function, in the same order inspect_notebook_data's
    own "functions" field returns them.
    """
    if api_key is None:
        api_key = DEFAULT_DEV_API_KEY

    data = inspect_notebook_data(notebook_path)

    functions = _filter_functions_by_name(data["functions"], only, exclude)

    reserved_names = set(data["reserved_name_conflicts"])

    base_url = f"http://{host}:{port}"

    commands = []

    for func in functions:

        name = func["name"]

        if name in reserved_names:
            continue

        payload = json.dumps(func.get("example_payload", {}))

        if _is_background_function(name):
            comment = (
                f'# {name} (background task -- POST only returns '
                f'{{"task_id": ...}}; poll GET /tasks/{{task_id}} for the '
                f"actual result)"
            )
        else:
            comment = f"# {name}"

        command = (
            f"{comment}\n"
            f"curl -X POST {base_url}/{name} \\\n"
            f'  -H "Content-Type: application/json" \\\n'
            f'  -H "X-API-Key: {api_key}" \\\n'
            f"  -d '{payload}'"
        )

        commands.append(command)

    return commands


def generate_postman_collection(
    notebook_path, host="localhost", port=8000, api_key=None,
    only=None, exclude=None, collection_name=None,
):
    """A Postman Collection v2.1.0 covering every function `notebook_path`
    would compile into an endpoint -- the same "try it before you compile
    it" coverage generate_curl_commands above already gives a terminal,
    for the far larger share of API consumers who reach for Postman
    instead of raw curl and have no way to turn a curl command back into
    a Postman request without pasting it into Postman's own "import from
    curl" flow by hand, once per endpoint.

    Mirrors generate_curl_commands wherever the two overlap: the same
    reserved-name-conflict skipping (a notebook containing one fails to
    compile at all, so a request targeting that path would never resolve
    to anything real), the same only/exclude filtering -- including the
    identical ValueError _filter_functions_by_name raises for both-given
    or an unrecognized name -- the same example_payload-as-body, and the
    same DEFAULT_DEV_API_KEY default. Works directly off
    inspect_notebook_data, exactly like generate_curl_commands does, so
    no compile step is required first.

    "host"/"port"/"api_key" become the collection's own "base_url"/
    "api_key" *variables* rather than being baked into each request's URL
    and header directly, so a caller can retarget every request at once
    (e.g. a different `serve` host/port, or a NOTEBOOK_API_KEY that's
    since been rotated) from Postman's own variable editor, without
    re-exporting the whole collection.

    A background/task_id-based function (see _is_background_function
    above) gets something a static curl command has no way to express at
    all: its submission request carries a Postman "test" script that
    captures the "task_id" field of its own response into a
    "{name}_task_id" collection variable -- scoped per function, not one
    shared "task_id", so running two different background functions'
    requests in the same collection doesn't clobber one function's
    captured id with another's -- and a paired "{name} - Task Status"
    request is already wired to poll GET /tasks/{{"{name}_task_id"}} with
    it, ready to send the moment the submission request has run once.
    Every "{name}_task_id" variable is declared up front (default "") so
    Postman doesn't warn about an undefined variable before that first
    run. A synchronous function's own request gets neither the script nor
    a companion request, since there's no task_id to capture.

    Returns a plain dict -- a valid Postman Collection v2.1.0 document
    once json.dump-ed -- rather than writing a file itself, the same
    "return data, let the caller decide where it goes" split
    inspect_notebook_data/generate_curl_commands already follow. Used by
    both `export-postman` (writes it to a local file) and POST
    /api/postman-preview (returns it directly as the response body).
    """
    if api_key is None:
        api_key = DEFAULT_DEV_API_KEY

    data = inspect_notebook_data(notebook_path)

    functions = _filter_functions_by_name(data["functions"], only, exclude)

    reserved_names = set(data["reserved_name_conflicts"])

    base_url = f"http://{host}:{port}"

    collection_variables = [
        {"key": "base_url", "value": base_url},
        {"key": "api_key", "value": api_key},
    ]

    items = []

    for func in functions:

        name = func["name"]

        if name in reserved_names:
            continue

        payload = func.get("example_payload", {})

        request = {
            "method": "POST",
            "header": [
                {"key": "Content-Type", "value": "application/json"},
                {"key": "X-API-Key", "value": "{{api_key}}"},
            ],
            "url": {
                "raw": "{{base_url}}/" + name,
                "host": ["{{base_url}}"],
                "path": [name],
            },
            "body": {
                "mode": "raw",
                "raw": json.dumps(payload, indent=2),
                "options": {"raw": {"language": "json"}},
            },
        }

        item = {"name": name, "request": request}

        if _is_background_function(name):

            task_id_var = f"{name}_task_id"
            collection_variables.append({"key": task_id_var, "value": ""})

            request["description"] = (
                'This POST only returns {"task_id": ...} immediately -- '
                f'run the paired "{name} - Task Status" request below '
                f"(its own {{{{{task_id_var}}}}} is captured automatically "
                "from this response) to see the actual result."
            )

            item["event"] = [{
                "listen": "test",
                "script": {
                    "type": "text/javascript",
                    "exec": [
                        "if (pm.response.code === 200) {",
                        "    const body = pm.response.json();",
                        "    if (body.task_id) {",
                        (
                            f'        pm.collectionVariables.set('
                            f'"{task_id_var}", body.task_id);'
                        ),
                        "    }",
                        "}",
                    ],
                },
            }]

            items.append(item)
            items.append({
                "name": f"{name} - Task Status",
                "request": {
                    "method": "GET",
                    "header": [{"key": "X-API-Key", "value": "{{api_key}}"}],
                    "url": {
                        "raw": "{{base_url}}/tasks/{{" + task_id_var + "}}",
                        "host": ["{{base_url}}"],
                        "path": ["tasks", "{{" + task_id_var + "}}"],
                    },
                },
            })

        else:
            items.append(item)

    return {
        "info": {
            "name": collection_name or Path(notebook_path).stem,
            "schema": (
                "https://schema.getpostman.com/json/collection/v2.1.0/"
                "collection.json"
            ),
        },
        "variable": collection_variables,
        "item": items,
    }
