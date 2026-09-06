import datetime
import functools
import hashlib
import importlib.metadata
import json
import keyword
import os
import re
import shutil
import sys
import pathlib
import threading

from pathlib import Path

# sys.stdlib_module_names (Python 3.10+) is the authoritative list of
# standard-library module names. The previous hand-picked set of 12 names
# missed the vast majority of the standard library (asyncio, random,
# logging, subprocess, csv, sqlite3, uuid, hashlib, threading, ...), so any
# notebook using one of them got that name written into requirements.txt
# as if it were a third-party PyPI package. That's not just noise: PyPI
# has real (unofficial, unrelated) packages published under some stdlib
# names -- e.g. `pip install asyncio` installs a bogus package that
# shadows the built-in module -- so this could actively break the
# generated app rather than just add a redundant line.
STANDARD_LIBS = set(sys.stdlib_module_names)

# This tool's own top-level package name -- derived from __name__ (this
# module's own fully-qualified name is "backend.compiler") rather than
# hardcoded, so it can't drift if this package is ever renamed. Used by
# package_name_for_output_dir's own-package collision check below.
THIS_TOOLS_OWN_PACKAGE_NAME = __name__.partition(".")[0]

# This tool's own version -- previously duplicated as two independent
# "0.1.0" string literals (backend/dashboard.py's own FastAPI(version=...)
# and its GET / root endpoint's own "version" field), with nothing
# enforcing they'd ever be bumped together, and no way for the CLI to
# report its own version at all (no --version flag existed, and
# backend/cli.py deliberately never imports from backend.dashboard --
# doing so would drag in that module's own FastAPI app and every
# unrelated subsystem router it mounts, just for a version string).
# backend/compiler.py is the one lightweight module both already import
# from at module level, so this is the one place a caller on either side
# reads it from -- bump this single constant and both dashboard.py's
# responses and the CLI's own --version follow automatically, with no way
# for them to drift out of sync with each other again.
NOTEBOOK_TO_API_VERSION = "0.1.0"

# Ensure project root is in sys.path for proper imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.parser.notebook_parser import (
    load_notebook,
    extract_code_cells
)

from backend.parser.ast_parser import (
    extract_functions_from_code,
    extract_imports_from_code,
    is_parseable_python,
    deduplicate_functions_by_name
)

from backend.generator.api_generator import (
    GENERATED_APP_ENV_VARS,
    generate_fastapi_code,
    write_generated_api
)

from backend.generator.docker_generator import (
    generate_dockerfile,
    generate_dockerignore,
    generate_docker_compose,
    generate_env_example,
    generate_readme
)

from backend.generator.kubernetes_generator import (
    generate_kubernetes_manifest,
)


@functools.lru_cache(maxsize=1)
def _installed_packages_distributions():
    """Cached importlib.metadata.packages_distributions() result: a map of
    top-level import name -> list of distribution names that provide it,
    for every distribution installed in the interpreter compiling this
    notebook (e.g. "cv2" -> ["opencv-python"], "fastapi" -> ["fastapi"]).

    Reads each installed distribution's own metadata (effectively its
    top_level.txt) rather than searching sys.path for anything that
    merely happens to be importable right now -- unlike
    importlib.util.find_spec, it can't pick up a local directory that
    just looks like a package. That distinction matters here specifically
    because this tool's own previous output directories (e.g.
    "generated", "test_generated") are exactly that: real, on-disk Python
    packages sitting in this project's own directory tree the moment
    they've been compiled once. Using find_spec instead would have made
    this tool's own documented default, `--output generated`, start
    failing the very next time it ran (see
    _installed_third_party_package_names below, built from this same
    call).

    Cached (this scan costs ~100ms and re-reads every installed
    distribution's metadata each time) since the set of installed
    packages doesn't change over the lifetime of a single compiling
    process -- the CLI's compile/serve/deploy commands each run once per
    process anyway, and the dashboard's compile-on-every-request path
    would otherwise re-pay this cost on every single POST /api/compile.
    Kept as its own function (rather than having each caller below call
    importlib.metadata.packages_distributions() directly) so the
    underlying scan only ever runs once per process no matter how many of
    them need it.
    """
    return importlib.metadata.packages_distributions()


def _installed_third_party_package_names():
    """Top-level import names of every distribution installed in the
    interpreter compiling this notebook (e.g. "fastapi", "numpy",
    "pandas", ...), used by package_name_for_output_dir's collision check
    below.
    """
    return frozenset(_installed_packages_distributions().keys())


def distribution_name_for_import(import_name):
    """The actual PyPI distribution name that provides `import_name`, if
    it's satisfied by something installed in the environment compiling
    this notebook, else `import_name` unchanged.

    A notebook's own `import` statement names a *module*, not necessarily
    the PyPI *distribution* that provides it -- `pip install <name>` only
    works for the latter, and the two frequently differ: `import cv2` is
    provided by "opencv-python", `import yaml` by "PyYAML", `import
    dateutil` by "python-dateutil", and so on. Before this, write_requirements
    (below) wrote the raw import name straight into requirements.txt
    unchanged, so `pip install -r requirements.txt` -- and from there
    every `deploy`/`docker build` -- failed outright for any notebook
    using one of these: confirmed against this exact environment, a
    notebook doing `import dateutil` (satisfied here by the installed
    "python-dateutil") produced a requirements.txt literally containing
    "dateutil", not a real PyPI package name. _pinned_requirement's own
    importlib.metadata.version() lookup didn't catch this either -- it
    resolves *distribution* names, not import names, so it raised
    PackageNotFoundError for "dateutil" too and silently fell back to
    writing that same wrong, unpinned name, with nothing in the output to
    indicate anything was off.

    _installed_packages_distributions() above is the authoritative
    reverse mapping for this: it reads each installed distribution's own
    metadata to report which top-level import names it actually provides.
    On the rare case more than one installed distribution provides the
    same import name, picks one deterministically (sorted, first). Falls
    back to `import_name` unchanged if it isn't installed here at all --
    preserving _pinned_requirement's own existing fallback for a
    dependency this tool has no way to introspect, rather than guessing
    at a distribution name it can't confirm.
    """
    distributions = _installed_packages_distributions().get(import_name)

    if not distributions:
        return import_name

    return sorted(distributions)[0]


def package_name_for_output_dir(output_dir):
    """The generated app imports its runtime module as
    `<package_name>.runtime.notebook_module`, so the output directory's
    basename must double as a valid Python package name. This was
    previously just hardcoded to "generated" everywhere, which silently
    broke the documented --output flag for any other directory: the
    runtime module ended up written to a fixed generated/runtime/ path
    while app.py (written wherever --output pointed) still imported from
    it by the fixed name "generated", regardless of where it actually
    landed on disk.
    """
    name = os.path.basename(os.path.normpath(output_dir))

    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f"Output directory {output_dir!r} (basename {name!r}) can't be "
            "used as a Python package name for the generated app's "
            "`import <name>.runtime.notebook_module` statement. Choose an "
            "--output directory whose final path segment is a valid Python "
            "identifier (letters, digits, underscores; not starting with a "
            "digit; not a reserved keyword like 'import')."
        )

    # Confirmed exploitable: `--output json` (or os/sys/time/re/... --
    # any real standard-library module name, already collected in
    # STANDARD_LIBS above for the identical name-collision hazard in
    # write_requirements) compiled without error, but the generated
    # app.py's `import json.runtime.notebook_module as notebook_module`
    # statement then resolved to the real, already-imported stdlib `json`
    # module instead of the locally compiled package -- Python's import
    # system finds and caches a standard-library module ahead of same-
    # named packages under the working directory. This isn't just a
    # cosmetic naming clash: `python -m uvicorn json.app:app` (what
    # `serve`, the generated Dockerfile's CMD, and any real deployment
    # all run) fails outright with "No module named 'json.app'" -- the
    # generated app is entirely unusable, with the failure only ever
    # surfacing later, disconnected from the --output choice that
    # actually caused it, and looking like a packaging bug rather than a
    # bad directory name.
    if name in STANDARD_LIBS:
        raise ValueError(
            f"Output directory {output_dir!r} (basename {name!r}) collides "
            f"with the Python standard library module {name!r} -- the "
            f"generated app's `import {name}.runtime.notebook_module` "
            "statement would resolve to that standard-library module "
            "instead of the locally compiled package. Choose a different "
            "--output directory whose final path segment isn't a standard "
            "library module name."
        )

    # Same import-shadowing hazard as the standard-library check above,
    # but for this tool's own top-level package -- the one this very
    # function (backend/compiler.py) lives inside. Confirmed exploitable:
    # `--output backend` passed the isidentifier()/keyword/STANDARD_LIBS
    # checks fine and compiled without error, but the generated app.py's
    # `import backend.runtime.notebook_module` statement would then
    # resolve to this tool's own real "backend" package instead of the
    # locally compiled one. Neither check above catches this: "backend"
    # isn't a standard-library module, and
    # importlib.metadata.packages_distributions() (used by the
    # already-installed-package check just below) doesn't know about it
    # either -- this project was never `pip install`ed, so it has no
    # distribution metadata of its own, the identical reason this tool's
    # own prior --output dirs like "generated" don't trip that check (see
    # _installed_third_party_package_names's own docstring). Unlike an
    # ordinary third-party package, though, this collision isn't merely
    # *possible* -- backend/compiler.py (this very module) is part of the
    # "backend" package, so it is unconditionally already imported and in
    # sys.modules by the time this function ever runs, in every single
    # invocation of this tool: the CLI, the dashboard, and any process
    # that has imported backend.compiler at all. Checked here, ahead of
    # the installed-third-party-package scan below, since it's a plain
    # string comparison against a known constant rather than needing that
    # scan's own ~100ms metadata read to answer a question this already
    # answers for free.
    if name == THIS_TOOLS_OWN_PACKAGE_NAME:
        raise ValueError(
            f"Output directory {output_dir!r} (basename {name!r}) collides "
            f"with this tool's own top-level package {name!r} -- the "
            f"generated app's `import {name}.runtime.notebook_module` "
            "statement would resolve to this tool's own package instead of "
            "the locally compiled one. Choose a different --output "
            "directory whose final path segment isn't this tool's own "
            "package name."
        )

    # Same import-shadowing hazard as the standard-library check above,
    # but for an already-installed *third-party* package rather than one
    # built into the interpreter. Confirmed exploitable: `--output
    # fastapi` (or numpy/pandas/httpx/... any package actually `pip
    # install`ed in the compiling environment, discovered via
    # _installed_third_party_package_names() above) compiled without
    # error, but the generated app.py's `import
    # fastapi.runtime.notebook_module` statement then resolved to the
    # real installed `fastapi` package instead of the locally compiled
    # one -- reproduced against a real `python -m uvicorn fastapi.app:app`,
    # which fails outright with "Could not import module 'fastapi.app'"
    # since the real package has no such submodule. It's worse still when
    # the notebook itself imports the same package: the runtime module's
    # own `import fastapi` (or whichever package collided) would then
    # resolve to the local, generated package -- which has none of the
    # real library's actual content -- breaking the notebook's own code,
    # not just app.py's outer import.
    if name in _installed_third_party_package_names():
        raise ValueError(
            f"Output directory {output_dir!r} (basename {name!r}) collides "
            f"with the already-installed Python package {name!r} -- the "
            f"generated app's `import {name}.runtime.notebook_module` "
            "statement would resolve to that installed package instead of "
            "the locally compiled one. Choose a different --output "
            "directory whose final path segment isn't an installed "
            "package name."
        )

    return name


def write_runtime_module(code_cells, output_dir):

    runtime_path = Path(output_dir) / "runtime" / "notebook_module.py"

    runtime_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined_code = "\n\n".join(code_cells)

    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write(combined_code)

    print("Runtime module generated.")


def _pinned_requirement(package_name):
    """Return "package_name==version" if `package_name`'s version can be
    resolved in the environment compiling the notebook, else just
    `package_name` unchanged.

    Without pinning, every generated app's requirements.txt listed bare
    package names ("fastapi", "pandas", ...), so `pip install -r
    requirements.txt` resolved whichever release happened to be latest at
    *deploy* time -- not the version the notebook was actually compiled
    and tested against. A breaking release published upstream between
    compile time and deploy time would silently change the generated
    app's behavior (or break it outright) with nothing in the generated
    output to indicate what had changed. Falling back to the bare name
    when the package isn't installed in this environment (e.g. a
    notebook imports a third-party library the compiling host doesn't
    have) preserves the previous behavior instead of failing the whole
    compile over a dependency this tool has no way to introspect.

    The PEP 440 local version segment (everything from the first "+"
    onward, e.g. "2.1.0+cu121" for a CUDA-specific torch build,
    "0.1.0+dirty" or "0.1.0.dev3+g1a2b3c4" from a setuptools-scm/git-based
    build, or whatever an editable `pip install -e .` records) is stripped
    before pinning. PyPI rejects any upload whose version contains one --
    it exists specifically to distinguish a locally-modified build from
    the public release it's based on, never to be redistributed itself --
    so importlib.metadata.version() reporting one here means this exact
    "package_name==version" string can never resolve against PyPI.
    Confirmed reproduced: installing a package under a version string
    containing "+", then compiling a notebook that imports it, wrote that
    unresolvable pin straight into requirements.txt, and `pip install -r
    requirements.txt` -- and from there every `deploy`/`docker build` --
    failed outright with "No matching distribution found for
    <package>==<version>+<local>", even though the public version right
    before the "+" is very likely actually installable. This can affect
    any dependency in the pin list, not just an unusual one deliberately
    reproducing it: this tool's own compiling environment is exactly as
    likely to have a locally-built or editable-installed package as any
    other, and nothing before this caught it.
    """
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return package_name

    public_version = version.partition("+")[0]

    return f"{package_name}=={public_version}"


def compiling_python_version():
    """The "<major>.<minor>" Python version of the interpreter running this
    compile, e.g. "3.11".

    _pinned_requirement (above) resolves every dependency's pinned version
    from *this* interpreter's installed packages. A Docker base image
    running a different Python than the one that produced those pins can
    silently break `docker build`'s `pip install -r requirements.txt` the
    moment a pinned package's wheels don't cover that Python version, or
    fall back to a source build that behaves differently from what was
    actually resolved and tested locally. Passed straight through to
    generate_dockerfile (see compile_notebook_to_api below) so the base
    image always matches the interpreter that produced requirements.txt,
    instead of the Dockerfile hardcoding a fixed version unrelated to it.
    """
    return f"{sys.version_info.major}.{sys.version_info.minor}"


# Recognizes a "# notebook-to-api: requires <spec>" comment directive
# anywhere in a code cell's raw source -- see
# _extract_explicit_requirements below for what it's for. Matched against
# each cell's raw text directly (not via the AST, unlike every other
# extraction step in this pipeline) since a comment carries no meaning
# for `ast.parse` to preserve in the first place; it's gone from the tree
# entirely. Requires the "#" to start the line (allowing leading
# whitespace, so it's usable indented inside a function body too) so an
# unrelated inline comment that merely happens to contain this phrase
# elsewhere in a line of code isn't matched by accident.
REQUIREMENT_DIRECTIVE_PATTERN = re.compile(
    r"^\s*#\s*notebook-to-api:\s*requires\s+(?P<spec>\S.*)$",
    re.MULTILINE,
)


def _extract_explicit_requirements(code_cells):
    """Extra requirements.txt lines a notebook author declares explicitly
    via a "# notebook-to-api: requires <spec>" comment directive, one per
    line, anywhere in any code cell -- in the order first seen, with
    exact-duplicate lines removed.

    write_requirements (below) only ever pins whatever a notebook's own
    `import` statements resolve to (see distribution_name_for_import) --
    but that resolution depends entirely on what's importable in the
    environment doing the compiling. A dependency the notebook actually
    needs at runtime but that isn't importable there at all (one only
    ever imported dynamically via importlib rather than a top-level
    `import` a static scan can see, one that's only installed in the
    deploy target's own image and never the compiling machine, or simply
    a private package/VCS URL/extras spec this tool has no way to guess
    on its own) had no way to be added to requirements.txt at all.

    Each matched <spec> is written to requirements.txt exactly as given
    -- not resolved via distribution_name_for_import or re-pinned via
    _pinned_requirement, since a caller writing this directive is already
    stating precisely what belongs there, the same as a hand-written
    requirements.txt line would.

    Raises ValueError if two *different* specs name the same package (via
    _explicit_requirement_package_name below, case-insensitively -- PyPI
    distribution names are themselves case-insensitive) -- e.g. one cell
    declaring "# notebook-to-api: requires numpy==1.24.0" and another
    "# notebook-to-api: requires numpy==1.26.0", commonly left behind
    after pinning a different version while iterating on a notebook.
    "seen"/exact-duplicate removal just above only catches the identical-
    line case; two distinct specs for the same package both survived it
    (and this function's own returned list) before this check existed,
    both then surviving resolve_requirements' own explicit-vs-auto-
    detected conflict resolution too (its own docstring explains that
    one) -- since resolve_requirements only ever drops an *auto-detected*
    import an explicit directive already names, it never had a reason to
    also dedupe explicit requirements against each other, that being this
    function's own job. Both lines landing side by side in
    requirements.txt is the identical "Double requirement given" pip
    failure resolve_requirements' own docstring already describes fixing
    for the auto-detected-vs-explicit case, just for two explicit
    directives instead -- confirmed exploitable before this check
    existed. Raised here (not in resolve_requirements, which has no way
    to tell an explicit spec's own conflicting sibling apart from an
    ordinary auto-detected/explicit collision it's already supposed to
    resolve) so the conflicting notebook fails this specific, actionable
    check -- rather than a raw pip error surfacing only much later, at
    `docker build` time, deep inside POST /api/deploy or the CLI's own
    `deploy`.
    """
    specs = []
    seen = set()
    spec_by_package_name = {}

    for cell in code_cells:

        for match in REQUIREMENT_DIRECTIVE_PATTERN.finditer(cell):

            spec = match.group("spec").strip()

            if not spec or spec in seen:
                continue

            package_name = _explicit_requirement_package_name(spec)

            if package_name is not None:

                normalized_name = package_name.lower()
                conflicting_spec = spec_by_package_name.get(normalized_name)

                if conflicting_spec is not None:
                    raise ValueError(
                        "Conflicting '# notebook-to-api: requires' "
                        f"directives for '{package_name}': "
                        f"'{conflicting_spec}' and '{spec}' -- pip refuses "
                        "two requirements for the same package "
                        "('Double requirement given'). Remove or reconcile "
                        "one of them."
                    )

                spec_by_package_name[normalized_name] = spec

            seen.add(spec)
            specs.append(spec)

    return specs


# Recognizes a "# notebook-to-api: exclude <import-name>" comment
# directive -- see _extract_excluded_imports below for what it's for.
# Same matching rules as REQUIREMENT_DIRECTIVE_PATTERN above (leading
# whitespace allowed, "#" must start the line) for the identical reason:
# usable indented inside a function body, and immune to an unrelated
# inline comment merely containing this phrase elsewhere in a line.
EXCLUDE_DIRECTIVE_PATTERN = re.compile(
    r"^\s*#\s*notebook-to-api:\s*exclude\s+(?P<name>\S+)\s*$",
    re.MULTILINE,
)


def _extract_excluded_imports(code_cells):
    """Import names a notebook author explicitly opts out of
    requirements.txt (and, via inspector.py's own identical use of this,
    every "dependencies" field derived from the same import scan) via a
    "# notebook-to-api: exclude <import-name>" comment directive, one per
    line, anywhere in any code cell.

    extract_third_party_imports (below) otherwise pins every third-party
    name any code cell's own `import` statement references, with no way
    to keep one out short of deleting the import (and whatever uses it)
    from the notebook entirely -- even for an import that's incidental to
    what actually gets compiled into an endpoint (e.g. a scratch cell
    doing `import pytest` to sanity-check a function inline, never
    imported by anything the generated app itself calls at runtime), or
    one already vendored/bundled into the deploy target's own image and
    never meant to be pip-installed there at all.

    `name` is matched exactly against an import's own top-level module
    name (the same name extract_third_party_imports/
    extract_imports_from_code already collect it under, before any
    distribution_name_for_import resolution) -- not a PyPI distribution
    name, since that's what a notebook's own `import` statement actually
    names, and what a caller writing this directive is looking at.
    Returns a set, the same membership-tested shape STANDARD_LIBS already
    is here, for the identical `imp not in ...` filter below.
    """
    excluded = set()

    for cell in code_cells:

        for match in EXCLUDE_DIRECTIVE_PATTERN.finditer(cell):

            name = match.group("name").strip()

            if name:
                excluded.add(name)

    return excluded


# Recognizes a "# notebook-to-api: private" comment directive immediately
# above a function definition (blank lines in between are tolerated, the
# same way a real editor/formatter might leave one) -- see
# _extract_private_function_names below for what it's for. Unlike
# REQUIREMENT_DIRECTIVE_PATTERN/EXCLUDE_DIRECTIVE_PATTERN above, this one
# is positional: it only matches when directly followed by a "def"/"async
# def" line, since which function it applies to is the entire point.
PRIVATE_FUNCTION_DIRECTIVE_PATTERN = re.compile(
    r"^[ \t]*#\s*notebook-to-api:\s*private\s*$"
    r"(?:\n[ \t]*\n)*"
    r"\n[ \t]*(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)


def _extract_private_function_names(code_cells):
    """Names of every function `code_cells` marks "# notebook-to-api:
    private" (immediately above its own `def`/`async def` line, see
    PRIVATE_FUNCTION_DIRECTIVE_PATTERN above), across all cells.

    _filter_functions_by_name's own docstring already spells out the gap
    this closes: "Every module-level function a notebook defines ...
    becomes a public API endpoint, with no previous way to opt any of
    them out" besides an external --only/--exclude flag a caller has to
    remember to pass on *every* compile/serve/watch/deploy invocation --
    renaming a helper doesn't help either, since extract_functions_from_code
    applies no leading-underscore (or any other) naming convention. A
    notebook author's own internal helper (data loading, validation,
    formatting, ...) had no way to declare, once, from inside the
    notebook itself, that it should never become an endpoint at all.

    Returns a set, the same membership-tested shape _extract_excluded_imports
    above already returns for an import name, for the identical `name in
    ...`/`name not in ...` filtering _drop_private_functions below (and
    every direct caller of this function) already needs.
    """
    private_names = set()

    for cell in code_cells:

        for match in PRIVATE_FUNCTION_DIRECTIVE_PATTERN.finditer(cell):

            private_names.add(match.group("name"))

    return private_names


def _drop_private_functions(functions, code_cells, only=None, exclude=None):
    """Drop every function `code_cells` marks private (see
    _extract_private_function_names above) from `functions` -- shared by
    every entry point that builds a functions list directly from
    code_cells (compile_notebook_to_api below, and app_preview_endpoint
    in routes/upload.py) so a private function is dropped identically
    everywhere, the same "can't drift" guarantee _filter_functions_by_name's
    own only/exclude handling already provides. Applied before
    _filter_functions_by_name itself, which has no access to code_cells
    (only to the already-extracted `functions` list) and so has no way to
    know which of them a notebook has marked private on its own.

    Returns (filtered_functions, adjusted_exclude): `exclude` has any
    already-private name filtered back out, so naming an already-private
    function via --exclude is the harmless no-op it actually is (the same
    "redundant is not an error" precedent this project's own sha256/tag
    filters already follow elsewhere) instead of _filter_functions_by_name's
    "not defined in this notebook" error, which would otherwise be
    misleading here -- the function *is* defined, just already excluded
    by directive.

    Raises ValueError -- the same type _filter_functions_by_name's own
    unknown-name error already raises -- if `only` explicitly names a
    private function: unlike `exclude` above, that's a direct
    contradiction of the notebook's own declared intent, not a redundant
    no-op, and deserves a clearer, more actionable error than the generic
    "unknown function name" a caller would otherwise get once the
    function is silently missing from `functions` altogether.
    """
    private_names = _extract_private_function_names(code_cells)

    if not private_names:
        return functions, exclude

    if only:

        conflicting = sorted(set(only) & private_names)

        if conflicting:
            raise ValueError(
                f"Function(s) {', '.join(conflicting)} are marked "
                '"# notebook-to-api: private" in the notebook and can\'t '
                "be exposed via only/--only. Remove the directive in the "
                "notebook to expose them."
            )

    if exclude:
        exclude = [name for name in exclude if name not in private_names]

    return (
        [func for func in functions if func["name"] not in private_names],
        exclude,
    )


def extract_third_party_imports(code_cells):
    """The raw, STANDARD_LIBS-filtered import names `code_cells` (already
    filtered to parseable cells, as compile_notebook_to_api's own
    `code_cells` already is) collect -- before any distribution-name
    resolution or version pinning, exactly the shape resolve_requirements
    (below) itself expects as its own "imports" argument.

    Factored out of compile_notebook_to_api, which used to assemble this
    same set/filter pass inline just before calling write_requirements --
    routes/upload.py's requirements-preview endpoint needs the identical
    "what does this notebook actually import" step to build a preview
    without writing anything to disk, and duplicating it inline a second
    time there would risk the exact kind of two-copies-silently-drift
    problem _third_party_dependencies' own docstring (backend/inspector.py)
    already describes happening once before, for a related but distinct
    computation.

    Also filters out anything _extract_excluded_imports (above) collects
    from the same `code_cells` -- applied here, in the one place both
    compile_notebook_to_api and the requirements-preview endpoint already
    share, rather than at each call site separately.
    """
    imports = set()

    for cell in code_cells:
        imports.update(extract_imports_from_code(cell))

    excluded = _extract_excluded_imports(code_cells)

    return [
        imp for imp in imports
        if imp not in STANDARD_LIBS and imp not in excluded
    ]


# Matches the leading package-name token of a requirement spec (PEP 508
# distribution name rules: letters/digits/"."/"_"/"-"), stopping at the
# first character that can only belong to a version specifier, extras
# bracket, or "@ <url>" direct-reference suffix -- see
# _explicit_requirement_package_name below for what this is for.
#
# The trailing lookahead is load-bearing, not cosmetic: without it, a bare
# VCS/URL spec with no "name @ " prefix at all (e.g.
# "git+https://example.com/pkg.git", one of the exact "no reliable way to
# name" examples this pattern's own docstring below already calls out)
# still matched -- "[A-Za-z0-9._-]*" has no reason to stop before the "+"
# any earlier than it does, so it happily captured "git" as though that
# were the actual package name. Confirmed: this pattern alone (with no
# lookahead) returned "git" for that exact input, silently
# mis-identifying an arbitrary, unrelated VCS spec as declaring a package
# literally named "git" -- the false "same package" match
# _explicit_requirement_package_name below exists to avoid, not manufacture.
# Anchoring the match to end only where PEP 508 actually allows a name to
# end (whitespace, "[", a version comparator, ";", "@", or end of string)
# makes this return None for that case instead, matching what the
# docstring already claims.
_REQUIREMENT_SPEC_NAME_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?=\s|\[|==|!=|<=|>=|~=|===|<|>|=|;|@|$)"
)


def _explicit_requirement_package_name(spec):
    """The package name `spec` (one "# notebook-to-api: requires <spec>"
    line, or an auto-pinned "name==version" one) declares, or None if
    `spec` doesn't start with one at all (e.g. a bare VCS URL with no
    "#egg=name" or "name @ " prefix, which this tool has no reliable way
    to name).

    Used by resolve_requirements below to tell whether an explicit
    directive and an auto-detected import are naming the *same* package,
    even though their own spec strings never match exactly (a pin can
    carry a version/extras/URL an auto-detected "name==version" line
    never would).
    """
    match = _REQUIREMENT_SPEC_NAME_PATTERN.match(spec)

    return match.group(1) if match else None


def resolve_requirements(imports, explicit_requirements=None):
    """The exact, sorted requirements.txt lines write_requirements (below)
    would write for `imports` (a notebook's own third-party imports, e.g.
    from extract_third_party_imports above) and `explicit_requirements`
    (see _extract_explicit_requirements) -- computed without writing
    anything to disk, so a caller can preview what a compile would
    produce there without actually running one.

    Factored out of write_requirements, which used to compute this same
    value immediately before writing it out; write_requirements below now
    just calls this and writes the result, so a preview built from this
    function can never drift from what an actual compile would produce.

    An auto-detected import whose own resolved distribution name matches
    an explicit requirement's own package name (see
    _explicit_requirement_package_name above) is dropped from the
    auto-pinned set entirely -- confirmed exploitable before this: a
    notebook importing a package directly (e.g. `import numpy`) while
    also declaring "# notebook-to-api: requires numpy==1.24.0" (to pin a
    specific version this tool's own auto-resolution wouldn't have
    chosen) got *both* lines written to requirements.txt side by side --
    the auto-detected "numpy==<installed-version>" and the explicit
    "numpy==1.24.0" -- two version-pinned lines for the same distribution,
    which pip treats as an unsatisfiable requirement and refuses outright
    (`pip install -r requirements.txt` failing with "Double requirement
    given"), breaking `deploy`'s own Docker build over exactly the kind
    of explicit override this directive exists to let a notebook author
    make. The explicit line always wins; nothing here validates that its
    own version/extras are otherwise sensible, the same "copied through
    verbatim, unvalidated" contract _extract_explicit_requirements'
    own docstring already establishes for it.
    """

    # watchdog is a dependency of this tool's own `serve` command (it
    # watches the notebook file for changes to trigger a hot recompile --
    # see backend/serve.py). The generated app itself never imports it
    # (see generator/api_generator.py); every compiled app was shipping it
    # as dead weight in requirements.txt and, from there, in its Docker
    # image.
    core_dependencies = [
        "fastapi",
        "uvicorn",
        "pydantic",
    ]

    final_deps = sorted(
        set(list(imports) + core_dependencies)
    )

    # Resolve each notebook import to the actual PyPI distribution name
    # that provides it before pinning -- see distribution_name_for_import's
    # own docstring for why this can't just pin the raw import name
    # directly. A set here (not a list) since two distinct import names
    # occasionally resolve to the same distribution (e.g. "attr" and
    # "attrs" are both provided by the "attrs" distribution), and this
    # must not write a duplicate requirements.txt line for it.
    distribution_names = sorted({
        distribution_name_for_import(dep) for dep in final_deps
    })

    explicit_requirements = explicit_requirements or []

    # Case-insensitive: PyPI distribution names are themselves
    # case-insensitive (pip normalizes "NumPy"/"numpy"/"nUmPy" to the
    # identical project), so a directive spelled differently than this
    # tool's own auto-resolved distribution_name_for_import result must
    # still be recognized as naming the same package.
    explicit_package_names = {
        name.lower()
        for name in (
            _explicit_requirement_package_name(spec)
            for spec in explicit_requirements
        )
        if name is not None
    }

    # Drops any auto-detected distribution an explicit directive already
    # names -- see this function's own docstring above for the conflicting-
    # pin this closes. Still deduplicated only by exact matching text
    # against *other* explicit requirements below, not by which PyPI
    # distribution a line actually refers to: this can't tell that
    # "opencv-python==4.9.0.80" and "opencv-python-headless==4.9.0.80"
    # ultimately provide the same `cv2` import, so declaring one doesn't
    # suppress the other if the notebook also imports it directly -- only
    # an auto-detected import colliding with an *explicit* requirement is
    # resolved in the explicit one's favor.
    pinned_deps = [
        _pinned_requirement(dep) for dep in distribution_names
        if dep.lower() not in explicit_package_names
    ]

    return sorted(set(pinned_deps) | set(explicit_requirements))


def write_requirements(imports, output_dir, explicit_requirements=None):

    requirements_path = os.path.join(
        output_dir,
        "requirements.txt"
    )

    all_deps = resolve_requirements(imports, explicit_requirements)

    with open(requirements_path, "w", encoding="utf-8") as f:
        for dep in all_deps:
            f.write(dep + "\n")

    print(
        f"requirements.txt generated with dependencies: {all_deps}"
    )


COMPILE_METADATA_FILENAME = ".compile_metadata.json"

# Serializes compile_notebook_to_api's multi-file writes to a given
# output_dir (see its own docstring below), and is also held by any
# dashboard route that reads that same directory's compiled output --
# POST /api/deploy's Docker build, POST /api/export-openapi's import of
# the compiled app, and GET /api/download's zip of it (see
# routes/upload.py) -- so none of them can observe a directory
# mid-write, torn between an old and a new compile. A plain
# threading.Lock, not a per-output_dir one: every dashboard process
# already funnels all of these through a single GENERATED_DIR, and the
# CLI's own compile/inspect/serve/deploy commands each run in their own
# process with nothing else to contend with, so one process-wide lock is
# exactly the right granularity without adding a locking scheme keyed by
# path.
COMPILE_LOCK = threading.Lock()


def hash_notebook_file(notebook_path):
    """SHA-256 of `notebook_path`'s raw bytes, as a hex digest.

    Used to detect when the notebook that produced the current
    `generated/` output has since been modified (see
    write_compile_metadata below and
    list_notebooks/_currently_compiled_notebook_metadata in
    routes/upload.py) -- a content hash survives a touch/copy/re-upload
    that doesn't actually change the bytes, unlike comparing mtimes,
    which would flag those as "changed" even though nothing meaningful
    did.
    """
    hasher = hashlib.sha256()

    with open(notebook_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


# Every file a real compile actually writes into output_dir, relative to
# it -- "app.py" itself is appended separately in
# _generated_files_sha256 below, since its own filename is a caller-
# supplied output_path, not a fixed literal the way these eight are.
# Deliberately a fixed list, not a generic directory walk: an operator's
# own unrelated file dropped into output_dir by hand (or a later POST
# /api/export-openapi/export-sdk's own openapi.json/sdk/, which a
# compile never produces and clear_stale_export_artifacts already treats
# as not part of "this compile's own output") must never affect this
# hash -- only these specific compile-produced artifacts should.
_GENERATED_OUTPUT_RELATIVE_PATHS = (
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",
    "kubernetes.yaml",
    ".env.example",
    "README.md",
    os.path.join("runtime", "notebook_module.py"),
)


def _generated_files_sha256(output_dir, app_filename="app.py"):
    """A single SHA-256 summarizing the exact set of files a real compile
    writes into `output_dir` -- the same "one filename:sha256 pair per
    file, sorted, combined into one hash" technique GET /api/download's
    own "X-Bundle-SHA256" header and GET /api/generated?checksums=true's
    own "bundle_sha256" (both backend/routes/upload.py) already use for
    an entire compiled bundle, just applied here to compiler.py's own
    fixed, known-at-compile-time file set (see
    _GENERATED_OUTPUT_RELATIVE_PATHS above) instead of a live directory
    walk -- reused (not duplicated) by write_compile_metadata below to
    record a baseline at compile time, and by GET /api/generated
    (routes/upload.py) to recompute the same hash later and compare
    against it.

    A file this project's own compile pipeline hasn't written yet (or
    that's since been deleted by hand) is silently skipped rather than
    raising -- this must never be the reason a real compile fails, since
    write_compile_metadata's own caller has already committed to
    treating this compile as successful by the time it calls this (see
    write_compile_metadata's own docstring for the "no exception past
    this point" discipline compile_notebook_to_api already follows). A
    missing file still changes the resulting hash from a compile that
    did produce it, so a deleted artifact is still detected as a change,
    just not as a hard failure.

    `app_filename` defaults to "app.py" -- every real caller compiles to
    "<output_dir>/app.py" (see compile_notebook's own docstring); passed
    explicitly rather than hardcoded so a caller computing this against
    an unusual output_path can still get an accurate hash.
    """
    hasher = hashlib.sha256()

    relative_paths = sorted(_GENERATED_OUTPUT_RELATIVE_PATHS + (app_filename,))

    for relative_path in relative_paths:

        file_path = Path(output_dir) / relative_path

        if not file_path.is_file():
            continue

        hasher.update(
            f"{relative_path}:{hash_notebook_file(str(file_path))}\n".encode("utf-8")
        )

    return hasher.hexdigest()


def write_compile_metadata(
    notebook_path, output_dir, content_path=None, version_id=None,
    app_filename="app.py",
):
    """Record which notebook produced the app in `output_dir`, its
    content hash at that moment, when, and -- if this compile came from
    one of that notebook's own previously snapshotted versions rather
    than its current content -- which one.

    "generated_files_sha256" (added alongside this same docstring's
    original three fields, not a separate change) is
    _generated_files_sha256(output_dir, app_filename) above -- a
    baseline hash over the exact compile-produced files themselves
    (app.py, requirements.txt, Dockerfile, .dockerignore,
    docker-compose.yml, .env.example, README.md, the runtime module), taken at the
    very end of a successful compile, once every one of those has
    actually been written. Every hash/staleness check this dashboard
    already had
    compared the *source notebook's* own content against what's
    currently compiled (see "source_notebook_sha256" below) -- but
    nothing recorded whether the compiled *output itself* still matches
    what that compile actually produced. A generated/app.py hand-edited
    directly on the server after compiling (to patch something in a
    hurry, say) previously left every metadata-driven consumer -- GET
    /api/notebooks, GET /api/generated, POST /api/deploy's own staleness
    check -- confidently reporting the compile as fully up to date with
    its source notebook, with no way to tell the *served* code had
    silently diverged from what that notebook actually compiles to.
    `app_filename` is threaded through from compile_notebook_to_api's
    own `output_path` (its basename) rather than assumed, so this stays
    accurate even for a caller compiling to something other than the
    "<output_dir>/app.py" every documented caller already uses.

    "compiled_version_id" (None for an ordinary compile of a notebook's
    own current content, the overwhelmingly common case) is the same
    "version_id" POST /api/compile's own response and compile-history
    entries already report for a version-pinned compile (see POST
    /api/compile's own docstring, backend/routes/upload.py) -- but until
    now that was only ever visible in that one request/response or a
    compile-history entry, never persisted alongside the compile it
    actually describes. An operator investigating "what is actually
    running right now" -- after the fact, possibly a long time later, or
    from a different process entirely (GET /api/notebooks, GET
    /api/generated, ...) -- had no way to tell a version-pinned compile
    apart from an ordinary one short of cross-referencing compile
    history by "source_notebook_sha256" and hoping nothing else ever
    produced the same hash.

    `content_path` (optional) is the file whose *bytes* actually got
    compiled and should be hashed -- `notebook_path` is always what gets
    recorded as "source_notebook" and stays the real, currently-uploaded
    notebook's own path. They differ only when compiling one of a
    notebook's own previously snapshotted versions (see POST /api/compile's
    own "version_id", backend/routes/upload.py): "source_notebook" still
    names the real notebook (so every "currently_compiled"/staleness check
    already keyed on that path -- _currently_compiled_notebook_metadata
    and friends -- keeps recognizing it), while the hash reflects the
    version's own content, not the notebook's current one. That mismatch
    is exactly the correct, honest outcome: the notebook's current content
    genuinely differs from what's actually served, so
    "notebook_changed_since_compile" correctly reports true. Defaults to
    `notebook_path` -- every existing caller compiling a notebook's own
    current content keeps hashing exactly what it already did.

    Without this, nothing -- on disk or via the API -- recorded which
    notebook a given `generated/` output actually came from. GET
    /api/notebooks could list every uploaded notebook, but had no way to
    say "this is the one currently reflected in generated/", so a
    dashboard frontend had to track that itself client-side: fragile
    (lost on refresh) and wrong the moment a second compile happens
    (from this browser tab or another) without it finding out.

    The content hash closes a related gap: even once "this is the
    currently-compiled notebook" was known, there was still no way to
    tell whether that notebook had since been edited and re-uploaded
    (e.g. via /api/upload?overwrite=true) *after* the compile that
    produced the current generated/ output -- silently leaving the
    served app stale relative to the notebook a caller might think it
    still matches.

    notebook_path is stored as an absolute path so it can be compared
    directly against a resolved upload path later (see
    list_notebooks/_currently_compiled_notebook_metadata in
    routes/upload.py) regardless of whether it was originally relative.
    """
    metadata_path = os.path.join(output_dir, COMPILE_METADATA_FILENAME)

    metadata = {
        "source_notebook": os.path.abspath(notebook_path),
        "source_notebook_sha256": hash_notebook_file(content_path or notebook_path),
        "compiled_at": (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        ),
        "compiled_version_id": version_id,
        "generated_files_sha256": _generated_files_sha256(output_dir, app_filename),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def update_compile_metadata_source_notebook(output_dir, new_source_notebook_path):
    """Update the "source_notebook" field recorded in `output_dir`'s
    .compile_metadata.json to `new_source_notebook_path`, leaving its
    "source_notebook_sha256" and "compiled_at" fields untouched.

    Used by PATCH /api/notebooks/{filename} (backend/routes/upload.py)
    when the notebook being renamed is the one that produced the app
    currently in output_dir. Renaming a file on disk doesn't change its
    bytes, so the recorded sha256 is still accurate -- but
    write_compile_metadata's "source_notebook" field is an absolute path
    baked in at compile time, and every "currently_compiled"/staleness
    check this dashboard makes
    (_currently_compiled_notebook_metadata/_currently_compiled_notebook_is_stale
    in routes/upload.py) resolves that exact path. Left untouched by a
    rename, it would keep pointing at a path that no longer exists,
    silently and permanently orphaning the currently-compiled app from
    its own source notebook -- indistinguishable, from the API's
    perspective, from that notebook having been deleted outright.

    A no-op (returns False) if output_dir has no .compile_metadata.json
    yet, or if it's missing/unreadable/corrupt -- nothing to update in
    that case, mirroring _currently_compiled_notebook_metadata's own
    best-effort handling of the same file. Callers are expected to hold
    COMPILE_LOCK, the same way every other read/write of this file
    already does.
    """
    metadata_path = os.path.join(output_dir, COMPILE_METADATA_FILENAME)

    if not os.path.isfile(metadata_path):
        return False

    try:

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    except (OSError, ValueError):
        return False

    metadata["source_notebook"] = os.path.abspath(new_source_notebook_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return True


def clear_stale_export_artifacts(output_dir):
    """Remove a previous compile's exported openapi.json/openapi.yaml and
    sdk/ directory from `output_dir`, if present.

    POST /api/export-openapi and POST /api/export-sdk (and the CLI's
    export-openapi/export-sdk commands, by default -- see 354bb6c) always
    write into output_dir alongside the compiled app itself. Recompiling
    a notebook overwrites app.py, the runtime module, requirements.txt,
    and the Dockerfile -- but previously left any openapi.json/
    openapi.yaml/sdk/ already sitting in output_dir completely untouched,
    silently describing the *previous* compile's endpoints instead.
    Confirmed: compiling a notebook exposing `add`, exporting its schema,
    then recompiling to expose `multiply` instead (without re-exporting)
    left openapi.json still listing "/add" and omitting "/multiply"
    entirely -- exactly the mismatch GET /api/download's zip and GET
    /api/generated/openapi.json would then hand a caller alongside the
    freshly compiled app.py that no longer matches either of them.

    Only ever called from the successful-compile path in
    compile_notebook_to_api below (after generate_fastapi_code has
    already succeeded), never on a failed compile -- a notebook that
    fails to compile must leave a previous good compile's exports
    untouched too, for the same reason it already leaves app.py,
    requirements.txt, and everything else untouched (see the
    generate_fastapi_code comment below).
    """
    output_path = Path(output_dir)

    for stale_export_file in ("openapi.json", "openapi.yaml"):
        (output_path / stale_export_file).unlink(missing_ok=True)

    stale_sdk_dir = output_path / "sdk"

    if stale_sdk_dir.is_dir():
        shutil.rmtree(stale_sdk_dir)


def _filter_functions_by_name(functions, only, exclude):
    """Restrict `functions` (already deduplicated by name) to the ones a
    caller actually wants compiled into endpoints, via at most one of
    `only` or `exclude` (each an iterable of function names, or falsy for
    "no filtering" -- the default).

    Every module-level function a notebook defines (barring a *args/
    **kwargs one -- see extract_functions_from_code) becomes a public API
    endpoint, with no previous way to opt any of them out: a notebook's
    own internal helper functions (data loading, validation, formatting,
    ...) were compiled into endpoints exactly like the functions actually
    meant to be part of the API's public surface. Renaming a helper
    doesn't help either -- extract_functions_from_code applies no
    leading-underscore (or any other) naming convention, so it's
    extracted and exposed identically to every other top-level function.

    `only` keeps just the named functions; `exclude` keeps every function
    except the named ones. Either way, this only changes what
    generate_fastapi_code (below) turns into endpoints -- write_runtime_module
    still writes *every* cell's code, filtered or not, so an excluded
    helper function stays fully callable from whichever exposed function
    actually calls it; it just doesn't get its own endpoint.

    Raises ValueError naming the unrecognized function(s) if `only`/
    `exclude` names one this notebook doesn't actually define -- the same
    "fail loudly on a name that doesn't do what the caller thinks"
    precedent ReservedFunctionNameError already sets for a different kind
    of name problem (generator/api_generator.py), rather than silently
    compiling as if a typo'd name had simply never been requested at all.
    """
    if only and exclude:
        raise ValueError(
            "only and exclude can't both be given -- choose one."
        )

    if not only and not exclude:
        return functions

    known_names = {func["name"] for func in functions}
    requested_names = set(only) if only else set(exclude)

    unknown_names = sorted(requested_names - known_names)

    if unknown_names:
        raise ValueError(
            f"{'only' if only else 'exclude'} names function(s) not defined "
            f"in this notebook: {', '.join(unknown_names)}. Available: "
            f"{', '.join(sorted(known_names)) or '(none)'}"
        )

    if only:
        return [func for func in functions if func["name"] in requested_names]

    return [func for func in functions if func["name"] not in requested_names]


def compile_notebook_to_api(
    notebook_path,
    output_path,
    only=None,
    exclude=None,
    source_notebook_path=None,
    version_id=None,
):

    # compile_notebook_to_api writes several files to output_dir
    # (runtime module, requirements.txt, app.py, Dockerfile,
    # .dockerignore, .compile_metadata.json) one after another with no
    # atomicity across the set. Every dashboard route that can trigger
    # this -- POST /api/compile chief among them -- is declared a plain
    # `def`, not `async def` (see routes/upload.py), specifically so a
    # slow compile runs in FastAPI's worker threadpool instead of
    # blocking the single event loop -- but that means two overlapping
    # POST /api/compile calls (two browser tabs, a retry racing the
    # original request, ...) can now genuinely execute this function in
    # two different threads at once, both writing into the *same*
    # GENERATED_DIR. Without serializing them, their writes interleave:
    # output_dir can end up with, say, one notebook's runtime module
    # alongside a different notebook's app.py, expecting functions the
    # runtime module doesn't define -- a corrupted, mismatched compile
    # output neither request actually produced on its own, with nothing
    # to indicate it happened.
    with COMPILE_LOCK:

        print(f"Starting compilation for: {notebook_path}")

        output_dir = os.path.dirname(output_path)

        os.makedirs(output_dir, exist_ok=True)

        package_name = package_name_for_output_dir(output_dir)

        notebook = load_notebook(notebook_path)

        code_cells = [
            cell for cell in extract_code_cells(notebook)
            if is_parseable_python(cell)
        ]

        # Computed here -- and let it raise (a ValueError, for two
        # conflicting "# notebook-to-api: requires" directives naming the
        # same package -- see its own docstring) -- before writing
        # anything to output_dir at all, the same "validate the notebook's
        # own content before any write" reasoning generate_fastapi_code's
        # own ReservedFunctionNameError check below already established.
        # Reused at the write_requirements call further down instead of
        # calling this a second time there, so a conflict is caught this
        # early rather than only surfacing after write_runtime_module has
        # already overwritten a previous successful compile's own runtime
        # module with this failing notebook's code -- the identical
        # inconsistent-output_dir failure mode that comment already
        # documents fixing for generate_fastapi_code's own checks.
        explicit_requirements = _extract_explicit_requirements(code_cells)

        functions = []

        for cell in code_cells:

            funcs = extract_functions_from_code(cell)

            functions.extend(funcs)

        functions = deduplicate_functions_by_name(functions)

        # Dropped before _filter_functions_by_name's own only/exclude
        # handling, so a function the notebook itself marks
        # "# notebook-to-api: private" never becomes an endpoint
        # regardless of --only/--exclude -- see _drop_private_functions'
        # own docstring for why this needs code_cells (not just the
        # already-extracted `functions` list) and can't simply live
        # inside _filter_functions_by_name itself.
        functions, exclude = _drop_private_functions(
            functions, code_cells, only, exclude
        )

        # Applied before generate_fastapi_code, and before the reserved-
        # name-collision check it performs, so an --only/--exclude typo is
        # reported as its own clear error rather than silently changing
        # which (if any) reserved-name collision generate_fastapi_code
        # happens to hit first.
        functions = _filter_functions_by_name(functions, only, exclude)

        # Generate the API code -- and let it raise (e.g.
        # ReservedFunctionNameError, generator/api_generator.py, for a
        # function name that collides with an identifier the generated app
        # itself defines) -- before writing anything to output_dir at all.
        # This used to run after write_runtime_module/write_requirements
        # below, so a notebook that failed this check still overwrote the
        # runtime module and requirements.txt from a previous successful
        # compile with content generated from the *failing* notebook, while
        # app.py, the Dockerfile, and .compile_metadata.json were left
        # untouched from that previous compile -- leaving output_dir in an
        # inconsistent state that matched neither the old nor the new
        # notebook (confirmed: recompiling a working app with a notebook that
        # has one reserved-name collision left its runtime module rewritten
        # to the broken notebook's code while app.py and
        # .compile_metadata.json still described the last working one).
        # Baked into the generated app itself as SOURCE_NOTEBOOK_SHA256,
        # returned by its own GET /info -- see generate_fastapi_code's own
        # docstring for why. Hashed from notebook_path (the exact content
        # actually being compiled here -- a version snapshot's own path
        # when this compile is pinned to one, not necessarily the
        # notebook's current content) so a running deployed container can
        # always be traced back to precisely what produced it, the same
        # content identity every other sha256 already tracked in this
        # project (deploy/compile history, GET /api/notebooks?sha256=,
        # GET /api/notebooks/duplicates) already uses.
        api_code = generate_fastapi_code(
            functions, package_name,
            source_notebook_sha256=hash_notebook_file(notebook_path),
            notebook_to_api_version=NOTEBOOK_TO_API_VERSION,
        )

        # generate_fastapi_code succeeding means this compile is now
        # guaranteed to go through -- from here on the only failures left
        # are I/O errors, not this notebook's own content -- so it's now
        # safe to clear out any openapi.json/openapi.yaml/sdk/ left over
        # from a previous compile's export (see its own docstring for why
        # this can't just be left alone).
        clear_stale_export_artifacts(output_dir)

        write_runtime_module(code_cells, output_dir)

        write_requirements(
            extract_third_party_imports(code_cells),
            output_dir,
            explicit_requirements=explicit_requirements
        )

        write_generated_api(
            api_code,
            output_path
        )

        # From here on, app.py (and its runtime module) already reflect
        # *this* compile's notebook -- if anything below raises
        # (generate_dockerfile/generate_dockerignore, or write_compile_metadata
        # itself), app.py has already moved on to the new notebook while
        # .compile_metadata.json, left untouched, would still describe
        # whichever notebook the *previous* successful compile actually
        # wrote it for. Confirmed reproduced: recompiling a working "add"
        # app with a notebook exposing "multiply" while generate_dockerfile
        # was made to raise (standing in for a real disk/permission
        # failure) left app.py and its runtime module already serving
        # multiply, while .compile_metadata.json's "source_notebook" still
        # pointed at the notebook that produced the *previous* compile --
        # silently wrong, not merely stale: every metadata-driven consumer
        # (GET /api/notebooks' "currently_compiled"/"compiled_at"/
        # "notebook_changed_since_compile", GET /api/generated's
        # "source_notebook_filename"/"source_notebook_exists") would
        # confidently report the wrong notebook as the one actually being
        # served, with nothing to indicate the mismatch. Removing
        # .compile_metadata.json here instead turns that into a correctly-
        # reported "unknown" state -- _currently_compiled_notebook_metadata
        # (routes/upload.py) already treats a missing metadata file exactly
        # this way (returns (None, None, None)), the same graceful
        # degradation it already applies when nothing has ever been
        # compiled at all. This also cleans up a metadata file
        # write_compile_metadata itself might have left partially written
        # (a mid-json.dump I/O failure), not just one from the two steps
        # before it.
        try:

            dockerfile_path = os.path.join(
                output_dir,
                "Dockerfile"
            )

            generate_dockerfile(
                dockerfile_path, package_name, compiling_python_version()
            )

            dockerignore_path = os.path.join(
                output_dir,
                ".dockerignore"
            )

            generate_dockerignore(dockerignore_path)

            docker_compose_path = os.path.join(
                output_dir,
                "docker-compose.yml"
            )

            generate_docker_compose(
                docker_compose_path, package_name, GENERATED_APP_ENV_VARS
            )

            env_example_path = os.path.join(
                output_dir,
                ".env.example"
            )

            generate_env_example(env_example_path, GENERATED_APP_ENV_VARS)

            kubernetes_manifest_path = os.path.join(
                output_dir,
                "kubernetes.yaml"
            )

            generate_kubernetes_manifest(
                kubernetes_manifest_path, package_name, GENERATED_APP_ENV_VARS
            )

            readme_path = os.path.join(
                output_dir,
                "README.md"
            )

            generate_readme(
                readme_path, package_name, functions, GENERATED_APP_ENV_VARS
            )

            write_compile_metadata(
                source_notebook_path or notebook_path,
                output_dir,
                content_path=notebook_path,
                version_id=version_id,
                app_filename=os.path.basename(output_path),
            )

        except Exception:

            metadata_path = os.path.join(output_dir, COMPILE_METADATA_FILENAME)

            if os.path.isfile(metadata_path):
                os.remove(metadata_path)

            raise

        print(
            f"Successfully generated FastAPI app at: {output_path}"
        )


def compile_notebook(
    notebook_path,
    output_dir,
    only=None,
    exclude=None,
    source_notebook_path=None,
    version_id=None,
):
    """
    Convenient wrapper for CLI.
    Generates the FastAPI app at <output_dir>/app.py.

    `only`/`exclude` are passed straight through to
    compile_notebook_to_api -- see _filter_functions_by_name's own
    docstring for what they do and why. `source_notebook_path`/
    `version_id` are passed straight through too -- see
    write_compile_metadata's own docstring.
    """

    output_path = os.path.join(
        output_dir,
        "app.py"
    )

    compile_notebook_to_api(
        notebook_path,
        output_path,
        only=only,
        exclude=exclude,
        source_notebook_path=source_notebook_path,
        version_id=version_id,
    )