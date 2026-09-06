import nbformat
import os
import re
import sys

# Ensure backend directory is in sys.path for robust imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backend.parser.ast_parser import extract_functions_from_code
except ImportError:
    from ast_parser import extract_functions_from_code

# IPython line magics (%foo), cell magics (%%foo) and shell escapes (!foo)
# are not valid Python syntax and are not preceded by executed code, so any
# line starting with them (ignoring leading whitespace) can be safely
# commented out.
_MAGIC_LINE_RE = re.compile(r"^(\s*)(%{1,2}|!)(?!=)")

# IPython's "dynamic object introspection" syntax -- ``obj?``/``obj??`` for
# an object's docstring/source, or the equivalent prefix form ``?obj``/
# ``??obj`` -- is just as common in real notebooks (typed while exploring,
# then left in a cell) and just as invalid as Python syntax, but wasn't
# covered by _MAGIC_LINE_RE above. `?` never appears in valid Python syntax
# outside of a string literal, so a line that consists of *only* an
# attribute-chain expression (optionally called) plus a leading/trailing
# "?"/"??" is unambiguously an introspection query, not code -- as opposed
# to matching any line merely containing a "?" (which would wrongly also
# match, and corrupt, plain code like `msg = "wait?"`).
_INTROSPECTION_PREFIX_RE = re.compile(
    r"^(\s*)\?{1,2}\s*[A-Za-z_][A-Za-z0-9_.]*(\(\))?\s*$"
)
_INTROSPECTION_SUFFIX_RE = re.compile(
    r"^(\s*)[A-Za-z_][A-Za-z0-9_.]*(\(\))?\s*\?{1,2}\s*$"
)

# Cell magics whose own *body* -- everything after the "%%name ..." line
# itself -- is never executed as Python in the notebook's own namespace at
# all: %%writefile writes it to a file instead of running it; %%bash/%%sh/
# %%perl/%%ruby/%%script run it as a *different* language via a subprocess;
# %%html/%%HTML/%%javascript/%%js/%%latex/%%svg/%%markdown render it as
# non-Python content. Contrast with %%time/%%timeit/%%capture/%%prun/
# %%debug, deliberately excluded here -- each of those *does* execute its
# own body as ordinary Python in the notebook's own namespace (timing it,
# capturing its output, profiling it, ...), so a function defined inside
# one of those really is callable from a later cell in the real kernel,
# unlike every magic name listed below.
NON_PYTHON_BODY_CELL_MAGICS = frozenset({
    "writefile", "bash", "sh", "perl", "ruby", "script",
    "html", "HTML", "javascript", "js", "latex", "svg", "markdown",
})

_NON_PYTHON_BODY_CELL_MAGIC_RE = re.compile(
    r"^\s*%%(" + "|".join(re.escape(name) for name in NON_PYTHON_BODY_CELL_MAGICS)
    + r")\b"
)


def detect_non_python_body_cell_magic(source):
    """The cell magic name (e.g. "writefile") if `source`'s own first
    non-blank line invokes one of NON_PYTHON_BODY_CELL_MAGICS, else None.

    Only the first non-blank line is checked -- a real Jupyter cell magic
    must be the cell's very first statement, so a "%%writefile" appearing
    later in the cell (inside a string, a comment, or simply a syntax
    error) is not one at all.
    """
    for line in source.split("\n"):

        if not line.strip():
            continue

        match = _NON_PYTHON_BODY_CELL_MAGIC_RE.match(line)

        return match.group(1) if match else None

    return None


def strip_magic_commands(source):
    """Comment out IPython magics, shell escapes, and object-introspection
    queries in notebook source.

    Real-world notebooks routinely contain lines like ``%matplotlib inline``,
    ``%%time``, ``!pip install pandas``, or ``train_model?`` (inline help/
    source lookup, via IPython's ``?``/``??`` operator). None of these are
    valid Python, so feeding a cell's raw source straight into ``ast.parse``
    (or writing it verbatim into the generated runtime module) blows up on
    almost any notebook exported from Jupyter -- and since ``ast.parse``
    parses a cell as a single unit, *any one* such line anywhere in the cell
    fails the whole cell, silently dropping every function it defines along
    with it (see is_parseable_python in ast_parser.py). Commenting the
    offending lines out instead keeps line numbers stable and preserves the
    rest of the cell as executable Python.

    A cell opening with a NON_PYTHON_BODY_CELL_MAGICS magic (see
    detect_non_python_body_cell_magic above) is handled differently: the
    *entire* cell is commented out, not just that first line. Confirmed
    exploitable before this: a "%%writefile helper.py" cell whose body
    happened to be syntactically valid Python (the overwhelmingly common
    real-world case -- %%writefile is routinely used to scaffold a .py
    module from inside a notebook) previously had only its own
    "%%writefile helper.py" line commented out, leaving the rest of the
    cell -- e.g. a `def greet(name): ...` -- completely untouched. That
    function is never actually defined in the *notebook's own* namespace
    in a real kernel at all (it's written to helper.py, never imported or
    executed there); calling it from a later cell in the real notebook
    raises NameError. This tool instead silently compiled and exposed it
    as a real, working POST /greet endpoint -- a fidelity gap between what
    the source notebook actually does and what got served, with no
    warning anywhere.
    """
    if detect_non_python_body_cell_magic(source) is not None:
        return "\n".join(
            f"# {line}" if line.strip() else line
            for line in source.split("\n")
        )

    cleaned_lines = []

    for line in source.split("\n"):
        match = (
            _MAGIC_LINE_RE.match(line)
            or _INTROSPECTION_PREFIX_RE.match(line)
            or _INTROSPECTION_SUFFIX_RE.match(line)
        )

        if match:
            indent = match.group(1)
            cleaned_lines.append(f"{indent}# {line.strip()}")
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def notebook_kernel_language(notebook):
    """The notebook's own declared kernel language (e.g. "python", "R",
    "julia"), lowercased, or None if it doesn't declare one at all.

    Checked in the same two places a real Jupyter frontend does, in the
    same order of authority: "kernelspec.language" (nbformat's own
    documented field for exactly this -- present on essentially every
    notebook a real Jupyter frontend ever writes) first, falling back to
    "language_info.name" (also standard, and sometimes present even when
    kernelspec.language is missing or a custom kernel name doesn't
    itself say much, e.g. "language": "R" under a kernelspec named
    "ir"). Both are optional per the nbformat spec, and a hand-built or
    stripped-down notebook -- most of this project's own test fixtures
    among them -- commonly omits both metadata blocks entirely; this
    returns None rather than guessing in that case, since there's no
    honest way to tell "wasn't recorded" apart from "is Python" in the
    metadata's own absence.
    """
    metadata = notebook.get("metadata") or {}

    kernelspec = metadata.get("kernelspec") or {}
    language = kernelspec.get("language")

    if not language:

        language_info = metadata.get("language_info") or {}
        language = language_info.get("name")

    return language.strip().lower() if language else None


def load_notebook(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # A real Jupyter notebook can carry any kernel at all -- R, Julia,
    # Scala, ... -- but this tool only ever extracts *Python* functions
    # from a cell's source (extract_functions_from_code parses it with
    # ast.parse) and only ever runs them as Python inside the generated
    # runtime module. Before this, uploading/compiling a genuinely
    # non-Python notebook wasn't rejected anywhere: every one of its code
    # cells simply failed is_parseable_python (a SyntaxError from
    # ast.parse on, say, R's own `f <- function(x) x + 1` syntax) and was
    # silently dropped -- compile_notebook_to_api "succeeded" with zero
    # extracted functions, producing a working-but-endpoint-less API with
    # nothing anywhere explaining *why* it exposed nothing. Raising here
    # instead -- a plain ValueError, already part of every call site's own
    # MALFORMED_NOTEBOOK_ERRORS/CLI_USER_FACING_ERRORS handling (see
    # routes/upload.py and cli.py), so this needs no new exception
    # handling of its own anywhere -- reports the real, specific reason up
    # front, at the exact same "validate before doing anything else" point
    # this project's own MALFORMED_NOTEBOOK_ERRORS docstring already
    # establishes for a malformed (not-even-valid-JSON) notebook.
    #
    # Silently permissive (no language declared at all) rather than
    # rejecting: a notebook's own kernelspec/language_info are both
    # optional per the nbformat spec, and assuming Python in their
    # absence preserves this function's previous behavior exactly for
    # every notebook that doesn't declare a language either way --
    # including, not incidentally, most of this project's own test
    # fixtures (nbformat.v4.new_notebook() sets neither by default).
    language = notebook_kernel_language(notebook)

    if language and language not in ("python", "python3", "python2"):

        raise ValueError(
            f"This notebook's kernel language is '{language}', not "
            "Python -- notebook-to-api only compiles Python notebooks "
            "into an API."
        )

    return notebook


def extract_code_cells(notebook):
    code_cells = []

    for cell in notebook.cells:
        if cell.cell_type == "code":
            code_cells.append(strip_magic_commands(cell.source))

    return code_cells


def extract_functions_from_notebook(notebook_path):
    notebook = load_notebook(notebook_path)
    code_cells = extract_code_cells(notebook)
    
    all_functions = []
    for code in code_cells:
        funcs = extract_functions_from_code(code)
        all_functions.extend(funcs)
    return all_functions


if __name__ == "__main__":
    # Resolve the path to sample.ipynb relative to this script to make it run from anywhere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(script_dir, "../../notebooks/sample.ipynb")

    print(f"Loading notebook from: {os.path.abspath(sample_path)}")
    notebook = load_notebook(sample_path)
    code_cells = extract_code_cells(notebook)

    for idx, code in enumerate(code_cells):
        print(f"\n--- CODE CELL {idx + 1} ---\n")
        print(code)

    print("\n--- EXTRACTED FUNCTIONS ---")
    funcs = extract_functions_from_notebook(sample_path)
    for func in funcs:
        print(func)

