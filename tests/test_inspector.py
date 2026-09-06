import json

import nbformat
import pytest

from backend.inspector import (
    DEFAULT_DEV_API_KEY,
    classify_notebook_diff,
    diff_notebook_functions,
    diff_notebook_source,
    generate_curl_commands,
    generate_postman_collection,
    inspect_notebook,
    inspect_notebook_data,
    print_compile_summary,
    print_notebook_diff,
    _aggregate_skipped_functions,
    _duplicate_function_names,
    _functions_without_docstrings,
    _list_generated_files,
    _reserved_name_conflicts,
)
from backend.generator.api_generator import RESERVED_INFRASTRUCTURE_NAMES


def _write_notebook(path, source):
    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(nbformat.v4.new_code_cell(source))
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)


def test_reserved_name_conflicts_flags_a_colliding_function_name():

    functions = [
        {"name": "health_check"},
        {"name": "add"},
    ]

    assert _reserved_name_conflicts(functions) == ["health_check"]


def test_reserved_name_conflicts_is_empty_when_nothing_collides():

    functions = [
        {"name": "add"},
        {"name": "subtract"},
    ]

    assert _reserved_name_conflicts(functions) == []


def test_reserved_name_conflicts_is_sorted_and_deduplicated():

    functions = [
        {"name": "verify_api_key"},
        {"name": "app"},
        {"name": "add"},
    ]

    assert _reserved_name_conflicts(functions) == sorted(
        {"verify_api_key", "app"}
    )


def test_inspect_notebook_data_reports_no_conflicts_for_a_clean_notebook(tmp_path):

    notebook_path = tmp_path / "clean.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["reserved_name_conflicts"] == []


def test_inspect_notebook_data_flags_a_reserved_name_before_compile_would(tmp_path):
    """Before this fix, /api/inspect and `inspect` had no idea
    generate_fastapi_code (backend/generator/api_generator.py) would
    later refuse to compile a function named "health_check" -- the first
    signal a notebook author got was a compile failure, not a preview.
    """

    notebook_path = tmp_path / "colliding.ipynb"
    _write_notebook(
        notebook_path,
        "def health_check() -> dict:\n    return {}\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["reserved_name_conflicts"] == ["health_check"]
    # The clean function must still be reported normally alongside it.
    assert {f["name"] for f in data["functions"]} == {"health_check", "add"}


def test_inspect_notebook_data_flags_every_reserved_infrastructure_name(tmp_path):
    """Exercises the full RESERVED_INFRASTRUCTURE_NAMES set (not just one
    example), so a future addition to that set is automatically covered
    by inspect too, without this test needing to be updated.
    """

    source = "\n\n".join(
        f"def {name}(): pass" for name in sorted(RESERVED_INFRASTRUCTURE_NAMES)
    )

    notebook_path = tmp_path / "all_reserved.ipynb"
    _write_notebook(notebook_path, source)

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert set(data["reserved_name_conflicts"]) == RESERVED_INFRASTRUCTURE_NAMES


def test_inspect_notebook_prints_a_reserved_name_conflict_warning(tmp_path, capsys):

    notebook_path = tmp_path / "colliding.ipynb"
    _write_notebook(
        notebook_path,
        "def verify_api_key() -> dict:\n    return {}\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Reserved Name Conflicts" in output
    assert "verify_api_key" in output


def test_inspect_notebook_omits_the_warning_section_for_a_clean_notebook(
    tmp_path, capsys
):

    notebook_path = tmp_path / "clean.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Reserved Name Conflicts" not in output


def test_inspect_notebook_prints_a_functions_own_docstring(tmp_path, capsys):
    """A notebook function's own docstring already becomes its compiled
    endpoint's OpenAPI description (see api_generator.py) -- but
    `inspect`, this tool's own "preview what compiling this notebook will
    do" report, never showed it at all, even though inspect_notebook_data
    (and `inspect --json`) already carried it.
    """

    notebook_path = tmp_path / "documented.ipynb"
    _write_notebook(
        notebook_path,
        (
            "def train_model(epochs: int) -> str:\n"
            '    """Train the classifier for the given number of epochs.\n\n'
            "    Returns a short accuracy summary.\n"
            '    """\n'
            "    return 'done'\n"
        ),
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Train the classifier for the given number of epochs." in output
    assert "Returns a short accuracy summary." in output


def test_inspect_notebook_omits_docstring_lines_for_an_undocumented_function(
    tmp_path, capsys
):

    notebook_path = tmp_path / "undocumented.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "1. add(a: int, b: int) -> int" in output
    # No stray docstring lines (blank or otherwise) inserted between the
    # route line and the example payload for a function with no
    # docstring at all.
    assert "   Route: POST /add\n   Example Payload:" in output


def test_inspect_notebook_data_reports_endpoints_and_flags_background_ones(tmp_path):
    """Before this fix, inspect_notebook_data -- the data behind both
    `inspect --json` and POST /api/inspect -- had no way to tell a caller
    which functions would compile into background/task_id-based endpoints
    vs synchronous ones. That distinction was only ever visible *after*
    compiling, via print_compile_summary or POST /api/compile's
    "endpoints" field.
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def train_model(epochs: int) -> str:\n    return 'done'\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    endpoints = {e["path"]: e for e in data["endpoints"]}

    assert endpoints["/add"] == {"path": "/add", "method": "POST", "is_async": False}
    assert endpoints["/train_model"] == {
        "path": "/train_model", "method": "POST", "is_async": True
    }


def test_inspect_notebook_data_endpoints_is_empty_for_a_notebook_with_no_functions(
    tmp_path
):

    notebook_path = tmp_path / "empty.ipynb"
    _write_notebook(notebook_path, "x = 1\n")

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["endpoints"] == []


def test_list_generated_files_excludes_pycache_directories(tmp_path):

    output_dir = tmp_path / "generated"
    (output_dir / "__pycache__").mkdir(parents=True)
    (output_dir / "__pycache__" / "app.cpython-314.pyc").write_bytes(b"\x00")
    (output_dir / "app.py").write_text("# app\n")

    assert sorted(_list_generated_files(output_dir)) == ["app.py"]


def test_list_generated_files_returns_an_empty_list_when_the_directory_does_not_exist(
    tmp_path,
):

    assert _list_generated_files(tmp_path / "does_not_exist") == []


def test_list_generated_files_excludes_compile_metadata(tmp_path):
    """.compile_metadata.json (write_compile_metadata, backend/compiler.py)
    is dashboard-internal bookkeeping -- read only by
    list_notebooks/_currently_compiled_notebook_metadata in
    routes/upload.py -- never a real compiled deliverable, and its
    "source_notebook" field is the source notebook's absolute filesystem
    path on the compiling server. Before this fix, it showed up in
    generated_files exactly like a real output file, from where it also
    flowed into GET /api/download's zip, GET /api/generated/{filename}'s
    preview, and -- worst of all -- the deployed Docker image itself.
    """

    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    (output_dir / "app.py").write_text("# app\n")
    (output_dir / ".compile_metadata.json").write_text(
        '{"source_notebook": "/home/someuser/private/nb.ipynb"}\n'
    )

    assert sorted(_list_generated_files(output_dir)) == ["app.py"]


def test_inspect_notebook_prints_generated_files_but_excludes_pycache(
    tmp_path, capsys
):

    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    (output_dir / "app.py").write_text("# app\n")

    pycache_dir = output_dir / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "app.cpython-314.pyc").write_bytes(b"\x00")

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    inspect_notebook(str(notebook_path), str(output_dir))

    output = capsys.readouterr().out
    assert "- app.py" in output
    assert "__pycache__" not in output
    assert ".pyc" not in output


def test_inspect_notebook_data_excludes_pycache_from_generated_files(tmp_path):
    """__pycache__ is created by Python itself the first time the compiled
    app or its runtime module gets imported (e.g. by `serve`, a prior
    `export-openapi` call, or a test suite) -- it is not part of what the
    compiler actually wrote, and its .pyc filenames are tied to whichever
    Python version happened to import it, so they aren't even stable
    across machines for the same compiled output. Before this fix, it
    still showed up in generated_files as if it were.
    """

    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    (output_dir / "app.py").write_text("# app\n")

    pycache_dir = output_dir / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "app.cpython-314.pyc").write_bytes(b"\x00")

    nested_pycache_dir = output_dir / "runtime" / "__pycache__"
    nested_pycache_dir.mkdir(parents=True)
    (nested_pycache_dir / "notebook_module.cpython-314.pyc").write_bytes(b"\x00")
    (output_dir / "runtime" / "notebook_module.py").write_text("# runtime\n")

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    data = inspect_notebook_data(str(notebook_path), str(output_dir))

    assert data["generated_files"] == ["app.py", "runtime/notebook_module.py"]


def test_inspect_notebook_prints_the_background_marker_next_to_its_route(
    tmp_path, capsys
):
    """Matches the same "[background]" marking print_compile_summary
    already prints after compiling (see
    test_print_compile_summary_lists_endpoints_and_flags_background_ones
    below) -- `inspect` is the tool's preview step and should show the
    same classification before compiling, not just after.
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def train_model(epochs: int) -> str:\n    return 'done'\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Route: POST /train_model  [background]" in output

    add_route_line = next(
        line for line in output.splitlines()
        if line.strip() == "Route: POST /add"
    )
    assert "[background]" not in add_route_line


def test_print_compile_summary_lists_endpoints_and_flags_background_ones(
    tmp_path, capsys
):
    """Shared by both `compile` and `serve` (see backend/cli.py and
    backend/serve.py) so what a caller sees is identical either way, and
    matches the same background/task_id-based marking POST /api/compile's
    "endpoints" field already uses.
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def train_model(epochs: int) -> str:\n    return 'done'\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Generated 2 endpoint(s):" in output
    assert "POST /add" in output
    assert "POST /train_model  [background]" in output

    add_line = next(
        line for line in output.splitlines() if line.strip() == "POST /add"
    )
    assert "[background]" not in add_line


def test_print_compile_summary_lists_third_party_dependencies(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "import pandas as pd\n\n"
        "def summarize(count: int) -> int:\n    return count * 2\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Dependencies: pandas" in output


def test_print_compile_summary_excludes_standard_library_imports_from_dependencies(
    tmp_path, capsys
):
    """Confirmed misleading before this fix: a notebook importing both a
    standard-library module and a real third-party one had *both* listed
    under "Dependencies" here -- even though write_requirements
    (backend/compiler.py) already excludes standard-library imports from
    requirements.txt via this exact same STANDARD_LIBS set. A notebook
    author reading "here's what this compile just produced" had no way
    to tell which of their imports would actually be installed without
    separately knowing which happen to be standard-library.
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "import os\n"
        "import pandas as pd\n\n"
        "def summarize(count: int) -> int:\n    return count * 2\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Dependencies: pandas" in output
    assert "os" not in output.split("Dependencies:")[1].split("\n")[0]


def test_inspect_notebook_data_dependencies_excludes_standard_library_imports(
    tmp_path,
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "import json\n"
        "import pandas as pd\n\n"
        "def summarize(count: int) -> int:\n    return count * 2\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["dependencies"] == ["pandas"]


def test_inspect_notebook_report_excludes_standard_library_imports(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "import sys\n"
        "import pandas as pd\n\n"
        "def summarize(count: int) -> int:\n    return count * 2\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    dependencies_section = output.split("Dependencies:")[1].split("Generated Files:")[0]
    assert "- pandas" in dependencies_section
    assert "- sys" not in dependencies_section


def test_inspect_notebook_data_dependencies_omits_an_excluded_import(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: exclude pandas\n"
        "import pandas as pd\n"
        "import nbformat\n\n"
        "def summarize(count: int) -> int:\n    return count * 2\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert "pandas" not in data["dependencies"]
    assert "nbformat" in data["dependencies"]


def test_inspect_notebook_report_omits_an_excluded_import(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: exclude pandas\n"
        "import pandas as pd\n"
        "import nbformat\n\n"
        "def summarize(count: int) -> int:\n    return count * 2\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    dependencies_section = output.split("Dependencies:")[1].split("Generated Files:")[0]
    assert "- pandas" not in dependencies_section
    assert "- nbformat" in dependencies_section


def test_inspect_notebook_data_reports_an_excluded_import_separately(tmp_path):
    """A "# notebook-to-api: exclude <import-name>" directive silently
    drops the named import from "dependencies" -- but before this,
    nothing reported *which* import(s) were dropped, indistinguishable
    from one simply never imported at all. The identical "silently
    dropped, but surfaced separately" precedent "private_functions"
    already sets for "# notebook-to-api: private".
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: exclude pandas\n"
        "import pandas as pd\n"
        "import nbformat\n\n"
        "def summarize(count: int) -> int:\n    return count * 2\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["excluded_imports"] == ["pandas"]
    assert "pandas" not in data["dependencies"]
    assert "nbformat" in data["dependencies"]


def test_inspect_notebook_data_excluded_imports_is_empty_for_a_clean_notebook(
    tmp_path,
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["excluded_imports"] == []


def test_inspect_notebook_report_lists_an_excluded_import_section(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: exclude pandas\n"
        "import pandas as pd\n\n"
        "def summarize(count: int) -> int:\n    return count\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Excluded Imports (opted out of requirements.txt):" in output
    excluded_section = output.split(
        "Excluded Imports (opted out of requirements.txt):"
    )[1].split("Dependencies:")[0]
    assert "- pandas" in excluded_section


def test_duplicate_function_names_finds_a_name_defined_more_than_once():

    functions = [
        {"name": "add"}, {"name": "subtract"}, {"name": "add"}, {"name": "add"},
    ]

    assert _duplicate_function_names(functions) == ["add"]


def test_duplicate_function_names_is_empty_when_every_name_is_unique():

    functions = [{"name": "add"}, {"name": "subtract"}]

    assert _duplicate_function_names(functions) == []


def test_inspect_notebook_data_reports_a_redefined_function_as_duplicate(tmp_path):
    """A notebook re-running an edited cell under the same function name
    is deduplicate_functions_by_name's (backend/parser/ast_parser.py) own
    documented common case -- it silently keeps only the last definition,
    matching what running the whole notebook top to bottom in a single
    kernel would do. But before this, nothing reported *that* a
    redefinition happened at all: an accidental one (a copy-pasted cell,
    a typo'd name reused by mistake) silently lost a function with no
    signal anywhere.
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def add(a: int, b: int) -> int:\n    return a * b\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["duplicate_functions"] == ["add"]
    # Only the last definition survives -- the same "last one wins"
    # behavior deduplicate_functions_by_name already documents.
    assert len(data["functions"]) == 1
    assert data["functions"][0]["name"] == "add"


def test_inspect_notebook_data_duplicate_functions_is_empty_for_a_clean_notebook(
    tmp_path,
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["duplicate_functions"] == []


def test_inspect_notebook_report_lists_a_duplicate_function_section(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def add(a: int, b: int) -> int:\n    return a * b\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert (
        "Duplicate Functions (redefined; only the last definition is compiled):"
        in output
    )
    duplicate_section = output.split(
        "Duplicate Functions (redefined; only the last definition is compiled):"
    )[1].split("Functions Found:")[0]
    assert "- add" in duplicate_section


def test_print_compile_summary_lists_a_duplicate_function(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def add(a: int, b: int) -> int:\n    return a * b\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert (
        "duplicate function(s) (redefined; only the last definition is compiled):"
        in output
    )
    assert "  add" in output


def test_print_compile_summary_omits_duplicate_functions_line_when_there_are_none(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "duplicate function" not in output


def test_inspect_notebook_data_omits_a_private_directive_marked_function(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: private\n"
        "def helper(x: int) -> int:\n    return x\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert [f["name"] for f in data["functions"]] == ["add"]
    assert data["private_functions"] == ["helper"]
    assert [e["path"] for e in data["endpoints"]] == ["/add"]


def test_inspect_notebook_data_private_functions_is_empty_for_a_clean_notebook(
    tmp_path,
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["private_functions"] == []


def test_inspect_notebook_report_omits_a_private_directive_marked_function(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: private\n"
        "def helper(x: int) -> int:\n    return x\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Private Functions (never exposed as an endpoint):" in output
    assert "- helper" in output
    functions_section = output.split("Functions Found:")[1].split("Dependencies:")[0]
    assert "helper" not in functions_section
    assert "add(" in functions_section


def test_inspect_notebook_omits_the_private_section_for_a_clean_notebook(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Private Functions" not in output


def test_print_compile_summary_omits_a_private_directive_marked_function(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: private\n"
        "def helper(x: int) -> int:\n    return x\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "POST /add" in output
    assert "POST /helper" not in output
    assert "Private 1 function(s) (never exposed as an endpoint):" in output
    assert "  helper" in output


def test_print_compile_summary_lists_an_excluded_import(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: exclude pandas\n"
        "import pandas as pd\n"
        "import nbformat\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    dependencies_line = next(
        line for line in output.splitlines() if line.startswith("Dependencies:")
    )
    assert "pandas" not in dependencies_line
    assert "Excluded 1 import(s) (opted out of requirements.txt):" in output
    assert "  pandas" in output


def test_print_compile_summary_omits_excluded_imports_line_when_there_are_none(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Excluded" not in output


def test_functions_without_docstrings_finds_a_function_with_none():

    functions = [
        {"name": "documented", "docstring": "Does a thing."},
        {"name": "undocumented", "docstring": None},
    ]

    assert _functions_without_docstrings(functions) == ["undocumented"]


def test_functions_without_docstrings_treats_a_blank_docstring_as_missing():
    """ast.get_docstring(clean=True) (via extract_functions_from_code)
    already normalizes an empty/all-whitespace docstring down to a falsy
    value -- this must treat that identically to no docstring at all.
    """

    functions = [{"name": "blank", "docstring": ""}]

    assert _functions_without_docstrings(functions) == ["blank"]


def test_functions_without_docstrings_is_empty_when_every_function_has_one():

    functions = [
        {"name": "add", "docstring": "Adds two numbers."},
        {"name": "subtract", "docstring": "Subtracts two numbers."},
    ]

    assert _functions_without_docstrings(functions) == []


def test_inspect_notebook_data_reports_a_function_without_a_docstring(tmp_path):
    """generate_fastapi_code (backend/generator/api_generator.py) already
    falls back to a generic, auto-generated OpenAPI description for a
    function with no docstring -- but before this, nothing on `inspect`
    or `compile`'s own summary said which endpoints would get that
    fallback instead of real documentation, the same "silently missing
    signal" precedent "private_functions"/"excluded_imports" already
    close for their own directives.
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def documented(a: int) -> int:\n"
        "    \"\"\"Doubles a.\"\"\"\n"
        "    return a * 2\n\n"
        "def undocumented(a: int) -> int:\n"
        "    return a + 1\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["functions_without_docstrings"] == ["undocumented"]


def test_inspect_notebook_data_functions_without_docstrings_is_empty_when_every_function_has_one(
    tmp_path,
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Adds a and b.\"\"\"\n"
        "    return a + b\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["functions_without_docstrings"] == []


def test_inspect_notebook_data_functions_without_docstrings_excludes_private_functions(
    tmp_path,
):
    """A private function never becomes an endpoint at all, so its own
    missing docstring is irrelevant -- the same reasoning that already
    excludes it from every other endpoint-relevant field here.
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: private\n"
        "def helper(a: int) -> int:\n    return a\n\n"
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Adds a and b.\"\"\"\n"
        "    return a + b\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["functions_without_docstrings"] == []


def test_inspect_notebook_report_lists_a_functions_without_docstrings_section(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def undocumented(a: int) -> int:\n    return a\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Functions Without Docstrings (will get a generic OpenAPI description):" in output
    section = output.split(
        "Functions Without Docstrings (will get a generic OpenAPI description):"
    )[1].split("Dependencies:")[0]
    assert "- undocumented" in section


def test_inspect_notebook_report_omits_functions_without_docstrings_section_when_none(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Adds a and b.\"\"\"\n"
        "    return a + b\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Functions Without Docstrings" not in output


def test_print_compile_summary_lists_a_function_without_a_docstring(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def undocumented(a: int) -> int:\n    return a\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "1 function(s) without a docstring (will get a generic OpenAPI description):" in output
    assert "  undocumented" in output


def test_print_compile_summary_omits_functions_without_docstrings_line_when_there_are_none(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Adds a and b.\"\"\"\n"
        "    return a + b\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "without a docstring" not in output


def test_inspect_notebook_data_dependencies_resolves_the_actual_distribution_name(
    tmp_path,
):
    """A notebook's `import` statement names a *module*, not necessarily
    the PyPI *distribution* that provides it -- write_requirements
    (backend/compiler.py) was fixed to resolve and pin the real
    distribution name (e.g. "multipart" -> "python-multipart") instead of
    the raw import name, but this function kept returning the raw import
    name unchanged. Confirmed: compiling a notebook importing "multipart"
    wrote "python-multipart==<version>" to requirements.txt, while
    inspect_notebook_data's own "dependencies" field still reported
    "multipart" -- a name that appears nowhere in the requirements.txt
    that same compile just produced.

    Uses python-multipart as the notebook's import for the same
    reliability reason test_requirements_resolves_an_import_name_to_its_
    actual_distribution_name (test_compiler.py) already documents: its
    import name ("multipart") differs from its distribution name
    ("python-multipart"), and it's a direct, guaranteed dependency of
    this very project.
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "import multipart\n\n"
        "def noop() -> int:\n    return 1\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["dependencies"] == ["python-multipart"]


def test_print_compile_summary_resolves_the_actual_distribution_name(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "import multipart\n\n"
        "def noop() -> int:\n    return 1\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Dependencies: python-multipart" in output


def test_inspect_notebook_report_resolves_the_actual_distribution_name(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "import multipart\n\n"
        "def noop() -> int:\n    return 1\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    dependencies_section = output.split("Dependencies:")[1].split("Generated Files:")[0]
    assert "- python-multipart" in dependencies_section


def test_aggregate_skipped_functions_reports_unsupported_signatures():

    code_cells = [
        "def unsupported(a, **kwargs):\n    return a\n",
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    ]

    skipped = _aggregate_skipped_functions(code_cells, {"add"})

    assert [s["name"] for s in skipped] == ["unsupported"]


def test_aggregate_skipped_functions_omits_names_that_ended_up_exposed():
    """A later cell can redefine a name that an earlier cell's unsupported
    version would otherwise be reported as skipped for -- exactly what
    running the whole notebook top to bottom in one kernel would do (see
    deduplicate_functions_by_name). Once that redefinition made it into
    the final, exposed function list, the earlier skip must not still be
    reported.
    """

    code_cells = [
        "def add(a, **kwargs):\n    return a\n",
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    ]

    skipped = _aggregate_skipped_functions(code_cells, {"add"})

    assert skipped == []


def test_aggregate_skipped_functions_is_sorted_and_deduplicated():

    code_cells = [
        "class Model:\n    def predict(self, x):\n        return x\n",
        "def outer():\n    def predict(y):\n        return y\n    return predict\n",
    ]

    skipped = _aggregate_skipped_functions(code_cells, set())

    assert [s["name"] for s in skipped] == ["predict"]


def test_inspect_notebook_data_reports_skipped_functions(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def unsupported(a, **kwargs):\n    return a\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["skipped_functions"] == [
        {
            "name": "unsupported",
            "reason": (
                "uses *args/**kwargs, which can't be represented as a "
                "fixed set of request fields"
            ),
        }
    ]
    assert {f["name"] for f in data["functions"]} == {"add"}


def test_inspect_notebook_data_skipped_functions_is_empty_for_a_clean_notebook(
    tmp_path,
):

    notebook_path = tmp_path / "clean.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    data = inspect_notebook_data(str(notebook_path), str(tmp_path / "generated"))

    assert data["skipped_functions"] == []


def test_inspect_notebook_prints_a_skipped_function_warning(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "class Model:\n    def predict(self, x):\n        return x\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Skipped Functions" in output
    assert "predict" in output
    assert "callable as a standalone endpoint" in output


def test_inspect_notebook_omits_the_skipped_section_for_a_clean_notebook(
    tmp_path, capsys
):

    notebook_path = tmp_path / "clean.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    inspect_notebook(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Skipped Functions" not in output


def test_print_compile_summary_lists_skipped_functions(tmp_path, capsys):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def unsupported(a, **kwargs):\n    return a\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Skipped 1 function(s)" in output
    assert "unsupported:" in output


def test_print_compile_summary_omits_skipped_line_when_there_are_none(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Skipped" not in output


def test_print_compile_summary_omits_dependencies_line_when_there_are_none(
    tmp_path, capsys
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    print_compile_summary(str(notebook_path), str(tmp_path / "generated"))

    output = capsys.readouterr().out
    assert "Dependencies:" not in output


def test_diff_notebook_functions_reports_added_functions(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(
        new_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n    return a * b\n",
    )

    diff = diff_notebook_functions(str(old_path), str(new_path))

    assert [f["name"] for f in diff["added"]] == ["multiply"]
    assert diff["removed"] == []
    assert diff["changed"] == []
    assert diff["unchanged"] == ["add"]


def test_diff_notebook_functions_reports_removed_functions(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(
        old_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )
    _write_notebook(new_path, "def add(a: int, b: int) -> int:\n    return a + b\n")

    diff = diff_notebook_functions(str(old_path), str(new_path))

    assert diff["added"] == []
    assert [f["name"] for f in diff["removed"]] == ["subtract"]
    assert diff["changed"] == []
    assert diff["unchanged"] == ["add"]


def test_diff_notebook_functions_reports_a_changed_signature(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(
        new_path,
        "def add(a: int, b: int, c: int = 0) -> int:\n    return a + b + c\n",
    )

    diff = diff_notebook_functions(str(old_path), str(new_path))

    assert diff["added"] == []
    assert diff["removed"] == []
    assert diff["unchanged"] == []
    assert len(diff["changed"]) == 1
    changed = diff["changed"][0]
    assert changed["name"] == "add"
    assert len(changed["old"]["args"]) == 2
    assert len(changed["new"]["args"]) == 3


def test_diff_notebook_functions_return_type_change_is_reported_as_changed(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(new_path, "def add(a: int, b: int) -> str:\n    return str(a + b)\n")

    diff = diff_notebook_functions(str(old_path), str(new_path))

    assert [c["name"] for c in diff["changed"]] == ["add"]


def test_diff_notebook_functions_async_change_is_reported_as_changed(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(
        new_path, "async def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    diff = diff_notebook_functions(str(old_path), str(new_path))

    assert [c["name"] for c in diff["changed"]] == ["add"]


def test_diff_notebook_functions_docstring_only_edit_is_not_a_change(tmp_path):
    """A function's docstring becomes its endpoint's OpenAPI description,
    not part of its actual request/response contract -- editing just that
    must not be reported the same way a genuine signature change is.
    """

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(
        old_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    _write_notebook(
        new_path,
        'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n',
    )

    diff = diff_notebook_functions(str(old_path), str(new_path))

    assert diff["changed"] == []
    assert diff["unchanged"] == ["add"]


def test_diff_notebook_functions_per_arg_docstring_description_edit_is_not_a_change(
    tmp_path,
):
    """Confirmed exploitable before this fix: extract_functions_from_code
    now attaches each parameter's own Google-style "Args:" description
    directly onto its arg dict (see _parse_docstring_arg_descriptions,
    backend/parser/ast_parser.py) -- _function_signature_key's own tuple
    comparison, unless it explicitly excludes that per-arg field the
    same way it already excludes the whole-function "docstring", reports
    a pure docstring reword (every type/default/kind byte-for-byte
    identical) as a genuine signature change, the exact "docstring edit
    reported as a breaking change" bug this function otherwise already
    exists to prevent for the whole-function case above.
    """

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(
        old_path,
        'def train(epochs: int) -> str:\n'
        '    """Train.\n\n    Args:\n        epochs: How many passes.\n    """\n'
        '    return "done"\n',
    )
    _write_notebook(
        new_path,
        'def train(epochs: int) -> str:\n'
        '    """Train.\n\n    Args:\n        epochs: Number of training passes.\n    """\n'
        '    return "done"\n',
    )

    diff = diff_notebook_functions(str(old_path), str(new_path))

    assert diff["changed"] == []
    assert diff["unchanged"] == ["train"]


def test_diff_notebook_functions_identical_notebooks_report_no_changes(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    diff = diff_notebook_functions(str(notebook_path), str(notebook_path))

    assert diff["added"] == []
    assert diff["removed"] == []
    assert diff["changed"] == []
    assert sorted(diff["unchanged"]) == ["add", "subtract"]


def test_classify_notebook_diff_removed_function_is_breaking(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(
        old_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )
    _write_notebook(new_path, "def add(a: int, b: int) -> int:\n    return a + b\n")

    diff = diff_notebook_functions(str(old_path), str(new_path))
    classification = classify_notebook_diff(diff)

    assert classification["compatible"] is False
    assert classification["breaking_changes"] == [{
        "type": "removed_endpoint",
        "name": "subtract",
        "detail": "Endpoint 'subtract' was removed.",
    }]


def test_classify_notebook_diff_added_function_is_not_breaking(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(
        new_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n    return a * b\n",
    )

    diff = diff_notebook_functions(str(old_path), str(new_path))
    classification = classify_notebook_diff(diff)

    assert classification == {"compatible": True, "breaking_changes": []}


def test_classify_notebook_diff_new_required_parameter_is_breaking(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(
        new_path, "def add(a: int, b: int, c: int) -> int:\n    return a + b + c\n"
    )

    diff = diff_notebook_functions(str(old_path), str(new_path))
    classification = classify_notebook_diff(diff)

    assert classification["compatible"] is False
    assert classification["breaking_changes"] == [{
        "type": "required_parameter_added",
        "name": "add",
        "detail": "New required parameter 'c' was added to 'add'.",
    }]


def test_classify_notebook_diff_new_optional_parameter_is_not_breaking(tmp_path):
    """The one case test_diff_notebook_functions_reports_a_changed_signature
    above already exercises for diff_notebook_functions itself -- a new
    parameter *with* a default doesn't change what an existing caller's
    already-valid request looks like, so it must not be reported as
    breaking here even though diff_notebook_functions still reports the
    function as "changed".
    """

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(
        new_path,
        "def add(a: int, b: int, c: int = 0) -> int:\n    return a + b + c\n",
    )

    diff = diff_notebook_functions(str(old_path), str(new_path))
    classification = classify_notebook_diff(diff)

    assert classification == {"compatible": True, "breaking_changes": []}


def test_classify_notebook_diff_removed_parameter_is_breaking(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(
        old_path, "def add(a: int, b: int, c: int = 0) -> int:\n    return a + b + c\n"
    )
    _write_notebook(new_path, "def add(a: int, b: int) -> int:\n    return a + b\n")

    diff = diff_notebook_functions(str(old_path), str(new_path))
    classification = classify_notebook_diff(diff)

    assert classification["compatible"] is False
    assert classification["breaking_changes"] == [{
        "type": "removed_parameter",
        "name": "add",
        "detail": "Parameter 'c' was removed from 'add'.",
    }]


def test_classify_notebook_diff_parameter_became_required_is_breaking(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(
        old_path, "def add(a: int, b: int = 0) -> int:\n    return a + b\n"
    )
    _write_notebook(new_path, "def add(a: int, b: int) -> int:\n    return a + b\n")

    diff = diff_notebook_functions(str(old_path), str(new_path))
    classification = classify_notebook_diff(diff)

    assert classification["compatible"] is False
    assert classification["breaking_changes"] == [{
        "type": "parameter_became_required",
        "name": "add",
        "detail": "Parameter 'b' of 'add' lost its default and is now required.",
    }]


def test_classify_notebook_diff_parameter_type_change_is_breaking(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(new_path, "def add(a: str, b: int) -> int:\n    return a + b\n")

    diff = diff_notebook_functions(str(old_path), str(new_path))
    classification = classify_notebook_diff(diff)

    assert classification["compatible"] is False
    assert classification["breaking_changes"] == [{
        "type": "parameter_type_changed",
        "name": "add",
        "detail": "Parameter 'a' of 'add' changed type from 'int' to 'str'.",
    }]


def test_classify_notebook_diff_return_type_change_is_breaking(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(new_path, "def add(a: int, b: int) -> str:\n    return str(a + b)\n")

    diff = diff_notebook_functions(str(old_path), str(new_path))
    classification = classify_notebook_diff(diff)

    assert classification["compatible"] is False
    assert classification["breaking_changes"] == [{
        "type": "return_type_changed",
        "name": "add",
        "detail": "'add' return type changed from 'int' to 'str'.",
    }]


def test_classify_notebook_diff_async_only_change_is_not_breaking(tmp_path):
    """test_diff_notebook_functions_async_change_is_reported_as_changed
    above already confirms diff_notebook_functions reports this as
    "changed" -- but sync/async-ness is invisible to an HTTP caller, so
    it must not count as a breaking change here.
    """

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(
        new_path, "async def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    diff = diff_notebook_functions(str(old_path), str(new_path))
    classification = classify_notebook_diff(diff)

    assert classification == {"compatible": True, "breaking_changes": []}


def test_classify_notebook_diff_no_changes_is_compatible(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n")

    diff = diff_notebook_functions(str(notebook_path), str(notebook_path))
    classification = classify_notebook_diff(diff)

    assert classification == {"compatible": True, "breaking_changes": []}


def test_diff_notebook_source_identical_notebooks_yields_no_lines(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    assert diff_notebook_source(str(notebook_path), str(notebook_path)) == []


def test_diff_notebook_source_reports_a_docstring_only_edit(tmp_path):
    """The one case diff_notebook_functions deliberately does NOT report
    (see test_diff_notebook_functions_docstring_only_edit_is_not_a_change
    above) -- diff_notebook_source must still show it, since the code
    genuinely did change even though the compiled signature didn't.
    """

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(
        new_path,
        'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n',
    )

    diff_lines = diff_notebook_source(str(old_path), str(new_path))

    assert any(line.startswith("+") and "Add two numbers" in line for line in diff_lines)


def test_diff_notebook_source_reports_added_and_removed_lines(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(new_path, "def multiply(a: int, b: int) -> int:\n    return a * b\n")

    diff_lines = diff_notebook_source(str(old_path), str(new_path))

    removed = [line for line in diff_lines if line.startswith("-") and "def add" in line]
    added = [line for line in diff_lines if line.startswith("+") and "def multiply" in line]
    assert removed
    assert added


def test_diff_notebook_source_uses_the_given_labels_not_the_raw_paths(tmp_path):

    old_path = tmp_path / "some" / "internal" / "old.ipynb"
    new_path = tmp_path / "some" / "internal" / "new.ipynb"
    old_path.parent.mkdir(parents=True)
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(new_path, "def add(a: int, b: int, c: int) -> int:\n    return a + b + c\n")

    diff_lines = diff_notebook_source(
        str(old_path), str(new_path),
        old_label="version 'v1'", new_label="current",
    )

    header = "\n".join(diff_lines[:2])
    assert "version 'v1'" in header
    assert "current" in header
    assert str(old_path) not in header
    assert str(new_path) not in header


def test_diff_notebook_source_defaults_labels_to_the_paths_themselves(tmp_path):

    old_path = tmp_path / "old.ipynb"
    new_path = tmp_path / "new.ipynb"
    _write_notebook(old_path, "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write_notebook(new_path, "def add(a: int, b: int, c: int) -> int:\n    return a + b + c\n")

    diff_lines = diff_notebook_source(str(old_path), str(new_path))

    header = "\n".join(diff_lines[:2])
    assert str(old_path) in header
    assert str(new_path) in header


def test_print_notebook_diff_reports_no_changes(capsys):

    print_notebook_diff(
        {"added": [], "removed": [], "changed": [], "unchanged": ["add"]}
    )

    output = capsys.readouterr().out
    assert "No changes to the compiled API surface." in output


def test_print_notebook_diff_prints_added_removed_and_changed(capsys):

    print_notebook_diff({
        "added": [{"name": "multiply"}],
        "removed": [{"name": "subtract"}],
        "changed": [{"name": "add", "old": {}, "new": {}}],
        "unchanged": [],
    })

    output = capsys.readouterr().out
    assert "Added 1 endpoint(s):" in output
    assert "POST /multiply" in output
    assert "Removed 1 endpoint(s):" in output
    assert "POST /subtract" in output
    assert "Changed 1 endpoint(s):" in output
    assert "POST /add" in output


def test_print_notebook_diff_omits_compatibility_verdict_without_those_keys(capsys):
    """A plain diff_notebook_functions dict -- no "compatible"/
    "breaking_changes" merged in -- must print exactly as before
    classify_notebook_diff existed.
    """

    print_notebook_diff(
        {"added": [], "removed": [], "changed": [], "unchanged": ["add"]}
    )

    output = capsys.readouterr().out
    assert "breaking change" not in output
    assert "compiled API's contract" not in output


def test_print_notebook_diff_prints_compatible_verdict(capsys):

    print_notebook_diff({
        "added": [], "removed": [], "changed": [], "unchanged": ["add"],
        "compatible": True, "breaking_changes": [],
    })

    output = capsys.readouterr().out
    assert "No breaking changes to the compiled API's contract." in output


def test_print_notebook_diff_prints_breaking_changes(capsys):

    print_notebook_diff({
        "added": [], "removed": [{"name": "subtract"}], "changed": [], "unchanged": [],
        "compatible": False,
        "breaking_changes": [{
            "type": "removed_endpoint",
            "name": "subtract",
            "detail": "Endpoint 'subtract' was removed.",
        }],
    })

    output = capsys.readouterr().out
    assert "1 breaking change(s) to the compiled API's contract:" in output
    assert "Endpoint 'subtract' was removed." in output


def test_generate_curl_commands_returns_one_command_per_function(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    commands = generate_curl_commands(str(notebook_path))

    assert len(commands) == 2
    assert any("curl -X POST http://localhost:8000/add" in c for c in commands)
    assert any("curl -X POST http://localhost:8000/subtract" in c for c in commands)


def test_generate_curl_commands_includes_the_example_payload_as_the_body(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    [command] = generate_curl_commands(str(notebook_path))

    assert '-d \'{"a": 0, "b": 0}\'' in command


def test_generate_curl_commands_includes_the_api_key_header(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    [default_command] = generate_curl_commands(str(notebook_path))
    assert f'-H "X-API-Key: {DEFAULT_DEV_API_KEY}"' in default_command

    [custom_command] = generate_curl_commands(str(notebook_path), api_key="mykey123")
    assert '-H "X-API-Key: mykey123"' in custom_command


def test_generate_curl_commands_respects_custom_host_and_port(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    [command] = generate_curl_commands(
        str(notebook_path), host="api.example.com", port=9000
    )

    assert "curl -X POST http://api.example.com:9000/add" in command


def test_generate_curl_commands_flags_a_background_function(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def train_model(epochs: int) -> str:\n    return 'done'\n"
    )

    [command] = generate_curl_commands(str(notebook_path))

    assert "background task" in command
    assert "task_id" in command
    assert "curl -X POST http://localhost:8000/train_model" in command


def test_generate_curl_commands_excludes_a_reserved_name_conflict(tmp_path):
    """generate_fastapi_code refuses to compile a notebook containing a
    reserved-name collision at all -- a curl command targeting that path
    would never resolve to anything real, so it must not be generated.
    """

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def health_check() -> dict:\n    return {}\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    commands = generate_curl_commands(str(notebook_path))

    assert len(commands) == 1
    assert "/add" in commands[0]
    assert not any("/health_check" in c for c in commands)


def test_generate_curl_commands_excludes_a_private_directive_marked_function(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "# notebook-to-api: private\n"
        "def helper(x: int) -> int:\n    return x\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    commands = generate_curl_commands(str(notebook_path))

    assert len(commands) == 1
    assert "/add" in commands[0]
    assert not any("/helper" in c for c in commands)


def test_generate_curl_commands_returns_an_empty_list_for_a_notebook_with_no_functions(
    tmp_path
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(notebook_path, "x = 1\n")

    assert generate_curl_commands(str(notebook_path)) == []


def test_generate_curl_commands_only_restricts_to_the_named_functions(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    commands = generate_curl_commands(str(notebook_path), only=["add"])

    assert len(commands) == 1
    assert "curl -X POST http://localhost:8000/add" in commands[0]


def test_generate_curl_commands_exclude_omits_the_named_functions(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    commands = generate_curl_commands(str(notebook_path), exclude=["subtract"])

    assert len(commands) == 1
    assert "curl -X POST http://localhost:8000/add" in commands[0]


def test_generate_curl_commands_rejects_only_and_exclude_together(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    with pytest.raises(ValueError):
        generate_curl_commands(str(notebook_path), only=["add"], exclude=["add"])


def test_generate_curl_commands_rejects_an_unknown_only_name(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    with pytest.raises(ValueError):
        generate_curl_commands(str(notebook_path), only=["does_not_exist"])


def test_generate_postman_collection_returns_one_item_per_function(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    collection = generate_postman_collection(str(notebook_path))

    names = [item["name"] for item in collection["item"]]
    assert names == ["add", "subtract"]


def test_generate_postman_collection_uses_the_notebook_stem_as_the_default_name(tmp_path):

    notebook_path = tmp_path / "my_notebook.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    collection = generate_postman_collection(str(notebook_path))

    assert collection["info"]["name"] == "my_notebook"
    assert collection["info"]["schema"] == (
        "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    )


def test_generate_postman_collection_respects_a_custom_collection_name(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    collection = generate_postman_collection(
        str(notebook_path), collection_name="My API"
    )

    assert collection["info"]["name"] == "My API"


def test_generate_postman_collection_includes_the_example_payload_as_the_body(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    [item] = generate_postman_collection(str(notebook_path))["item"]

    assert item["request"]["method"] == "POST"
    assert item["request"]["url"]["raw"] == "{{base_url}}/add"
    assert json.loads(item["request"]["body"]["raw"]) == {"a": 0, "b": 0}


def test_generate_postman_collection_sets_base_url_and_api_key_variables(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    default_collection = generate_postman_collection(str(notebook_path))
    variables = {v["key"]: v["value"] for v in default_collection["variable"]}
    assert variables["base_url"] == "http://localhost:8000"
    assert variables["api_key"] == DEFAULT_DEV_API_KEY

    custom_collection = generate_postman_collection(
        str(notebook_path), host="api.example.com", port=9000, api_key="mykey123"
    )
    variables = {v["key"]: v["value"] for v in custom_collection["variable"]}
    assert variables["base_url"] == "http://api.example.com:9000"
    assert variables["api_key"] == "mykey123"


def test_generate_postman_collection_includes_the_api_key_header(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    [item] = generate_postman_collection(str(notebook_path))["item"]

    headers = {h["key"]: h["value"] for h in item["request"]["header"]}
    assert headers["X-API-Key"] == "{{api_key}}"


def test_generate_postman_collection_adds_a_task_status_request_for_a_background_function(
    tmp_path
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def train_model(epochs: int) -> str:\n    return 'done'\n"
    )

    collection = generate_postman_collection(str(notebook_path))

    names = [item["name"] for item in collection["item"]]
    assert names == ["train_model", "train_model - Task Status"]

    submission_item, status_item = collection["item"]

    assert "task_id" in submission_item["request"]["description"]
    assert submission_item["event"][0]["listen"] == "test"
    assert any(
        "train_model_task_id" in line
        for line in submission_item["event"][0]["script"]["exec"]
    )

    assert status_item["request"]["method"] == "GET"
    assert status_item["request"]["url"]["raw"] == (
        "{{base_url}}/tasks/{{train_model_task_id}}"
    )

    variables = {v["key"]: v["value"] for v in collection["variable"]}
    assert variables["train_model_task_id"] == ""


def test_generate_postman_collection_omits_task_status_request_for_a_synchronous_function(
    tmp_path
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    [item] = generate_postman_collection(str(notebook_path))["item"]

    assert "event" not in item
    assert "description" not in item["request"]


def test_generate_postman_collection_excludes_a_reserved_name_conflict(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def health_check() -> dict:\n    return {}\n\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )

    items = generate_postman_collection(str(notebook_path))["item"]

    assert len(items) == 1
    assert items[0]["name"] == "add"


def test_generate_postman_collection_only_restricts_to_the_named_functions(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    items = generate_postman_collection(str(notebook_path), only=["add"])["item"]

    assert [item["name"] for item in items] == ["add"]


def test_generate_postman_collection_exclude_omits_the_named_functions(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path,
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n    return a - b\n",
    )

    items = generate_postman_collection(str(notebook_path), exclude=["subtract"])["item"]

    assert [item["name"] for item in items] == ["add"]


def test_generate_postman_collection_rejects_only_and_exclude_together(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    with pytest.raises(ValueError):
        generate_postman_collection(str(notebook_path), only=["add"], exclude=["add"])


def test_generate_postman_collection_rejects_an_unknown_only_name(tmp_path):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(
        notebook_path, "def add(a: int, b: int) -> int:\n    return a + b\n"
    )

    with pytest.raises(ValueError):
        generate_postman_collection(str(notebook_path), only=["does_not_exist"])


def test_generate_postman_collection_returns_an_empty_item_list_for_a_notebook_with_no_functions(
    tmp_path
):

    notebook_path = tmp_path / "nb.ipynb"
    _write_notebook(notebook_path, "x = 1\n")

    assert generate_postman_collection(str(notebook_path))["item"] == []
