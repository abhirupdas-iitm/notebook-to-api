import nbformat
import pytest

from backend.parser.ast_parser import extract_functions_from_code
from backend.parser.notebook_parser import (
    load_notebook,
    extract_code_cells,
    strip_magic_commands,
    detect_non_python_body_cell_magic,
    notebook_kernel_language,
)


def test_extract_code_cells():

    notebook = load_notebook(
        "notebooks/sample.ipynb"
    )

    code_cells = extract_code_cells(notebook)

    assert len(code_cells) > 0


def test_strip_magic_commands_comments_out_line_magic():

    source = "%matplotlib inline\nimport matplotlib.pyplot as plt"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# %matplotlib inline\nimport matplotlib.pyplot as plt"


def test_strip_magic_commands_comments_out_cell_magic():

    source = "%%time\nx = 1 + 1"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# %%time\nx = 1 + 1"


def test_strip_magic_commands_comments_out_shell_escape():

    source = "!pip install pandas\nimport pandas as pd"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# !pip install pandas\nimport pandas as pd"


def test_strip_magic_commands_preserves_indentation():

    source = "if True:\n    %timeit x\n    y = 1"

    cleaned = strip_magic_commands(source)

    assert cleaned == "if True:\n    # %timeit x\n    y = 1"


def test_strip_magic_commands_leaves_plain_code_untouched():

    source = "def add(a, b):\n    return a + b"

    assert strip_magic_commands(source) == source


def test_strip_magic_commands_does_not_touch_modulo_operator():

    source = "remainder = 10 % 3"

    assert strip_magic_commands(source) == source


def test_detect_non_python_body_cell_magic_recognizes_writefile():

    assert detect_non_python_body_cell_magic(
        "%%writefile helper.py\ndef greet(name):\n    return name"
    ) == "writefile"


def test_detect_non_python_body_cell_magic_recognizes_other_non_python_magics():

    for magic, first_line in (
        ("bash", "%%bash"),
        ("sh", "%%sh"),
        ("perl", "%%perl"),
        ("ruby", "%%ruby"),
        ("script", "%%script bash"),
        ("html", "%%html"),
        ("HTML", "%%HTML"),
        ("javascript", "%%javascript"),
        ("js", "%%js"),
        ("latex", "%%latex"),
        ("svg", "%%svg"),
        ("markdown", "%%markdown"),
    ):
        assert detect_non_python_body_cell_magic(f"{first_line}\nsome content") == magic


def test_detect_non_python_body_cell_magic_ignores_magics_that_do_run_as_python():
    """%%time/%%timeit/%%capture/%%prun/%%debug all execute their own body
    as ordinary Python in the notebook's own namespace (timing it,
    capturing its output, profiling it, ...) -- unlike %%writefile/%%bash/
    etc., a function defined inside one of these really is callable from
    a later cell in the real kernel.
    """

    for first_line in ("%%time", "%%timeit", "%%capture", "%%prun", "%%debug"):
        assert detect_non_python_body_cell_magic(f"{first_line}\nx = 1") is None


def test_detect_non_python_body_cell_magic_returns_none_for_plain_code():

    assert detect_non_python_body_cell_magic("def add(a, b):\n    return a + b") is None


def test_detect_non_python_body_cell_magic_skips_leading_blank_lines():

    assert detect_non_python_body_cell_magic("\n\n%%writefile x.py\npass") == "writefile"


def test_strip_magic_commands_comments_out_the_entire_writefile_cell():
    """Confirmed exploitable before this fix: %%writefile writes its own
    body to a file instead of executing it -- a function defined inside
    one is never actually defined in the *notebook's own* namespace in a
    real kernel at all. Only the "%%writefile ..." line itself was
    commented out before this, leaving a syntactically-valid-Python body
    (the overwhelmingly common real case: %%writefile is routinely used
    to scaffold a .py module) completely untouched and compilable.
    """

    source = "%%writefile helper.py\ndef greet(name):\n    return name"

    cleaned = strip_magic_commands(source)

    assert cleaned == (
        "# %%writefile helper.py\n# def greet(name):\n#     return name"
    )
    assert extract_functions_from_code(cleaned) == []


def test_strip_magic_commands_comments_out_the_entire_bash_cell():

    source = "%%bash\npip install pandas\necho done"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# %%bash\n# pip install pandas\n# echo done"


def test_strip_magic_commands_preserves_blank_lines_in_a_writefile_cell():

    source = "%%writefile helper.py\ndef greet(name):\n\n    return name"

    cleaned = strip_magic_commands(source)

    assert cleaned == (
        "# %%writefile helper.py\n# def greet(name):\n\n#     return name"
    )


def test_strip_magic_commands_still_executes_time_magic_body():
    """Contrast with the %%writefile/%%bash cases above -- %%time really
    does run its own body as Python in the notebook's own namespace, so
    only its own magic line is commented out, exactly as before.
    """

    source = "%%time\nx = 1 + 1"

    assert strip_magic_commands(source) == "# %%time\nx = 1 + 1"


def test_extract_code_cells_strips_magics_from_notebook():

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "%matplotlib inline\ndef plot():\n    return 1"
        )
    )

    code_cells = extract_code_cells(notebook)

    assert len(code_cells) == 1
    assert "# %matplotlib inline" in code_cells[0]
    assert "def plot():" in code_cells[0]


def test_strip_magic_commands_comments_out_prefix_introspection():

    source = "?len\nx = 1"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# ?len\nx = 1"


def test_strip_magic_commands_comments_out_double_prefix_introspection():

    source = "??train_model\nx = 1"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# ??train_model\nx = 1"


def test_strip_magic_commands_comments_out_suffix_introspection():

    source = "train_model?\nx = 1"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# train_model?\nx = 1"


def test_strip_magic_commands_comments_out_double_suffix_introspection():

    source = "train_model??\nx = 1"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# train_model??\nx = 1"


def test_strip_magic_commands_comments_out_dotted_suffix_introspection():

    source = "np.random.seed?"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# np.random.seed?"


def test_strip_magic_commands_comments_out_called_suffix_introspection():

    source = "pd.DataFrame()?"

    cleaned = strip_magic_commands(source)

    assert cleaned == "# pd.DataFrame()?"


def test_strip_magic_commands_introspection_preserves_indentation():

    source = "if True:\n    train_model?\n    y = 1"

    cleaned = strip_magic_commands(source)

    assert cleaned == "if True:\n    # train_model?\n    y = 1"


def test_strip_magic_commands_leaves_question_mark_inside_string_untouched():
    """A "?" is not valid Python syntax anywhere outside of a string
    literal or comment, so any line consisting of *only* an
    attribute-chain expression plus a leading/trailing "?"/"??" is
    unambiguously an IPython introspection query. A line that merely
    *contains* a "?" as part of a larger, otherwise-valid statement (e.g.
    a string literal ending in "?") must not be touched.
    """

    source = 'msg = "wait?"'

    assert strip_magic_commands(source) == source


def test_strip_magic_commands_leaves_assignment_ending_in_digit_and_question_mark_untouched():

    source = "x = 5?"

    assert strip_magic_commands(source) == source


def test_extract_code_cells_strips_introspection_query_from_notebook():

    notebook = nbformat.v4.new_notebook()

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "train_model?\ndef train_model():\n    return 1"
        )
    )

    code_cells = extract_code_cells(notebook)

    assert len(code_cells) == 1
    assert "# train_model?" in code_cells[0]
    assert "def train_model():" in code_cells[0]


def _write_notebook_file(path, notebook):
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)


def test_notebook_kernel_language_returns_none_without_kernelspec_or_language_info():
    """Both fields are optional per the nbformat spec, and a hand-built or
    stripped-down notebook -- most of this project's own test fixtures
    among them, since nbformat.v4.new_notebook() sets neither by default
    -- commonly omits them entirely. There's no honest way to tell
    "wasn't recorded" apart from "is Python" in their own absence, so
    this must return None rather than guessing.
    """

    notebook = nbformat.v4.new_notebook()

    assert notebook_kernel_language(notebook) is None


def test_notebook_kernel_language_reads_kernelspec_language():

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "ir", "display_name": "R", "language": "R",
    }

    assert notebook_kernel_language(notebook) == "r"


def test_notebook_kernel_language_falls_back_to_language_info_name():

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["language_info"] = {"name": "Julia", "version": "1.9"}

    assert notebook_kernel_language(notebook) == "julia"


def test_notebook_kernel_language_prefers_kernelspec_over_language_info():

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "ir", "display_name": "R", "language": "R",
    }
    notebook.metadata["language_info"] = {"name": "python"}

    assert notebook_kernel_language(notebook) == "r"


def test_load_notebook_accepts_a_notebook_with_no_declared_language(tmp_path):
    """The overwhelmingly common case for this project's own test
    fixtures (see notebook_kernel_language's own docstring) -- must keep
    working exactly as before this feature.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(
        nbformat.v4.new_code_cell("def add(a, b):\n    return a + b")
    )

    path = tmp_path / "nb.ipynb"
    _write_notebook_file(path, notebook)

    loaded = load_notebook(str(path))

    assert loaded is not None


def test_load_notebook_accepts_a_notebook_declaring_python(tmp_path):

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "python3", "display_name": "Python 3", "language": "python",
    }

    path = tmp_path / "nb.ipynb"
    _write_notebook_file(path, notebook)

    loaded = load_notebook(str(path))

    assert loaded is not None


def test_load_notebook_rejects_a_non_python_kernel(tmp_path):
    """Confirmed exploitable before this feature: every cell of a
    genuinely non-Python notebook (R's own `f <- function(x) x + 1`
    syntax, say) simply failed is_parseable_python and was silently
    dropped -- compile_notebook_to_api "succeeded" with zero extracted
    functions, producing a working-but-endpoint-less API with nothing
    anywhere explaining why it exposed nothing.
    """

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "ir", "display_name": "R", "language": "R",
    }
    notebook.cells.append(
        nbformat.v4.new_code_cell("f <- function(x) x + 1")
    )

    path = tmp_path / "nb.ipynb"
    _write_notebook_file(path, notebook)

    with pytest.raises(ValueError, match="not Python"):
        load_notebook(str(path))


def test_load_notebook_rejects_a_non_python_language_info_with_no_kernelspec(
    tmp_path
):

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["language_info"] = {"name": "julia", "version": "1.9"}

    path = tmp_path / "nb.ipynb"
    _write_notebook_file(path, notebook)

    with pytest.raises(ValueError, match="julia"):
        load_notebook(str(path))


def test_load_notebook_error_message_names_the_actual_language(tmp_path):

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "ir", "display_name": "R", "language": "R",
    }

    path = tmp_path / "nb.ipynb"
    _write_notebook_file(path, notebook)

    with pytest.raises(ValueError) as exc_info:
        load_notebook(str(path))

    assert "'r'" in str(exc_info.value)