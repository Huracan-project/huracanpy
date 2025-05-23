import pathlib
import warnings

import pytest

import nbformat
import nbclient


docs_path = pathlib.Path(__file__).parent / "../docs/"
all_notebooks = [(str(fname),) for fname in docs_path.glob("examples/*.ipynb")] + [
    (str(fname),) for fname in docs_path.glob("user_guide/*.ipynb")
]


@pytest.mark.skipif(
    "not config.getoption('--docs')",
    reason="ipynb's are slow to run. Specify --docs explicitly to test execution",
)
@pytest.mark.parametrize(("notebook",), all_notebooks)
def test_notebook_runs(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        nbclient.execute(nb, Xfrozen_modules="off")
    # out = ExecutePreprocessor().preprocess(nb)
