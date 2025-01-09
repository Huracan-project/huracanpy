# Documentation

## Docstrings
The API documentation is built from docstrings written with the code.
We use [numpy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) docstrings.
If you are adding a function or class, give it a numpy-style docstring and add the name of the function/class to the relevant .rst file in `docs/api/`, and it will be included in the documentation.

## Documentation webpages

### Sphinx
The present documentation is built with [sphinx](https://www.sphinx-doc.org/en/master/). If you installed `huracanPy` following the developers' instruction, you should have the following package installed in your environment:

- `sphinx`
- `pydata-sphinx-theme`
- `nbsphinx`
- `sphinx-copybutton`

All documentations files are in the `docs` folder at the root of the package.
They are either `.rst` (reStructured Text), `.md` (markdown), or `.ipynb` (Jupyter Notebook, which includes MarkDown) files.

:::{hint}
To build the documentation locally, run `make html` in this folder. This is useful to preview your changes before you commit them. 
:::


To learn how to use these tools, you may want to check out the following resources:
- reStructuredText
  - [Wikipedia page](https://en.wikipedia.org/wiki/ReStructuredText)
  - [Sphinx tutorial](https://sphinx-tutorial.readthedocs.io/step-1/)
  - [Sphinx documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
- Jupyter notebooks
  - [Jupyter website](https://jupyter.org/)
- Markdown
  - [Jupyter doc](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html)
  - [Markdown Guide](https://www.markdownguide.org/)
  - [Markdown tutorial](https://www.markdowntutorial.com/)
  - [MyST doc](https://myst-parser.readthedocs.io/en/stable/)
- Sphinx
  - [Documentation](https://www.sphinx-doc.org/en/master/index.html)
  - [Tutorial](https://sphinx-tutorial.readthedocs.io/)

You can also check out the source code of documentation pages you like to find how to do a specific thing.
