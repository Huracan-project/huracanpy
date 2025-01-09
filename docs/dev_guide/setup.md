# Setup Developer's environment

First create your own fork of huracanpy from the [github page](https://github.com/Huracan-project/huracanpy).
"Create new fork" from the dropdown.
Then clone your repository e.g.

```shell
    git clone /https://github.com/{your-username}/huracanpy.git
```

To install your copy locally run
```shell
    pip install -e .[dev, docs]
```

The "[dev]" argument installs the following optional packages that are useful for
contributing to development

1. **pytest**

    We use [pytest](https://docs.pytest.org/en/latest/) to run automated tests. If you
    add a new feature, it would be good to also add tests to check that feature is
    working and keeps working in the future. You can also run `pytest` from the top
    level directory of the package to check that your changes haven't broken anything.

2. **ruff**

    We use [ruff](<https://docs.astral.sh/ruff/) to automatically keep the style of the code consistent, so we don't have to worry about it.

    To check that your code passes you can run `ruff check` and `ruff format --check`.

    To automatically fix differences run `ruff check --fix` and `ruff format`.

3. **pre-commit**

    You can use [pre-commit](https://pre-commit.com/) to automate the formatting done by ruff.

    After running `pre-commit install` at the top level directory, any future git commits will automatically run the ruff formatting on any files you have changes.


The "[docs]" argument installs extra dependencies needed to build the documentation locally.
If you are adding or modifying functions it will probably be useful to also update the documentation.
For further instructions see {doc}`this page <doc>`.
