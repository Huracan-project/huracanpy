# Tests in HuracanPy

## Running tests
Tests are run using [pytest](https://docs.pytest.org/en/latest/). Simply run
`pytest` from the top of your repository and it will run all the existing tests.

If you want to run the tests across multiple python versions use [tox](https://tox.wiki/en/stable/).
Instead, run `tox` from the top of your repository, and tox will build a virtual environment for each python version supported by huracanpy and run the tests for each version.
Note, the python versions need to be installed on your system for `tox` to build the environments.
I used `conda` environments and symlinks to achieve this.
I don't know if that's how you're supposed to do it.

Generally `pytest` should be sufficient for testing.
GitHub will run the tests over different python versions for pull requests, so `tox` can be useful if GitHub gives unexpected failures.

## Writing tests
Tests are kept in the `tests` folder.
The structure of this folder largely mimics the structure of huracanpy, but with folders/files/functions named `test_*`.
To add your test, add an appropriately named function/file/folder and use some kind of `assert`
statement to check the results.
Typically we are using
- `assert` - The old classic
- `np.testing.assert_equal` - Check two numpy arrays or array-like objects are equivalent, including if NaNs are present
- `np.testing.assert_allclose` - Check two numpy arrays or array-like objects are equivalent within a given tolerance for floating-point errors
- `xr.testing.assert_identical` - Check that two xarray objects are the same, including all metadata

In `tests/conftest.py` are some pre-defined fixtures that can be useful for testing.
The simplest use is to include the name of the fixture as an argument to your test function, and then the fixture can be used in that test.
See the [pytest documentation](https://docs.pytest.org/en/stable/explanation/fixtures.html) for more details.
