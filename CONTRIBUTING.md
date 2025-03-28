# Contributing guidelines

If you run into any issues when using the library, feel free to raise a new Github issue
with code that reproduces the bug you've encountered.

If you would like to contribute to the package, feel free to open a PR !

## Pull Request Checklist

There are few rules when it comes to raising a new PR to this repository, as long as the
code runs and it would be a good addition to the package then you can raise a PR.

The package is currently light on tests and so these aren't mandatory (but highly recommended),
however if you are adding a new provider it would be great if you could add a test.

To run the tests, simply run:

```bash
uv pip install -e ".[test]"

uv run pytest
```