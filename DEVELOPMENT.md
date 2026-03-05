# Development Guide

## Publishing to PyPI

### Prerequisites

1. A [PyPI account](https://pypi.org/account/register/) and an [API token](https://pypi.org/manage/account/token/).
2. [maturin](https://github.com/PyO3/maturin) installed (`pip install maturin`).
3. [twine](https://github.com/pypa/twine) installed (`pip install twine`) -- only needed for manual uploads.

### One-Time Setup

Store your PyPI API token so maturin/twine can authenticate:

```bash
# Option A: environment variable
export MATURIN_PYPI_TOKEN=pypi-AgEI...

# Option B: keyring (persistent)
python -m keyring set https://upload.pypi.org/legacy/ __token__
```

### Building Wheels

maturin compiles the Rust extension and packages it into a Python wheel:

```bash
# Build a wheel for the current platform and Python version
maturin build --release

# Build wheels for multiple Python versions (requires each version installed)
maturin build --release --interpreter python3.10 python3.11 python3.12

# Build an sdist (source distribution) as well
maturin sdist
```

Built artefacts appear in `target/wheels/`.

### Publishing

#### Test PyPI (do this first)

```bash
# Build and publish to Test PyPI in one step
maturin publish --repository testpypi

# Or manually with twine
twine upload --repository testpypi target/wheels/*.whl

# Verify installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ rusty-sketches
```

#### Production PyPI

```bash
# Build and publish to PyPI in one step
maturin publish

# Or manually with twine
twine upload target/wheels/*.whl
```

### Cross-Platform Wheels with CI

For publishing wheels across Linux (manylinux), macOS (x86_64 + arm64), and Windows,
use a GitHub Actions workflow with maturin's official action:

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

permissions:
  contents: read

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          args: --release --out dist
          manylinux: auto
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # trusted publishing
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheels-*
          merge-multiple: true
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
```

This uses [trusted publishing](https://docs.pypi.org/trusted-publishers/) so no API
token is needed -- configure it once in PyPI project settings.

### Versioning

The package version is read from `Cargo.toml` (the `version` field). maturin
automatically uses this as the Python package version. To release a new version:

1. Update `version` in `Cargo.toml`.
2. Commit and tag: `git tag v0.3.0`.
3. Push the tag: `git push origin v0.3.0`.
4. Create a GitHub release from the tag -- the CI workflow publishes automatically.

### Import Name vs Package Name

The PyPI package name (what you `pip install`) and the import name (what you
`import` in Python) are independent. This is configured in `pyproject.toml`:

```toml
[project]
name = "rusty-sketches"  # <-- pip install rusty-sketches

[tool.maturin]
module-name = "sketches"  # <-- import sketches
```

Users always do `import sketches` regardless of the PyPI package name.
