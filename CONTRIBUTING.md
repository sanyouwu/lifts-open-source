# Contributing to LIFTS

Thank you for helping improve LIFTS. Bug reports, documentation fixes, tests, and focused implementation changes are welcome.

## Set up a development environment

Fork and clone the standalone repository, then run from its root:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

For a CPU-only PyTorch environment, install the appropriate CPU wheel from the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before installing LIFTS.

## Make a focused change

- Open an issue before undertaking a large API or model-design change.
- Keep public names and examples under the LIFTS name.
- Preserve the `(N, p, T)` input convention and document any intentional API change.
- Add or update tests for behavior changes and regression fixes.
- Keep tests small, deterministic, and runnable on CPU. Do not require access to private data, a GPU, or external services.
- Preserve copyright and license headers when adapting third-party code, and update `THIRD_PARTY_NOTICES.md` when needed.

Use clear Python names and docstrings, and follow the surrounding style. Avoid unrelated formatting or generated-file changes in the same pull request.

## Validate the change

Run the same core checks as continuous integration:

```bash
pytest -q
python -m build
python -m twine check dist/*
```

For notebook changes, restart the kernel and run all cells from top to bottom. A public notebook must not depend on absolute local paths, private repositories, credentials, or untracked data. Keep large cell outputs and generated figures out of commits unless they are essential documentation assets.

## Submit a pull request

In the pull request description:

- explain the problem and the chosen change;
- list the commands used to validate it;
- call out user-visible API, dependency, or numerical changes;
- link the relevant issue, if one exists; and
- include before/after figures only when they help reviewers assess a visualization change.

By submitting a contribution, you agree that it may be distributed under the repository's [MIT License](LICENSE). Contributions that include third-party material must be compatible with that license and retain all required attribution.
