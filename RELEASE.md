# Releasing LIFTS

This document describes releases from the standalone repository at <https://github.com/sanyouwu/lifts>.

## Important version note

PyPI already contains `lifts` version `0.1.0`, and PyPI does not permit a published filename or version to be overwritten. The source metadata in this repository intentionally remains at `0.1.0` while the public-repository snapshot is prepared. **Do not try to upload this snapshot as `0.1.0`.**

Before the first release made from `sanyouwu/lifts`, update `version` in `pyproject.toml` to `0.1.1` (or a later unused version). Every subsequent release also requires a new version.

## PyPI trusted publishing setup

Use a PyPI Trusted Publisher so the GitHub Actions publishing job obtains a short-lived identity token instead of storing a long-lived PyPI token.

In the PyPI project settings for `lifts`, configure the publisher with:

- Owner: `sanyouwu`
- Repository: `lifts`
- Workflow filename: the exact filename of the publishing workflow, for example `publish.yml`
- Environment: `pypi`

Repository and workflow names are exact and case-sensitive. The GitHub publishing job must use the `pypi` environment, grant `id-token: write`, and invoke `pypa/gh-action-pypi-publish`. The included `ci.yml` validates distributions but deliberately does not publish them.

The `pypi` GitHub environment should require approval if releases need a manual gate. No PyPI API token is needed for Trusted Publishing.

## Prepare a release

1. Confirm that the target version does not already exist on [PyPI](https://pypi.org/project/lifts/#history).
2. Update `version` in `pyproject.toml`. For the first standalone-repository release, use at least `0.1.1`.
3. Update user-facing documentation and release notes for the change.
4. Run the checks below from a clean checkout of the repository.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
python -m build
python -m twine check dist/*
```

The build produces a source distribution and wheel under `dist/`. Inspect their filenames and confirm that both contain the intended version. Installing the wheel in a fresh virtual environment is recommended before publishing.

## Publish from GitHub

1. Commit the version and release changes and merge them into the default branch.
2. Create a GitHub release with a matching tag, for example `v0.1.1`.
3. Allow the dedicated publishing workflow to build from that tag or release event.
4. Approve the `pypi` environment deployment if environment protection is enabled.
5. Verify the new files and installation instructions on PyPI.

Creating a tag alone does not upload a package unless a publishing workflow is installed and configured for that event.

## Optional TestPyPI rehearsal

TestPyPI is useful when changing packaging metadata or the release workflow. Configure a separate Trusted Publisher and GitHub environment for TestPyPI, and pass its repository URL to the publishing action. TestPyPI and production PyPI have separate accounts, projects, and publisher settings.

## Manual upload fallback

Trusted Publishing is preferred. If a maintainer must upload manually, build from a clean checkout and run:

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Use `__token__` as the username and a scoped PyPI API token as the password. Never commit a token, place it in a notebook, or add it to a GitHub Actions file.
