# Contributing

Thanks for your interest in improving the Whale MRI Witness Pipeline! This guide outlines how to propose
changes, report issues, and share new benchmarks.

## Ground rules

- **Be respectful.** Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- **Prefer issues first.** Open a GitHub issue describing the bug or feature before starting
  significant work.
- **Keep changes focused.** Smaller pull requests with tests and documentation updates are
  easier to review.
- **Run the smoke suite.** Ensure `python -m unittest discover -s tests` passes locally before
  submitting a pull request.

## Development setup

1. Fork the repository and clone your fork.
2. Create a virtual environment (conda or venv) and install dependencies:
   ```bash
   pip install -r paper_ready/requirements.txt
   ```
3. Point `PYTHONPATH` at the source tree:
   ```bash
   export PYTHONPATH="$(pwd)/paper_ready/src"
   ```
4. Run the smoke tests:
   ```bash
   python -m unittest discover -s tests
   ```

## Proposing changes

1. Create a feature branch from `main`.
2. Make your changes, including:
   - Tests for new behaviour when applicable.
   - Documentation updates (README, examples, notebooks).
3. Run `python -m unittest discover -s tests`.
4. Commit your work and push the branch to your fork.
5. Open a pull request against `main`, linking to the relevant issue.

## Benchmarks and data

- Do not commit raw MRI volumes or large point-cloud datasets. Use manifests and helper scripts
  instead.
- Place derived CSV summaries under `examples/sample_outputs/` or `artifacts/` (git-ignored by
  default).
- Describe reproduction steps in the PR description or in `analysis/` notes.

## Releasing

1. Update `CHANGELOG.md` (if present) and the version number.
2. Tag the release (e.g., `v0.1.0`) after the CI workflow passes.
3. Attach the compiled `paper/main.pdf` and curated CSV bundles to the GitHub release.

We’re excited to see what you build—thanks for contributing!
