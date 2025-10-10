# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Roadmap, governance, and distribution documentation.
- Docker images for standard and fast MRI pipelines.

## [0.2.0] - 2025-10-09
### Added
- `whale.ai` helpers for embedding-to-persistence workflows, including a torch `WitnessFeatureLayer` and batch summarisation utilities.
- AI optional dependency extra (`pip install "whale-tda[ai]"`) and unit tests covering fast vs regular modes.
- Repository layout documentation and legacy workspace archive with compatibility shim for `paper_ready_tests` imports.

### Changed
- Allow selection between fast (dim-1) and regular (dim-2) witness runs through a new `tda_mode` argument across AI helpers.
- Increased default package version to 0.2.0 in preparation for PyPI distribution.
- Tidied top-level repository structure while retaining all CSV artifacts.

## [0.1.0] - 2025-09-28
### Added
- Whale MRI witness pipeline library extracted from research workspace.
- Synthetic MRI smoke tests and CLI runners for `deep_dive` and `deep_dive_fast`.
- Comprehensive README with quickstart, Colab demo, and AI assistance notice.
- Contribution guidelines, Code of Conduct, and citation metadata.
- Continuous integration workflow running the smoke suite on GitHub Actions.

### Changed
- README tagline and badges reflecting the new CI and Colab demo.

### Removed
- N/A

[Unreleased]: https://github.com/jorgeLRW/whale/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jorgeLRW/whale/releases/tag/v0.2.0
[0.1.0]: https://github.com/jorgeLRW/whale/releases/tag/v0.1.0
