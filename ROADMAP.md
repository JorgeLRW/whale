# Whale Roadmap

## 0.2 (Next minor release)

- **ArXiv paper polish**: incorporate reviewer feedback, regenerate figures, and ship a camera-ready PDF.
- **Notebook refresh**: add parameter exploration widgets to the Colab demo and include the point-cloud benchmark workflow.
- **Packaging polish**: publish the `whale-mri` package to TestPyPI, gather install feedback, then promote to PyPI.

## 0.3

- **Performance profiling**: integrate optional GPU-accelerated nearest-neighbour search via FAISS/HNSWLib switches.
- **CLI ergonomics**: add `whale` console entry point with subcommands for `deep-dive`, `fast`, and `pointcloud` flows.
- **Dataset registry**: provide friendly manifests and helper scripts for additional public MRI datasets (ABIDE, OASIS).

## 1.0

- **Full reproducibility bundles**: container images for every benchmark configuration and a Makefile to run them.
- **Visualization UI**: lightweight dashboard for interactive persistence diagram exploration.
- **Community governance**: expand maintainer group, document voting procedures, and introduce a technical steering committee if contributor base grows.

Roadmap milestones are tracked through GitHub projects. If you have feature requests, open an issue and propose where it should land.
