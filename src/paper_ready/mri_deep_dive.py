"""Compatibility wrapper for the MRI deep-dive CLI.

The canonical implementation now lives in :mod:`whale.mri.deep_dive`. This
module keeps the original import surface for scripts that still reference
``paper_ready.mri_deep_dive``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from whale.mri.deep_dive import (  # noqa: F401 - re-exported symbols
    add_result_row,
    build_parser,
    cli as _cli,
    main as _main,
    run,
)

__all__ = [
    "add_result_row",
    "build_parser",
    "main",
    "cli",
    "run",
]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point preserved for backwards compatibility."""

    _main(argv)


def cli(argv: Optional[Sequence[str]] = None) -> None:
    """Console-script compatible entry point."""

    _cli(argv)


if __name__ == "__main__":  # pragma: no cover - legacy CLI behaviour
    main()
