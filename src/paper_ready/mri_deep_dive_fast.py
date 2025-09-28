"""Compatibility shim for the fast MRI deep-dive CLI."""

from __future__ import annotations

from typing import Optional, Sequence

from whale.mri.deep_dive_fast import (  # noqa: F401 - re-exported symbols
    build_parser,
    cli as _cli,
    main as _main,
    run,
)

from whale.mri.deep_dive import add_result_row  # noqa: F401 - legacy import path

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
