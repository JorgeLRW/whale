"""MRI-specific CLI helpers built on top of the Whale pipeline."""

from .deep_dive import (
    add_result_row,
    build_parser as build_standard_parser,
    cli as cli_standard,
    main as main_standard,
    run as run_standard_job,
)
from .deep_dive_fast import (
    build_parser as build_fast_parser,
    cli as cli_fast,
    main as main_fast,
    run as run_fast_job,
)

__all__ = [
    "add_result_row",
    "build_standard_parser",
    "main_standard",
    "cli_standard",
    "run_standard_job",
    "build_fast_parser",
    "main_fast",
    "cli_fast",
    "run_fast_job",
]
