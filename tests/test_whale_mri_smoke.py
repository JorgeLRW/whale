"""Smoke tests for the Whale MRI deep-dive command-line interfaces.

These exercises rely solely on the synthetic MRI phantom to avoid optional
runtime dependencies such as nibabel or gudhi. They aim to ensure the CLI
entry-points execute end-to-end with lightweight parameter settings.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from whale.mri import deep_dive, deep_dive_fast


class WhaleMriSmokeTests(unittest.TestCase):
    def test_standard_deep_dive_synthetic_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="whale-mri-standard-") as tmpdir:
            output_path = Path(tmpdir) / "standard.csv"
            parser = deep_dive.build_parser()
            args = parser.parse_args(
                [
                    "--synthetic",
                    "--seed",
                    "123",
                    "--methods",
                    "random",
                    "--rips-points",
                    "0",
                    "--max-points",
                    "2000",
                    "--m",
                    "64",
                    "--out",
                    str(output_path),
                ]
            )

            rows = deep_dive.run(args)
            self.assertGreater(len(rows), 0)
            self.assertTrue(output_path.exists(), "Output CSV should be created")

            with output_path.open(newline="", encoding="utf8") as handle:
                reader = csv.DictReader(handle)
                csv_rows = list(reader)
            self.assertEqual(len(rows), len(csv_rows))
            self.assertIn("method", csv_rows[0])
            self.assertEqual(csv_rows[0]["method"], "random")

    def test_fast_deep_dive_synthetic_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="whale-mri-fast-") as tmpdir:
            output_path = Path(tmpdir) / "fast.csv"
            parser = deep_dive_fast.build_parser()
            args = parser.parse_args(
                [
                    "--synthetic",
                    "--seed",
                    "321",
                    "--methods",
                    "random",
                    "--rips-points",
                    "0",
                    "--max-points",
                    "5000",
                    "--thin-ratio",
                    "0.5",
                    "--min-points",
                    "500",
                    "--m",
                    "96",
                    "--out",
                    str(output_path),
                ]
            )

            rows = deep_dive_fast.run(args)
            self.assertGreater(len(rows), 0)
            self.assertTrue(output_path.exists(), "Output CSV should be created")

            with output_path.open(newline="", encoding="utf8") as handle:
                reader = csv.DictReader(handle)
                csv_rows = list(reader)
            self.assertEqual(len(rows), len(csv_rows))
            self.assertEqual(csv_rows[0]["method"], "random")


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    unittest.main()
