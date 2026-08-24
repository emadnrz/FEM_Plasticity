"""Run tests, NorSand checks, matching, and 2-D drained FE comparisons."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from comparison import run_all_comparisons  # noqa: E402
from fe_demo import run_fe_demo  # noqa: E402
from paper_figure1_verification import run_figure1_verification  # noqa: E402


def run_tests() -> None:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    run_tests()
    run_figure1_verification(ROOT / "results")
    run_all_comparisons(ROOT / "results")
    run_fe_demo(ROOT / "results")
    print(f"Results written to {ROOT / 'results'}")


if __name__ == "__main__":
    main()
