"""Regenerate results/RESULTS.md.

Tables 3.1/3.3 are recomputed; Table 3.2 and the follow-up study are read from the budget sweep's
CSVs. Run the sweep first if results/story_cube.csv is missing or stale:

    python analysis/extensive_sweep.py      # (or --fine / --max)
    python -m qmetrology.reproduce
"""
import os
import sys


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "analysis"))
    import make_results
    make_results.main()


if __name__ == "__main__":
    main()
