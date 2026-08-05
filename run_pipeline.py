"""
run_pipeline.py - one command to run the whole fraud-analytics pipeline.

    python run_pipeline.py            # generate synthetic data + run everything
    python run_pipeline.py --real     # use data/creditcard.csv if you downloaded it

Steps: generate -> ingest/clean -> EDA -> features -> train -> evaluate -> RCA
       -> A/B test -> dashboard.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")


def run(mod, *args):
    print(f"\n{'='*70}\n>>> {mod} {' '.join(args)}\n{'='*70}")
    rc = subprocess.call([sys.executable, os.path.join(SRC, mod), *args])
    if rc != 0:
        print(f"!! step {mod} failed (rc={rc})", file=sys.stderr)
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use downloaded data/creditcard.csv")
    ap.add_argument("--n", type=int, default=284807, help="synthetic rows")
    args = ap.parse_args()

    if not args.real:
        run("generate_synthetic_data.py", "--n", str(args.n))
    else:
        if not os.path.exists(os.path.join(HERE, "data", "creditcard.csv")):
            print("--real set but data/creditcard.csv missing. Run: python src/download_data.py")
            sys.exit(1)

    run("ingest_clean.py")
    run("eda.py")
    run("features.py")
    run("train.py")
    run("evaluate.py")
    run("rca.py")
    run("ab_test.py")
    run("build_dashboard.py")
    print("\nDONE. Open dashboard/index.html and see reports/ for all outputs.")


if __name__ == "__main__":
    main()
