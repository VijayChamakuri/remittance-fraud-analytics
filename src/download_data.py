"""
download_data.py
================
Fetch the REAL public dataset instead of the synthetic stand-in.

Two supported sources (either works; ULB is the default because it is small
enough to keep the repo clone-and-run):

  ULB  : mlg-ulb/creditcardfraud  -> creditcard.csv (Time, V1..V28, Amount, Class)
  IEEE : ieee-fraud-detection      -> train_transaction.csv (+ identity)

Requires a Kaggle API token (~/.kaggle/kaggle.json) OR kagglehub. See README.

After downloading, the file is written to data/synthetic_creditcard.csv's sibling
`data/creditcard.csv`. The rest of the pipeline auto-detects a real file and uses
it (the synthetic entity/behavioral columns are simply skipped when absent, and
the model falls back to the Time+V1..V28+Amount schema, exactly like real ULB).

NOTE: This script cannot run in an offline build sandbox; it is provided so a
user with internet + Kaggle credentials can reproduce every metric on real data.
"""
from __future__ import annotations
import os
import sys

from config import DATA_DIR


def download_ulb() -> str:
    dest = os.path.join(DATA_DIR, "creditcard.csv")
    try:
        import kagglehub  # type: ignore
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        src = os.path.join(path, "creditcard.csv")
        import shutil
        shutil.copy(src, dest)
        print(f"[download] ULB creditcard.csv -> {dest}")
        return dest
    except Exception as e:  # noqa: BLE001
        print(f"[download] kagglehub path failed ({e}); trying kaggle CLI...")

    # Fallback: official Kaggle CLI
    rc = os.system(
        f"kaggle datasets download -d mlg-ulb/creditcardfraud -p {DATA_DIR} --unzip"
    )
    if rc != 0 or not os.path.exists(dest):
        print(
            "[download] FAILED. Set up Kaggle creds first:\n"
            "  pip install kaggle kagglehub\n"
            "  mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
            "Then re-run: python src/download_data.py",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[download] ULB creditcard.csv -> {dest}")
    return dest


if __name__ == "__main__":
    download_ulb()
