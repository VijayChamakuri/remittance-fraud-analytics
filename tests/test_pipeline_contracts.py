from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import build  # noqa: E402
from generate_synthetic_data import FRAUD_RATE, generate  # noqa: E402
from train import time_split  # noqa: E402


class FraudPipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.synthetic = generate(n=12_000, seed=42)

    def test_generator_is_deterministic_and_explicitly_synthetic(self) -> None:
        repeated = generate(n=12_000, seed=42)
        assert_frame_equal(self.synthetic, repeated)
        expected_fraud = round(len(self.synthetic) * FRAUD_RATE)
        self.assertEqual(int(self.synthetic["Class"].sum()), expected_fraud)
        self.assertIn("fraud_mode", self.synthetic.columns)
        self.assertEqual(
            set(self.synthetic.loc[self.synthetic["Class"] == 0, "fraud_mode"]),
            {"none"},
        )

    def test_time_split_is_chronological_and_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "Time": list(reversed(range(100))),
                "Class": [0] * 100,
                "fraud_mode": ["none"] * 100,
            }
        )
        train, validation, test = time_split(frame)
        self.assertEqual((len(train), len(validation), len(test)), (60, 10, 30))
        self.assertLessEqual(train["Time"].max(), validation["Time"].min())
        self.assertLessEqual(validation["Time"].max(), test["Time"].min())
        self.assertFalse(set(train.index) & set(validation.index))
        self.assertFalse(set(validation.index) & set(test.index))

    def test_feature_history_excludes_the_current_transaction(self) -> None:
        frame = self.synthetic.sort_values(["card_id", "Time"]).reset_index(drop=True)
        featured, feature_names = build(frame)
        first_for_card = frame.groupby("card_id", sort=False).cumcount() == 0
        self.assertTrue((featured.loc[first_for_card, "card_hist_txn_count"] == 0).all())
        self.assertNotIn("Class", feature_names)
        self.assertNotIn("fraud_mode", feature_names)
        self.assertNotIn("Time", feature_names)


if __name__ == "__main__":
    unittest.main()
