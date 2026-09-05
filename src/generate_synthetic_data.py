"""
generate_synthetic_data.py
===========================
Deterministic generator for a ULB / IEEE-CIS *style* card-fraud dataset used to
demonstrate a fraud-analytics pipeline in an OFFLINE build environment.

WHY SYNTHETIC (read this):
--------------------------
This repository was built in a sandbox with no access to Kaggle / GitHub, so the
real ULB "creditcard.csv" (284,807 rows) and the IEEE-CIS dataset could not be
downloaded during the build. To keep the repo *clone-and-run* and to keep every
reported metric HONEST (computed on real data, not typed by hand), the pipeline
runs on a deterministic synthetic dataset that reproduces:

  * the ULB schema: Time, V1..V28 (anonymized PCA-style signals), Amount, Class
  * the real class imbalance (~0.17% fraud)
  * IEEE-CIS-style *entity* fields the anonymized ULB set lacks (card_id,
    device, corridor, channel, account age, session clickstream) so that
    velocity / entity-aggregation / behavioral features can be demonstrated.

The four fraud "modes" are engineered to map onto cross-border remittance risks:
  1. stolen_instrument  -> stolen card/instrument used in a new corridor
  2. account_takeover   -> ATO: device change + high velocity on an aged account
  3. mule               -> mule account: brand-new account, cash-out pattern
  4. structuring        -> many just-under-threshold sends to evade limits

IMPORTANT ON HONESTY: signal strengths below are fixed for reproducibility, but
they were iteratively adjusted after an early near-separable version to create
more overlap and a more useful demonstration benchmark. They were not estimated
from remittance outcomes. Whatever the current model achieves is reported as-is.
To reproduce on the REAL ULB dataset instead, run `python src/download_data.py`
(needs a Kaggle API token); the entire downstream pipeline is column-compatible.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

RNG_SEED = 42
N_DEFAULT = 284_807          # exact ULB row count
FRAUD_RATE = 0.00172         # ULB has 492 / 284807 = 0.1727%
TIME_SPAN_S = 172_800        # 2 days, like ULB
STRUCTURING_THRESHOLD = 1000.0

DEVICE_TYPES = ["ios_app", "android_app", "web_desktop", "web_mobile"]
CHANNELS = ["app", "web"]
MERCHANT_CATS = ["p2p_transfer", "bill_pay", "airtime_topup", "cash_pickup", "bank_deposit"]
# Remittance corridors (send_country -> receive_country)
CORRIDORS = ["US->MX", "US->IN", "US->PH", "US->NG", "UK->NG", "US->GT", "CA->IN", "AU->PH"]


def _softabs_lognormal(rng, mean_log, sigma_log, size):
    return np.exp(rng.normal(mean_log, sigma_log, size))


def generate(n: int = N_DEFAULT, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(round(n * FRAUD_RATE)))
    n_legit = n - n_fraud
    is_fraud = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # ---- Entities: a pool of cards, each with a behavioral profile ----------
    n_cards = max(2000, n // 6)
    card_pool = np.arange(n_cards)
    # legit txns pick cards ~ zipf-ish (some cards very active)
    card_pop = rng.zipf(1.6, size=n).astype(float)
    card_pop = np.clip(card_pop, 1, 500)
    card_prob = card_pop / card_pop.sum()
    card_id = rng.choice(card_pool, size=n, p=None)  # uniform base; fraud reassigned below

    # per-card home profile
    card_home_corridor = rng.choice(CORRIDORS, size=n_cards)
    card_home_device = rng.choice(DEVICE_TYPES, size=n_cards)
    card_typical_amt_log = rng.normal(3.1, 0.5, size=n_cards)  # ~ $22 median like ULB
    card_account_age = rng.integers(5, 1500, size=n_cards)     # days since signup

    # ---- Timestamps (seconds from start) ------------------------------------
    # Fraud is spread across the whole window (so the out-of-time holdout has a
    # stable number of fraud cases), with only a MILD night-hour skew.
    t_legit = rng.integers(0, TIME_SPAN_S, size=n_legit)
    t_fraud = rng.integers(0, TIME_SPAN_S, size=n_fraud)
    night = rng.random(n_fraud) < 0.25   # 25% of fraud pushed into 0-6am local
    t_fraud = np.where(
        night,
        rng.integers(0, 6 * 3600, size=n_fraud) + rng.choice([0, 86400], n_fraud),
        t_fraud,
    )
    t_fraud = np.clip(t_fraud, 0, TIME_SPAN_S - 1)
    time_s = np.concatenate([t_legit, t_fraud]).astype(int)

    # ---- Assign fraud modes -------------------------------------------------
    modes = np.array(["none"] * n, dtype=object)
    fraud_idx = np.where(is_fraud == 1)[0]
    mode_choice = rng.choice(
        ["stolen_instrument", "account_takeover", "mule", "structuring"],
        size=n_fraud, p=[0.34, 0.30, 0.18, 0.18],
    )
    modes[fraud_idx] = mode_choice

    # ---- Base behavioral columns (legit defaults) ---------------------------
    dev = card_home_device[card_id].copy()
    corr = card_home_corridor[card_id].copy()
    acct_age = card_account_age[card_id].astype(float).copy()
    channel = np.where(np.isin(dev, ["ios_app", "android_app"]), "app", "web")
    mcat = rng.choice(MERCHANT_CATS, size=n, p=[0.42, 0.18, 0.15, 0.15, 0.10])

    # Amount: lognormal around each card's typical, like ULB (mean ~88, median ~22)
    amount = _softabs_lognormal(rng, card_typical_amt_log[card_id], 0.6, n)
    amount = np.round(np.clip(amount, 0.5, 25_000), 2)

    # Clickstream / behavioral (session)
    n_clicks = rng.poisson(9, size=n).astype(float) + 1
    session_s = np.round(rng.gamma(2.2, 45, size=n), 1)  # session duration seconds
    device_change = np.zeros(n, dtype=int)  # device != card home device this session
    corridor_new = np.zeros(n, dtype=int)   # first time this card uses this corridor

    # small natural rate of benign device change / new corridor
    device_change = (rng.random(n) < 0.04).astype(int)
    dev = np.where(device_change == 1, rng.choice(DEVICE_TYPES, size=n), dev)
    corridor_new = (rng.random(n) < 0.06).astype(int)
    corr = np.where(corridor_new == 1, rng.choice(CORRIDORS, size=n), corr)

    # ---- Fraud-mode behavioral overrides ------------------------------------
    def midx(name):
        return np.where(modes == name)[0]

    # NOTE: overrides below are PROBABILISTIC and overlap heavily with legit
    # behavior on purpose. Legit traffic also shows device changes, new corridors,
    # high amounts and new accounts, so no single behavioral flag is a giveaway.

    # stolen_instrument: somewhat higher amount, often (not always) new corridor/device
    ix = midx("stolen_instrument")
    amount[ix] = np.round(amount[ix] * rng.uniform(1.5, 6.0, ix.size), 2)  # elevated vs card norm
    corridor_new[ix] = (rng.random(ix.size) < 0.6).astype(int)
    corr[ix] = np.where(corridor_new[ix] == 1, rng.choice(CORRIDORS, size=ix.size), corr[ix])
    device_change[ix] = (rng.random(ix.size) < 0.45).astype(int)

    # account_takeover: usually device change + rushed session; amount moderately up
    ix = midx("account_takeover")
    amount[ix] = np.round(amount[ix] * rng.uniform(1.3, 4.0, ix.size), 2)
    dc = rng.random(ix.size) < 0.6
    device_change[ix] = dc.astype(int)
    dev[ix] = np.where(dc, rng.choice(DEVICE_TYPES, size=ix.size), dev[ix])
    acct_age[ix] = rng.integers(120, 1500, size=ix.size)
    n_clicks[ix] = rng.poisson(6, size=ix.size) + 1  # somewhat rushed

    # mule: often new account + cash-out categories, but new legit accounts exist too
    ix = midx("mule")
    newacct = rng.random(ix.size) < 0.7
    acct_age[ix] = np.where(newacct, rng.integers(0, 45, size=ix.size), acct_age[ix])
    mcat[ix] = np.where(rng.random(ix.size) < 0.7,
                        rng.choice(["cash_pickup", "bank_deposit"], size=ix.size), mcat[ix])
    amount[ix] = np.round(amount[ix] * rng.uniform(1.2, 3.0, ix.size), 2)

    # structuring: ~60% just-under-threshold; concentrated on some cards for velocity
    ix = midx("structuring")
    near = rng.random(ix.size) < 0.6
    amount[ix] = np.where(
        near,
        np.round(rng.uniform(STRUCTURING_THRESHOLD * 0.9, STRUCTURING_THRESHOLD * 0.999, ix.size), 2),
        amount[ix],
    )
    str745 = rng.choice(card_pool[: max(50, n_cards // 200)], size=ix.size)
    card_id[ix] = str745
    amount = np.clip(amount, 0.5, 25_000)

    # Legit traffic that mimics fraud (hard negatives): some legit sends are large,
    # to a new corridor, on a new device, or from new accounts.
    legit_ix = np.where(is_fraud == 0)[0]
    big_legit = legit_ix[rng.random(legit_ix.size) < 0.03]
    amount[big_legit] = np.round(amount[big_legit] * rng.uniform(2, 8, big_legit.size), 2)
    amount = np.clip(amount, 0.5, 25_000)
    near_legit = legit_ix[rng.random(legit_ix.size) < 0.01]
    amount[near_legit] = np.round(rng.uniform(900, 999, near_legit.size), 2)

    # ---- Velocity features (needs ordering by card then time) ---------------
    df = pd.DataFrame({
        "card_id": card_id, "time_s": time_s, "amount": amount,
        "is_fraud": is_fraud, "mode": modes,
    })
    df = df.sort_values(["card_id", "time_s"]).reset_index()  # keep orig index
    df["txn_gap_s"] = df.groupby("card_id")["time_s"].diff()
    df["card_txn_rank"] = df.groupby("card_id").cumcount() + 1

    # 1h rolling count per card, vectorized via searchsorted within each group
    def _count_1h(g):
        t = g["time_s"].values  # already sorted ascending by time within card
        lo = np.searchsorted(t, t - 3600, side="left")
        idx = np.arange(len(t))
        return pd.Series(idx - lo + 1, index=g.index)

    df["txn_count_1h"] = df.groupby("card_id", group_keys=False).apply(
        _count_1h, include_groups=False)
    df = df.sort_values("index")
    txn_gap_s = df["txn_gap_s"].fillna(999999).values
    card_txn_rank = df["card_txn_rank"].values
    txn_count_1h = df["txn_count_1h"].values

    # Amplify velocity for ATO + structuring, but only for SOME rows (overlap)
    boost_mask = np.isin(modes, ["account_takeover", "structuring"]) & (rng.random(n) < 0.6)
    txn_count_1h = txn_count_1h + boost_mask * rng.integers(1, 5, size=n)

    # ---- V1..V28: anonymized PCA-style signals ------------------------------
    # Legit ~ N(0,1). Fraud gets MODEST class-conditional mean shifts on a subset
    # of components. Effect sizes are deliberately small with heavy overlap and
    # substantial label noise, so the benchmark is not separable. These fixed
    # values are demonstration assumptions, not estimates from remittance data.
    V = rng.normal(0, 1, size=(n, 28))
    shift = np.zeros(28)
    informative = {2: -0.95, 3: -0.8, 4: 0.85, 9: -0.7, 10: -0.9, 11: 0.8,
                   13: -0.6, 16: -0.65, 17: -0.85, 18: 0.6, 6: -0.5, 7: -0.45}
    for comp, s in informative.items():
        shift[comp] = s
    # mode-specific pattern so RCA can still separate modes
    mode_pattern = {
        "stolen_instrument": {13: -1.0, 17: -1.1},
        "account_takeover": {4: 1.2, 11: 1.1},
        "mule": {9: -1.1, 10: -1.2},
        "structuring": {2: -1.2, 3: -1.0},
    }
    fmask = is_fraud == 1
    # per-fraud-row random gain so effect strength varies (adds overlap)
    gain = rng.uniform(0.5, 1.2, size=(n, 1))
    V[fmask] += shift * gain[fmask]
    for name, patt in mode_pattern.items():
        ixm = midx(name)
        for comp, s in patt.items():
            V[ixm, comp] += s * gain[ixm, 0]
    # label noise: ~12% of fraud looks fully benign; ~0.2% of legit looks fraud-like
    benign_looking = fmask & (rng.random(n) < 0.12)
    V[benign_looking] -= shift * rng.uniform(0.7, 1.1, size=(benign_looking.sum(), 1))
    hard_neg = (~fmask) & (rng.random(n) < 0.002)
    V[hard_neg] += shift * rng.uniform(0.5, 0.9, size=(hard_neg.sum(), 1))
    # weak correlation with amount (like ULB where V carries some amount signal)
    V[:, 0] += (np.log1p(amount) - np.log1p(amount).mean()) / np.log1p(amount).std() * 0.25

    Vcols = {f"V{i+1}": np.round(V[:, i], 6) for i in range(28)}

    out = pd.DataFrame({
        "Time": time_s,
        **Vcols,
        "Amount": amount,
        # --- IEEE-CIS-style entity/behavioral fields (ULB lacks these) -------
        "card_id": card_id,
        "device_type": dev,
        "channel": channel,
        "corridor": corr,
        "merchant_category": mcat,
        "account_age_days": acct_age.astype(int),
        "device_change": device_change,
        "corridor_new": corridor_new,
        "n_clicks_session": n_clicks.astype(int),
        "session_duration_s": session_s,
        "txn_gap_s": np.round(txn_gap_s, 1),
        "card_txn_rank": card_txn_rank.astype(int),
        "txn_count_1h": txn_count_1h.astype(int),
        "fraud_mode": modes,   # ground-truth mode label (for RCA only; NOT a model feature)
        "Class": is_fraud,
    })

    # shuffle rows so order isn't class-sorted
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--out", type=str, default="data/synthetic_creditcard.csv")
    ap.add_argument("--sample-out", type=str, default="data/sample_1000.csv")
    args = ap.parse_args()

    df = generate(args.n, args.seed)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    df.head(1000).to_csv(args.sample_out, index=False)
    n_fraud = int(df["Class"].sum())
    print(f"[generate] wrote {len(df):,} rows -> {args.out}")
    print(f"[generate] fraud: {n_fraud:,} ({n_fraud/len(df)*100:.3f}%)")
    print(f"[generate] columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
