"""How much of the model-selection table is signal vs test-window luck?

This project scores everything on one fixed test window (2023-01-01 →
2024-12-31), so a 1 cm RMSE gap between two models might just be which storms
happened to land after the cutoff. There is no cross-validation here to average
that out (an earlier rolling-origin build explored that separately), so we
quantify the single-window uncertainty directly with a moving-block bootstrap.

Run:
    ./.venv/bin/python notebooks/test_window_noise.py

Three reports, all on the primary-buoy (`solo`) model so they are directly
comparable:

1. **Absolute RMSE 95% CI** per horizon for the shipped Ridge-on-primary model
   — "how precisely does this one window pin the model's skill?" Block length is
   two weeks so storm-scale autocorrelation is preserved.

2. **Paired RMSE-difference CI** (HGB − Ridge) per horizon. Because both models
   see the same storms their errors are correlated, so the paired difference is
   resolved far more tightly than the overlapping absolute CIs suggest — this is
   the right tool for "is model A actually better than model B here?"

3. **dev-nuke config cross-check.** An earlier rolling-origin model selection
   settled on primary-only `Ridge(alpha=10)` and a direct-on-level
   `HGB(max_iter=300, lr=0.06, min_samples_leaf=200)`. We re-run those configs
   here and compare to our `Ridge(alpha=1)` / residual-`HGB` to confirm whether
   they would change anything (they do not — the gaps are inside the band).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
import forecast as fc
from forecast import summarise
from horizon_sweep import COMBOS, HGB_KW, TEST_START, build_combo

HORIZONS_H = [6, 12, 24, 36, 48, 72]
BLOCK = 672          # 2 weeks of 30-min steps — exceeds storm-event dependence
N_BOOT = 2000
SEED = 0

# dev-nuke frozen recipes (primary-only, horizon-independent).
DN_RIDGE_ALPHA = 10.0
DN_HGB = dict(max_iter=300, learning_rate=0.06, min_samples_leaf=200, max_depth=None,
              max_leaf_nodes=31, l2_regularization=0.0, early_stopping=True, random_state=42)


def _rmse(y, p):
    return summarise(y, p)["RMSE"]


def _block_index(n, rng):
    """Indices for one moving-block resample of length n."""
    nblocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=nblocks)
    return (starts[:, None] + np.arange(BLOCK)).ravel()[:n]


def abs_ci(err):
    """95% CI of RMSE from a residual series via moving-block bootstrap."""
    rng = np.random.default_rng(SEED)
    e = err[~np.isnan(err)]
    sq = e ** 2
    boot = np.array([np.sqrt(sq[_block_index(len(sq), rng)].mean()) for _ in range(N_BOOT)])
    return float(np.sqrt(sq.mean())), *np.percentile(boot, [2.5, 97.5])


def paired_ci(eA, eB):
    """95% CI of RMSE(eA) - RMSE(eB) with the SAME blocks drawn for both."""
    rng = np.random.default_rng(SEED)
    mask = ~np.isnan(eA) & ~np.isnan(eB)
    a, b = eA[mask] ** 2, eB[mask] ** 2
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = _block_index(len(a), rng)
        boot[k] = np.sqrt(a[idx].mean()) - np.sqrt(b[idx].mean())
    return float(np.sqrt(a.mean()) - np.sqrt(b.mean())), *np.percentile(boot, [2.5, 97.5])


def main() -> None:
    solo = next(c for c in COMBOS if c["name"] == "solo")
    wave, X, _ = build_combo(solo)
    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)

    abs_rows, paired_rows, dn_rows = [], [], []
    for h in HORIZONS_H:
        y = fc.make_target(wave, horizon_steps=fc.hours_to_steps(h))
        pX, pW = int(X.index.searchsorted(ts)), int(wave.index.searchsorted(ts))
        X_tr, X_te = X.iloc[:pX], X.iloc[pX:]
        y_tr, y_te = y.iloc[:pX], y.iloc[pX:]
        wave_tr, wave_te = wave["hsig_m"].iloc[:pW], wave["hsig_m"].iloc[pW:]
        yte = y_te.to_numpy()

        pre = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_tr)
        Xtr_i, Xte_i = pre.transform(X_tr), pre.transform(X_te)
        Xtr_r, Xte_r = X_tr[pre.kept_columns_], X_te[pre.kept_columns_]
        rm = ~y_tr.isna()

        ridge1 = Ridge(alpha=1.0).fit(Xtr_i[rm.to_numpy()], y_tr[rm]).predict(Xte_i)
        ridge10 = Ridge(alpha=DN_RIDGE_ALPHA).fit(Xtr_i[rm.to_numpy()], y_tr[rm]).predict(Xte_i)

        # ours: HGB on the persistence residual
        yres = y_tr - wave_tr
        rmask = ~yres.isna() & ~Xtr_r.isna().any(axis=1)
        hgb_ours = wave_te.to_numpy() + HistGradientBoostingRegressor(**HGB_KW).fit(
            Xtr_r.loc[rmask].to_numpy(), yres.loc[rmask].to_numpy()).predict(Xte_r.to_numpy())
        # dev-nuke: HGB direct on the level (its actual recipe)
        lmask = ~y_tr.isna() & ~Xtr_r.isna().any(axis=1)
        hgb_dn = HistGradientBoostingRegressor(**DN_HGB).fit(
            Xtr_r.loc[lmask].to_numpy(), y_tr.loc[lmask].to_numpy()).predict(Xte_r.to_numpy())

        pt, lo, hi = abs_ci(yte - ridge1)
        abs_rows.append((h, pt, lo, hi, (hi - lo) / 2 * 100))

        d, plo, phi = paired_ci(yte - hgb_ours, yte - ridge1)
        sig = "yes" if (plo > 0 or phi < 0) else "tie"
        paired_rows.append((h, d * 100, plo * 100, phi * 100, sig))

        dn_rows.append((h, (_rmse(yte, ridge10) - _rmse(yte, ridge1)) * 100,
                        (_rmse(yte, hgb_dn) - _rmse(yte, hgb_ours)) * 100))

    print("\n1) Absolute RMSE 95% CI — Ridge on primary buoy (block bootstrap, 2-wk)")
    print(f"   {'h':>3} {'RMSE':>7} {'lo':>7} {'hi':>7} {'±cm':>6}")
    for h, pt, lo, hi, half in abs_rows:
        print(f"   {h:>3} {pt:7.3f} {lo:7.3f} {hi:7.3f} {half:6.1f}")

    print("\n2) Paired RMSE diff (HGB − Ridge), primary buoy, 95% CI")
    print(f"   {'h':>3} {'Δcm':>7} {'lo':>7} {'hi':>7}  real?")
    for h, d, lo, hi, sig in paired_rows:
        print(f"   {h:>3} {d:7.2f} {lo:7.2f} {hi:7.2f}  {sig}")

    print("\n3) dev-nuke configs − ours (cm); both inside the band above = nothing to import")
    print(f"   {'h':>3} {'ridge α10−α1':>13} {'hgb dn(lvl)−ours':>17}")
    for h, dr, dh in dn_rows:
        print(f"   {h:>3} {dr:13.2f} {dh:12.2f}")


if __name__ == "__main__":
    main()
