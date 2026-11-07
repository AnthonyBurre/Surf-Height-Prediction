"""Held-out 2025 scoring of the README's pre-committed candidates.

Run:  ./.venv/bin/python notebooks/heldout_2025.py [--no-log]

The README's "Real world performance" section commits, before 2025 was seen,
to three (horizon, model) pairs:

  - h=12h  HGB-on-persistence-residual  on ``tweed_mc`` (mooloolaba + tweed-heads + mountain-creek)
  - h=24h  Ridge(alpha=1)               on ``solo``     (mooloolaba only)
  - h=48h  Ridge(alpha=1)               on ``solo``

Scoring is a re-fit of the *same recipe on the same training data* — the
``horizon_sweep.run_cell`` code path verbatim: train on rows before
2023-01-01, robust-scaled mean-imputed ``Preprocessor`` for Ridge, native-NaN
HGB on ``y(t+h) − y(t)``. Only the scored window changes: 2025-01-01 →
2025-12-31 instead of 2023-01-01 → 2024-12-31. 2023–24 rows are neither
trained on nor scored.

Two stages:

1. **Reproduction check** — rebuild the 2015–2024 window from the freshly
   downloaded CSVs and re-score each candidate on the original 2023–24 test
   window, comparing to the ``hsweep_*`` records in ``experiments.jsonl``. The
   QLD portal silently republishes past years (README "Upstream revisions
   note"), so a mismatch here means the snapshot moved, not the model.
   Also asserts the training matrix built from the 2025-extended window is
   identical to the 2024 one (features are past-only, so extending the
   window must not touch training rows).

2. **Held-out 2025** — score each candidate plus persistence and
   hour-of-day climatology on 2025, with a two-week moving-block bootstrap
   95% CI on RMSE and on the paired RMSE gap to the best no-model baseline
   (same helpers as ``test_window_noise.py``). Logged to
   ``experiments.jsonl`` as ``heldout2025_<combo>_h<H>h_<model>``; names
   already present are not relogged.
"""
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
import forecast as fc
from forecast import summarise
from horizon_sweep import COMBOS, HGB_KW, RIDGE_KW, TEST_START
from test_window_noise import abs_ci, paired_ci

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

HELDOUT_YEAR = 2025
HELDOUT_START = f"{HELDOUT_YEAR}-01-01"
HELDOUT_END = f"{HELDOUT_YEAR + 1}-01-01"  # exclusive

# (horizon_h, combo, model) — the README's pre-committed list. Do not edit
# after seeing 2025 numbers.
CANDIDATES: list[tuple[int, str, str]] = [
    (12, "tweed_mc", "hgb"),
    (24, "solo",     "ridge"),
    (48, "solo",     "ridge"),
]

COMBO_BY_NAME = {c["name"]: c for c in COMBOS}


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_localize(fc.SOURCE_TZ)


def build(combo_name: str, year_max: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    combo = COMBO_BY_NAME[combo_name]
    wave, nb, wind = fc.load_sources(
        neighbours=combo["neighbours"], wind_stations=combo["wind"],
        year_min=combo["year_min"], year_max=year_max,
    )
    X = fc.build_design(wave, nb, wind, kind="engineered")
    if not X.index.equals(wave.index):
        raise AssertionError("design matrix and wave frame must share an index")
    return wave, X, ["mooloolaba"] + combo["neighbours"] + combo["wind"]


def score(
    model_kind: str, horizon_h: int, wave: pd.DataFrame, X: pd.DataFrame,
    test_start: str, test_end: str | None,
) -> dict:
    """Fit on rows < TEST_START, predict rows in [test_start, test_end).

    Mirrors ``horizon_sweep.run_cell`` (Ridge / HGB-residual) and
    ``horizon_sweep.compute_baselines`` (persistence / climatology-hour).
    """
    h_steps = fc.hours_to_steps(horizon_h)
    y = fc.make_target(wave, horizon_steps=h_steps)
    idx = X.index
    tr = idx < _ts(TEST_START)
    te = idx >= _ts(test_start)
    if test_end is not None:
        te &= idx < _ts(test_end)

    X_tr, X_te = X.loc[tr], X.loc[te]
    y_tr, y_te = y.loc[tr], y.loc[te]
    hs_tr, hs_te = wave["hsig_m"].loc[tr], wave["hsig_m"].loc[te]

    persist = fc.PersistenceForecaster().fit(hs_tr.to_frame(), y_tr)
    pp = np.asarray(persist.predict(hs_te.to_frame()), dtype=float)
    clim = fc.ClimatologyHourForecaster(horizon_steps=h_steps).fit(hs_tr.to_frame(), y_tr)
    cp = np.asarray(clim.predict(hs_te.to_frame()), dtype=float)

    preproc = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_tr)
    if model_kind == "ridge":
        res = fc.evaluate(
            Ridge(**RIDGE_KW), preproc.transform(X_tr), y_tr,
            preproc.transform(X_te), y_te, baseline_preds=pp,
        )
        preds, model = res.predictions, res.model
    elif model_kind == "hgb":
        X_tr_raw, X_te_raw = X_tr[preproc.kept_columns_], X_te[preproc.kept_columns_]
        y_res = y_tr - hs_tr
        mask = ~y_res.isna() & ~X_tr_raw.isna().any(axis=1)
        model = HistGradientBoostingRegressor(**HGB_KW)
        model.fit(X_tr_raw.loc[mask].to_numpy(), y_res.loc[mask].to_numpy())
        preds = hs_te.to_numpy() + model.predict(X_te_raw.to_numpy())
    else:
        raise ValueError(model_kind)

    return {
        "y": y_te.to_numpy(), "model": preds, "persistence": pp, "climatology": cp,
        "fitted": model, "X_tr": X_tr, "X_te": X_te,
    }


def _logged_metrics(name: str) -> dict | None:
    log = fc.read_log()
    rows = log[log["name"] == name]
    return None if rows.empty else rows.iloc[-1]["metrics"]


def reproduction_check() -> bool:
    print("=" * 72)
    print("1. Reproduction check — fresh snapshot vs logged 2023–24 scores")
    print("=" * 72)
    ok = True
    for h, combo, kind in CANDIDATES:
        wave24, X24, _ = build(combo, year_max=2024)
        r = score(kind, h, wave24, X24, TEST_START, None)
        for label, preds in (("persistence", r["persistence"]), (kind, r["model"])):
            name = f"hsweep_{combo}_h{h}h_{label}"
            logged = _logged_metrics(name)
            now = summarise(r["y"], preds)["RMSE"]
            if logged is None:
                print(f"  {name:<36} now {now*100:6.2f} cm   (no logged record)")
                continue
            d = (now - logged["RMSE"]) * 100
            flag = "ok" if abs(d) < 0.05 else "DRIFT"
            ok &= flag == "ok"
            print(f"  {name:<36} logged {logged['RMSE']*100:6.2f}  now {now*100:6.2f}  "
                  f"Δ {d:+.2f} cm  {flag}")

        # Extending the window to 2025 must leave the training matrix untouched.
        _, X25, _ = build(combo, year_max=HELDOUT_YEAR)
        tr = X24.index < _ts(TEST_START)
        pd.testing.assert_frame_equal(X24.loc[tr], X25.loc[X25.index < _ts(TEST_START)])
    print(f"  training matrices identical across 2024/2025 builds: yes")
    return ok


def heldout(log: bool) -> list[dict]:
    print("\n" + "=" * 72)
    print(f"2. Held-out {HELDOUT_YEAR} — train < {TEST_START}, score "
          f"{HELDOUT_START} → {HELDOUT_YEAR}-12-31")
    print("=" * 72)
    existing = set(fc.read_log()["name"]) if log else set()
    snapshot = date.today().isoformat()
    rows = []
    for h, combo, kind in CANDIDATES:
        wave, X, sources = build(combo, year_max=HELDOUT_YEAR)
        if wave.index.max() < _ts(f"{HELDOUT_YEAR}-12-31"):
            raise RuntimeError(f"{combo}: data ends {wave.index.max()} — {HELDOUT_YEAR} incomplete")
        r = score(kind, h, wave, X, HELDOUT_START, HELDOUT_END)
        y, pm, pp, cp = r["y"], r["model"], r["persistence"], r["climatology"]

        m = summarise(y, pm, y_pred_baseline=pp)
        p_rmse = summarise(y, pp)["RMSE"]
        c_rmse = summarise(y, cp)["RMSE"]
        best_name, best_preds = (("persistence", pp) if p_rmse <= c_rmse
                                 else ("climatology", cp))
        skill_best = fc.skill_score(y, pm, best_preds)
        _, lo, hi = abs_ci(pm - y)
        gap, glo, ghi = paired_ci(pm - y, best_preds - y)
        row = {
            "h": h, "combo": combo, "model": kind, "RMSE": m["RMSE"], "ci": (lo, hi),
            "persistence": p_rmse, "climatology": c_rmse, "best_baseline": best_name,
            "skill_vs_persistence": m["SkillVsBaseline"], "skill_vs_best": skill_best,
            "gap_vs_best": (gap, glo, ghi),
        }
        rows.append(row)

        print(f"\n  h={h}h  {kind.upper()} on {combo}  ({np.isfinite(y).sum():,} scored rows)")
        print(f"    model        RMSE {m['RMSE']*100:6.2f} cm  95% CI [{lo*100:.1f}, {hi*100:.1f}]"
              f"  MAE {m['MAE']*100:.2f}  bias {m['Bias']*100:+.2f}")
        print(f"    persistence  RMSE {p_rmse*100:6.2f} cm")
        print(f"    climatology  RMSE {c_rmse*100:6.2f} cm")
        print(f"    skill vs persistence {m['SkillVsBaseline']:+.3f}   vs best ({best_name}) "
              f"{skill_best:+.3f}")
        print(f"    paired RMSE gap vs {best_name}: {gap*100:+.2f} cm  "
              f"95% CI [{glo*100:+.2f}, {ghi*100:+.2f}]")

        if not log:
            continue
        te_idx = r["X_te"].index
        tr_idx = r["X_tr"].index
        extra = {
            "heldout_year": HELDOUT_YEAR, "horizon_h": h,
            "horizon_steps": fc.hours_to_steps(h), "combo": combo,
            "committed_candidate": True, "data_snapshot": snapshot,
            "window": f"{wave.index.min().date()}:{wave.index.max().date()}",
        }
        recs = [
            (f"heldout{HELDOUT_YEAR}_{combo}_h{h}h_{kind}", m, pm, r["fitted"], sources,
             {**extra, "imputation": "mean", "scaling": "robust" if kind == "ridge" else None,
              "skill_vs_best_baseline": skill_best, "best_baseline": best_name,
              "rmse_gain_vs_best_baseline": 1 - m["RMSE"] / min(p_rmse, c_rmse),
              "rmse_ci95": [lo, hi], "paired_gap_vs_best_ci95": [glo, ghi],
              **({"hgb_mode": "persistence_residual", "nan_handling": "native_hgb"}
                 if kind == "hgb" else {})}),
            (f"heldout{HELDOUT_YEAR}_{combo}_h{h}h_persistence", summarise(y, pp), pp,
             None, ["mooloolaba"], extra),
            (f"heldout{HELDOUT_YEAR}_{combo}_h{h}h_climatology", summarise(y, cp, pp), cp,
             None, ["mooloolaba"], extra),
        ]
        for name, metrics, preds, fitted, src, ex in recs:
            if name in existing:
                print(f"    (already logged: {name})")
                continue
            fc.log_run(
                fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=fitted),
                data_sources=src, train_index=tr_idx, test_index=te_idx,
                n_features=r["X_tr"].shape[1] if fitted is not None else 1,
                model_class=None if fitted is not None else name.rsplit("_", 1)[-1],
                extra=ex,
            )
    return rows


def main() -> None:
    log = "--no-log" not in sys.argv
    reproduced = reproduction_check()
    if not reproduced:
        print("\n  NOTE: snapshot drift — 2023–24 scores no longer match the log. "
              "2025 numbers below are still a valid blind test of the recipe, but "
              "compare them to the re-scored 2023–24 values above, not the README table.")
    rows = heldout(log)

    print("\n" + "=" * 72)
    print("README table rows")
    print("=" * 72)
    for r in rows:
        # README convention: skill = RMSE reduction vs the best no-model baseline.
        base = r[r["best_baseline"]]
        print(f"| {HELDOUT_YEAR} | {r['h']}h | {'HGB' if r['model'] == 'hgb' else 'Ridge'} "
              f"| {r['RMSE']*100:.1f} | {base*100:.1f} ({r['best_baseline']}) "
              f"| {(1 - r['RMSE']/base)*100:+.1f}% |")


if __name__ == "__main__":
    main()
