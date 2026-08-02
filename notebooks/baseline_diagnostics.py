"""Residual diagnostics for the two non-ML baselines.

Run:  ./.venv/bin/python notebooks/baseline_diagnostics.py

Fits Persistence and ClimatologyHour on the pre-2023 portion of Mooloolaba
and scores them on the same 2023-01-01 → 2024-12-31 AEST window used by
the linear-model sweep — so the headline persistence RMSE here is the same
number the sweep table is quoted against, and climatology provides the
"regress to the diurnal mean" floor.

Saves:
  baseline_residuals.png — 2 stacked residual time series (raw + 7-day
                            rolling mean) plus a residual-density panel.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecast as fc

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

TEST_START = "2023-01-01"
HORIZON_H = 24
HORIZON_STEPS = HORIZON_H * 2  # 30-min cadence

# Plot colour per baseline — kept consistent across all panels.
COLORS = {
    "Persistence":      "#1f77b4",
    "Climatology hour": "#2ca02c",
}


def _build_baselines() -> list[tuple[str, object]]:
    return [
        ("Persistence",      fc.PersistenceForecaster()),
        ("Climatology hour", fc.ClimatologyHourForecaster(horizon_steps=HORIZON_STEPS)),
    ]


def _residual_summary(resid: pd.Series) -> dict[str, float]:
    r = resid.dropna().to_numpy()
    return {
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "mae":  float(np.mean(np.abs(r))),
        "bias": float(np.mean(r)),
    }


def main() -> None:
    wave = fc.load_data(buoy="mooloolaba")
    wave = fc.restrict_to_years(wave, None, 2024)

    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    train_mask = wave.index < ts
    test_mask  = wave.index >= ts

    y_full  = fc.make_target(wave, horizon_steps=HORIZON_STEPS)
    y_train = y_full.loc[train_mask & y_full.notna()]
    y_test  = y_full.loc[test_mask  & y_full.notna()]
    X_train = wave.loc[y_train.index]

    print(f"train window  : {X_train.index.min().date()} → {X_train.index.max().date()}  ({len(X_train):,} rows)")
    print(f"test  window  : {y_test.index.min().date()} → {y_test.index.max().date()}  ({len(y_test):,} rows)")
    print()

    results = []
    for name, model in _build_baselines():
        model.fit(X_train, y_train)
        # Predict over the full series, then take only the test-window
        # slice. Persistence and ClimatologyHour both work on any X; this
        # also keeps the call shape uniform if a shift-based baseline is
        # ever reintroduced.
        preds_full = pd.Series(model.predict(wave), index=wave.index, name=name)
        preds = preds_full.loc[y_test.index]
        resid = (y_test - preds).rename(name)
        stats = _residual_summary(resid)
        results.append({"name": name, "preds": preds, "resid": resid, **stats})
        print(f"  {name:18s}  RMSE {stats['rmse']:.4f}   MAE {stats['mae']:.4f}   Bias {stats['bias']:+.4f}")

    # ------------------------------------------------------------------ #
    # Figure: 3 stacked residual time series + density panel
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [1, 1, 1.15]},
    )
    roll_steps = 7 * 48  # 7-day rolling window at 30-min cadence
    ymax = max(np.nanmax(np.abs(r["resid"].to_numpy())) for r in results)
    ylim = (-ymax * 1.05, ymax * 1.05)

    for ax, r in zip(axes[:2], results):
        c = COLORS[r["name"]]
        resid = r["resid"].dropna()
        ax.plot(resid.index, resid.values, color=c, linewidth=0.5, alpha=0.35)
        ax.plot(resid.rolling(roll_steps, min_periods=roll_steps // 4).mean(),
                color="black", linewidth=1.4, label="7-day rolling mean")
        ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
        ax.set_ylim(ylim)
        ax.set(ylabel="y − ŷ (m)",
               title=f"{r['name']}   RMSE {r['rmse']:.4f} m,  MAE {r['mae']:.4f} m,  Bias {r['bias']:+.4f} m")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    # Bottom: residual densities (both overlaid). Same x-range so the
    # spread/tail difference reads at a glance.
    ax = axes[2]
    edges = np.linspace(-2.5, 2.5, 121)
    for r in results:
        resid = r["resid"].dropna().to_numpy()
        ax.hist(resid, bins=edges, density=True, alpha=0.42,
                label=f"{r['name']}  (σ={resid.std():.3f})",
                color=COLORS[r["name"]])
    ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set(xlabel="residual y − ŷ (m)", ylabel="density",
           title="Residual distribution — 2023–2024 test window")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Baseline residuals on the 2023-01-01 → 2024-12-31 test window  "
        f"(Mooloolaba hsig_m, {HORIZON_H}-h horizon)",
        fontsize=12, y=1.002,
    )
    fig.tight_layout()
    out = FIG_DIR / "baseline_residuals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
