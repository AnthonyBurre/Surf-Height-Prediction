"""Horizon sweep — how each (architecture, source) combo performs across horizons.

Run:  ./.venv/bin/python notebooks/horizon_sweep.py

Loops six forecast horizons (6h, 12h, 24h, 36h, 48h, 72h) over four source
combos that came out of the linear sweep:

  - solo      — mooloolaba only, no neighbours, no wind  (2015-2024)
  - tweed_mc  — mooloolaba + tweed-heads + mountain-creek wind  (2015-2024)
  - baseline  — mooloolaba + 5 neighbour buoys + 3 wind stations  (2015-2024)
  - wide      — mooloolaba + 7 neighbour buoys + 4 wind stations  (2019-2024)

For each combo the feature matrix is built ONCE (horizon-independent); only
the target column and `evaluate` call are redone per horizon. Models: Ridge,
Lasso, HGB-on-persistence-residual, a nanmean ensemble of the three, plus
the four sequence forecasters (RNN/GRU/LSTM/TCN) using the best hyperparams
picked from the narrow + wide sequence sweeps. Sequence models only run on
the `baseline` and `wide` combos — they need a full circular-encoded input
frame and the seq sweep only fitted those two feature sets.

All runs share the same pinned test-window cutoff (2023-01-01 AEST) so skill
scores at any one horizon are directly comparable across combos. Persistence
is computed per-horizon (it varies a lot — the autocorrelation collapses).

Saves:
  horizon_sweep.png — single panel: RMSE vs horizon, one line per model
                       family (Ridge / HGB-on-residual / GRU). Each family is
                       drawn at its lowest RMSE across the hand-picked source
                       combos, against the better of the two no-model
                       baselines (persistence ∪ climatology). The figure is
                       deliberately reduced to these four lines — the headline
                       comparison is family-vs-family, not combo-vs-combo.
  experiments.jsonl entries under names 'hsweep_<combo>_h<H>h_<model>' and
                       'hsweep_seq_<combo>_h<H>h_<arch>'.

Linear cells are not relogged on rerun (log=False) — the first run already
laid down 120 hsweep entries and the in-memory metrics are what the plot
uses. Sequence cells log every time.

Iterate on the chart without paying the 90-min sweep cost:
    ./.venv/bin/python notebooks/horizon_sweep.py --plot-only
which reads results from experiments.jsonl and just re-renders the figure.
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge

import forecast as fc
from forecast import summarise

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

HORIZONS_H: list[int] = [6, 12, 24, 36, 48, 72]
TEST_START = "2023-01-01"

COMBOS: list[dict] = [
    {
        "name": "solo",
        "year_min": None, "year_max": 2024,
        "neighbours": [],
        "wind":       [],
    },
    {
        "name": "tweed_mc",
        "year_min": None, "year_max": 2024,
        "neighbours": ["tweed-heads"],
        "wind":       ["mountain-creek"],
    },
    {
        "name": "baseline",
        "year_min": None, "year_max": 2024,
        "neighbours": ["caloundra", "brisbane", "gold-coast", "north-moreton-bay", "tweed-heads"],
        "wind":       ["mountain-creek", "deception-bay", "lytton"],
    },
    {
        "name": "wide",
        "year_min": 2019, "year_max": 2024,
        "neighbours": ["caloundra", "brisbane", "gold-coast", "north-moreton-bay",
                       "palm-beach", "tweed-heads", "wide-bay"],
        "wind":       ["mountain-creek", "deception-bay", "lytton", "southport"],
    },
]

COMBO_COLORS = {
    "solo":     "#1f77b4",
    "tweed_mc": "#ff7f0e",
    "baseline": "#2ca02c",
    "wide":     "#d62728",
}

# Two baselines plotted as gray reference lines on every architecture panel.
# Persistence is the project's headline baseline (skill ≡ 0 by definition);
# climatology-hour is the "regress to the diurnal mean" floor that any real
# model must beat. Naming them here keeps the plot labels honest.
BASELINE_STYLES = {
    "persistence":      {"linestyle": "-",  "color": "#000000", "linewidth": 1.6, "alpha": 0.9},
    "climatology_hour": {"linestyle": ":",  "color": "#555555", "linewidth": 1.5, "alpha": 0.85},
}
LINEAR_ARCHITECTURES = ["ridge", "lasso", "hgb"]
SEQ_ARCHITECTURES    = ["rnn", "gru", "lstm", "tcn"]
ARCHITECTURES        = LINEAR_ARCHITECTURES + SEQ_ARCHITECTURES

# Combos that have a fitted sequence-sweep config. Solo / tweed_mc don't —
# they'd need their own seq sweep first (probably not interesting; the
# sequence models depend on neighbour breadth to win).
SEQ_COMBOS = ["baseline", "wide"]

# Hyperparams — kept exactly in step with the linear playground defaults
# so the h=12 row of this sweep reproduces the v1 baseline numbers in the
# README's "Linear models" table.
RIDGE_KW = {"alpha": 1.0}
LASSO_KW = {"alpha": 0.001, "max_iter": 10000}
HGB_KW   = {
    "max_iter": 800, "learning_rate": 0.03, "max_depth": 6,
    "min_samples_leaf": 50, "l2_regularization": 1.0, "random_state": 42,
    "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 40,
}

# Best (combo, arch) sequence-model config from the seq sweeps. Each value
# is the kwargs dict that build_seq_model() unpacks into the matching
# Forecaster constructor. Picked by lowest RMSE within each (set, arch)
# group of the seqsweep_<set>_<arch>_* rows in experiments.jsonl.
SEQ_CONFIGS: dict[tuple[str, str], dict] = {
    # --- baseline (narrow) ---
    ("baseline", "rnn"):  {"seq_len": 48, "hidden": 128, "num_layers": 2,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.0},
    ("baseline", "gru"):  {"seq_len": 48, "hidden":  64, "num_layers": 1,
                            "epochs": 2, "weight_decay": 0.0,  "rnn_dropout": 0.0},
    ("baseline", "lstm"): {"seq_len": 48, "hidden": 128, "num_layers": 1,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.0},
    ("baseline", "tcn"):  {"seq_len": 48, "hidden": 128, "num_layers": 2,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.2},
    # --- wide ---
    ("wide",     "rnn"):  {"seq_len": 48, "hidden": 256, "num_layers": 2,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.1},
    ("wide",     "gru"):  {"seq_len": 48, "hidden":  64, "num_layers": 1,
                            "epochs": 2, "weight_decay": 0.0,  "rnn_dropout": 0.0},
    ("wide",     "lstm"): {"seq_len": 48, "hidden": 128, "num_layers": 2,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.1},
    ("wide",     "tcn"):  {"seq_len": 48, "hidden": 128, "num_layers": 4,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.2},
}

SEQ_BATCH_SIZE = 512
SEQ_LR = 1e-3
SEQ_SEED = 42
SEQ_SCALER = "robust"


# ---------------------------------------------------------------------------
# Feature build (horizon-independent — done once per combo)
# ---------------------------------------------------------------------------

def build_combo(combo: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (wave_df, X_features, source_labels) for a combo."""
    wave, neighbours, wind = fc.load_sources(
        neighbours=combo["neighbours"], wind_stations=combo["wind"],
        year_min=combo["year_min"], year_max=combo["year_max"],
    )
    X = fc.build_design(wave, neighbours, wind, kind="engineered")
    sources = ["mooloolaba"] + combo["neighbours"] + combo["wind"]
    return wave, X, sources


def build_seq_combo(combo: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Same window as build_combo but emits the circular-encoded seq frame.

    Sequence forecasters window their own input over time and expect raw
    channels + sin/cos direction columns, NOT the lag/rolling feature matrix
    used by the linear/tree models.
    """
    wave, neighbours, wind = fc.load_sources(
        neighbours=combo["neighbours"], wind_stations=combo["wind"],
        year_min=combo["year_min"], year_max=combo["year_max"],
    )
    X = fc.build_design(wave, neighbours, wind, kind="raw")
    sources = ["mooloolaba"] + combo["neighbours"] + combo["wind"]
    return wave, X, sources


# ---------------------------------------------------------------------------
# Non-ML baselines — Mooloolaba-only, combo-independent
# ---------------------------------------------------------------------------

def compute_baselines(wave_full: pd.DataFrame, horizon_h: int) -> dict[str, dict[str, float]]:
    """Persistence + ClimatologyHour at one horizon.

    Both only need the Mooloolaba ``hsig_m`` column, so they're computed
    once per horizon (not once per combo). Returned skill is vs Persistence.
    """
    horizon_steps = fc.hours_to_steps(horizon_h)
    y = fc.make_target(wave_full, horizon_steps=horizon_steps)
    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos_w = _pinned_split(wave_full.index, ts)
    X_p_tr = wave_full[["hsig_m"]].iloc[:pos_w]
    X_p_te = wave_full[["hsig_m"]].iloc[pos_w:]
    y_tr, y_te = y.iloc[:pos_w], y.iloc[pos_w:]

    # Persistence — the reference for climatology's skill score
    persist = fc.evaluate(
        fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te,
        name=f"persistence_h{horizon_h}h",
    )
    pp = persist.predictions

    # ClimatologyHour — learn hour-of-target-time mean from train
    ch = fc.ClimatologyHourForecaster(horizon_steps=horizon_steps)
    ch.fit(X_p_tr, y_tr)
    ch_preds = ch.predict(X_p_te)

    return {
        "persistence":      {**persist.metrics, "SkillVsBaseline": 0.0},
        "climatology_hour": summarise(y_te.to_numpy(), ch_preds, y_pred_baseline=pp),
    }


# ---------------------------------------------------------------------------
# One (combo, horizon) cell — fit/predict/score/log every model
# ---------------------------------------------------------------------------

def _pinned_split(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    return int(idx.searchsorted(ts))


def run_cell(
    combo: dict,
    horizon_h: int,
    wave: pd.DataFrame,
    X: pd.DataFrame,
    sources: list[str],
    *,
    log: bool,
) -> dict[str, dict[str, float]]:
    """Fit ridge/lasso/hgb-residual/ensemble at one horizon. Return per-model metrics."""
    horizon_steps = fc.hours_to_steps(horizon_h)
    y = fc.make_target(wave, horizon_steps=horizon_steps)

    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos   = _pinned_split(X.index, ts)
    pos_w = _pinned_split(wave.index, ts)
    X_tr, X_te = X.iloc[:pos], X.iloc[pos:]
    y_tr, y_te = y.iloc[:pos], y.iloc[pos:]
    wave_tr = wave["hsig_m"].iloc[:pos_w]
    wave_te = wave["hsig_m"].iloc[pos_w:]
    X_p_tr = wave[["hsig_m"]].iloc[:pos_w]
    X_p_te = wave[["hsig_m"]].iloc[pos_w:]

    preproc = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_tr)
    X_tr_imp = preproc.transform(X_tr)
    X_te_imp = preproc.transform(X_te)
    X_tr_raw = X_tr[preproc.kept_columns_]  # HGB-native NaN
    X_te_raw = X_te[preproc.kept_columns_]

    name_prefix = f"hsweep_{combo['name']}_h{horizon_h}h"
    window_str = f"{wave.index.min().date()}:{wave.index.max().date()}"
    extra = {
        "window": window_str, "imputation": "mean", "scaling": "robust",
        "horizon_h": horizon_h, "horizon_steps": horizon_steps,
        "combo": combo["name"], "n_neighbours": len(combo["neighbours"]),
        "wind_stations": combo["wind"],
    }

    # --- Persistence (per-horizon — gets worse as h grows) ---------------
    persist = fc.evaluate_and_log(
        fc.PersistenceForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name=f"{name_prefix}_persistence",
        data_sources=["mooloolaba"], extra={"window": window_str, "horizon_h": horizon_h},
        log=log,
    )
    pp = persist.predictions

    # --- Ridge ------------------------------------------------------------
    ridge = fc.evaluate_and_log(
        Ridge(**RIDGE_KW),
        X_tr_imp, y_tr, X_te_imp, y_te,
        name=f"{name_prefix}_ridge", baseline_preds=pp,
        data_sources=sources, extra=extra, log=log,
    )

    # --- Lasso ------------------------------------------------------------
    lasso = fc.evaluate_and_log(
        Lasso(**LASSO_KW),
        X_tr_imp, y_tr, X_te_imp, y_te,
        name=f"{name_prefix}_lasso", baseline_preds=pp,
        data_sources=sources, extra=extra, log=log,
    )

    # --- HGB on persistence residual --------------------------------------
    # y_residual = y(t+h) - y(t).  At long h this delta has more variance than
    # the level itself, which is exactly why letting HGB learn the delta beats
    # learning the level directly.
    y_res = y_tr - wave_tr
    res_mask = ~y_res.isna() & ~X_tr_raw.isna().any(axis=1)
    hgb_model = HistGradientBoostingRegressor(**HGB_KW)
    hgb_model.fit(X_tr_raw.loc[res_mask].to_numpy(), y_res.loc[res_mask].to_numpy())
    hgb_preds = wave_te.to_numpy() + hgb_model.predict(X_te_raw.to_numpy())
    hgb_metrics = summarise(y_te.to_numpy(), hgb_preds, y_pred_baseline=pp)
    hgb_result = fc.EvaluationResult(
        name=f"{name_prefix}_hgb", metrics=hgb_metrics,
        predictions=hgb_preds, model=hgb_model,
    )
    if log:
        fc.log_run(
            hgb_result, data_sources=sources,
            train_index=X_tr.index, test_index=X_te.index, n_features=X_tr.shape[1],
            extra={**extra, "hgb_mode": "persistence_residual", "nan_handling": "native_hgb"},
        )

    # --- Ensemble (nanmean of the three) ---------------------------------
    members = [ridge.predictions, lasso.predictions, hgb_preds]
    ens_preds = np.nanmean(np.vstack(members), axis=0)
    ens_metrics = summarise(y_te.to_numpy(), ens_preds, y_pred_baseline=pp)
    ens_result = fc.EvaluationResult(
        name=f"{name_prefix}_ensemble", metrics=ens_metrics,
        predictions=ens_preds, model=None,
    )
    if log:
        fc.log_run(
            ens_result, data_sources=sources,
            train_index=X_tr.index, test_index=X_te.index, n_features=X.shape[1],
            model_class="NanMeanEnsemble",
            extra={**extra, "members": [r.name for r in (ridge, lasso, hgb_result)],
                   "combiner": "nanmean"},
        )

    return {
        "persistence": persist.metrics,
        "ridge":       ridge.metrics,
        "lasso":       lasso.metrics,
        "hgb":         hgb_metrics,
        "ensemble":    ens_metrics,
    }


# ---------------------------------------------------------------------------
# Sequence cell — fit one (arch, combo, horizon)
# ---------------------------------------------------------------------------

def _build_seq_model(arch: str, cfg: dict):
    common = dict(
        seq_len=cfg["seq_len"], epochs=cfg["epochs"],
        batch_size=SEQ_BATCH_SIZE, lr=SEQ_LR, seed=SEQ_SEED,
        device=fc.auto_device(), verbose=False, scaler=SEQ_SCALER,
        weight_decay=cfg["weight_decay"],
    )
    if arch == "tcn":
        return fc.TCNForecaster(
            channels=(cfg["hidden"],) * cfg["num_layers"],
            dropout=cfg["rnn_dropout"], **common,
        )
    cls = {"rnn": fc.SimpleRNNForecaster,
           "gru": fc.GRUForecaster,
           "lstm": fc.LSTMForecaster}[arch]
    return cls(hidden=cfg["hidden"], num_layers=cfg["num_layers"],
               rnn_dropout=cfg["rnn_dropout"], **common)


def run_seq_cell(
    combo: dict,
    horizon_h: int,
    arch: str,
    cfg: dict,
    wave: pd.DataFrame,
    X_seq: pd.DataFrame,
    sources: list[str],
    *,
    log: bool,
) -> dict[str, float]:
    """Fit one sequence forecaster at one horizon; return metrics dict.

    Persistence at this (combo, horizon) is computed locally so skill is
    against the same baseline as the linear runs. Re-uses the pinned
    TEST_START split exactly like run_cell so seq numbers slot into the
    same plot rows.
    """
    horizon_steps = fc.hours_to_steps(horizon_h)
    y = fc.make_target(wave, horizon_steps=horizon_steps)

    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos   = _pinned_split(X_seq.index, ts)
    pos_w = _pinned_split(wave.index,   ts)
    X_tr, X_te = X_seq.iloc[:pos], X_seq.iloc[pos:]
    y_tr, y_te = y.iloc[:pos],     y.iloc[pos:]
    X_p_tr = wave[["hsig_m"]].iloc[:pos_w]
    X_p_te = wave[["hsig_m"]].iloc[pos_w:]

    X_tr_imp, X_te_imp = fc.mean_impute(X_tr, X_te)
    persist = fc.evaluate(
        fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te,
        name=f"persistence_h{horizon_h}h_{combo['name']}",
    )
    pp = persist.predictions

    model = _build_seq_model(arch, cfg)
    t0 = time.time()
    model.fit(X_tr_imp, y_tr)
    preds = model.predict(X_te_imp)
    elapsed = time.time() - t0
    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=pp)

    name = f"hsweep_seq_{combo['name']}_h{horizon_h}h_{arch}"
    if log:
        fc.log_run(
            fc.EvaluationResult(name=name, metrics=metrics,
                                predictions=preds, model=model),
            data_sources=sources,
            train_index=X_tr.index, test_index=X_te.index,
            n_features=X_seq.shape[1],
            extra={
                "combo": combo["name"], "horizon_h": horizon_h,
                "horizon_steps": horizon_steps, "feature_mode": "raw",
                "seq_len": cfg["seq_len"], "hidden": cfg["hidden"],
                "num_layers": cfg["num_layers"], "epochs": cfg["epochs"],
                "lr": SEQ_LR, "batch_size": SEQ_BATCH_SIZE,
                "scaler": SEQ_SCALER, "device": fc.auto_device(),
                "imputation": "mean",
                "weight_decay": cfg["weight_decay"],
                "rnn_dropout": cfg["rnn_dropout"],
                "elapsed_min": round(elapsed / 60, 2),
            },
        )
    return metrics


# ---------------------------------------------------------------------------
# Plot — best model per family vs horizon. Reshapes the hsweep_/seqsweep_ rows
# in experiments.jsonl down to one line per model family (Ridge / HGB / GRU),
# each at its lowest RMSE across the hand-picked source combos, plus the better
# of the two no-model baselines. Deliberately four lines: the headline
# comparison is family-vs-family, not combo-vs-combo.
# ---------------------------------------------------------------------------

import re

FAMILIES = ["ridge", "hgb", "gru"]
_FAMILY_LABEL = {"ridge": "Ridge", "hgb": "HGB-on-residual", "gru": "GRU"}
_FAMILY_STYLE = {
    "ridge": {"color": "#1f77b4", "marker": "o"},
    "hgb":   {"color": "#ff7f0e", "marker": "s"},
    "gru":   {"color": "#2ca02c", "marker": "^"},
}

# ridge/hgb come from the linear hsweep; GRU on baseline/wide from the seq
# hsweep; GRU on tweed_mc from its own hyperparameter sweep (many configs, so
# reduced by lowest RMSE). Lasso/RNN/LSTM/TCN and the ablation "rec" combo are
# intentionally left out of the headline chart.
_PAT_LIN = re.compile(r"^hsweep_(\w+)_h(\d+)h_(ridge|hgb)$")
_PAT_SEQ = re.compile(r"^hsweep_seq_(\w+)_h(\d+)h_(gru)$")
_PAT_TWEED = re.compile(r"^seqsweep_tweed_mc_(gru)_.*_h(\d+)h$")

_COMBO_DISPLAY = {"baseline": "5b+3w", "tweed_mc": "tweed_mc",
                  "solo": "solo", "wide": "wide"}


def _family_best_per_horizon() -> pd.DataFrame:
    """Lowest-RMSE run per (family, horizon) across the hand-picked combos.

    ridge/hgb and the baseline/wide GRU have one config per combo, so the most
    recent log row wins; the tweed_mc GRU is a config sweep, reduced by lowest
    RMSE. Returns columns ``family``, ``horizon_h``, ``RMSE``, and the
    ``combo`` that produced each per-family best (used by the prose to show
    that wider feature sets only win at short lead).
    """
    rows: list[dict] = []
    for _, r in fc.find_runs(name_prefix="hsweep_").iterrows():
        m = _PAT_LIN.match(r["name"]) or _PAT_SEQ.match(r["name"])
        if not m:
            continue
        rows.append({"family": m.group(3), "horizon_h": int(m.group(2)),
                     "combo": m.group(1), "RMSE": r["metrics"]["RMSE"],
                     "ts": r["timestamp"]})
    df = (pd.DataFrame(rows)
            .sort_values("ts")
            .drop_duplicates(["family", "horizon_h", "combo"], keep="last")
            .drop(columns="ts"))

    tweed: list[dict] = []
    for _, r in fc.find_runs(name_prefix="seqsweep_tweed_mc_").iterrows():
        m = _PAT_TWEED.match(r["name"])
        if not m:
            continue
        tweed.append({"family": m.group(1), "horizon_h": int(m.group(2)),
                      "combo": "tweed_mc", "RMSE": r["metrics"]["RMSE"]})
    if tweed:
        tw = (pd.DataFrame(tweed)
                .sort_values("RMSE")
                .drop_duplicates(["family", "horizon_h", "combo"], keep="first"))
        df = pd.concat([df, tw], ignore_index=True)

    return (df.sort_values("RMSE")
              .drop_duplicates(["family", "horizon_h"], keep="first")
              .reset_index(drop=True))


def _baseline_envelope(horizons: list[int]) -> pd.DataFrame:
    """Better-of-persistence/climatology RMSE per horizon (the no-model floor).

    Mooloolaba-only and cheap, so recomputed on every plot. Persistence wins
    at short lead; climatology takes over from ~24h on.
    """
    wave_full = fc.restrict_to_years(fc.load_data(buoy="mooloolaba"), None, 2024)
    rows = []
    for h in horizons:
        b = compute_baselines(wave_full, h)
        p, c = b["persistence"]["RMSE"], b["climatology_hour"]["RMSE"]
        rows.append({"horizon_h": h, "RMSE": float(min(p, c)),
                     "source": "persistence" if p < c else "climatology"})
    return pd.DataFrame(rows)


def save_horizon_chart() -> None:
    """Render the best-model-per-family chart to ``figures/horizon_sweep.png``."""
    best = _family_best_per_horizon()
    horizons = sorted(best["horizon_h"].unique())
    env = _baseline_envelope(horizons)

    print("\n=== Best model per family per horizon (rec combo excluded) ===")
    for h in horizons:
        cells = []
        for fam in FAMILIES:
            row = best[(best["family"] == fam) & (best["horizon_h"] == h)]
            if not row.empty:
                combo = _COMBO_DISPLAY.get(row["combo"].iloc[0], row["combo"].iloc[0])
                cells.append(f"{_FAMILY_LABEL[fam]} {row['RMSE'].iloc[0]:.4f} ({combo})")
        print(f"  h={h:>3}h  " + "   ".join(cells))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for fam in FAMILIES:
        sub = best[best["family"] == fam].sort_values("horizon_h")
        ax.plot(sub["horizon_h"], sub["RMSE"], linewidth=1.8, markersize=8,
                label=_FAMILY_LABEL[fam], **_FAMILY_STYLE[fam])
    ax.plot(env["horizon_h"], env["RMSE"], color="#555555", linestyle="--",
            linewidth=1.4, marker="", label="best no-model baseline")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("RMSE (m)")
    ax.set_xticks(horizons)
    ax.set_title(
        "Mooloolaba significant wave height — best model per family vs horizon\n"
        "pinned 2023-01-01 → 2024-12-31 test window",
        fontsize=11,
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "horizon_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    results: dict[str, dict[int, dict[str, dict[str, float]]]] = {}

    # --- Non-ML baselines first (cheap; combo-independent) ---------------
    wave_full = fc.restrict_to_years(fc.load_data(buoy="mooloolaba"), None, 2024)
    baselines: dict[int, dict[str, dict[str, float]]] = {}
    print(f"\n--- baselines (Mooloolaba {wave_full.index.min().date()} → {wave_full.index.max().date()}) ---")
    for h in HORIZONS_H:
        baselines[h] = compute_baselines(wave_full, h)
        b = baselines[h]
        print(
            f"  h={h:>3}h  "
            f"persistence RMSE {b['persistence']['RMSE']:.4f}  |  "
            f"climatology-hour  {b['climatology_hour']['RMSE']:.4f} (skill {b['climatology_hour']['SkillVsBaseline']:+.4f})"
        )

    for combo in COMBOS:
        print(f"\n{'=' * 70}\nCombo: {combo['name']}  (year_min={combo['year_min']}, "
              f"neighbours={len(combo['neighbours'])}, wind={len(combo['wind'])})\n{'=' * 70}")
        wave, X, sources = build_combo(combo)
        print(f"  window  {wave.index.min().date()} → {wave.index.max().date()}   "
              f"rows {len(wave):,}   features {X.shape[1]}")

        per_h: dict[int, dict[str, dict[str, float]]] = {}
        for h in HORIZONS_H:
            print(f"\n  --- h={h}h ---")
            t_cell = time.time()
            # log=False: 120 linear hsweep entries are already in
            # experiments.jsonl from the first run; rerunning is fast and
            # gives us fresh metrics, but logging again would duplicate.
            per_h[h] = run_cell(combo, h, wave, X, sources, log=False)
            ens = per_h[h]["ensemble"]
            pst = per_h[h]["persistence"]
            print(f"  [{time.time() - t_cell:5.1f}s] "
                  f"persistence RMSE {pst['RMSE']:.4f}  |  "
                  f"ensemble RMSE {ens['RMSE']:.4f}  Skill {ens['SkillVsBaseline']:+.4f}")
        results[combo["name"]] = per_h

    print(f"\nLinear sweep time: {time.time() - t0:.1f}s")

    # --- Sequence models (only for combos with a fitted seq config) ------
    t_seq = time.time()
    seq_combos = [c for c in COMBOS if c["name"] in SEQ_COMBOS]
    for combo in seq_combos:
        print(f"\n{'=' * 70}\nSeq combo: {combo['name']}\n{'=' * 70}")
        wave, X_seq, sources = build_seq_combo(combo)
        print(f"  seq window  {wave.index.min().date()} → {wave.index.max().date()}   "
              f"rows {len(wave):,}   features {X_seq.shape[1]}")
        for arch in SEQ_ARCHITECTURES:
            cfg = SEQ_CONFIGS[(combo["name"], arch)]
            print(f"\n  === {arch.upper()}  h{cfg['hidden']} L{cfg['num_layers']} "
                  f"ep{cfg['epochs']} wd{cfg['weight_decay']:g} do{cfg['rnn_dropout']:g} ===")
            for h in HORIZONS_H:
                t_cell = time.time()
                metrics = run_seq_cell(combo, h, arch, cfg, wave, X_seq, sources, log=True)
                results[combo["name"]][h][arch] = metrics
                print(f"    h={h:>3}h  [{time.time() - t_cell:6.1f}s]  "
                      f"RMSE {metrics['RMSE']:.4f}  Skill {metrics['SkillVsBaseline']:+.4f}")
    print(f"\nSeq sweep time: {time.time() - t_seq:.1f}s")
    print(f"Total sweep time: {time.time() - t0:.1f}s")

    # --- Summary table for the log -----------------------------------------
    print("\nSummary (ensemble RMSE / skill by horizon):")
    horizons = HORIZONS_H
    header = "horizon  " + "  ".join(f"{c['name']:>10}" for c in COMBOS)
    print(header)
    for h in horizons:
        skills = [f"{results[c['name']][h]['ensemble']['SkillVsBaseline']:+.4f}" for c in COMBOS]
        rmses  = [f"{results[c['name']][h]['ensemble']['RMSE']:.4f}"             for c in COMBOS]
        print(f"  h={h:>3}h  " + "  ".join(f"{r}" for r in rmses))
        print(f"   skill  " + "  ".join(f"{s}" for s in skills))

    save_horizon_chart()


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        # Replot from experiments.jsonl without re-running the 90-min sweep.
        save_horizon_chart()
    else:
        main()
