"""Linear / tree model playground for +12h hsig_m.

Run:  ./.venv/bin/python notebooks/linear_playground.py

A single CONFIG dict at the top controls everything: data window, neighbour
buoys, wind stations, FeatureConfig knobs, which models to run (Ridge / Lasso /
HGB), HGB residual-target mode, and an optional nanmean ensemble. Edit it in
place and re-run — no other code changes needed.

Each completed run appends to experiments.jsonl with a name derived from the
CONFIG, so back-to-back runs stay distinguishable.


Tips
----
- Set a model key to False to skip it, True to use the defaults below, or a
  dict to override specific hyperparameters (merged with the defaults).
- ``hgb_residual_target=True`` trains HGB on y - persistence(y) then adds
  persistence back at predict time; often improves HGB skill slightly.
- ``ensemble=True`` computes a nanmean of every enabled model's predictions.
- Year trimming is done BEFORE the wind overlap restriction, so
  ``year_max=2024`` + a non-empty ``wind_stations`` just restricts to the
  selected stations' wind window (currently 2015-2024 for both).
- Persistence is computed on the same test window as all other models.
"""

import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge

import forecast as fc
from forecast import summarise

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

# ---------------------------------------------------------------------------
# CONFIG — edit me
# ---------------------------------------------------------------------------

CONFIG: dict = {
    # --- data window ---
    "primary_buoy": "mooloolaba",  # any key in qld_ckan.wave.constants.BUOYS; download with `python -m qld_ckan wave --buoy NAME`
    "year_min":     None,   # None = data start; e.g. 2024 for the short multi-buoy window
    "year_max":     2024,   # None = data end;   e.g. 2024 to match the wind window

    # Optional explicit test-window cutoff (tz-aware string or pd.Timestamp).
    # None = natural chronological 80/20. Set this when comparing different
    # feature sets / windows: e.g. anchor a wide-set (2019-2024) run on the
    # narrow set's natural split timestamp so both score on the same test
    # slice. Rows strictly before the cutoff are train; rows at/after are test.
    "test_start":   None,

    # --- extra sources ---
    "neighbours":     ["caloundra", "brisbane", "gold-coast", "north-moreton-bay", "tweed-heads"],     # any subset of: "brisbane", "caloundra", "gold-coast", "north-moreton-bay", "palm-beach", "tweed-heads", "wide-bay"
    "wind_stations":  ["mountain-creek", "deception-bay", "lytton"],  # any subset of: "mountain-creek", "deception-bay", "lytton", "southport"; [] disables wind

    # --- feature engineering (FeatureConfig knobs) ---
    # Set to None to use the package defaults.
    "lag_steps":              None,   # default: [1,2,3,6,12,24,48,96,144]
    "roll_windows":           None,   # default: [12,24,48,96]
    "delta_steps":            None,   # default: [6,12,24,48]
    "neighbour_lag_steps":    None,   # default: [1,2,3,6,12,24]
    "neighbour_roll_windows": None,   # default: [6,12,24]

    # --- models ---
    # True  → run with default hyperparams shown in MODELS below
    # dict  → run, merging the dict over the defaults
    # False → skip
    "ridge":  True,
    "lasso":  True,
    "hgb":    True,

    # HGB-specific: train on y − persistence(y) instead of y
    "hgb_residual_target": True,

    # nanmean ensemble of all enabled model predictions
    "ensemble": True,

    # feature scaling for the linear models (Ridge/Lasso); HGB is left raw.
    # None disables; "robust" or "standard".
    "scaling": "robust",

    # --- run / logging ---
    "run_name":     "lineopt",  # log entries get suffixed with active models + sources
    "log_to_jsonl": True,
}

# ---------------------------------------------------------------------------
# Model registry: factory + defaults + optional reporter callback.
# ``supports_residual`` flags models that participate in residual-target mode.
# ---------------------------------------------------------------------------


def _report_lasso_coefs(model, columns: pd.Index) -> None:
    coef = model.coef_
    n_nonzero = int((coef != 0).sum())
    print(f"  Non-zero coefficients: {n_nonzero} / {len(coef)}")
    top = (pd.Series(np.abs(coef), index=columns)
           .sort_values(ascending=False).head(10))
    print("  Top 10 by |coef|:")
    for feat, val in top.items():
        print(f"    {feat:42s}  {val:.4f}")


def _report_feature_importance(model, columns: pd.Index) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    top = (pd.Series(model.feature_importances_, index=columns)
           .sort_values(ascending=False).head(10))
    print("  Top 10 features by importance:")
    for feat, val in top.items():
        print(f"    {feat:42s}  {val:.4f}")


# Reporter takes (fitted_model, X_train.columns) and prints a model-specific
# summary; None means no extra reporting beyond the metrics line.
MODELS: dict[str, dict] = {
    "ridge": {
        "factory":  Ridge,
        "defaults": {"alpha": 1.0},
        "reporter": None,
        "supports_residual": False,
        # HGB handles NaN natively; linear models need the imputed matrix.
        "needs_imputed": True,
    },
    "lasso": {
        "factory":  Lasso,
        "defaults": {"alpha": 0.001, "max_iter": 10000},
        "reporter": _report_lasso_coefs,
        "supports_residual": False,
        "needs_imputed": True,
    },
    "hgb": {
        "factory":  HistGradientBoostingRegressor,
        "defaults": {
            "max_iter": 800, "learning_rate": 0.03, "max_depth": 6,
            "min_samples_leaf": 50, "l2_regularization": 1.0, "random_state": 42,
            "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 40,
        },
        "reporter": _report_feature_importance,
        "supports_residual": True,
        "needs_imputed": False,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_hyperparams(cfg_value: bool | dict, defaults: dict) -> dict | None:
    """Return merged hyperparams, or None if the model is disabled."""
    if cfg_value is False:
        return None
    if cfg_value is True:
        return defaults.copy()
    return {**defaults, **cfg_value}


def _build_features(
    wave: pd.DataFrame,
    neighbours: dict[str, pd.Series],
    wind: pd.DataFrame | None,
    cfg: dict,
) -> pd.DataFrame:
    fc_kwargs: dict = {}
    for key in ("lag_steps", "roll_windows", "delta_steps",
                "neighbour_lag_steps", "neighbour_roll_windows"):
        if cfg.get(key) is not None:
            fc_kwargs[key] = cfg[key]
    feat_cfg = fc.FeatureConfig(**fc_kwargs) if fc_kwargs else None
    return fc.build_design(wave, neighbours, wind, kind="engineered", config=feat_cfg)


def _run_model(
    key: str,
    kw: dict,
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_te: pd.DataFrame, y_te: pd.Series,
    baseline_preds: np.ndarray,
    run_name: str,
    sources: list[str],
    extra: dict,
    *,
    log: bool,
    residual_train: pd.Series | None = None,
    residual_test: pd.Series | None = None,
) -> fc.EvaluationResult:
    """Fit one model and (optionally) log the run.

    Residual mode: when ``residual_train`` is supplied (and the spec
    allows it), train on ``y - residual_train`` and add ``residual_test``
    back at predict time. Used for HGB-on-residuals where the model only
    has to learn the *delta* over persistence.
    """
    spec = MODELS[key]
    name = f"{run_name}_{key}"
    use_residual = residual_train is not None and spec["supports_residual"]
    if spec["supports_residual"]:
        mode_str = "persistence_residual" if use_residual else "direct"
        print(f"\n=== {name}  ({mode_str}) ===")
    else:
        print(f"\n=== {name} ===")

    t0 = time.time()
    if use_residual:
        y_res = y_tr - residual_train
        model = spec["factory"](**kw)
        mask = ~y_res.isna()
        model.fit(X_tr.loc[mask].to_numpy(), y_res.loc[mask].to_numpy())
        preds = residual_test.to_numpy() + model.predict(X_te.to_numpy())
        result = fc.EvaluationResult(
            name=name,
            metrics=summarise(y_te.to_numpy(), preds, y_pred_baseline=baseline_preds),
            predictions=preds,
            model=model,
        )
        if log:
            fc.log_run(
                result, data_sources=sources,
                train_index=X_tr.index, test_index=X_te.index,
                n_features=X_tr.shape[1],
                extra={**extra, "hgb_mode": "persistence_residual",
                       "nan_handling": "native_hgb"},
            )
    else:
        result = fc.evaluate_and_log(
            spec["factory"](**kw),
            X_tr, y_tr, X_te, y_te,
            name=name,
            baseline_preds=baseline_preds,
            data_sources=sources,
            extra=extra,
            log=log,
        )
    print(f"  [{time.time() - t0:5.1f}s] {name}")

    reporter: Callable | None = spec["reporter"]
    if reporter is not None:
        reporter(result.model, X_tr.columns)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CONFIG
    log = cfg["log_to_jsonl"]

    wave, neighbours, wind = fc.load_sources(
        buoy=cfg["primary_buoy"],
        neighbours=cfg["neighbours"], wind_stations=cfg["wind_stations"],
        year_min=cfg["year_min"], year_max=cfg["year_max"],
    )

    print(f"window         : {wave.index.min().date()} → {wave.index.max().date()}")
    print(f"rows           : {len(wave):,}")

    X = _build_features(wave, neighbours, wind, cfg)
    y = fc.make_target(wave)
    X_p = wave[["hsig_m"]]

    if cfg.get("test_start") is None:
        X_tr, X_te, y_tr, y_te = fc.chronological_split(X, y)
        X_p_tr, X_p_te, _, _   = fc.chronological_split(X_p, y)
    else:
        ts = pd.Timestamp(cfg["test_start"])
        if ts.tzinfo is None:
            ts = ts.tz_localize(fc.SOURCE_TZ)
        pos = X.index.searchsorted(ts)
        X_tr, X_te = X.iloc[:pos], X.iloc[pos:]
        y_tr, y_te = y.iloc[:pos], y.iloc[pos:]
        pos_p = X_p.index.searchsorted(ts)
        X_p_tr, X_p_te = X_p.iloc[:pos_p], X_p.iloc[pos_p:]
        print(f"test_start     : {ts} (override of natural 80/20 split)")
    # One stateful preprocessor: drop → impute → (optional) scale. Fit on
    # train, transform both. Pickle-able alongside the model for held-out
    # year scoring (see Preprocessor docstring).
    preproc = fc.Preprocessor(max_nan_frac=0.5, scaling=cfg.get("scaling") or None).fit(X_tr)
    X_tr_imp = preproc.transform(X_tr)
    X_te_imp = preproc.transform(X_te)
    # Drop list is now visible on the preprocessor for downstream introspection.
    X_tr = X_tr[preproc.kept_columns_]
    X_te = X_te[preproc.kept_columns_]

    nan_pct = X.isna().mean().mul(100)
    worst = nan_pct.sort_values(ascending=False).head(3).round(2).to_dict()
    print(f"features       : {X.shape[1]}  |  train: {len(X_tr):,}  test: {len(X_te):,}")
    print(f"top NaN cols   : {worst}\n")

    run_name = fc.compose_run_name(
        cfg["run_name"],
        wind_stations=cfg["wind_stations"],
        neighbours=cfg["neighbours"],
        neighbour_chars=4,
    )
    sources = [cfg["primary_buoy"]] + cfg["neighbours"] + cfg["wind_stations"]
    window_str = f"{wave.index.min().date()}:{wave.index.max().date()}"
    extra_base: dict = {
        "window": window_str,
        "imputation": "mean",
        "scaling": cfg.get("scaling") or "none",
        "n_neighbours": len(cfg["neighbours"]),
        "wind_stations": cfg["wind_stations"],
    }

    print("=== Persistence baseline ===")
    persist_name = f"{run_name}_persistence"
    t0 = time.time()
    persist = fc.evaluate_and_log(
        fc.PersistenceForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name=persist_name,
        data_sources=[cfg["primary_buoy"]],
        extra={"window": window_str},
        log=log,
    )
    print(f"  [{time.time() - t0:5.1f}s] {persist_name}")
    pp = persist.predictions
    print(f"persistence    : RMSE {persist.metrics['RMSE']:.4f}\n")

    # Residual baselines (current hsig_m), only used if residual mode is on.
    wave_tr = wave["hsig_m"].loc[X_tr.index]
    wave_te = wave["hsig_m"].loc[X_te.index]

    results: list[fc.EvaluationResult] = []
    for key, spec in MODELS.items():
        kw = _resolve_hyperparams(cfg.get(key, False), spec["defaults"])
        if kw is None:
            continue
        X_tr_use = X_tr_imp if spec["needs_imputed"] else X_tr
        X_te_use = X_te_imp if spec["needs_imputed"] else X_te
        residual = key == "hgb" and cfg.get("hgb_residual_target", False)
        r = _run_model(
            key, kw,
            X_tr_use, y_tr, X_te_use, y_te,
            pp, run_name, sources, extra_base,
            log=log,
            residual_train=wave_tr if residual else None,
            residual_test=wave_te if residual else None,
        )
        results.append(r)

    if cfg.get("ensemble") and len(results) >= 2:
        ens_preds = np.nanmean(np.vstack([r.predictions for r in results]), axis=0)
        ens_name  = f"{run_name}_ensemble"
        ens = fc.EvaluationResult(
            name=ens_name,
            metrics=summarise(y_te.to_numpy(), ens_preds, y_pred_baseline=pp),
            predictions=ens_preds,
            model=None,
        )
        if log:
            fc.log_run(ens, data_sources=sources,
                       train_index=X_tr.index, test_index=X_te.index,
                       n_features=X.shape[1],
                       model_class="NanMeanEnsemble",
                       extra={**extra_base,
                              "members": [r.name for r in results],
                              "combiner": "nanmean"})
        results.append(ens)

    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"{'='*60}")
    table = fc.compare([persist] + results).round(4)
    print(table.to_string())

    if log:
        recent = fc.recent_runs(cfg["run_name"], n=10)
        if len(recent) > 1:
            print("\nrecent playground runs (most recent last):")
            for _, row in recent.iterrows():
                m = row["metrics"]
                ex = row.get("extra") or {}
                ws = ex.get("wind_stations", "?")
                tag = f"w={ws} nb={ex.get('n_neighbours','?')}"
                skill = m.get("SkillVsBaseline", 0)
                print(f"  {row['name']:60s}  {tag:20s}  RMSE {m['RMSE']:.4f}  Skill {skill:+.4f}")


if __name__ == "__main__":
    main()
