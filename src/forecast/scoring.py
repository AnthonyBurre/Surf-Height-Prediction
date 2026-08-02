"""Scoring: regression metrics plus the fit/predict/score harness.

Two layers, in the order you read them:

- **Metrics** (``mae``, ``rmse``, ``bias``, ``skill_score``, ``summarise``) —
  pure functions over numpy/pandas inputs that ignore NaN rows. Use them
  directly to score predictions you already have.
- **Harness** (``evaluate``, ``compare``, ``EvaluationResult``) — fit a model
  on finite-feature rows, predict, and ``summarise`` the result. The harness
  drops NaN rows so every model sees a consistent training subset regardless
  of how many lag/rolling features it uses.

The model contract the harness accepts (``Forecaster``) lives in
``forecast.baselines``, alongside the simplest concrete models. To score *and*
record a run, see ``forecast.runlog`` (``evaluate_and_log``).
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baselines import Forecaster


# --------------------------------------------------------------------------- #
# Metrics — pure, NaN-aware functions over numpy/pandas inputs.
# --------------------------------------------------------------------------- #


def _align(*arrays) -> tuple[np.ndarray, ...]:
    """Coerce inputs to float arrays and mask out any row with a NaN.

    Accepts two or more arrays of identical shape; returns the same number
    of arrays, all clipped to the rows where every input was finite.
    """
    cast = [np.asarray(a, dtype=float) for a in arrays]
    shape = cast[0].shape
    for a in cast[1:]:
        if a.shape != shape:
            raise ValueError(f"shape mismatch: {[a.shape for a in cast]}")
    mask = np.ones(shape, dtype=bool)
    for a in cast:
        mask &= ~np.isnan(a)
    return tuple(a[mask] for a in cast)


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else float("nan")


def rmse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else float("nan")


def bias(y_true, y_pred) -> float:
    """Mean signed error; positive = systematic over-prediction."""
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(yp - yt)) if yt.size else float("nan")


def skill_score(y_true, y_pred, y_pred_baseline) -> float:
    """1 - MSE(model) / MSE(baseline). Positive ⇒ beats baseline, 1.0 = perfect.

    Using MSE (not MAE) because skill scores are conventionally defined on
    squared error, and it's what matches the RMSE that regression models
    optimise.
    """
    yt, yp, yb = _align(y_true, y_pred, y_pred_baseline)
    if not yt.size:
        return float("nan")
    mse_model = float(np.mean((yt - yp) ** 2))
    mse_base = float(np.mean((yt - yb) ** 2))
    if mse_base <= 1e-12:
        return float("nan")
    return 1.0 - mse_model / mse_base


def summarise(y_true, y_pred, y_pred_baseline=None) -> dict[str, float]:
    """One-shot metrics dict. Skill is included iff a baseline is provided."""
    out: dict[str, float] = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }
    if y_pred_baseline is not None:
        out["SkillVsBaseline"] = skill_score(y_true, y_pred, y_pred_baseline)
    return out


# --------------------------------------------------------------------------- #
# Harness — fit on finite rows, predict, summarise.
# --------------------------------------------------------------------------- #


@dataclass
class EvaluationResult:
    name: str
    metrics: dict[str, float]
    predictions: np.ndarray  # aligned with X_test.index (NaN where input had NaN)
    model: Any


def _finite_row_mask(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Rows where every feature and the target are finite."""
    return (~X.isna().any(axis=1) & ~y.isna()).to_numpy()


def evaluate(
    model: Forecaster,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    name: str | None = None,
    baseline_preds: np.ndarray | None = None,
) -> EvaluationResult:
    """Fit on finite-feature rows, predict on finite-feature rows, pad back.

    Why this split:
      - Models that need no lags (e.g. Persistence) get every training row.
      - Models with 48-step lags lose the first 48 rows; that's fine —
        the mask drops them and everyone else still aligns on the index.
      - sklearn raises on NaN in X at predict time, so we mask there too
        and fill NaN into the skipped rows. The returned array is still
        length-aligned with X_test.index for metric computation.
    """
    train_mask = _finite_row_mask(X_train, y_train)
    model.fit(X_train.loc[train_mask], y_train.loc[train_mask])

    predict_mask = (~X_test.isna().any(axis=1)).to_numpy()
    preds = np.full(len(X_test), np.nan)
    if predict_mask.any():
        preds[predict_mask] = np.asarray(
            model.predict(X_test.loc[predict_mask]), dtype=float
        )
    metrics = summarise(y_test, preds, y_pred_baseline=baseline_preds)
    return EvaluationResult(
        name=name or type(model).__name__,
        metrics=metrics,
        predictions=preds,
        model=model,
    )


def compare(results: list[EvaluationResult]) -> pd.DataFrame:
    """Stack per-model metrics into a sorted DataFrame (lowest RMSE first)."""
    rows = [{"model": r.name, **r.metrics} for r in results]
    df = pd.DataFrame(rows).set_index("model")
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE")
    return df
