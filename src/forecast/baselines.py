"""Baseline forecasters. sklearn-style fit/predict so they drop into the same harness as regressors.

Why these matter: an ML model is only useful if it beats the dumbest
thing that also knows the autocorrelation structure of the series.
Persistence is that thing for any process with strong short-horizon
autocorrelation; skill-score-vs-persistence is the honest metric.
"""
from typing import Protocol, Self

import numpy as np
import pandas as pd

from .config import HORIZON_STEPS, SAMPLING_FREQ_MINUTES, TARGET_COL


class Forecaster(Protocol):
    """The fit/predict contract every model in this package satisfies.

    sklearn estimators, the baselines below, and the sequence models in
    ``neural`` all duck-type to this — which is what lets the ``scoring``
    harness accept any of them. Defined here, alongside the simplest
    concrete models, rather than in the harness that merely consumes it.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class PersistenceForecaster:
    """ŷ_{t+h} = y_t — assume the future equals the present.

    Requires ``target_col`` to be present in X at prediction time (which it
    is in our setup: forecasts are made from the unshifted feature frame).
    """

    def __init__(self, target_col: str = TARGET_COL) -> None:
        self.target_col = target_col

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceForecaster":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.target_col].to_numpy()


class ClimatologyHourForecaster:
    """ŷ_{t+h} = training-set mean of target grouped by hour-of-day at t+h.

    Captures only the diurnal climatology — no weather, no persistence.
    Useful as a "regress to the mean" floor that a real model must beat.
    """

    def __init__(
        self,
        horizon_steps: int = HORIZON_STEPS,
        sampling_freq_minutes: int = SAMPLING_FREQ_MINUTES,
    ) -> None:
        self.horizon_steps = horizon_steps
        self.sampling_freq_minutes = sampling_freq_minutes
        self._hourly_mean: pd.Series | None = None
        self._global_mean: float | None = None

    def _forecast_hour(self, index: pd.DatetimeIndex) -> np.ndarray:
        offset = pd.Timedelta(minutes=self.horizon_steps * self.sampling_freq_minutes)
        return (index + offset).hour.to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClimatologyHourForecaster":
        hour = self._forecast_hour(X.index)
        mask = ~y.isna().to_numpy()
        self._hourly_mean = (
            pd.Series(y.to_numpy()[mask])
            .groupby(hour[mask])
            .mean()
        )
        self._global_mean = float(y[mask].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._hourly_mean is None:
            raise RuntimeError("ClimatologyHourForecaster.predict called before fit")
        hour = self._forecast_hour(X.index)
        preds = pd.Series(hour).map(self._hourly_mean).to_numpy()
        # Fall back to the global mean for any hour not seen during fit (rare
        # but possible on short training sets).
        return np.where(np.isnan(preds), self._global_mean, preds)
