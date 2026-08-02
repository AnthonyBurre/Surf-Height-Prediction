"""Feature engineering building blocks for wave-height forecasting.

All functions are pure — they take a DataFrame and return a new one —
so they compose in any order without surprising side effects.

Leakage rule: these helpers only look at times ``<= t`` for the row
indexed at ``t``. Pair with ``data.make_target`` (which shifts by
``-horizon``) to keep the forecast origin at ``t``.
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .config import CIRCULAR_COLS


@dataclass
class FeatureConfig:
    """Canonical knobs for lag/rolling/momentum feature construction.

    A single instance shared across an experiment guarantees that every
    model in that run sees the same feature set, making results directly
    comparable.
    """
    # Primary buoy lags/rolling/momentum (in 30-min steps)
    lag_steps: list[int]    = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 96, 144])
    roll_windows: list[int] = field(default_factory=lambda: [12, 24, 48, 96])
    delta_steps: list[int]  = field(default_factory=lambda: [6, 12, 24, 48])
    # Neighbour-column lags/rolling (applied by add_neighbour_features)
    neighbour_lag_steps: list[int]    = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    neighbour_roll_windows: list[int] = field(default_factory=lambda: [6, 12, 24])


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Append ``{col}_lag_{k}`` for every ``(col, k)`` in ``columns x lags``.

    Lag ``k`` is in 30-minute steps (so ``k=2`` means 1 hour ago). A
    positive lag looks backward; the produced columns have NaN for the
    first ``k`` rows where history is unavailable.
    """
    new_cols: dict[str, pd.Series] = {}
    for col in columns:
        for lag in lags:
            new_cols[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    stats: tuple[str, ...] = ("mean", "std"),
) -> pd.DataFrame:
    """Append rolling-window aggregates, right-aligned and lagged by 1 step.

    The shift-by-one is a convention, not strictly a leakage fix at the
    current 12h horizon: ``hsig_m[t]`` is a legitimate feature at forecast
    time ``t``, so an unshifted rolling window over ``[t-w+1, t]`` doesn't
    expose the future label ``hsig_m[t+12h]``. We shift anyway for three
    reasons:

    1. **Robustness to horizon changes.** If ``HORIZON_STEPS`` is ever set
       to 0 (nowcasting), an unshifted window that includes ``t`` would
       leak the label directly. Shifting keeps the module safe by default.
    2. **Symmetry with ``add_lag_features``**, which only ever looks at
       strictly-past values. Treating rolling stats as "summary of the
       lags" makes the whole feature family interpretable as past-only.
    3. **Less collinearity with the raw column.** An unshifted ``roll4_mean``
       is ``0.25·hsig_m[t] + 0.75·(past)`` — nearly redundant with the raw
       feature already in the frame. The shifted version is independent.
    """
    valid_stats = {"mean", "std", "min", "max", "median"}
    unknown = set(stats) - valid_stats
    if unknown:
        raise ValueError(f"Unknown stats: {sorted(unknown)}; valid: {sorted(valid_stats)}")

    new_cols: dict[str, pd.Series] = {}
    for col in columns:
        for w in windows:
            r = df[col].shift(1).rolling(window=w, min_periods=max(1, w // 2))
            for stat in stats:
                new_cols[f"{col}_roll{w}_{stat}"] = getattr(r, stat)()
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def encode_circular(
    df: pd.DataFrame,
    periods: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Replace each circular feature with a sin/cos pair.

    ``periods`` maps name → period; the name can be either:
      - a column already in ``df`` (replaced with ``{name}_sin``/``{name}_cos``), or
      - one of the virtual names ``"hour"`` (period 24) or ``"doy"``
        (period 365.25), which are derived from the DatetimeIndex and added
        as new columns.

    With ``periods=None``: encode every ``CIRCULAR_COLS`` column present in
    ``df`` with period 360.

    Sin/cos pairs let a linear model represent "similar values are close"
    without a one-hot blowup, and avoid the Dec-31 → Jan-1 (or 359° → 1°)
    cliff. Apply this first in the pipeline so any downstream lag/rolling
    features see the encoded representation.
    """
    if periods is None:
        periods = {c: 360.0 for c in CIRCULAR_COLS if c in df.columns}

    out = df.copy()
    for name, period in periods.items():
        if name == "hour":
            values = df.index.hour + df.index.minute / 60.0
        elif name == "doy":
            values = df.index.day_of_year.to_numpy(dtype=float)
        elif name in df.columns:
            values = df[name].to_numpy()
        else:
            raise ValueError(
                f"encode_circular: {name!r} not in df.columns and not a "
                f"recognised virtual name (hour, doy)."
            )
        rad = 2 * np.pi * values / period
        out[f"{name}_sin"] = np.sin(rad)
        out[f"{name}_cos"] = np.cos(rad)
        if name in df.columns:
            out = out.drop(columns=name)
    return out


def add_momentum(
    df: pd.DataFrame,
    columns: list[str],
    deltas: list[int],
) -> pd.DataFrame:
    """Append ``{col}_delta_{d} = col(t) - col(t-d)`` for each (col, d).

    Captures the rate of change / trend direction over multiple timescales.
    ``d`` is in 30-minute steps.
    """
    new_cols: dict[str, pd.Series] = {}
    for col in columns:
        for d in deltas:
            new_cols[f"{col}_delta_{d}"] = df[col] - df[col].shift(d)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_buoy_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Full primary-buoy feature matrix: circular + time + lags + rolling + momentum.

    ``df`` should contain only the primary buoy's columns (no neighbour
    columns). Call ``add_neighbour_features`` on the result to append
    cross-buoy inputs. Works for any QLD wave buoy — the column names
    referenced (``hsig_m``, ``hmax_m``, ``tp_s``, ``tz_s``, ``peak_dir_deg``)
    are the standard CKAN wave schema, shared across all buoys in the network.
    """
    if config is None:
        config = FeatureConfig()
    return (
        df.pipe(encode_circular,
                periods={"peak_dir_deg": 360.0, "hour": 24.0, "doy": 365.25})
          .pipe(add_lag_features,
                columns=["hsig_m", "hmax_m", "tp_s", "tz_s"],
                lags=config.lag_steps)
          .pipe(add_rolling_features,
                columns=["hsig_m", "tp_s", "hmax_m"],
                windows=config.roll_windows,
                stats=("mean", "std", "min", "max"))
          .pipe(add_momentum,
                columns=["hsig_m", "tp_s", "hmax_m"],
                deltas=config.delta_steps)
    )


def add_neighbour_features(
    X: pd.DataFrame,
    source_df: pd.DataFrame,
    columns: list[str],
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Append raw + lag + rolling features for each neighbour column.

    Args:
        X: existing feature matrix to extend (typically from
            ``build_buoy_features``).
        source_df: DataFrame containing the raw neighbour columns, indexed
            on the same DatetimeIndex as ``X``.
        columns: column names in ``source_df`` to use as neighbour inputs.
        config: feature configuration; defaults to ``FeatureConfig()``.

    The raw value at t (lag 0) plus ``config.neighbour_lag_steps`` shifted
    copies and ``config.neighbour_roll_windows`` rolling mean/std are added
    for each column.
    """
    if config is None:
        config = FeatureConfig()
    new_cols: dict[str, pd.Series] = {}
    for col in columns:
        new_cols[col] = source_df[col]
        for lag in config.neighbour_lag_steps:
            new_cols[f"{col}_lag_{lag}"] = source_df[col].shift(lag)
        for w in config.neighbour_roll_windows:
            r = source_df[col].shift(1).rolling(window=w, min_periods=max(1, w // 2))
            new_cols[f"{col}_roll{w}_mean"] = r.mean()
            new_cols[f"{col}_roll{w}_std"]  = r.std()
    return pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)


def build_seq_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal input frame for sequence models: circular encoding + time features.

    No lag or rolling columns — the sequence model windows its own input
    over ``seq_len`` steps and is expected to learn temporal structure
    itself. Pass the result directly to an ``LSTMForecaster`` (or GRU/TCN).
    """
    return df.pipe(
        encode_circular,
        periods={"peak_dir_deg": 360.0, "hour": 24.0, "doy": 365.25},
    )


def assemble_inputs(
    wave: pd.DataFrame,
    neighbours: dict[str, pd.Series],
    wind: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge neighbour series into wave; return ``(merged, neighbour_cols, wind_cols)``.

    Each neighbour series becomes a ``{name}_hsig_m`` column on a copy of
    ``wave``. Wind columns aren't merged (they have a different cadence and
    are usually treated separately by the caller); their names are returned
    so downstream feature engineering can target them by group.
    """
    merged = wave.copy()
    neighbour_cols: list[str] = []
    for name, series in neighbours.items():
        col = f"{name}_hsig_m"
        merged[col] = series
        neighbour_cols.append(col)
    wind_cols = list(wind.columns) if wind is not None else []
    return merged, neighbour_cols, wind_cols


def build_design(
    wave: pd.DataFrame,
    neighbours: dict[str, pd.Series],
    wind: pd.DataFrame | None = None,
    *,
    kind: Literal["engineered", "raw"] = "engineered",
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build the model-ready design matrix from a primary buoy + chosen sources.

    The one canonical "pick a feature style + data sources → X" builder; every
    notebook/sweep funnels through here instead of re-deriving the chain.
    ``wave`` is the primary buoy frame, ``neighbours`` maps each neighbour slug
    to its ``hsig_m`` series (as ``load_neighbours`` returns), and ``wind`` is
    the optional sin/cos-encoded wind frame (``load_wind``). All three must
    share one index — use ``data.load_sources`` (or ``SourceBundle.subset``),
    which also clips them to their overlap, to obtain them.

    ``kind`` selects the feature style:

    - ``"engineered"`` — lag/rolling/momentum matrix for linear/tree models
      (``build_buoy_features`` on the primary buoy, ``add_neighbour_features``
      for neighbour and wind columns). Raw wind bearings (``*_deg``) are
      excluded — their sin/cos pair already carries direction.
    - ``"raw"`` — circular + time channels only (``build_seq_features``) with
      neighbour/wind columns appended verbatim. Sequence models window this
      themselves, so no lag/rolling columns are added.

    ``config`` tunes the engineered lag/rolling/momentum windows and is ignored
    when ``kind="raw"``; ``None`` uses ``FeatureConfig()`` defaults.
    """
    merged, neighbour_cols, _wind_cols = assemble_inputs(wave, neighbours, wind)

    if kind == "raw":
        X = build_seq_features(merged)
        if wind is not None:
            for col in wind.columns:
                X[col] = wind[col]
        return X

    if kind == "engineered":
        primary_only = merged[[c for c in merged.columns if c not in neighbour_cols]]
        X = build_buoy_features(primary_only, config=config)
        if neighbour_cols:
            X = add_neighbour_features(X, merged, neighbour_cols, config=config)
        if wind is not None:
            wind_cols = [c for c in wind.columns if not c.endswith("_deg")]
            X = add_neighbour_features(X, wind, wind_cols, config=config)
        return X

    raise ValueError(f"kind must be 'engineered' or 'raw', got {kind!r}")
