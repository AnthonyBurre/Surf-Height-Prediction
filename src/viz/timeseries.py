"""Time-series plots — shared primitives for any time-indexed series.

These functions are deliberately context-free: they work equally well on
a raw downloaded channel (post-download EDA) and on a model's prediction
or residual series (post-modeling diagnostics). Single-source and multi-
source overlays share the same API.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def _steps_per_hour(series: pd.Series, sampling_freq_min: int | None) -> float:
    """How many samples per hour. Inferred from the index unless overridden."""
    if sampling_freq_min is not None:
        return 60.0 / sampling_freq_min
    if len(series.index) < 2:
        raise ValueError("Cannot infer sampling frequency from < 2 rows; pass sampling_freq_min.")
    # Median delta survives occasional gaps that trip pd.infer_freq on tz-aware indexes.
    delta = pd.Series(series.index).diff().dropna().median()
    if delta <= pd.Timedelta(0):
        raise ValueError("Non-positive median timestep; pass sampling_freq_min explicitly.")
    return pd.Timedelta(hours=1) / delta


def plot_series(
    series: pd.Series,
    *,
    title: str = "",
    ylabel: str = "",
    source_label: str | None = None,
    ax: Axes | None = None,
    **plot_kwargs,
) -> Axes:
    """Plot a single time series. ``source_label`` shows up in the legend.

    Intentionally thin — use :func:`plot_multi_source` when comparing
    multiple buoys / sources on one axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    series.plot(ax=ax, label=source_label, **plot_kwargs)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if source_label:
        ax.legend()
    return ax


def plot_multi_source(
    sources: dict[str, pd.Series | pd.DataFrame],
    *,
    column: str | None = None,
    title: str = "",
    ylabel: str = "",
    ax: Axes | None = None,
    alpha: float = 0.7,
) -> Axes:
    """Overlay the same variable from multiple sources on one axis.

    Parameters
    ----------
    sources
        Mapping from source label (e.g. buoy name, reanalysis product) to a
        Series or DataFrame. If DataFrames, ``column`` must be given and is
        extracted from each.
    column
        Which column to plot when ``sources`` holds DataFrames.

    Example
    -------
    >>> plot_multi_source(
    ...     {"mooloolaba": mool_df, "brisbane": bris_df},
    ...     column="hsig_m",
    ...     title="Significant wave height — Mooloolaba vs Brisbane",
    ... )
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))

    for label, data in sources.items():
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("column= is required when sources are DataFrames")
            data = data[column]
        data.plot(ax=ax, alpha=alpha, label=label)

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return ax


def autocorrelation_curve(
    series: pd.Series,
    *,
    max_hours: int = 72,
    step_hours: float = 1.0,
    sampling_freq_min: int | None = None,
    highlight_hours: list[float] | None = None,
    threshold: float | None = 0.5,
    source_label: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot autocorrelation vs lag-in-hours, with optional horizon markers.

    ``highlight_hours`` draws a dashed vertical line at each value —
    handy for annotating the forecast horizon. ``threshold`` draws a
    horizontal reference line (default 0.5 — commonly treated as the
    boundary between "useful" and "weak" correlation).
    """
    steps_per_hour = _steps_per_hour(series, sampling_freq_min)
    hours = np.arange(0, max_hours + step_hours, step_hours)
    acf = [series.autocorr(lag=int(round(h * steps_per_hour))) for h in hours]

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hours, acf, marker=".", linewidth=1.2, label=source_label)

    for h in highlight_hours or []:
        ax.axvline(h, color="red", linestyle="--", alpha=0.6, label=f"{h}h horizon")
    if threshold is not None:
        ax.axhline(threshold, color="grey", linestyle=":", alpha=0.6, label=f"r = {threshold}")
    ax.axhline(0, color="black", linewidth=0.6)

    title = "Autocorrelation"
    if source_label:
        title = f"{title} — {source_label}"
    ax.set(xlabel="Lag (hours)", ylabel="Autocorrelation", title=title)
    if highlight_hours or threshold is not None or source_label:
        ax.legend()
    return ax
