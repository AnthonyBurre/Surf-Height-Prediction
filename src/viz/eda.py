"""Post-download exploratory plots.

Charts that only make sense *before* a forecasting model exists — used to
screen candidate features and decide which neighbour sources are worth
pulling in:

1. ``feature_horizon_heatmap``  — *within one source*: how each channel
   correlates with a target at a range of forecast horizons.
2. ``cross_source_correlation`` — *across sources*: how the same
   variable correlates across multiple buoys / products at a given lag.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def _steps(hours: float, sampling_freq_min: int) -> int:
    return int(round(hours * 60.0 / sampling_freq_min))


def feature_horizon_heatmap(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str] | None = None,
    horizons_h: tuple[int, ...] = (1, 3, 6, 12, 24, 48, 72),
    sampling_freq_min: int = 30,
    source_label: str | None = None,
    ax: Axes | None = None,
) -> tuple[Axes, pd.DataFrame]:
    """corr(feature at t, target at t+h) — returns axes and the matrix.

    ``feature_cols`` defaults to every numeric column in ``df`` other than
    ``target_col``. The returned DataFrame is the underlying correlation
    grid (useful for downstream comparison across sources).
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]

    grid = pd.DataFrame(
        index=feature_cols,
        columns=[f"+{h}h" for h in horizons_h],
        dtype=float,
    )
    for h in horizons_h:
        future = df[target_col].shift(-_steps(h, sampling_freq_min))
        for feat in feature_cols:
            grid.loc[feat, f"+{h}h"] = df[feat].corr(future)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 0.5 * len(feature_cols) + 2))
    sns.heatmap(
        grid,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    title = f"corr(feature at t, {target_col} at t+h)"
    if source_label:
        title = f"{title} — {source_label}"
    ax.set(title=title, xlabel="Forecast horizon", ylabel="")
    return ax, grid


def cross_source_correlation(
    sources: dict[str, pd.Series],
    *,
    lag_hours: float = 0.0,
    sampling_freq_min: int = 30,
    source_label: str | None = None,
    ax: Axes | None = None,
) -> tuple[Axes, pd.DataFrame]:
    """Correlation matrix of the same variable across multiple sources / buoys.

    All series are inner-joined on their index before correlating, so
    pass raw (possibly non-overlapping) series — the function aligns
    them. ``lag_hours`` shifts every column *except* the first by that
    amount, letting you probe "does buoy A 3h ago predict buoy B now?"
    (useful when one buoy is systematically upstream of another).
    """
    if len(sources) < 2:
        raise ValueError("cross_source_correlation needs at least two sources")

    lag_steps = _steps(lag_hours, sampling_freq_min)
    columns = {}
    for i, (name, s) in enumerate(sources.items()):
        if not s.index.is_unique:
            s = s[~s.index.duplicated(keep="first")]
        columns[name] = s if i == 0 else s.shift(lag_steps)
    aligned = pd.concat(columns.values(), axis=1, join="inner", keys=columns.keys())
    aligned.columns = list(columns.keys())
    corr = aligned.corr()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    title = "Cross-source correlation"
    if lag_hours:
        title = f"{title}  (non-anchor sources lagged by {lag_hours}h)"
    if source_label:
        title = f"{title} — {source_label}"
    ax.set_title(title)
    return ax, corr
