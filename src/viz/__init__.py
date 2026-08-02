"""Plotting and exploratory visualisation — source-agnostic.

Organised by *when in the pipeline* you reach for the chart:

- ``timeseries``   — shared primitives for any time-indexed series.
                     Used both post-download (raw buoy / wind channels,
                     multi-source overlays, autocorrelation) and post-
                     modeling (prediction traces, residual series).
- ``eda``          — post-download exploratory plots that only make
                     sense *before* a model exists: feature-horizon
                     screening and cross-source correlation.
- ``results``      — post-experiment plots that consume a long-format
                     DataFrame of run results (typically pulled out of
                     ``experiments.jsonl`` via ``forecast.find_runs``).
                     Domain-agnostic: the caller does the name parsing
                     and labelling; this module just draws the chart.

Every function accepts an optional ``ax`` so panels can be composed into
a larger ``plt.subplots`` grid. For multi-source work (buoy comparisons,
overlaying atmospheric channels on buoy channels, etc.) the convention
is a ``dict[str, pd.Series]`` or ``dict[str, pd.DataFrame]`` keyed by
source label — that label ends up in the legend / row heading.
"""
from .eda import cross_source_correlation, feature_horizon_heatmap
from .results import plot_horizon_winners
from .timeseries import autocorrelation_curve, plot_multi_source, plot_series

__all__ = [
    # timeseries (shared primitives)
    "autocorrelation_curve",
    "plot_multi_source",
    "plot_series",
    # eda (post-download)
    "cross_source_correlation",
    "feature_horizon_heatmap",
    # results (post-experiment)
    "plot_horizon_winners",
]
