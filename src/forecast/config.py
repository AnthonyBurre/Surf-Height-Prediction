"""Shared constants for the forecast package.

``HORIZON_HOURS`` is the *default* lead time for single-horizon callers
(notably ``make_target``). The headline evaluation is a sweep over
6h / 12h / 24h / 36h / 48h / 72h via ``notebooks/horizon_sweep.py``;
``make_target`` and the forecasters all accept a ``horizon_steps``
override so non-default horizons don't require touching this file.
"""

# Wave buoy observations arrive every 30 minutes. A 12-hour horizon
# is therefore 24 steps ahead; use ``hours_to_steps`` for other values.
SAMPLING_FREQ_MINUTES = 30
HORIZON_HOURS = 12
HORIZON_STEPS = HORIZON_HOURS * 60 // SAMPLING_FREQ_MINUTES  # 24


def hours_to_steps(hours: int) -> int:
    """Convert a forecast horizon in hours to 30-min steps.

    The sweep notebooks parametrise by hours (6, 12, …, 72) but the
    forecasters and ``make_target`` work in steps. Centralising the
    conversion avoids the ``h * 2`` magic number scattered through
    ``horizon_sweep.py`` and any future multi-horizon callers.
    """
    return hours * 60 // SAMPLING_FREQ_MINUTES

TARGET_COL = "hsig_m"

FEATURE_COLS = [
    "hsig_m",
    "hmax_m",
    "tz_s",
    "tp_s",
    "peak_dir_deg",
    "sst_c",
]

# Wave direction is a compass bearing — jumps from 359° to 1° are 2° apart,
# not 358°. Any model that treats it as a plain float will learn that
# discontinuity as signal. Encode as (sin, cos) instead.
CIRCULAR_COLS = ["peak_dir_deg"]
