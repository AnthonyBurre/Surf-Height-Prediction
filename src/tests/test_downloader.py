import pandas as pd
import pytest

from qld_ckan.wave.downloader import _normalize_columns


@pytest.mark.parametrize(
    "year, input_cols, expected_cols",
    [
        # pre-2017: Dir_Tp TRUE → Peak Direction
        (2015,
         ["Date/Time", "Hs", "Dir_Tp TRUE", "SST"],
         ["Date/Time", "Hs", "Peak Direction", "SST"]),
        # boundary: 2016 still gets the pre-2017 rename
        (2016,
         ["Date/Time", "Hs", "Dir_Tp TRUE", "SST"],
         ["Date/Time", "Hs", "Peak Direction", "SST"]),
        # 2017+: Dir_Tp TRUE is left alone (it shouldn't appear, but if it
        # somehow does, don't rename — the canonical name applies only to
        # the pre-2017 era).
        (2017,
         ["Date/Time", "Dir_Tp TRUE"],
         ["Date/Time", "Dir_Tp TRUE"]),
        # mid-era: already-canonical columns pass through unchanged
        (2017,
         ["Date/Time", "Hs", "Hmax", "Tz", "Tp", "Peak Direction", "SST"],
         ["Date/Time", "Hs", "Hmax", "Tz", "Tp", "Peak Direction", "SST"]),
        # unit suffixes get stripped
        (2022,
         ["Date/Time (AEST)", "Hs (m)", "Hmax (m)", "Peak Direction (degrees)", "SST (degrees C)"],
         ["Date/Time", "Hs", "Hmax", "Peak Direction", "SST"]),
        # noop when no suffixes are present
        (2019,
         ["Date/Time", "Hs", "Peak Direction"],
         ["Date/Time", "Hs", "Peak Direction"]),
    ],
)
def test_normalize_columns(year, input_cols, expected_cols):
    df = pd.DataFrame(columns=input_cols)
    result = _normalize_columns(df, year)
    assert list(result.columns) == expected_cols
