import pandas as pd

from qld_ckan.wind.constants import STATIONS
from qld_ckan.wind.pipeline import clean


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def test_clean_combines_date_and_time_into_aest_index(wind_raw_records):
    df = clean(pd.DataFrame(wind_raw_records).drop(columns=["_id"]))
    # Raw input is naive AEST; cleaned index is tagged Australia/Brisbane
    # so the row reads as 2024-01-01 00:00 AEST (same physical instant as
    # 2023-12-31 14:00 UTC).
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00", tz="Australia/Brisbane")
    assert str(df.index.tz) == "Australia/Brisbane"
    assert df.index.name == "datetime"


def test_clean_renames_wind_columns_and_drops_pollutants(wind_raw_records):
    df = clean(pd.DataFrame(wind_raw_records).drop(columns=["_id"]))
    assert list(df.columns) == [
        "wind_dir_deg",
        "wind_speed_ms",
        "wind_sigma_theta_deg",
        "wind_speed_std_ms",
    ]


def test_clean_reindexes_onto_hourly_grid():
    # Drop the middle of three records; clean should fill the gap with NaN.
    records = [
        {"Date": "2024-06-01T00:00:00", "Time": "00:00", "Wind Speed (m/s)": 1.0, "Wind Direction (degTN)": 90},
        {"Date": "2024-06-01T00:00:00", "Time": "02:00", "Wind Speed (m/s)": 1.5, "Wind Direction (degTN)": 95},
    ]
    df = clean(pd.DataFrame(records))
    assert len(df) == 3  # 00:00, 01:00, 02:00 AEST
    assert df["wind_speed_ms"].isna().sum() == 1
    assert (df.index[1] - df.index[0]) == pd.Timedelta("1h")


def test_clean_drops_duplicate_timestamps(wind_raw_records):
    duplicated = wind_raw_records + [wind_raw_records[0]]
    df = clean(pd.DataFrame(duplicated).drop(columns=["_id"]))
    assert df.index.is_unique


def test_clean_coerces_string_numerics():
    records = [
        {"Date": "2024-06-01T00:00:00", "Time": "00:00", "Wind Speed (m/s)": "1.5", "Wind Direction (degTN)": "90"},
    ]
    df = clean(pd.DataFrame(records))
    assert df["wind_speed_ms"].dtype.kind == "f"
    assert df["wind_speed_ms"].iloc[0] == 1.5


def test_clean_parses_ddmmyyyy_date_format():
    # Some yearly resources (e.g. Mountain Creek 2019, Deception Bay 2015-2019)
    # emit Date as DD/MM/YYYY rather than ISO. Without dayfirst handling, dates
    # past the 12th of each month coerce to NaT and the year is silently lost.
    records = [
        {"Date": "01/03/2019", "Time": "12:00", "Wind Speed (m/s)": 2.0, "Wind Direction (degTN)": 90},
        {"Date": "15/03/2019", "Time": "12:00", "Wind Speed (m/s)": 2.5, "Wind Direction (degTN)": 95},
        {"Date": "31/03/2019", "Time": "12:00", "Wind Speed (m/s)": 3.0, "Wind Direction (degTN)": 100},
    ]
    df = clean(pd.DataFrame(records))
    # All three day-first dates must round-trip (not just day ≤ 12).
    parsed_days = sorted({ts.day for ts in df.dropna(subset=["wind_speed_ms"]).index})
    assert parsed_days == [1, 15, 31]


def test_stations_registry_includes_both_stations():
    # Smoke check that the multi-station registry is populated. Both Mountain
    # Creek and Deception Bay should expose 10 years of CKAN UUIDs.
    assert "mountain-creek" in STATIONS
    assert "deception-bay" in STATIONS
    for slug in ("mountain-creek", "deception-bay"):
        years = STATIONS[slug]
        assert len(years) >= 10
        assert all(isinstance(rid, str) and len(rid) == 36 for rid in years.values())


def test_clean_tolerates_missing_optional_columns():
    # 2024 records lack Air Temperature; ensure clean handles a frame that
    # only carries the two core wind fields.
    minimal = [
        {"Date": "2024-06-01T00:00:00", "Time": "00:00", "Wind Speed (m/s)": 2.0, "Wind Direction (degTN)": 100},
    ]
    df = clean(pd.DataFrame(minimal))
    assert "wind_speed_ms" in df.columns
    assert "wind_sigma_theta_deg" not in df.columns  # not in input → not in output
