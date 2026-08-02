import pandas as pd

import qld_ckan

from .constants import RESOURCE_IDS

# Re-exported so ``patch("qld_ckan.wave.downloader._session")``-style tests can
# replace the session for this sub-package without affecting the wind side.
_session = qld_ckan.session
_DATASTORE_BATCH = qld_ckan.DATASTORE_BATCH


def _normalize_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Standardise column names to the pre-COLUMN_RENAME_MAP schema.

    Two schema breaks exist across years:
      - All years may carry ' (unit)' suffixes (confirmed from 2022 onwards in
        both CSV files and the Datastore); stripping is a safe no-op otherwise.
      - 2015-2016: wave direction column is 'Dir_Tp TRUE' instead of 'Peak Direction'.
    """
    df = df.rename(columns={col: col.split(" (")[0] for col in df.columns})
    if year < 2017:
        df = df.rename(columns={"Dir_Tp TRUE": "Peak Direction"})
    return df


def fetch_year_datastore(year: int, resource_id: str) -> pd.DataFrame:
    """Fetch all records for one year from the CKAN Datastore API.

    Returns a DataFrame with a parsed 'Date/Time' column and standardised
    column names (pre-COLUMN_RENAME_MAP), ready for concatenation.
    """
    records = qld_ckan.paginate_records(_session(), resource_id)
    df = pd.DataFrame(records).drop(columns=["_id"], errors="ignore")
    df = _normalize_columns(df, year)
    # Date/Time arrives in two flavours across the network: ISO 8601 for
    # most years, and Australian d/m/Y with mixed leading-zero conventions
    # (e.g. tweed-heads 2019: "1/01/2019 0:00"). format="mixed", dayfirst=True
    # handles both without relying on pandas's per-series auto-inference,
    # which silently swapped to %m/%d/%Y on some short samples.
    df["Date/Time"] = pd.to_datetime(
        df["Date/Time"], format="mixed", dayfirst=True, errors="raise"
    )
    return df


def fetch_all(resource_ids: dict[int, str] | None = None) -> list[pd.DataFrame]:
    """Download every year, skipping any that return 404."""
    if resource_ids is None:
        resource_ids = RESOURCE_IDS
    return qld_ckan.fetch_all_years(resource_ids, fetch_year_datastore)
