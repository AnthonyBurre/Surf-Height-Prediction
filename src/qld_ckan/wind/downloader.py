import pandas as pd

import qld_ckan

from .constants import RESOURCE_IDS

# Re-exported so ``patch("qld_ckan.wind.downloader._session")``-style tests can
# replace the session for this sub-package without affecting the wave side.
_session = qld_ckan.session
_DATASTORE_BATCH = qld_ckan.DATASTORE_BATCH


# Pre-2015 column-name variants → modern (2015+) names. Applied per-year so
# that the unify step concatenates frames with a single canonical column
# layout — otherwise the variants would survive as parallel columns and
# pipeline.clean's rename would produce duplicate ``wind_dir_deg`` /
# ``wind_sigma_theta_deg`` columns (pandas would silently disambiguate to
# ``.1``-suffixed names). 2010 emits ``Wind Direction`` without the
# ``(degTN)`` suffix; 2010-2013 spell the sigma column ``(degrees)``
# rather than ``(deg)``.
_PRE_2015_RENAMES: dict[str, str] = {
    "Wind Direction": "Wind Direction (degTN)",
    "Wind Sigma Theta (degrees)": "Wind Sigma Theta (deg)",
}


def fetch_year_datastore(year: int, resource_id: str) -> pd.DataFrame:
    """Fetch all records for one year from the CKAN Datastore API.

    Returns a DataFrame with raw column names as exposed by the datastore
    (normalised to the modern 2015+ schema for older years); cleaning
    (rename, timestamp build, regrid) happens in pipeline.clean.
    """
    records = qld_ckan.paginate_records(_session(), resource_id)
    df = pd.DataFrame(records).drop(columns=["_id"], errors="ignore")
    if year < 2015:
        df = df.rename(columns=_PRE_2015_RENAMES)
    return df


def fetch_all(resource_ids: dict[int, str] | None = None) -> list[pd.DataFrame]:
    """Download every year, skipping any that return 404."""
    if resource_ids is None:
        resource_ids = RESOURCE_IDS
    return qld_ckan.fetch_all_years(resource_ids, fetch_year_datastore)
