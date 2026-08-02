"""Shared CKAN Datastore client for the QLD government open-data portal.

Two source families ride on top of the same retrying session / paginated GET /
404-skip loop:

- ``qld_ckan.wave``  — wave-buoy network (Mooloolaba, Brisbane, …).
- ``qld_ckan.wind``  — air-quality / met stations carrying 10 m wind.

Each sub-package owns its own per-year schema normalisation; everything else
(the transport, the year-loop, frame stitching) is shared here.
"""
import logging
from functools import cache

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

CKAN_API_BASE = "https://www.data.qld.gov.au/api/3/action"
DATASTORE_BATCH = 32000


@cache
def session() -> requests.Session:
    """Singleton session with retry/backoff for transient 5xx and connection errors.

    Built on first use — no HTTP machinery is constructed at import time.
    404 is not retried: callers treat it as a legitimate "resource missing"
    signal and skip the year.
    """
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def paginate_records(
    http: requests.Session,
    resource_id: str,
    *,
    base_url: str = CKAN_API_BASE,
    batch_size: int = DATASTORE_BATCH,
    timeout: float = 30.0,
) -> list[dict]:
    """Fetch every record for a CKAN datastore resource, paginating as needed."""
    records: list[dict] = []
    offset = 0
    while True:
        response = http.get(
            f"{base_url}/datastore_search",
            params={"resource_id": resource_id, "limit": batch_size, "offset": offset},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()["result"]
        batch = result["records"]
        records.extend(batch)
        if len(records) >= result["total"] or not batch:
            break
        offset += len(batch)
    return records


def filter_resource_years(
    resource_ids: dict[int, str],
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict[int, str]:
    """Return ``resource_ids`` keyed only on years in ``[year_min, year_max]``.

    Bounds are inclusive; either bound can be ``None`` to leave that side open.
    Filtering is on the dict key, which is the resource's nominal year label —
    for multi-year bundle resources (e.g. North Moreton Bay's 2010-2015 file
    keyed as ``2010``), that's the bundle's first year. Callers that want
    partial-bundle coverage should set ``year_min`` to the bundle's start
    year rather than a mid-bundle year.
    """
    return {
        y: r for y, r in resource_ids.items()
        if (year_min is None or y >= year_min)
        and (year_max is None or y <= year_max)
    }


def fetch_all_years(resource_ids, fetch_one):
    """Apply ``fetch_one(year, rid)`` across a year→rid map, skipping any 404s.

    ``fetch_one`` is expected to raise ``requests.HTTPError`` on transport
    failure; a 404 is treated as "this year's resource is missing" and the
    loop continues. Anything else re-raises.

    Emits a final WARNING listing any years that were skipped — a stale
    resource ID otherwise produces a silent year-shaped hole in the dataset.
    """
    frames = []
    missing: list[int] = []
    for year, rid in resource_ids.items():
        logger.info("Fetching %d...", year)
        try:
            frames.append(fetch_one(year, rid))
            logger.info("  OK (%d)", year)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("Skipping %d: resource not found", year)
                missing.append(year)
            else:
                raise
    if missing:
        logger.warning(
            "Fetched %d/%d years; missing: %s",
            len(frames), len(resource_ids), missing,
        )
    return frames


def unify_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concat per-year DataFrames into one, raising if nothing was downloaded.

    The wave and wind pipelines both call this immediately after ``fetch_all``;
    keeping it here means the "no frames means nothing was fetched" contract
    has one source of truth.
    """
    if not frames:
        raise ValueError("No data was downloaded; cannot build unified dataset.")
    return pd.concat(frames, ignore_index=True)
