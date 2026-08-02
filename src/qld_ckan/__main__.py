"""Unified entry point: ``python -m qld_ckan {wave,wind} [...]``.

  python -m qld_ckan wave  --buoy mooloolaba    [--output PATH] [--year-min/--year-max Y]
  python -m qld_ckan wind  --station mountain-creek [--output PATH] [--year-min/--year-max Y]
"""
import argparse
import logging

from .wave.pipeline import run as wave_run
from .wind.pipeline import run as wind_run


def _add_year_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--year-min", type=int, default=None,
        help="Earliest year to include (inclusive; default: registry minimum).",
    )
    p.add_argument(
        "--year-max", type=int, default=None,
        help="Latest year to include (inclusive; default: registry maximum).",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m qld_ckan",
        description="Download and unify QLD CKAN observation feeds (wave buoys, wind stations).",
    )
    sub = parser.add_subparsers(dest="source", required=True)

    p_wave = sub.add_parser("wave", help="Wave-buoy network (Mooloolaba, Brisbane, …).")
    p_wave.add_argument(
        "--buoy", default="mooloolaba",
        help="Buoy slug to download (default: mooloolaba).",
    )
    p_wave.add_argument(
        "--output", default=None,
        help="Output CSV path (default: data/{buoy}_wave_data_{years}.csv).",
    )
    _add_year_args(p_wave)

    p_wind = sub.add_parser("wind", help="QLD AWS station 10 m wind feed.")
    p_wind.add_argument(
        "--station", default="mountain-creek",
        help="Station slug to download (default: mountain-creek).",
    )
    p_wind.add_argument(
        "--output", default=None,
        help="Output CSV path (default: data/{station}_wind_data_{years}.csv).",
    )
    _add_year_args(p_wind)

    args = parser.parse_args()
    if args.source == "wave":
        wave_run(
            buoy=args.buoy, output_path=args.output,
            year_min=args.year_min, year_max=args.year_max,
        )
    elif args.source == "wind":
        wind_run(
            station=args.station, output_path=args.output,
            year_min=args.year_min, year_max=args.year_max,
        )


if __name__ == "__main__":
    main()
