"""Command-line interface.

Examples
--------
    cbb-similarity archetypes
    cbb-similarity comps --team Duke --season 2025-26 --group shooting --radar
    cbb-similarity comps --team Michigan --season 41097 --same-season
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from cbb_similarity.config import METRIC_GROUPS, OUTPUT_DIR, STATS_PATH
from cbb_similarity.data.loader import load_stat_labels, load_stats, resolve_season
from cbb_similarity.models.similarity import generate_comps_table
from cbb_similarity.pipeline import build_features, run_pipeline
from cbb_similarity.visualization.plots import plot_radar_comparison

logger = logging.getLogger("cbb_similarity")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbb-similarity",
        description="College basketball team archetypes and similarity comps.",
    )
    parser.add_argument("--data", type=Path, default=STATS_PATH, help="Path to combined stats CSV.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    arch = sub.add_parser("archetypes", help="Fit archetype models and write outputs.")
    arch.add_argument("--no-plots", action="store_true", help="Skip writing HTML plots.")

    comps = sub.add_parser("comps", help="Find the most similar team-seasons to a team.")
    comps.add_argument("--team", required=True, help='Team market name, e.g. "Duke".')
    comps.add_argument(
        "--season", required=True, help='Season name prefix ("2025-26") or competitionId.'
    )
    comps.add_argument(
        "--group",
        choices=sorted(METRIC_GROUPS),
        default="overall",
        help="Metric group to compare on (default: overall).",
    )
    comps.add_argument("-n", type=int, default=10, help="Number of comps (default: 10).")
    comps.add_argument("--same-season", action="store_true", help="Only compare within season.")
    comps.add_argument("--radar", action="store_true", help="Also write a radar chart (HTML).")
    return parser


def _run_comps(args: argparse.Namespace) -> None:
    stats = load_stats(args.data)
    season_id = resolve_season(stats, args.season)
    features = build_features(stats)
    metrics = list(METRIC_GROUPS[args.group])

    table = generate_comps_table(
        features.scaled,
        features.ranks,
        team=args.team,
        season_id=season_id,
        metrics=metrics,
        n=args.n,
        same_season=args.same_season,
    )

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.drop(columns=["hexColor1", "hexColor2"]).round(3).to_string(index=False))

    out_dir = args.output_dir / "comps"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(args.team)}_{season_id}_{args.group}"
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    logger.info("Wrote %s", out_dir / f"{stem}.csv")

    if args.radar:
        title = f"{table['Team'].iloc[0]} {args.group.title()} Comparison"
        fig = plot_radar_comparison(table, metrics, load_stat_labels(), title=title)
        fig.write_html(out_dir / f"{stem}_radar.html", include_plotlyjs="cdn")
        logger.info("Wrote %s", out_dir / f"{stem}_radar.html")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.command == "archetypes":
            run_pipeline(args.data, args.output_dir, save_plots=not args.no_plots)
        elif args.command == "comps":
            _run_comps(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
