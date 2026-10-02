"""Loading and validating the team-season stats table and metric labels."""

from __future__ import annotations

import difflib
import logging
from pathlib import Path

import pandas as pd

from cbb_similarity.config import (
    ALL_METRICS,
    DATA_DICTIONARY_PATH,
    ID_COLUMNS,
    LABEL_OVERRIDES,
    SEASON_COL,
    SEASON_NAME_COL,
    STATS_PATH,
    TEAM_COL,
    TEAM_ID_COL,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the input data does not match the expected schema."""


def load_stats(path: str | Path = STATS_PATH) -> pd.DataFrame:
    """Load the combined team-season stats table.

    Parameters
    ----------
    path
        CSV with one row per team per season, containing ``ID_COLUMNS`` and
        every metric in ``ALL_METRICS``.

    Returns
    -------
    pd.DataFrame
        Validated stats with a fresh ``RangeIndex``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    stats = pd.read_csv(path)
    _validate_stats(stats)
    logger.info(
        "Loaded %d team-seasons across %d seasons from %s",
        len(stats),
        stats[SEASON_COL].nunique(),
        path,
    )
    return stats.reset_index(drop=True)


def _validate_stats(stats: pd.DataFrame) -> None:
    missing = [c for c in (*ID_COLUMNS, *ALL_METRICS) if c not in stats.columns]
    if missing:
        raise DataValidationError(f"Stats table is missing required columns: {missing}")

    dupes = stats.duplicated([TEAM_ID_COL, SEASON_COL])
    if dupes.any():
        raise DataValidationError(
            f"Found {dupes.sum()} duplicate (teamId, competitionId) rows; "
            "expected one row per team per season."
        )

    null_counts = stats[list(ALL_METRICS)].isna().sum()
    if null_counts.any():
        raise DataValidationError(
            f"Metric columns contain nulls: {null_counts[null_counts > 0].to_dict()}"
        )


def load_stat_labels(path: str | Path = DATA_DICTIONARY_PATH) -> dict[str, str]:
    """Return a mapping of metric key -> human-readable label."""
    dictionary = pd.read_csv(path)
    labels = dict(zip(dictionary["key"], dictionary["label2"], strict=True))
    labels.update(LABEL_OVERRIDES)
    return labels


def resolve_season(stats: pd.DataFrame, season: int | str) -> int:
    """Resolve a season given as a competitionId (``41097``) or name prefix (``"2025-26"``)."""
    seasons = stats[[SEASON_COL, SEASON_NAME_COL]].drop_duplicates()

    if str(season).isdigit() and int(season) in set(seasons[SEASON_COL]):
        return int(season)

    matches = seasons[seasons[SEASON_NAME_COL].astype(str).str.startswith(str(season))]
    if len(matches) == 1:
        return int(matches[SEASON_COL].iloc[0])

    available = ", ".join(sorted(seasons[SEASON_NAME_COL].astype(str).str[:7]))
    raise ValueError(f"Unknown or ambiguous season {season!r}. Available: {available}")


def find_team_row(stats: pd.DataFrame, team: str, season_id: int) -> int:
    """Return the positional row index of ``team`` (by ``teamMarket``) in ``season_id``."""
    mask = (stats[TEAM_COL] == team) & (stats[SEASON_COL] == season_id)
    positions = mask.to_numpy().nonzero()[0]
    if len(positions) == 0:
        candidates = stats.loc[stats[SEASON_COL] == season_id, TEAM_COL].unique()
        suggestions = difflib.get_close_matches(team, candidates, n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"No team {team!r} found in season {season_id}.{hint}")
    return int(positions[0])
