"""Season-relative feature engineering.

Teams are always compared relative to their own season so that league-wide
trends (e.g. the rise in three-point rate) don't dominate the models.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from cbb_similarity.config import ID_COLUMNS, SEASON_COL


def scale_by_season(
    stats: pd.DataFrame,
    metrics: Sequence[str],
    season_col: str = SEASON_COL,
) -> pd.DataFrame:
    """Standardize ``metrics`` to z-scores within each season.

    Equivalent to fitting a ``StandardScaler`` per season. Zero-variance
    columns are scaled to 0 rather than NaN. Row order and index are preserved.
    """
    cols = list(metrics)
    grouped = stats.groupby(season_col)[cols]
    mean = grouped.transform("mean")
    std = grouped.transform("std", ddof=0).replace(0, 1.0)

    scaled = stats.copy()
    scaled[cols] = (stats[cols] - mean) / std
    return scaled


def percentile_rank_by_season(
    stats: pd.DataFrame,
    metrics: Sequence[str],
    season_col: str = SEASON_COL,
) -> pd.DataFrame:
    """Return ID columns plus each metric's percentile rank (0-1] within its season."""
    ranks = stats.groupby(season_col)[list(metrics)].rank(pct=True)
    return pd.concat([stats[list(ID_COLUMNS)], ranks], axis=1)
