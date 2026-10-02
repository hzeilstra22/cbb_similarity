"""Nearest-neighbour team comparisons ("comps") using cosine similarity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from cbb_similarity.config import SEASON_COL, SEASON_NAME_COL, TEAM_COL
from cbb_similarity.data.loader import find_team_row


class TeamSimilarityModel:
    """Finds the team-seasons most similar to a given team-season.

    Similarity is cosine similarity over season-standardized metrics, so it
    captures the *shape* of a team's profile rather than its magnitude.

    Parameters
    ----------
    metrics
        Feature columns to compare on.
    """

    def __init__(self, metrics: Sequence[str]) -> None:
        self.metrics = list(metrics)
        self._features: np.ndarray | None = None
        self._teams: pd.DataFrame | None = None

    def fit(self, scaled: pd.DataFrame) -> TeamSimilarityModel:
        """Store the standardized feature matrix for ``scaled`` team-seasons."""
        self._features = scaled[self.metrics].to_numpy(dtype=float)
        self._teams = scaled.reset_index(drop=True)
        return self

    def most_similar(
        self,
        team: str,
        season_id: int,
        n: int = 10,
        same_season: bool = False,
    ) -> pd.DataFrame:
        """Return the target team followed by its ``n`` most similar team-seasons.

        Returns
        -------
        pd.DataFrame
            Columns ``row`` (position in the fitted frame) and ``similarity``,
            sorted by similarity, with the target team in the first row.
        """
        if self._features is None or self._teams is None:
            raise RuntimeError("Call fit() before most_similar().")

        target = find_team_row(self._teams, team, season_id)
        sims = cosine_similarity(self._features[[target]], self._features)[0]

        candidates = np.ones(len(sims), dtype=bool)
        candidates[target] = False
        if same_season:
            candidates &= (self._teams[SEASON_COL] == season_id).to_numpy()

        rows = np.flatnonzero(candidates)
        top = rows[np.argsort(-sims[rows], kind="stable")[:n]]
        ordered = np.concatenate([[target], top])
        return pd.DataFrame({"row": ordered, "similarity": sims[ordered]})


def generate_comps_table(
    scaled: pd.DataFrame,
    ranks: pd.DataFrame,
    team: str,
    season_id: int,
    metrics: Sequence[str],
    n: int = 10,
    same_season: bool = False,
) -> pd.DataFrame:
    """Build a comparison table of the teams most similar to ``team`` in ``season_id``.

    Similarity is computed on ``scaled`` features; the metric values shown are
    season percentile ranks from ``ranks``. ``scaled`` and ``ranks`` must be
    row-aligned (both are derived from the same stats frame).

    Returns
    -------
    pd.DataFrame
        ``Team``, ``similarity``, team colours and one column per metric. The
        first row is the target team itself (similarity 1.0).
    """
    if len(scaled) != len(ranks) or not scaled.index.equals(ranks.index):
        raise ValueError("`scaled` and `ranks` must be row-aligned.")

    model = TeamSimilarityModel(metrics).fit(scaled)
    neighbours = model.most_similar(team, season_id, n=n, same_season=same_season)

    table = ranks.iloc[neighbours["row"]].reset_index(drop=True)
    table.insert(0, "similarity", neighbours["similarity"].to_numpy())
    table.insert(
        0,
        "Team",
        table[TEAM_COL] + " " + table[SEASON_NAME_COL].astype(str).str[:7],
    )
    return table[["Team", "similarity", "hexColor1", "hexColor2", *metrics]]
