import numpy as np
import pandas as pd
import pytest

from cbb_similarity.config import ALL_METRICS


@pytest.fixture
def stats() -> pd.DataFrame:
    """Synthetic stats: 3 seasons x 30 teams with every required column.

    Rows are shuffled across seasons to catch any code that assumes sorted input.
    """
    rng = np.random.default_rng(42)
    seasons = {1001: "2023-24 Men's Basketball", 1002: "2024-25 Men's Basketball",
               1003: "2025-26 Men's Basketball"}  # fmt: skip
    rows = []
    for season_id, season_name in seasons.items():
        for team in range(30):
            row = {
                "teamId": team,
                "competitionId": season_id,
                "teamMarket": f"Team {team}",
                "teamName": f"Mascot {team}",
                "competitionName": season_name,
                # Same school shares colours across seasons, like the real data
                "hexColor1": f"#{team:02x}0000",
                "hexColor2": f"#00{team:02x}00",
            }
            # Season-level shift so season-relative scaling actually matters
            row.update({m: rng.normal(loc=season_id - 1000, scale=1.0) for m in ALL_METRICS})
            rows.append(row)
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
