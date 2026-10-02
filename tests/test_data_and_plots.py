import pytest

from cbb_similarity.config import SHOOTING_METRICS
from cbb_similarity.data.loader import DataValidationError, load_stats, resolve_season
from cbb_similarity.models.similarity import generate_comps_table
from cbb_similarity.pipeline import build_features
from cbb_similarity.visualization.plots import plot_radar_comparison


def test_load_stats_rejects_missing_columns(stats, tmp_path):
    path = tmp_path / "stats.csv"
    stats.drop(columns=["fg3Pct"]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="fg3Pct"):
        load_stats(path)


def test_load_stats_rejects_duplicate_team_seasons(stats, tmp_path):
    path = tmp_path / "stats.csv"
    stats.pipe(lambda df: df._append(df.iloc[[0]])).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="duplicate"):
        load_stats(path)


@pytest.mark.parametrize("season", [1002, "1002", "2024-25"])
def test_resolve_season(stats, season):
    assert resolve_season(stats, season) == 1002


def test_resolve_season_unknown(stats):
    with pytest.raises(ValueError, match="Available"):
        resolve_season(stats, "1999-00")


def test_radar_handles_same_school_in_comps(stats):
    """Regression: the notebook version crashed when two rows shared team colours."""
    # Make Team 5's 2024-25 season identical to its 2025-26 season so both appear
    features = build_features(stats)
    target = features.stats.query("teamMarket == 'Team 5' and competitionId == 1003").index[0]
    twin = features.stats.query("teamMarket == 'Team 5' and competitionId == 1002").index[0]
    features.scaled.loc[twin, list(SHOOTING_METRICS)] = features.scaled.loc[
        target, list(SHOOTING_METRICS)
    ]

    table = generate_comps_table(
        features.scaled, features.ranks, "Team 5", 1003, list(SHOOTING_METRICS)
    )
    assert table["Team"].iloc[1] == "Team 5 2024-25"

    fig = plot_radar_comparison(table, list(SHOOTING_METRICS), n_teams=3)
    assert len(fig.data) == 3
    assert fig.layout.title.text == "Team 5 2025-26 Comparison"
