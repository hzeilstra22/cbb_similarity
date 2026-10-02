import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cbb_similarity.config import ALL_METRICS, ID_COLUMNS
from cbb_similarity.features.preprocessing import percentile_rank_by_season, scale_by_season


def test_scale_by_season_matches_per_season_standard_scaler(stats):
    metrics = list(ALL_METRICS)
    scaled = scale_by_season(stats, metrics)

    for _, idx in stats.groupby("competitionId").groups.items():
        expected = StandardScaler().fit_transform(stats.loc[idx, metrics])
        np.testing.assert_allclose(scaled.loc[idx, metrics].to_numpy(), expected, atol=1e-10)


def test_scale_by_season_preserves_order_and_does_not_mutate(stats):
    original = stats.copy()
    scaled = scale_by_season(stats, ALL_METRICS)

    assert scaled.index.equals(stats.index)
    assert (scaled["teamMarket"] == stats["teamMarket"]).all()
    pd.testing.assert_frame_equal(stats, original)


def test_scale_by_season_handles_zero_variance(stats):
    stats["pace"] = 70.0
    scaled = scale_by_season(stats, ["pace"])
    assert (scaled["pace"] == 0).all()


def test_percentile_rank_by_season(stats):
    ranks = percentile_rank_by_season(stats, ALL_METRICS)

    assert list(ranks.columns) == [*ID_COLUMNS, *ALL_METRICS]
    assert ranks[list(ALL_METRICS)].gt(0).all().all()
    assert ranks[list(ALL_METRICS)].le(1).all().all()
    # Every season has its own top-ranked team
    assert (ranks.groupby("competitionId")["pace"].max() == 1.0).all()
