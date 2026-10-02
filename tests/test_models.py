import pytest

from cbb_similarity.config import ARCHETYPE_SPECS, SHOOTING_METRICS
from cbb_similarity.models.archetypes import ArchetypeModel
from cbb_similarity.models.similarity import generate_comps_table
from cbb_similarity.pipeline import build_features, fit_archetypes


@pytest.fixture
def features(stats):
    return build_features(stats)


def test_comps_table_starts_with_target_team(features):
    table = generate_comps_table(
        features.scaled, features.ranks, "Team 5", 1003, list(SHOOTING_METRICS), n=10
    )

    assert len(table) == 11
    assert table["Team"].iloc[0] == "Team 5 2025-26"
    assert table["similarity"].iloc[0] == pytest.approx(1.0)
    assert table["similarity"].iloc[1:].is_monotonic_decreasing
    assert list(table.columns) == ["Team", "similarity", "hexColor1", "hexColor2",
                                   *SHOOTING_METRICS]  # fmt: skip


def test_comps_table_same_season_returns_full_n(features):
    table = generate_comps_table(
        features.scaled, features.ranks, "Team 5", 1003, list(SHOOTING_METRICS),
        n=10, same_season=True,
    )  # fmt: skip

    assert len(table) == 11
    assert table["Team"].str.endswith("2025-26").all()


def test_comps_table_unknown_team_suggests_matches(features):
    with pytest.raises(ValueError, match="Did you mean"):
        generate_comps_table(features.scaled, features.ranks, "Team 55", 1003, ["pace"])


def test_archetype_model_assigns_named_clusters(features):
    spec = ARCHETYPE_SPECS["shooting"]
    model = ArchetypeModel(spec).fit(features.scaled)
    out = model.transform(features.scaled)

    assert out.index.equals(features.scaled.index)
    assert set(out["shooting_archetype"]) <= set(spec.cluster_names.values())
    assert out["shooting_archetype"].notna().all()
    assert len(model.top_component_features(n=2)) == 2


def test_fit_archetypes_is_deterministic(features):
    first = fit_archetypes(features).assignments
    second = fit_archetypes(features).assignments
    assert first.equals(second)
    for name in ARCHETYPE_SPECS:
        assert {f"{name}_archetype", f"{name}_x", f"{name}_y"} <= set(first.columns)
