"""End-to-end workflow: load data -> build features -> fit archetype models -> save outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cbb_similarity.config import (
    ALL_METRICS,
    ARCHETYPE_SPECS,
    ID_COLUMNS,
    OUTPUT_DIR,
    SEASON_NAME_COL,
    STATS_PATH,
    TEAM_COL,
    ArchetypeSpec,
)
from cbb_similarity.data.loader import load_stats
from cbb_similarity.features.preprocessing import percentile_rank_by_season, scale_by_season
from cbb_similarity.models.archetypes import ArchetypeModel
from cbb_similarity.visualization.plots import plot_archetypes

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Row-aligned views of the same team-seasons."""

    stats: pd.DataFrame  # raw metrics
    scaled: pd.DataFrame  # z-scores within season (model inputs)
    ranks: pd.DataFrame  # percentiles within season (for display)


@dataclass
class ArchetypeResults:
    features: FeatureSet
    assignments: pd.DataFrame  # ID columns + per-model cluster/archetype/x/y
    models: dict[str, ArchetypeModel] = field(default_factory=dict)
    profiles: dict[str, pd.DataFrame] = field(default_factory=dict)


def build_features(stats: pd.DataFrame) -> FeatureSet:
    """Compute season-standardized features and season percentile ranks."""
    return FeatureSet(
        stats=stats,
        scaled=scale_by_season(stats, ALL_METRICS),
        ranks=percentile_rank_by_season(stats, ALL_METRICS),
    )


def fit_archetypes(
    features: FeatureSet,
    specs: dict[str, ArchetypeSpec] = ARCHETYPE_SPECS,
) -> ArchetypeResults:
    """Fit every archetype model and collect assignments and cluster profiles."""
    assignments = features.ranks[list(ID_COLUMNS)].copy()
    results = ArchetypeResults(features=features, assignments=assignments)

    for name, spec in specs.items():
        model = ArchetypeModel(spec).fit(features.scaled)
        model_assignments = model.transform(features.scaled)
        results.assignments = results.assignments.join(model_assignments)
        results.models[name] = model
        results.profiles[name] = model.cluster_profile(features.ranks, model_assignments)

        pc1, pc2 = model.top_component_features()
        logger.info("[%s] PC1 drivers: %s | PC2 drivers: %s", name, _fmt(pc1), _fmt(pc2))

    return results


def save_outputs(results: ArchetypeResults, output_dir: Path, save_plots: bool = True) -> None:
    """Write assignments, cluster profiles and (optionally) interactive HTML plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "team_archetypes.csv"
    results.assignments.to_csv(assignments_path, index=False)
    logger.info("Wrote %s", assignments_path)

    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(exist_ok=True)
    for name, profile in results.profiles.items():
        profile.round(3).to_csv(profile_dir / f"{name}_profile.csv")

    if not save_plots:
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for name, model in results.models.items():
        spec = model.spec
        fig = plot_archetypes(
            results.assignments,
            x_col=f"{name}_x",
            y_col=f"{name}_y",
            color_col=f"{name}_archetype",
            x_title=spec.x_title,
            y_title=spec.y_title,
            title=spec.plot_title,
            hover_cols=[TEAM_COL, SEASON_NAME_COL],
        )
        path = plot_dir / f"{name}_archetypes.html"
        fig.write_html(path, include_plotlyjs="cdn")
        logger.info("Wrote %s", path)


def run_pipeline(
    data_path: str | Path = STATS_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    save_plots: bool = True,
) -> ArchetypeResults:
    """Run the full archetype workflow and write results to ``output_dir``."""
    stats = load_stats(data_path)
    results = fit_archetypes(build_features(stats))
    save_outputs(results, Path(output_dir), save_plots=save_plots)
    return results


def _fmt(features: list[tuple[str, float]]) -> str:
    return ", ".join(f"{name} ({loading:+.2f})" for name, loading in features)
