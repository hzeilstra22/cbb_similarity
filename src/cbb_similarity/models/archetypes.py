"""KMeans team archetypes with a 2-D PCA projection for visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from cbb_similarity.config import ArchetypeSpec


class ArchetypeModel:
    """Clusters team-seasons into named archetypes for one metric group.

    Parameters
    ----------
    spec
        Metrics, cluster count, random seed and cluster names for this model.
    """

    def __init__(self, spec: ArchetypeSpec) -> None:
        self.spec = spec
        self.kmeans = KMeans(n_clusters=spec.n_clusters, random_state=spec.random_state)
        self.pca = PCA(n_components=2)

    @property
    def metrics(self) -> list[str]:
        return list(self.spec.metrics)

    def fit(self, scaled: pd.DataFrame) -> ArchetypeModel:
        """Fit KMeans and PCA on season-standardized metrics."""
        features = scaled[self.metrics].to_numpy(dtype=float)
        self.kmeans.fit(features)
        self.pca.fit(features)
        return self

    def transform(self, scaled: pd.DataFrame) -> pd.DataFrame:
        """Return cluster id, archetype name and PCA coordinates per team-season.

        Output columns are prefixed with the model name, e.g. ``shooting_archetype``.
        """
        features = scaled[self.metrics].to_numpy(dtype=float)
        clusters = self.kmeans.predict(features)
        coords = self.pca.transform(features)

        prefix = self.spec.name
        return pd.DataFrame(
            {
                f"{prefix}_cluster": clusters,
                f"{prefix}_archetype": pd.Series(clusters).map(self.spec.cluster_names),
                f"{prefix}_x": coords[:, 0],
                f"{prefix}_y": coords[:, 1],
            },
            index=scaled.index,
        )

    def top_component_features(self, n: int = 2) -> list[list[tuple[str, float]]]:
        """Return the ``n`` highest-|loading| features for each PCA component.

        Useful for labelling the axes of the archetype scatter plot.
        """
        result = []
        for loadings in self.pca.components_:
            idx = np.argsort(np.abs(loadings))[::-1][:n]
            result.append([(self.metrics[i], float(loadings[i])) for i in idx])
        return result

    def cluster_profile(self, ranks: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
        """Mean season-percentile of each metric per archetype, plus team counts.

        Use this to sanity-check that ``cluster_names`` still describe the clusters.
        """
        name_col = f"{self.spec.name}_archetype"
        frame = ranks[self.metrics].join(assignments[[name_col]])
        profile = frame.groupby(name_col)[self.metrics].mean()
        profile.insert(0, "n_teams", frame.groupby(name_col).size())
        return profile
