from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class SegmentationResult:
    n_clusters: int
    silhouette: float
    cluster_sizes: pd.Series
    cluster_profile: pd.DataFrame


class SegmentationEngine:
    """
    Industry-agnostic behavioral segmentation via KMeans clustering.

    Works on any numeric feature matrix — discovers natural groupings in
    the data instead of relying on fixed, hand-picked thresholds.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
    ) -> None:

        self.n_clusters = n_clusters

        self.random_state = random_state

        self.scaler = StandardScaler()

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )

        self._feature_names: list[str] | None = None

    def fit_predict(
        self,
        features: pd.DataFrame,
    ) -> tuple[pd.Series, SegmentationResult]:

        self._feature_names = list(features.columns)

        scaled = self.scaler.fit_transform(features)

        labels = self.model.fit_predict(scaled)

        cluster_labels = pd.Series(
            labels,
            index=features.index,
            name="segment",
        )

        silhouette = round(silhouette_score(scaled, labels), 3)

        cluster_sizes = cluster_labels.value_counts().sort_index()

        cluster_profile = (
            features.assign(segment=cluster_labels)
            .groupby("segment")
            .mean()
            .round(2)
        )

        result = SegmentationResult(
            n_clusters=self.n_clusters,
            silhouette=silhouette,
            cluster_sizes=cluster_sizes,
            cluster_profile=cluster_profile,
        )

        return cluster_labels, result

    def assign(self, features: pd.DataFrame) -> pd.Series:
        """
        Assigns cluster labels to new rows using the already-fitted model.
        """

        if self._feature_names is None:
            raise RuntimeError("Call fit_predict() before assign().")

        scaled = self.scaler.transform(features[self._feature_names])

        labels = self.model.predict(scaled)

        return pd.Series(
            labels,
            index=features.index,
            name="segment",
        )

    @staticmethod
    def name_clusters_by_rank(
        cluster_profile: pd.DataFrame,
        rank_by: str,
        names: list[str],
    ) -> dict[int, str]:
        """
        Orders clusters by a chosen column (e.g. an engagement or value
        score) and maps them to human-readable names, lowest to highest.
        Keeps the engine itself industry-agnostic — the caller decides
        what "rank_by" means and what names apply for their domain.
        """

        if len(names) != len(cluster_profile):
            raise ValueError("names must match the number of clusters.")

        ordered = cluster_profile[rank_by].sort_values().index

        return dict(zip(ordered, names))
