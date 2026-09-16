from __future__ import annotations

from analytics.music.feature_engineering import MusicFeatureEngineering
from shared.models.segmentation_engine import SegmentationEngine

FEATURE_COLUMNS = [
    "listening_hours",
    "playlist_saves",
    "concerts_attended",
    "merchandise_spend",
    "days_since_last_activity",
    "loyalty_score",
    "engagement_score",
]

TIER_NAMES = ["Casual", "Regular", "Loyal", "Champion"]


def main() -> None:

    engineered = MusicFeatureEngineering()

    dataframe = engineered.build_features()

    features = dataframe[FEATURE_COLUMNS]

    engine = SegmentationEngine(n_clusters=4)

    labels, result = engine.fit_predict(features)

    print(f"Silhouette score: {result.silhouette}")
    print()
    print("Cluster sizes:")
    print(result.cluster_sizes)
    print()
    print("Cluster profile (mean feature values per segment):")
    print(result.cluster_profile)

    name_map = SegmentationEngine.name_clusters_by_rank(
        result.cluster_profile,
        rank_by="engagement_score",
        names=TIER_NAMES,
    )

    print()
    print("Segment name mapping:")
    print(name_map)

    dataframe["discovered_tier"] = labels.map(name_map)

    print()
    print("Old rule-based tier vs. newly discovered tier (first 10 rows):")
    print(
        dataframe[["user_id", "loyalty_tier", "discovered_tier"]].head(10)
    )


if __name__ == "__main__":

    main()
