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

CANDIDATE_K_VALUES = [2, 3, 4, 5, 6]


def main() -> None:

    engineered = MusicFeatureEngineering()

    dataframe = engineered.build_features()

    features = dataframe[FEATURE_COLUMNS]

    print("Silhouette score by number of clusters:")
    print()

    for k in CANDIDATE_K_VALUES:

        engine = SegmentationEngine(n_clusters=k)

        _, result = engine.fit_predict(features)

        print(f"k={k}: silhouette={result.silhouette}")


if __name__ == "__main__":

    main()
