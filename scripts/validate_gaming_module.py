from __future__ import annotations

import pandas as pd

from analytics.gaming.feature_engineering import GamingFeatureEngineering
from shared.models.churn_predictor import ChurnPredictor
from shared.models.cohort_analyzer import CohortAnalyzer
from shared.models.segmentation_engine import SegmentationEngine

FEATURE_COLUMNS = [
    "sessions_per_week",
    "avg_session_minutes",
    "levels_completed",
    "in_game_purchases",
    "guild_member",
    "days_since_last_session",
    "engagement_score",
]

AS_OF = pd.Timestamp("2026-09-01")


def d_n_retention(dataframe: pd.DataFrame, day: int) -> float:
    """
    Fraction of players still active `day` days after their first
    session, restricted to players observed long enough to fairly
    assess retention at that horizon (avoids survivorship bias from
    players who joined too recently to have reached day N yet).
    """

    first_session = pd.to_datetime(dataframe["first_session_date"])

    churn_date = pd.to_datetime(dataframe["churn_date"])

    days_observed = (AS_OF - first_session).dt.days

    eligible = days_observed >= day

    still_active = churn_date.isna() | (
        churn_date > first_session + pd.Timedelta(days=day)
    )

    return round((still_active[eligible].mean()) * 100, 1)


def main() -> None:

    engineered = GamingFeatureEngineering()

    dataframe = engineered.build_features()

    features = dataframe[FEATURE_COLUMNS]

    target = dataframe["churned"]

    print("=" * 60)
    print("CHURN PREDICTION — same ChurnPredictor as prior industries")
    print("=" * 60)

    predictor = ChurnPredictor()

    result = predictor.fit(features, target)

    print(f"AUC:       {result.auc}")
    print(f"Precision: {result.precision}")
    print(f"Recall:    {result.recall}")
    print()
    print("Feature importance:")
    print(result.feature_importance)

    print()
    print("=" * 60)
    print("SEGMENTATION — same SegmentationEngine as prior industries")
    print("=" * 60)

    engine = SegmentationEngine(n_clusters=4)

    labels, seg_result = engine.fit_predict(features)

    print(f"Silhouette score: {seg_result.silhouette}")
    print()
    print("Cluster sizes:")
    print(seg_result.cluster_sizes)

    print()
    print("=" * 60)
    print("COHORT ANALYSIS — same CohortAnalyzer as prior industries")
    print("=" * 60)

    analyzer = CohortAnalyzer(period="M")

    cohort_result = analyzer.build_retention_table(
        dataframe,
        signup_col="first_session_date",
        churn_col="churn_date",
        as_of=AS_OF,
    )

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(cohort_result.retention_table)

    print()
    print("=" * 60)
    print("D1 / D7 / D30 RETENTION — gaming-standard KPI")
    print("=" * 60)

    for day in (1, 7, 30):
        print(f"D{day} retention: {d_n_retention(dataframe, day)}%")


if __name__ == "__main__":

    main()
