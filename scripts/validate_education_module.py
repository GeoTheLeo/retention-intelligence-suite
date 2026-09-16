from __future__ import annotations

import pandas as pd

from analytics.education.feature_engineering import EducationFeatureEngineering
from shared.models.churn_predictor import ChurnPredictor
from shared.models.cohort_analyzer import CohortAnalyzer
from shared.models.segmentation_engine import SegmentationEngine

FEATURE_COLUMNS = [
    "tuition_per_term",
    "lms_hours_weekly",
    "assignments_submitted",
    "advising_sessions_attended",
    "tutoring_hours_used",
    "days_since_last_login",
    "academic_engagement_score",
    "support_utilization",
]


def main() -> None:

    engineered = EducationFeatureEngineering()

    dataframe = engineered.build_features()

    features = dataframe[FEATURE_COLUMNS]

    target = dataframe["dropped_out"]

    print("=" * 60)
    print("CHURN (DROPOUT) PREDICTION — same ChurnPredictor as Music")
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
    print("SEGMENTATION — same SegmentationEngine as Music")
    print("=" * 60)

    engine = SegmentationEngine(n_clusters=4)

    labels, seg_result = engine.fit_predict(features)

    print(f"Silhouette score: {seg_result.silhouette}")
    print()
    print("Cluster sizes:")
    print(seg_result.cluster_sizes)

    print()
    print("=" * 60)
    print("COHORT ANALYSIS — same CohortAnalyzer as Music")
    print("=" * 60)

    analyzer = CohortAnalyzer(period="M")

    cohort_result = analyzer.build_retention_table(
        dataframe,
        signup_col="enrollment_date",
        churn_col="withdrawal_date",
        as_of=pd.Timestamp("2026-09-01"),
    )

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(cohort_result.retention_table)


if __name__ == "__main__":

    main()
