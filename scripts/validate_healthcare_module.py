from __future__ import annotations

import pandas as pd

from analytics.healthcare.feature_engineering import HealthcareFeatureEngineering
from shared.models.churn_predictor import ChurnPredictor
from shared.models.cohort_analyzer import CohortAnalyzer
from shared.models.segmentation_engine import SegmentationEngine

FEATURE_COLUMNS = [
    "monthly_care_cost",
    "medication_adherence_rate",
    "followup_appointments_attended",
    "care_plan_checkins",
    "remote_monitoring_hours",
    "days_since_last_contact",
    "care_engagement_score",
    "clinical_complexity_score",
]


def main() -> None:

    engineered = HealthcareFeatureEngineering()

    dataframe = engineered.build_features()

    features = dataframe[FEATURE_COLUMNS]

    target = dataframe["readmitted"]

    print("=" * 60)
    print("READMISSION PREDICTION — same ChurnPredictor as Music/Education")
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
    print("SEGMENTATION — same SegmentationEngine as Music/Education")
    print("=" * 60)

    engine = SegmentationEngine(n_clusters=4)

    labels, seg_result = engine.fit_predict(features)

    print(f"Silhouette score: {seg_result.silhouette}")
    print()
    print("Cluster sizes:")
    print(seg_result.cluster_sizes)

    print()
    print("=" * 60)
    print("COHORT ANALYSIS — same CohortAnalyzer as Music/Education")
    print("=" * 60)

    analyzer = CohortAnalyzer(period="M")

    cohort_result = analyzer.build_retention_table(
        dataframe,
        signup_col="discharge_date",
        churn_col="readmission_date",
        as_of=pd.Timestamp("2026-09-01"),
    )

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(cohort_result.retention_table)


if __name__ == "__main__":

    main()
