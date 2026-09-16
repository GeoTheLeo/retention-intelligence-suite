from __future__ import annotations

import pandas as pd

from analytics.ecommerce.feature_engineering import EcommerceFeatureEngineering
from shared.models.churn_predictor import ChurnPredictor
from shared.models.cohort_analyzer import CohortAnalyzer
from shared.models.segmentation_engine import SegmentationEngine

FEATURE_COLUMNS = [
    "total_orders",
    "avg_order_value",
    "total_spend",
    "days_since_last_order",
    "cart_abandonment_count",
    "email_open_rate",
    "engagement_score",
]


def main() -> None:

    engineered = EcommerceFeatureEngineering()

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
        signup_col="first_purchase_date",
        churn_col="churn_date",
        as_of=pd.Timestamp("2026-09-01"),
    )

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(cohort_result.retention_table)


if __name__ == "__main__":

    main()
