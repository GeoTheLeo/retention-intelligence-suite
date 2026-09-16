from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.utils.data_loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "business"
    / "ecommerce"
    / "dataset_manifest.yaml"
)

DATASET = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "ecommerce_customers.csv"
)


class EcommerceFeatureEngineering:
    """
    Creates business-facing analytical features from raw customer data,
    built around RFM (Recency, Frequency, Monetary) - the standard
    retail framework, rather than a single blended "loyalty" score.
    """

    def __init__(self) -> None:

        loader = DataLoader(MANIFEST)

        self.df = loader.load_csv(DATASET)

    def build_features(self) -> pd.DataFrame:

        df = self.df.copy()

        # -----------------------------
        # RFM Scores (1-5 quintile scale, standard retail convention)
        # -----------------------------
        # Recency: lower days-since-order is better, so quintiles are
        # reversed relative to Frequency and Monetary.

        df["recency_score"] = pd.qcut(
            df["days_since_last_order"],
            q=5,
            labels=[5, 4, 3, 2, 1],
        ).astype(int)

        df["frequency_score"] = pd.qcut(
            df["total_orders"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

        df["monetary_score"] = pd.qcut(
            df["avg_order_value"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

        df["rfm_score"] = (
            df["recency_score"]
            + df["frequency_score"]
            + df["monetary_score"]
        )

        # -----------------------------
        # Engagement Score (behavioral, feeds the shared models)
        # -----------------------------

        engagement = (
            df["frequency_score"] * 8
            + df["recency_score"] * 8
            - df["cart_abandonment_count"] * 3
            + df["email_open_rate"] * 0.3
        )

        engagement = engagement.clip(lower=0)

        engagement = (
            engagement / engagement.max()
        ) * 100

        df["engagement_score"] = engagement.round(1)

        # -----------------------------
        # Customer Value Tier
        # -----------------------------
        # Built from RFM directly, not the engagement score - a
        # high-value customer isn't necessarily a highly "engaged" one
        # (e.g. an infrequent big-ticket buyer).

        df["value_tier"] = pd.cut(
            df["rfm_score"],
            bins=[0, 6, 9, 12, 15],
            labels=[
                "At Risk",
                "Needs Attention",
                "Loyal",
                "Champion",
            ],
        )

        # -----------------------------
        # Churn Risk (rule-based — superseded by ChurnPredictor)
        # -----------------------------

        conditions = [
            df["days_since_last_order"] >= 150,
            df["days_since_last_order"] >= 90,
            df["days_since_last_order"] >= 45,
        ]

        choices = [
            "High",
            "Medium",
            "Elevated",
        ]

        df["churn_risk"] = "Low"

        for condition, value in zip(conditions, choices):

            df.loc[condition, "churn_risk"] = value

        return df


def main() -> None:

    features = EcommerceFeatureEngineering()

    df = features.build_features()

    print()

    print(
        df[
            [
                "customer_id",
                "rfm_score",
                "engagement_score",
                "value_tier",
                "churn_risk",
            ]
        ]
    )


if __name__ == "__main__":

    main()
