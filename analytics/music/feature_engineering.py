from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.utils.data_loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "business"
    / "music"
    / "dataset_manifest.yaml"
)

DATASET = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "music_users.csv"
)


class MusicFeatureEngineering:
    """
    Creates business-facing analytical features from raw customer data.
    """

    def __init__(self) -> None:

        loader = DataLoader(MANIFEST)

        self.df = loader.load_csv(DATASET)

    def build_features(self) -> pd.DataFrame:

        df = self.df.copy()

        # -----------------------------
        # Revenue Band
        # -----------------------------

        df["revenue_band"] = pd.cut(
            df["monthly_revenue"],
            bins=[-1, 0, 15, 22, 100],
            labels=[
                "Free",
                "Low",
                "Medium",
                "High",
            ],
        )

        # -----------------------------
        # Loyalty Tier
        # -----------------------------

        loyalty_score = (
            df["listening_hours"] * 0.40
            + df["playlist_saves"] * 0.30
            + df["concerts_attended"] * 8
            + df["merchandise_spend"] * 0.05
        )

        df["loyalty_score"] = loyalty_score.round(1)

        df["loyalty_tier"] = pd.cut(
            loyalty_score,
            bins=[0, 20, 40, 60, 1000],
            labels=[
                "Casual",
                "Regular",
                "Loyal",
                "Champion",
            ],
        )

        # -----------------------------
        # Engagement Score
        # -----------------------------

        engagement = (
            df["listening_hours"] * 0.50
            + df["playlist_saves"] * 0.25
            + (60 - df["days_since_last_activity"]) * 0.25
        )

        engagement = engagement.clip(lower=0)

        engagement = (
            engagement / engagement.max()
        ) * 100

        df["engagement_score"] = engagement.round(1)

        # -----------------------------
        # Churn Risk
        # -----------------------------

        conditions = [
            df["days_since_last_activity"] >= 60,
            df["days_since_last_activity"] >= 30,
            df["days_since_last_activity"] >= 14,
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

    features = MusicFeatureEngineering()

    df = features.build_features()

    print()

    print(
        df[
            [
                "user_id",
                "engagement_score",
                "revenue_band",
                "loyalty_tier",
                "churn_risk",
            ]
        ]
    )


if __name__ == "__main__":

    main()