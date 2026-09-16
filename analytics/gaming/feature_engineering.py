from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.utils.data_loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "business"
    / "gaming"
    / "dataset_manifest.yaml"
)

DATASET = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "gaming_players.csv"
)


class GamingFeatureEngineering:
    """
    Creates business-facing analytical features from raw player data.

    Keeps monetization (in-game purchases) and social ties (guild
    membership) as separate signals from engagement - a whale can be a
    light player, and a highly engaged player can spend nothing.
    """

    def __init__(self) -> None:

        loader = DataLoader(MANIFEST)

        self.df = loader.load_csv(DATASET)

    def build_features(self) -> pd.DataFrame:

        df = self.df.copy()

        # -----------------------------
        # Engagement Score (session behavior only)
        # -----------------------------

        engagement = (
            df["sessions_per_week"] * 5
            + df["avg_session_minutes"] * 0.5
            + df["levels_completed"] * 0.3
            - df["days_since_last_session"] * 0.5
        )

        engagement = engagement.clip(lower=0)

        engagement = (
            engagement / engagement.max()
        ) * 100

        df["engagement_score"] = engagement.round(1)

        # -----------------------------
        # Player Value Tier (monetization, not engagement)
        # -----------------------------

        df["player_value_tier"] = pd.cut(
            df["in_game_purchases"],
            bins=[-1, 0, 50, 200, 100000],
            labels=[
                "Non-Paying",
                "Occasional Spender",
                "High Spender",
                "Whale",
            ],
        )

        # -----------------------------
        # Churn Risk (rule-based — superseded by ChurnPredictor)
        # -----------------------------
        # Guild membership softens the thresholds, reflecting its real
        # protective effect against churn.

        threshold_adjustment = df["guild_member"] * 10

        conditions = [
            df["days_since_last_session"]
            >= (40 + threshold_adjustment),
            df["days_since_last_session"]
            >= (25 + threshold_adjustment),
            df["days_since_last_session"]
            >= (12 + threshold_adjustment),
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

    features = GamingFeatureEngineering()

    df = features.build_features()

    print()

    print(
        df[
            [
                "player_id",
                "engagement_score",
                "player_value_tier",
                "guild_member",
                "churn_risk",
            ]
        ]
    )


if __name__ == "__main__":

    main()
