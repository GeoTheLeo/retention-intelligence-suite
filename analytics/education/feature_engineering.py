from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.utils.data_loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "business"
    / "education"
    / "dataset_manifest.yaml"
)

DATASET = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "education_students.csv"
)


class EducationFeatureEngineering:
    """
    Creates business-facing analytical features from raw student data.
    """

    def __init__(self) -> None:

        loader = DataLoader(MANIFEST)

        self.df = loader.load_csv(DATASET)

    def build_features(self) -> pd.DataFrame:

        df = self.df.copy()

        # -----------------------------
        # Academic Engagement Score
        # -----------------------------
        # Weighted toward LMS activity and assignment completion, with
        # a recency penalty — this reflects how present the student is
        # in the actual coursework, not how much they spend.

        engagement = (
            df["lms_hours_weekly"] * 0.45
            + df["assignments_submitted"] * 1.5
            + (45 - df["days_since_last_login"]) * 0.30
        )

        engagement = engagement.clip(lower=0)

        engagement = (
            engagement / engagement.max()
        ) * 100

        df["academic_engagement_score"] = engagement.round(1)

        # -----------------------------
        # Support Utilization
        # -----------------------------
        # Deliberately kept separate from engagement rather than folded
        # in: using tutoring or advising doesn't mean the same thing as
        # attending class. A student can be highly engaged and still
        # rely heavily on support, or barely use it at all.

        support = (
            df["tutoring_hours_used"] * 0.6
            + df["advising_sessions_attended"] * 5
        )

        df["support_utilization"] = support.round(1)

        # -----------------------------
        # Academic Standing
        # -----------------------------

        df["academic_standing"] = pd.cut(
            engagement,
            bins=[-1, 25, 50, 75, 1000],
            labels=[
                "At Risk",
                "Adequate",
                "Good Standing",
                "Exemplary",
            ],
        )

        # -----------------------------
        # Dropout Risk (rule-based — superseded by ChurnPredictor)
        # -----------------------------

        conditions = [
            df["days_since_last_login"] >= 35,
            df["days_since_last_login"] >= 21,
            df["days_since_last_login"] >= 10,
        ]

        choices = [
            "High",
            "Medium",
            "Elevated",
        ]

        df["dropout_risk"] = "Low"

        for condition, value in zip(conditions, choices):

            df.loc[condition, "dropout_risk"] = value

        return df


def main() -> None:

    features = EducationFeatureEngineering()

    df = features.build_features()

    print()

    print(
        df[
            [
                "student_id",
                "academic_engagement_score",
                "support_utilization",
                "academic_standing",
                "dropout_risk",
            ]
        ]
    )


if __name__ == "__main__":

    main()
