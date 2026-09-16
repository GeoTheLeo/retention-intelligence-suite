from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.utils.data_loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "business"
    / "healthcare"
    / "dataset_manifest.yaml"
)

DATASET = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "healthcare_patients.csv"
)


class HealthcareFeatureEngineering:
    """
    Creates business-facing analytical features from raw patient data.

    Deliberately keeps behavioral engagement and clinical complexity as
    two separate signals rather than blending them into one score - a
    highly engaged patient with several chronic conditions is not the
    same risk profile as a disengaged patient with none, even if they
    land on the same combined number.
    """

    def __init__(self) -> None:

        loader = DataLoader(MANIFEST)

        self.df = loader.load_csv(DATASET)

    def build_features(self) -> pd.DataFrame:

        df = self.df.copy()

        # -----------------------------
        # Care Engagement Score (behavioral)
        # -----------------------------

        engagement = (
            df["medication_adherence_rate"] * 0.35
            + df["followup_appointments_attended"] * 8
            + df["care_plan_checkins"] * 4
            + (40 - df["days_since_last_contact"]) * 0.5
        )

        engagement = engagement.clip(lower=0)

        engagement = (
            engagement / engagement.max()
        ) * 100

        df["care_engagement_score"] = engagement.round(1)

        # -----------------------------
        # Clinical Complexity Score
        # -----------------------------
        # Independent of behavior - reflects how medically complex the
        # patient is, not how well they're following their care plan.

        complexity = (
            df["chronic_condition_count"] / df["chronic_condition_count"].max()
        ) * 100

        df["clinical_complexity_score"] = complexity.round(1)

        # -----------------------------
        # Care Priority Tier
        # -----------------------------
        # Combines both axes rather than collapsing them into one
        # number - a care coordinator needs to know *why* someone is
        # high priority, not just that they are.

        def _priority(row: pd.Series) -> str:

            low_engagement = row["care_engagement_score"] < 50
            high_complexity = row["chronic_condition_count"] >= 2

            if low_engagement and high_complexity:
                return "Priority Outreach"

            if low_engagement or high_complexity:
                return "Monitor"

            return "Low Priority"

        df["care_priority_tier"] = df.apply(_priority, axis=1)

        # -----------------------------
        # Readmission Risk (rule-based — superseded by ChurnPredictor)
        # -----------------------------

        conditions = [
            (df["days_since_last_contact"] >= 30)
            & (df["chronic_condition_count"] >= 2),
            (df["days_since_last_contact"] >= 30)
            | (df["chronic_condition_count"] >= 3),
            (df["days_since_last_contact"] >= 15)
            | (df["chronic_condition_count"] >= 1),
        ]

        choices = [
            "High",
            "Medium",
            "Elevated",
        ]

        df["readmission_risk"] = "Low"

        for condition, value in zip(conditions, choices):

            df.loc[condition, "readmission_risk"] = value

        return df


def main() -> None:

    features = HealthcareFeatureEngineering()

    df = features.build_features()

    print()

    print(
        df[
            [
                "patient_id",
                "care_engagement_score",
                "clinical_complexity_score",
                "care_priority_tier",
                "readmission_risk",
            ]
        ]
    )


if __name__ == "__main__":

    main()
