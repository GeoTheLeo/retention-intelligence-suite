from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

N_PATIENTS = 2000

AS_OF_DATE = pd.Timestamp("2026-09-01")

DISCHARGE_WINDOW_DAYS = 365

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample"
    / "healthcare_patients.csv"
)

PROGRAM_CONFIG = {
    "Chronic Care Management": {
        "probability": 0.45,
        "monthly_care_cost": 450.0,
    },
    "Post-Surgical Recovery": {
        "probability": 0.35,
        "monthly_care_cost": 800.0,
    },
    "Behavioral Health": {
        "probability": 0.20,
        "monthly_care_cost": 350.0,
    },
}


def generate_healthcare_patients(
    n_patients: int = N_PATIENTS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic patient-readmission dataset.

    Unlike Music and Education, readmission risk here is driven by two
    *independent* factors, not one: care-plan engagement (behavioral,
    same noisy-proxy design as the other industries) and chronic
    condition count (a clinical complexity factor unrelated to
    behavior). This mirrors how real readmission risk models work —
    a highly engaged patient with several chronic conditions can still
    carry meaningful baseline risk.
    """

    rng = np.random.default_rng(seed)

    true_engagement = rng.beta(a=2.0, b=2.0, size=n_patients)

    chronic_condition_count = np.clip(
        rng.poisson(lam=1.5, size=n_patients),
        0,
        6,
    )

    programs = list(PROGRAM_CONFIG.keys())

    program_probabilities = [
        PROGRAM_CONFIG[p]["probability"] for p in programs
    ]

    program_type = rng.choice(
        programs,
        size=n_patients,
        p=program_probabilities,
    )

    monthly_care_cost = np.array(
        [PROGRAM_CONFIG[p]["monthly_care_cost"] for p in program_type]
    )

    medication_adherence_rate = np.clip(
        rng.normal(loc=true_engagement * 100, scale=12, size=n_patients),
        0,
        100,
    ).round(1)

    followup_appointments_attended = np.clip(
        rng.normal(loc=true_engagement * 6, scale=1.2, size=n_patients),
        0,
        6,
    ).round().astype(int)

    care_plan_checkins = rng.poisson(
        lam=true_engagement * 4,
        size=n_patients,
    )

    remote_monitoring_hours = np.clip(
        rng.normal(loc=true_engagement * 25, scale=6, size=n_patients),
        0,
        None,
    ).round(1)

    days_since_last_contact = np.clip(
        rng.normal(loc=(1 - true_engagement) * 40, scale=8, size=n_patients),
        0,
        None,
    ).round().astype(int)

    readmission_logit = (
        -4.3
        + 4.0 * (1 - true_engagement)
        + 0.60 * chronic_condition_count
        + rng.normal(loc=0, scale=0.6, size=n_patients)
    )

    readmission_probability = 1 / (1 + np.exp(-readmission_logit))

    readmitted = (
        rng.uniform(size=n_patients) < readmission_probability
    ).astype(int)

    discharge_offset_days = rng.integers(
        low=1,
        high=DISCHARGE_WINDOW_DAYS + 1,
        size=n_patients,
    )

    discharge_date = AS_OF_DATE - pd.to_timedelta(
        discharge_offset_days, unit="D"
    )

    days_observed = discharge_offset_days

    # Readmission, when it happens, tends to cluster in the first 30-90
    # days post-discharge rather than being spread evenly across the
    # whole observed window - this matches real readmission timing.
    tenure_alpha = 1.2
    tenure_beta = 3.0 + 2.0 * true_engagement

    tenure_fraction = rng.beta(tenure_alpha, tenure_beta, size=n_patients)

    tenure_days = np.round(tenure_fraction * days_observed).astype(int)

    readmission_date = discharge_date + pd.to_timedelta(
        tenure_days, unit="D"
    )

    readmission_date = pd.Series(readmission_date).where(
        readmitted == 1, pd.NaT
    )

    return pd.DataFrame(
        {
            "patient_id": np.arange(3001, 3001 + n_patients),
            "program_type": program_type,
            "monthly_care_cost": monthly_care_cost,
            "chronic_condition_count": chronic_condition_count,
            "medication_adherence_rate": medication_adherence_rate,
            "followup_appointments_attended": followup_appointments_attended,
            "care_plan_checkins": care_plan_checkins,
            "remote_monitoring_hours": remote_monitoring_hours,
            "days_since_last_contact": days_since_last_contact,
            "discharge_date": discharge_date.strftime("%Y-%m-%d"),
            "readmission_date": readmission_date.dt.strftime("%Y-%m-%d"),
            "readmitted": readmitted,
        }
    )


def main() -> None:

    dataframe = generate_healthcare_patients()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(dataframe):,} rows to {OUTPUT_PATH}")

    print(f"Readmission rate: {dataframe['readmitted'].mean():.1%}")


if __name__ == "__main__":

    main()
