from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

N_STUDENTS = 2000

AS_OF_DATE = pd.Timestamp("2026-09-01")

SIGNUP_WINDOW_DAYS = 365

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample"
    / "education_students.csv"
)

ENROLLMENT_CONFIG = {
    "Certificate": {"probability": 0.25, "tuition_per_term": 1200.0},
    "Undergraduate": {"probability": 0.55, "tuition_per_term": 4500.0},
    "Graduate": {"probability": 0.20, "tuition_per_term": 6000.0},
}


def generate_education_students(
    n_students: int = N_STUDENTS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic student-retention dataset.

    Same design principle as the Music generator: every observable
    feature is a noisy reading of a hidden "true academic engagement"
    value, and dropout is derived from that same hidden value plus its
    own independent noise — not a deterministic formula.
    """

    rng = np.random.default_rng(seed)

    true_engagement = rng.beta(a=2.0, b=2.0, size=n_students)

    enrollment_types = list(ENROLLMENT_CONFIG.keys())

    enrollment_probabilities = [
        ENROLLMENT_CONFIG[t]["probability"] for t in enrollment_types
    ]

    enrollment_type = rng.choice(
        enrollment_types,
        size=n_students,
        p=enrollment_probabilities,
    )

    tuition_per_term = np.array(
        [ENROLLMENT_CONFIG[t]["tuition_per_term"] for t in enrollment_type]
    )

    lms_hours_weekly = np.clip(
        rng.normal(loc=true_engagement * 15, scale=3, size=n_students),
        0,
        None,
    ).round(1)

    assignments_submitted = np.clip(
        rng.normal(loc=true_engagement * 20, scale=3, size=n_students),
        0,
        20,
    ).round().astype(int)

    advising_sessions_attended = rng.poisson(
        lam=true_engagement * 2,
        size=n_students,
    )

    tutoring_hours_used = np.clip(
        rng.normal(loc=true_engagement * 20, scale=5, size=n_students),
        0,
        None,
    ).round(1)

    days_since_last_login = np.clip(
        rng.normal(loc=(1 - true_engagement) * 45, scale=8, size=n_students),
        0,
        None,
    ).round().astype(int)

    dropout_logit = (
        -3.6
        + 5.0 * (1 - true_engagement)
        + rng.normal(loc=0, scale=0.8, size=n_students)
    )

    dropout_probability = 1 / (1 + np.exp(-dropout_logit))

    dropped_out = (
        rng.uniform(size=n_students) < dropout_probability
    ).astype(int)

    signup_offset_days = rng.integers(
        low=1,
        high=SIGNUP_WINDOW_DAYS + 1,
        size=n_students,
    )

    enrollment_date = AS_OF_DATE - pd.to_timedelta(
        signup_offset_days, unit="D"
    )

    days_observed = signup_offset_days

    tenure_alpha = 1 + 3 * true_engagement
    tenure_beta = 1 + 3 * (1 - true_engagement)

    tenure_fraction = rng.beta(tenure_alpha, tenure_beta, size=n_students)

    tenure_days = np.round(tenure_fraction * days_observed).astype(int)

    withdrawal_date = enrollment_date + pd.to_timedelta(
        tenure_days, unit="D"
    )

    withdrawal_date = pd.Series(withdrawal_date).where(
        dropped_out == 1, pd.NaT
    )

    return pd.DataFrame(
        {
            "student_id": np.arange(2001, 2001 + n_students),
            "enrollment_type": enrollment_type,
            "tuition_per_term": tuition_per_term,
            "lms_hours_weekly": lms_hours_weekly,
            "assignments_submitted": assignments_submitted,
            "advising_sessions_attended": advising_sessions_attended,
            "tutoring_hours_used": tutoring_hours_used,
            "days_since_last_login": days_since_last_login,
            "enrollment_date": enrollment_date.strftime("%Y-%m-%d"),
            "withdrawal_date": withdrawal_date.dt.strftime("%Y-%m-%d"),
            "dropped_out": dropped_out,
        }
    )


def main() -> None:

    dataframe = generate_education_students()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(dataframe):,} rows to {OUTPUT_PATH}")

    print(f"Dropout rate: {dataframe['dropped_out'].mean():.1%}")


if __name__ == "__main__":

    main()
