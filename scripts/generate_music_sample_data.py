from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

N_USERS = 2000

AS_OF_DATE = pd.Timestamp("2026-09-01")

SIGNUP_WINDOW_DAYS = 365

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample"
    / "music_users.csv"
)

TIER_CONFIG = {
    "Free": {"probability": 0.40, "monthly_revenue": 0.00},
    "Student": {"probability": 0.20, "monthly_revenue": 9.99},
    "Premium": {"probability": 0.40, "monthly_revenue": 18.99},
}


def generate_music_users(
    n_users: int = N_USERS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic music-fan dataset for churn modeling.

    Every observable feature is a noisy reading of a hidden "true
    engagement" value, and churn is likewise derived from true engagement
    plus its own independent noise. This mirrors how real behavioral data
    works — observable signals are imperfect proxies for the underlying
    state — so the model has to genuinely earn its accuracy rather than
    fit a deterministic formula.
    """

    rng = np.random.default_rng(seed)

    true_engagement = rng.beta(a=2.0, b=2.0, size=n_users)

    tiers = list(TIER_CONFIG.keys())

    tier_probabilities = [
        TIER_CONFIG[tier]["probability"] for tier in tiers
    ]

    subscription_tier = rng.choice(
        tiers,
        size=n_users,
        p=tier_probabilities,
    )

    monthly_revenue = np.array(
        [TIER_CONFIG[tier]["monthly_revenue"] for tier in subscription_tier]
    )

    listening_hours = np.clip(
        rng.normal(loc=true_engagement * 60, scale=10, size=n_users),
        0,
        None,
    ).round(1)

    playlist_saves = np.clip(
        rng.normal(loc=true_engagement * 40, scale=7, size=n_users),
        0,
        None,
    ).round().astype(int)

    concerts_attended = rng.poisson(
        lam=true_engagement * 3,
        size=n_users,
    )

    merchandise_spend = np.clip(
        rng.normal(loc=true_engagement * 300, scale=70, size=n_users),
        0,
        None,
    ).round(2)

    days_since_last_activity = np.clip(
        rng.normal(loc=(1 - true_engagement) * 70, scale=12, size=n_users),
        0,
        None,
    ).round().astype(int)

    churn_logit = (
        -4.0
        + 5.0 * (1 - true_engagement)
        + rng.normal(loc=0, scale=0.8, size=n_users)
    )

    churn_probability = 1 / (1 + np.exp(-churn_logit))

    churned = (rng.uniform(size=n_users) < churn_probability).astype(int)

    signup_offset_days = rng.integers(
        low=1,
        high=SIGNUP_WINDOW_DAYS + 1,
        size=n_users,
    )

    signup_date = AS_OF_DATE - pd.to_timedelta(signup_offset_days, unit="D")

    days_observed = signup_offset_days

    # Higher true_engagement -> churns later in their observed lifetime
    # (or not at all). Lower engagement -> churns earlier. This mirrors
    # real subscription behavior: disengaged users tend to leave soon
    # after signing up, not years later.
    tenure_alpha = 1 + 3 * true_engagement
    tenure_beta = 1 + 3 * (1 - true_engagement)

    tenure_fraction = rng.beta(tenure_alpha, tenure_beta, size=n_users)

    tenure_days = np.round(tenure_fraction * days_observed).astype(int)

    churn_date = signup_date + pd.to_timedelta(tenure_days, unit="D")

    churn_date = pd.Series(churn_date).where(churned == 1, pd.NaT)

    return pd.DataFrame(
        {
            "user_id": np.arange(1001, 1001 + n_users),
            "subscription_tier": subscription_tier,
            "monthly_revenue": monthly_revenue,
            "listening_hours": listening_hours,
            "playlist_saves": playlist_saves,
            "concerts_attended": concerts_attended,
            "merchandise_spend": merchandise_spend,
            "days_since_last_activity": days_since_last_activity,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "churn_date": churn_date.dt.strftime("%Y-%m-%d"),
            "churned": churned,
        }
    )


def main() -> None:

    dataframe = generate_music_users()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(dataframe):,} rows to {OUTPUT_PATH}")

    print(f"Churn rate: {dataframe['churned'].mean():.1%}")


if __name__ == "__main__":

    main()
