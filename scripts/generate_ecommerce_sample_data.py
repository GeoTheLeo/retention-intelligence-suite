from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

N_CUSTOMERS = 2000

AS_OF_DATE = pd.Timestamp("2026-09-01")

ACQUISITION_WINDOW_DAYS = 365

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample"
    / "ecommerce_customers.csv"
)

# Real-world finding this mirrors: paid-acquired customers tend to churn
# faster than organic/referral customers, and email subscribers tend to
# be slightly more loyal than average.
CHANNEL_CONFIG = {
    "Organic": {"probability": 0.30, "churn_modifier": 0.0},
    "Paid Search": {"probability": 0.25, "churn_modifier": 0.5},
    "Social Media": {"probability": 0.25, "churn_modifier": 0.3},
    "Email": {"probability": 0.20, "churn_modifier": -0.2},
}


def generate_ecommerce_customers(
    n_customers: int = N_CUSTOMERS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic e-commerce customer dataset, built
    around RFM (Recency, Frequency, Monetary) rather than a single
    "engagement" axis. Monetary value (average order value) is
    deliberately semi-independent of loyalty/frequency - a customer can
    be a high-value infrequent shopper or a low-value frequent one,
    which is a realistic RFM distinction the other industries don't
    have a direct equivalent of.
    """

    rng = np.random.default_rng(seed)

    true_loyalty = rng.beta(a=2.0, b=2.0, size=n_customers)

    channels = list(CHANNEL_CONFIG.keys())

    channel_probabilities = [
        CHANNEL_CONFIG[c]["probability"] for c in channels
    ]

    acquisition_channel = rng.choice(
        channels,
        size=n_customers,
        p=channel_probabilities,
    )

    channel_modifier = np.array(
        [CHANNEL_CONFIG[c]["churn_modifier"] for c in acquisition_channel]
    )

    # Frequency - driven by loyalty
    total_orders = np.clip(
        rng.normal(loc=true_loyalty * 18, scale=4, size=n_customers),
        1,
        None,
    ).round().astype(int)

    # Monetary - semi-independent "spend tier", not purely loyalty-driven
    spend_tier = rng.gamma(shape=2.0, scale=25.0, size=n_customers)

    avg_order_value = np.clip(
        spend_tier + rng.normal(loc=0, scale=8, size=n_customers),
        10,
        None,
    ).round(2)

    # Recency - inverse of loyalty
    days_since_last_order = np.clip(
        rng.normal(loc=(1 - true_loyalty) * 160, scale=25, size=n_customers),
        0,
        None,
    ).round().astype(int)

    cart_abandonment_count = rng.poisson(
        lam=(1 - true_loyalty) * 5,
        size=n_customers,
    )

    email_open_rate = np.clip(
        rng.normal(loc=true_loyalty * 100, scale=15, size=n_customers),
        0,
        100,
    ).round(1)

    churn_logit = (
        -3.6
        + 4.2 * (1 - true_loyalty)
        + channel_modifier
        + rng.normal(loc=0, scale=0.7, size=n_customers)
    )

    churn_probability = 1 / (1 + np.exp(-churn_logit))

    churned = (rng.uniform(size=n_customers) < churn_probability).astype(int)

    acquisition_offset_days = rng.integers(
        low=1,
        high=ACQUISITION_WINDOW_DAYS + 1,
        size=n_customers,
    )

    first_purchase_date = AS_OF_DATE - pd.to_timedelta(
        acquisition_offset_days, unit="D"
    )

    days_observed = acquisition_offset_days

    tenure_alpha = 1 + 3 * true_loyalty
    tenure_beta = 1 + 3 * (1 - true_loyalty)

    tenure_fraction = rng.beta(tenure_alpha, tenure_beta, size=n_customers)

    tenure_days = np.round(tenure_fraction * days_observed).astype(int)

    churn_date = first_purchase_date + pd.to_timedelta(
        tenure_days, unit="D"
    )

    churn_date = pd.Series(churn_date).where(churned == 1, pd.NaT)

    total_spend = (total_orders * avg_order_value).round(2)

    return pd.DataFrame(
        {
            "customer_id": np.arange(4001, 4001 + n_customers),
            "acquisition_channel": acquisition_channel,
            "total_orders": total_orders,
            "avg_order_value": avg_order_value,
            "total_spend": total_spend,
            "days_since_last_order": days_since_last_order,
            "cart_abandonment_count": cart_abandonment_count,
            "email_open_rate": email_open_rate,
            "first_purchase_date": first_purchase_date.strftime("%Y-%m-%d"),
            "churn_date": churn_date.dt.strftime("%Y-%m-%d"),
            "churned": churned,
        }
    )


def main() -> None:

    dataframe = generate_ecommerce_customers()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(dataframe):,} rows to {OUTPUT_PATH}")

    print(f"Churn rate: {dataframe['churned'].mean():.1%}")


if __name__ == "__main__":

    main()
