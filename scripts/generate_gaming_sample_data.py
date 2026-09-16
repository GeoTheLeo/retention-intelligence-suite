from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

N_PLAYERS = 2000

AS_OF_DATE = pd.Timestamp("2026-09-01")

FIRST_SESSION_WINDOW_DAYS = 365

GUILD_MEMBERSHIP_RATE = 0.35

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample"
    / "gaming_players.csv"
)

PLATFORM_CONFIG = {
    "Mobile": 0.50,
    "PC": 0.30,
    "Console": 0.20,
}

# A small minority of "whales" drive disproportionate revenue - a
# well-documented real pattern in gaming monetization.
MONETIZATION_CONFIG = {
    "Free-to-Play": {"probability": 0.70, "spend_scale": 0.0},
    "Occasional Spender": {"probability": 0.25, "spend_scale": 15.0},
    "Whale": {"probability": 0.05, "spend_scale": 180.0},
}


def generate_gaming_players(
    n_players: int = N_PLAYERS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic player-retention dataset.

    Guild membership is drawn independently of engagement and acts as
    a protective factor against churn - mirroring one of the most
    well-documented findings in real gaming retention research: social
    ties reduce churn independent of how much someone actually plays.
    """

    rng = np.random.default_rng(seed)

    true_engagement = rng.beta(a=2.0, b=2.0, size=n_players)

    platforms = list(PLATFORM_CONFIG.keys())

    platform = rng.choice(
        platforms,
        size=n_players,
        p=list(PLATFORM_CONFIG.values()),
    )

    tiers = list(MONETIZATION_CONFIG.keys())

    tier_probabilities = [
        MONETIZATION_CONFIG[t]["probability"] for t in tiers
    ]

    monetization_tier = rng.choice(
        tiers,
        size=n_players,
        p=tier_probabilities,
    )

    spend_scale = np.array(
        [MONETIZATION_CONFIG[t]["spend_scale"] for t in monetization_tier]
    )

    guild_member = (
        rng.uniform(size=n_players) < GUILD_MEMBERSHIP_RATE
    ).astype(int)

    sessions_per_week = np.clip(
        rng.normal(loc=true_engagement * 12, scale=2.5, size=n_players),
        0,
        None,
    ).round(1)

    avg_session_minutes = np.clip(
        rng.normal(loc=true_engagement * 45, scale=10, size=n_players),
        0,
        None,
    ).round(1)

    levels_completed = np.clip(
        rng.normal(loc=true_engagement * 80, scale=15, size=n_players),
        0,
        None,
    ).round().astype(int)

    # Purchases scale with monetization tier, with some noise, largely
    # independent of raw engagement — a whale can be a light player.
    in_game_purchases = np.clip(
        rng.gamma(shape=1.5, scale=np.maximum(spend_scale, 1e-6)),
        0,
        None,
    ).round(2)

    in_game_purchases = np.where(
        monetization_tier == "Free-to-Play", 0.0, in_game_purchases
    )

    days_since_last_session = np.clip(
        rng.normal(loc=(1 - true_engagement) * 50, scale=9, size=n_players),
        0,
        None,
    ).round().astype(int)

    churn_logit = (
        -3.7
        + 4.3 * (1 - true_engagement)
        - 0.9 * guild_member
        + rng.normal(loc=0, scale=0.7, size=n_players)
    )

    churn_probability = 1 / (1 + np.exp(-churn_logit))

    churned = (rng.uniform(size=n_players) < churn_probability).astype(int)

    first_session_offset_days = rng.integers(
        low=1,
        high=FIRST_SESSION_WINDOW_DAYS + 1,
        size=n_players,
    )

    first_session_date = AS_OF_DATE - pd.to_timedelta(
        first_session_offset_days, unit="D"
    )

    days_observed = first_session_offset_days

    # Gaming churn is front-loaded: players who are going to quit
    # usually quit within days or weeks, not months, unlike subscription
    # cancellations which spread more evenly across the observed
    # window. Modeled as an absolute-day exponential instead of a
    # fraction of tenure, capped at each player's actual observed window.
    tenure_scale_days = 6 + 20 * (1 - true_engagement)

    raw_tenure_days = rng.exponential(scale=tenure_scale_days)

    tenure_days = np.minimum(
        np.round(raw_tenure_days).astype(int),
        days_observed,
    )

    churn_date = first_session_date + pd.to_timedelta(
        tenure_days, unit="D"
    )

    churn_date = pd.Series(churn_date).where(churned == 1, pd.NaT)

    return pd.DataFrame(
        {
            "player_id": np.arange(5001, 5001 + n_players),
            "platform": platform,
            "monetization_tier": monetization_tier,
            "sessions_per_week": sessions_per_week,
            "avg_session_minutes": avg_session_minutes,
            "levels_completed": levels_completed,
            "in_game_purchases": in_game_purchases,
            "guild_member": guild_member,
            "days_since_last_session": days_since_last_session,
            "first_session_date": first_session_date.strftime("%Y-%m-%d"),
            "churn_date": churn_date.dt.strftime("%Y-%m-%d"),
            "churned": churned,
        }
    )


def main() -> None:

    dataframe = generate_gaming_players()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(dataframe):,} rows to {OUTPUT_PATH}")

    print(f"Churn rate: {dataframe['churned'].mean():.1%}")

    print(
        "Churn rate, guild members: "
        f"{dataframe.loc[dataframe['guild_member'] == 1, 'churned'].mean():.1%}"
    )

    print(
        "Churn rate, non-guild members: "
        f"{dataframe.loc[dataframe['guild_member'] == 0, 'churned'].mean():.1%}"
    )


if __name__ == "__main__":

    main()
