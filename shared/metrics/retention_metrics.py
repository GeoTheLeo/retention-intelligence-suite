from __future__ import annotations

import pandas as pd


class RetentionMetrics:
    """
    Reusable retention metrics shared across all industries.
    """

    @staticmethod
    def customer_retention_rate(
        active_start: int,
        active_end: int,
        new_customers: int,
    ) -> float:
        """
        ((Customers at End - New Customers) / Customers at Start) × 100
        """

        if active_start == 0:
            return 0.0

        return round(
            ((active_end - new_customers) / active_start) * 100,
            2,
        )

    @staticmethod
    def churn_rate(
        customers_lost: int,
        active_start: int,
    ) -> float:

        if active_start == 0:
            return 0.0

        return round(
            (customers_lost / active_start) * 100,
            2,
        )

    @staticmethod
    def average_revenue_per_user(
        revenue: float,
        active_users: int,
    ) -> float:

        if active_users == 0:
            return 0.0

        return round(
            revenue / active_users,
            2,
        )

    @staticmethod
    def customer_lifetime_value(
        average_purchase: float,
        purchases_per_year: float,
        retention_years: float,
    ) -> float:

        return round(
            average_purchase
            * purchases_per_year
            * retention_years,
            2,
        )

    @staticmethod
    def monthly_active_users(
        dataframe: pd.DataFrame,
        user_column: str = "user_id",
    ) -> int:

        return dataframe[user_column].nunique()

    @staticmethod
    def weekly_active_users(
        dataframe: pd.DataFrame,
        user_column: str = "user_id",
    ) -> int:

        return dataframe[user_column].nunique()