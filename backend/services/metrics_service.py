from __future__ import annotations

from dataclasses import dataclass

from shared.metrics.retention_metrics import RetentionMetrics
from shared.utils.industry_registry import INDUSTRY_REGISTRY


@dataclass(slots=True)
class DashboardMetrics:
    retention_rate: float
    churn_rate: float
    customer_lifetime_value: float
    average_revenue_per_user: float
    engagement_score: float


class MetricsService:
    """
    Provides real, industry-specific executive metrics for the
    dashboard, computed from each industry's actual generated dataset
    via the shared framework - not placeholder values.
    """

    def get_metrics(self, industry: str) -> DashboardMetrics:

        config = INDUSTRY_REGISTRY[industry]

        dataframe = config.loader()

        total = len(dataframe)

        churned = int(dataframe[config.target_column].sum())

        active = total - churned

        churn_rate = RetentionMetrics.churn_rate(
            customers_lost=churned,
            active_start=total,
        )

        retention_rate = RetentionMetrics.customer_retention_rate(
            active_start=total,
            active_end=active,
            new_customers=0,
        )

        arpu = RetentionMetrics.average_revenue_per_user(
            revenue=dataframe[config.revenue_column].sum(),
            active_users=total,
        )

        # CLV ≈ ARPU / churn rate — a standard, defensible approximation:
        # expected customer lifetime (in periods) is the inverse of the
        # churn rate. Floor the churn fraction to avoid a divide-by-zero
        # blowup if an industry ever has near-zero churn.
        churn_fraction = max(churned / total, 0.01)

        expected_lifetime_periods = 1 / churn_fraction

        clv = RetentionMetrics.customer_lifetime_value(
            average_purchase=arpu,
            purchases_per_year=1,
            retention_years=expected_lifetime_periods,
        )

        engagement_score = round(
            dataframe[config.engagement_column].mean(), 1
        )

        return DashboardMetrics(
            retention_rate=retention_rate,
            churn_rate=churn_rate,
            customer_lifetime_value=clv,
            average_revenue_per_user=arpu,
            engagement_score=engagement_score,
        )
