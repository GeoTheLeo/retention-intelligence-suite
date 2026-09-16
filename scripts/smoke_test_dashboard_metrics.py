from __future__ import annotations

from backend.services.metrics_service import MetricsService
from shared.utils.industry_registry import INDUSTRY_REGISTRY


def main() -> None:

    service = MetricsService()

    for industry in INDUSTRY_REGISTRY:

        metrics = service.get_metrics(industry)

        print(f"--- {industry} ---")
        print(f"Retention rate: {metrics.retention_rate:.1f}%")
        print(f"Churn rate:     {metrics.churn_rate:.1f}%")
        print(f"ARPU:           {metrics.average_revenue_per_user:.2f}")
        print(f"CLV:            {metrics.customer_lifetime_value:,.0f}")
        print(f"Engagement:     {metrics.engagement_score:.1f}")
        print()


if __name__ == "__main__":

    main()
