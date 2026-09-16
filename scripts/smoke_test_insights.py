from __future__ import annotations

from backend.insights.insight_engine import InsightEngine
from backend.services.metrics_service import MetricsService
from shared.utils.industry_registry import INDUSTRY_REGISTRY


def main() -> None:

    service = MetricsService()

    engine = InsightEngine()

    for industry in INDUSTRY_REGISTRY:

        metrics = service.get_metrics(industry)

        insights = engine.generate(metrics)

        print(f"--- {industry} (CLV={metrics.customer_lifetime_value:,.0f}, "
              f"engagement={metrics.engagement_score:.1f}, "
              f"retention={metrics.retention_rate:.1f}%) ---")

        if not insights:
            print("  (no insights triggered)")

        for insight in insights:
            print(f"  [{insight.priority}] {insight.title}")

        print()


if __name__ == "__main__":

    main()
