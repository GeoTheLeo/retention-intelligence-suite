from __future__ import annotations

from dataclasses import dataclass

from backend.services.metrics_service import DashboardMetrics


@dataclass(slots=True)
class ExecutiveInsight:
    title: str
    priority: str
    summary: str
    recommendation: str


class InsightEngine:
    """
    Turns raw metrics into executive-readable insights.

    Thresholds are calibrated against the actual observed range across
    all five industries (retention 68-80%, engagement 34-51 on each
    industry's own 0-100 scale) rather than arbitrary numbers - and the
    value check uses a CLV/ARPU ratio (mathematically 1/churn rate)
    instead of a raw currency amount, so it's comparable across
    industries with very different absolute price points instead of
    just detecting "which industry charges more."
    """

    RETENTION_THRESHOLD = 78.0
    ENGAGEMENT_THRESHOLD = 45.0
    VALUE_RATIO_THRESHOLD = 4.0

    def generate(
        self,
        metrics: DashboardMetrics,
    ) -> list[ExecutiveInsight]:

        insights: list[ExecutiveInsight] = []

        if metrics.retention_rate >= self.RETENTION_THRESHOLD:

            insights.append(
                ExecutiveInsight(
                    title="Retention Performance",
                    priority="Low",
                    summary=(
                        f"Retention rate of {metrics.retention_rate:.1f}% is "
                        "strong relative to the other industries in this "
                        "suite, indicating existing engagement strategies "
                        "are working."
                    ),
                    recommendation=(
                        "Continue monitoring, and document what's working "
                        "here before assuming it will transfer to a "
                        "lower-retention segment without adaptation."
                    ),
                )
            )

        if metrics.engagement_score < self.ENGAGEMENT_THRESHOLD:

            insights.append(
                ExecutiveInsight(
                    title="Engagement Decline",
                    priority="High",
                    summary=(
                        f"Engagement score of {metrics.engagement_score:.1f} "
                        "is on the low end of the observed range and is "
                        "the strongest leading indicator of future churn."
                    ),
                    recommendation=(
                        "Launch targeted retention outreach for the "
                        "lowest-engagement segment specifically, not the "
                        "population as a whole."
                    ),
                )
            )

        value_ratio = (
            metrics.customer_lifetime_value
            / max(metrics.average_revenue_per_user, 0.01)
        )

        if value_ratio >= self.VALUE_RATIO_THRESHOLD:

            insights.append(
                ExecutiveInsight(
                    title="Strong Unit Economics",
                    priority="Medium",
                    summary=(
                        f"Customer lifetime value is {value_ratio:.1f}x "
                        "average revenue per user, meaning customers stay "
                        "long enough to be worth significantly more than "
                        "a single transaction — a genuinely healthy churn "
                        "rate, not just a high headline price."
                    ),
                    recommendation=(
                        "Protect this by keeping acquisition cost per "
                        "customer well below the lifetime value, not just "
                        "below the first-period revenue."
                    ),
                )
            )

        if not insights:

            insights.append(
                ExecutiveInsight(
                    title="Stable, Unremarkable Performance",
                    priority="Low",
                    summary=(
                        f"Retention ({metrics.retention_rate:.1f}%), "
                        f"engagement ({metrics.engagement_score:.1f}), and "
                        "unit economics are all in the middle of the "
                        "observed range — nothing currently stands out as "
                        "either a strength or a risk worth escalating."
                    ),
                    recommendation=(
                        "No immediate action required. Revisit after the "
                        "next reporting cycle to check whether any metric "
                        "has moved toward one of the flagged thresholds."
                    ),
                )
            )

        return insights
