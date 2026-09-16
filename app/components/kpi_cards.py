from __future__ import annotations

import streamlit as st

from backend.services.metrics_service import MetricsService
from shared.utils.industry_registry import INDUSTRY_REGISTRY


def render(industry: str) -> None:

    metrics = MetricsService().get_metrics(industry)

    churn_label = INDUSTRY_REGISTRY[industry].churn_label

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Retention Rate",
        f"{metrics.retention_rate:.1f}%",
    )

    c2.metric(
        churn_label,
        f"{metrics.churn_rate:.1f}%",
    )

    c3.metric(
        "Customer Lifetime Value (CLV)",
        f"€{metrics.customer_lifetime_value:,.0f}",
    )

    c4.metric(
        "Average Revenue Per User (ARPU)",
        f"€{metrics.average_revenue_per_user:.2f}",
    )

    c5.metric(
        "Engagement Score",
        f"{metrics.engagement_score:.1f}",
    )