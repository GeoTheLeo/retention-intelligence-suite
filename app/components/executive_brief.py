from __future__ import annotations

import streamlit as st

from backend.services.metrics_service import MetricsService
from shared.utils.industry_registry import INDUSTRY_REGISTRY
from shared.utils.yaml_loader import load_business_problem


def render(industry: str) -> None:

    config = INDUSTRY_REGISTRY[industry]

    metrics = MetricsService().get_metrics(industry)

    manifest = load_business_problem(config.module_id)

    business_problem = manifest["business_context"]["business_problem"].strip()

    st.subheader(f"{industry} Executive Brief")

    st.info(
        f"""
### Business Objective

{business_problem}

### Executive Summary

Current {config.churn_label.lower()} rate stands at
**{metrics.churn_rate:.1f}%**, with an overall retention rate of
**{metrics.retention_rate:.1f}%**. Average engagement score across the
population is **{metrics.engagement_score:.1f}**, and estimated
customer lifetime value is **€{metrics.customer_lifetime_value:,.0f}**
per customer at current retention levels.

### Business Risk

Every point of churn rate compounds directly into lost lifetime value
— at the current churn rate, expected customer tenure is roughly
**{100 / max(metrics.churn_rate, 1):.1f}** periods before disengagement.

### Executive Priority

Use the segment breakdown below to focus retention spend on the
highest-risk segment first, rather than spreading effort evenly across
the full population.
"""
    )
