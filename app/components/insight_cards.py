from __future__ import annotations

import streamlit as st

from backend.insights.insight_engine import InsightEngine
from backend.services.metrics_service import MetricsService


def render(industry: str) -> None:

    metrics = MetricsService().get_metrics(industry)

    insights = InsightEngine().generate(metrics)

    st.subheader("Executive Findings")

    for insight in insights:

        if insight.priority == "High":

            st.error(
                f"""
### {insight.title}

{insight.summary}

**Recommendation**

{insight.recommendation}
"""
            )

        elif insight.priority == "Medium":

            st.warning(
                f"""
### {insight.title}

{insight.summary}

**Recommendation**

{insight.recommendation}
"""
            )

        else:

            st.success(
                f"""
### {insight.title}

{insight.summary}

**Recommendation**

{insight.recommendation}
"""
            )