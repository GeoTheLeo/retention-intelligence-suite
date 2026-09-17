from __future__ import annotations

import streamlit as st

from app.components import (
    ai_panel,
    executive_brief,
    header,
    insight_cards,
    kpi_cards,
    recommendation_panel,
    segment_chart,
    sidebar,
    trend_chart,
)


def render() -> None:

    header.render()

    industry = sidebar.render()

    st.header(f"{industry} Executive Dashboard")

    executive_brief.render(industry)

    st.divider()

    kpi_cards.render(industry)

    st.divider()

    insight_cards.render(industry)

    st.divider()

    left, right = st.columns(2)

    with left:
        trend_chart.render(industry)

    with right:
        segment_chart.render(industry)

    st.divider()

    recommendation_panel.render(industry)

    st.divider()

    ai_panel.render(industry)