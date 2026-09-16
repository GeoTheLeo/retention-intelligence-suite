from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from shared.models.cohort_analyzer import CohortAnalyzer
from shared.utils.industry_registry import INDUSTRY_REGISTRY


def render(industry: str) -> None:

    config = INDUSTRY_REGISTRY[industry]

    dataframe = config.loader()

    analyzer = CohortAnalyzer(period="M")

    result = analyzer.build_retention_table(
        dataframe,
        signup_col=config.signup_column,
        churn_col=config.churn_column,
    )

    # Blend all cohorts into one overall retention curve (mean retention
    # rate per month-offset, across every cohort that has data for that
    # offset) - a single trend line an executive can read at a glance,
    # rather than a full per-cohort table.
    overall_trend = result.retention_table.mean(axis=0, skipna=True)

    trend_df = pd.DataFrame(
        {
            "Months Since Signup": overall_trend.index,
            "Retention Rate": overall_trend.values,
        }
    )

    fig = px.line(
        trend_df,
        x="Months Since Signup",
        y="Retention Rate",
        markers=True,
        title=f"{industry}: Retention Trend (Blended Across All Cohorts)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )
