from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from shared.models.segmentation_engine import SegmentationEngine
from shared.utils.industry_registry import INDUSTRY_REGISTRY


def render(industry: str) -> None:

    config = INDUSTRY_REGISTRY[industry]

    dataframe = config.loader()

    features = dataframe[config.feature_columns]

    engine = SegmentationEngine(n_clusters=4)

    labels, result = engine.fit_predict(features)

    name_map = SegmentationEngine.name_clusters_by_rank(
        result.cluster_profile,
        rank_by=config.engagement_column,
        names=config.segment_names,
    )

    named_labels = labels.map(name_map)

    counts = (
        named_labels.value_counts()
        .reindex(config.segment_names)
        .reset_index()
    )

    counts.columns = ["Segment", "Customers"]

    fig = px.bar(
        counts,
        x="Segment",
        y="Customers",
        title=f"{industry}: Segments (K=4, silhouette={result.silhouette})",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )
