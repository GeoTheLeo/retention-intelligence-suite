from __future__ import annotations

import streamlit as st

from shared.models.segmentation_engine import SegmentationEngine
from shared.utils.industry_registry import INDUSTRY_REGISTRY
from shared.utils.yaml_loader import load_business_problem


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

    lowest_segment = config.segment_names[0]

    at_risk_share = round(
        (named_labels == lowest_segment).mean() * 100, 1
    )

    churn_rate = round(dataframe[config.target_column].mean() * 100, 1)

    st.subheader("Executive Recommendations")

    st.markdown(
        f"""
### Immediate Actions

- **{at_risk_share}%** of the {industry} population falls in the
  **{lowest_segment}** segment — prioritize outreach here first.
- Current {config.churn_label.lower()} rate is **{churn_rate}%** —
  use the risk score from the churn model to target the highest-risk
  individuals within that segment specifically, not the whole group.
- Re-run segmentation on a rolling basis; segment membership shifts as
  behavior changes, and a static one-time label goes stale.

---

### Strategic Actions

- Track whether interventions targeted at the {lowest_segment} segment
  actually move people up a tier over time, not just whether the
  overall rate improves.
- Feed real intervention outcomes back into the model as they
  accumulate, so risk scoring reflects what's been tried, not just
  historical behavior.
- Extend the same shared framework to additional segments or products
  within {industry} before considering a new industry entirely — reuse
  before expansion.
"""
    )

    manifest = load_business_problem(config.module_id)

    playbook = manifest["recommendations"]

    st.markdown("---")

    st.markdown(f"### {industry} Business Playbook")

    st.caption(
        "Sourced from this module's business plugin manifest, not "
        "recomputed from data — the domain-specific playbook for this "
        "industry, distinct from the model-driven actions above."
    )

    immediate_items = "\n".join(
        f"- {item}" for item in playbook["immediate"]
    )

    strategic_items = "\n".join(
        f"- {item}" for item in playbook["strategic"]
    )

    st.markdown(
        f"""
**Immediate**

{immediate_items}

**Strategic**

{strategic_items}
"""
    )
