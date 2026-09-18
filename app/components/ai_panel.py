from __future__ import annotations

import streamlit as st

from backend.services.advisor_service import AdvisorService
from shared.utils.industry_registry import INDUSTRY_REGISTRY
from shared.utils.yaml_loader import load_business_problem


def render(industry: str) -> None:

    config = INDUSTRY_REGISTRY[industry]

    manifest = load_business_problem(config.module_id)

    advisor_info = manifest["ai_advisor"]

    st.subheader(f"AI Retention Advisor — {advisor_info['persona']}")

    question = st.text_input(
        "Ask an executive business question",
        placeholder=advisor_info["example_questions"][0],
    )

    if st.button("Ask Advisor"):

        if question:

            with st.spinner("Analyzing..."):

                answer = AdvisorService().answer(industry, question)

            st.info(answer)

        else:

            st.warning("Please enter a business question.")
