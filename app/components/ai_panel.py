from __future__ import annotations

import streamlit as st

from backend.services.advisor_service import AdvisorService


def render(industry: str) -> None:

    st.subheader("AI Retention Advisor")

    question = st.text_input(
        "Ask an executive business question",
        placeholder="Why are customers churning?",
    )

    if st.button("Ask Advisor"):

        if question:

            with st.spinner("Analyzing..."):

                answer = AdvisorService().answer(industry, question)

            st.info(answer)

        else:

            st.warning("Please enter a business question.")
