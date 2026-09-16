from __future__ import annotations

import streamlit as st


def render() -> None:

    st.subheader("AI Retention Advisor")

    question = st.text_input(

        "Ask an executive business question",

        placeholder="Why are Premium fans leaving?",

    )

    if st.button("Ask Advisor"):

        if question:

            st.info(
                "AI Advisor implementation begins in Sprint 10."
            )

        else:

            st.warning(
                "Please enter a business question."
            )