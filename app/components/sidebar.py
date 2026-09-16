from __future__ import annotations

import streamlit as st

from shared.utils.plugin_loader import PluginLoader


def render() -> str:

    plugins = PluginLoader().discover()

    st.sidebar.title("Industries")

    names = [plugin.name for plugin in plugins]

    selected = st.sidebar.radio(

        "Select Industry",

        names,

        label_visibility="collapsed",

    )

    st.sidebar.divider()

    st.sidebar.success("AI Advisor Ready")

    return selected