from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard.executive_dashboard import render


st.set_page_config(
    page_title="Retention Intelligence Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

render()