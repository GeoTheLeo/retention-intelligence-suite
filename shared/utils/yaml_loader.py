from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUSINESS_DIR = PROJECT_ROOT / "business"


@lru_cache(maxsize=None)
def load_business_problem(module_id: str) -> dict[str, Any]:
    """
    Loads a module's business_problem.yaml manifest - the executive
    framing, KPIs, and AI advisor persona/example questions that drive
    real dashboard copy, instead of hardcoded generic text.
    """

    path = BUSINESS_DIR / module_id / "business_problem.yaml"

    with path.open("r", encoding="utf-8") as file:

        return yaml.safe_load(file) or {}
