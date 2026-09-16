from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.metrics.engagement_metrics import EngagementMetrics
from shared.metrics.retention_metrics import RetentionMetrics
from shared.utils.data_loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "business"
    / "music"
    / "dataset_manifest.yaml"
)


def main() -> None:
    """
    Demonstration of the shared metrics engine.

    A real dataset will replace this sample DataFrame in the next sprint.
    """

    loader = DataLoader(MANIFEST)
    loader.show_available_datasets()

    sample = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "listening_duration": [180, 240, 210, 120, 300],
            "completed": [1, 1, 0, 1, 1],
            "skipped": [0, 0, 1, 0, 0],
        }
    )

    print("\n==============================")
    print("Music Retention Metrics Demo")
    print("==============================")

    print(
        f"MAU: {RetentionMetrics.monthly_active_users(sample)}"
    )

    print(
        f"Average Session: "
        f"{EngagementMetrics.average_session_duration(sample)} sec"
    )

    print(
        f"Completion Rate: "
        f"{EngagementMetrics.completion_rate(sample)}%"
    )

    print(
        f"Skip Rate: "
        f"{EngagementMetrics.skip_rate(sample)}%"
    )


if __name__ == "__main__":
    main()