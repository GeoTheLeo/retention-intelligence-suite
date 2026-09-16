from __future__ import annotations

import pandas as pd

from analytics.music.feature_engineering import MusicFeatureEngineering
from shared.models.cohort_analyzer import CohortAnalyzer


def main() -> None:

    engineered = MusicFeatureEngineering()

    dataframe = engineered.build_features()

    analyzer = CohortAnalyzer(period="M")

    result = analyzer.build_retention_table(
        dataframe,
        as_of=pd.Timestamp("2026-09-01"),
    )

    print("Cohort sizes:")
    print(result.cohort_sizes)

    print()
    print("Retention table (rows = signup cohort, columns = months since signup):")

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(result.retention_table)


if __name__ == "__main__":

    main()
