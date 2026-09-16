from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class CohortRetentionResult:
    retention_table: pd.DataFrame
    cohort_sizes: pd.Series


class CohortAnalyzer:
    """
    Industry-agnostic cohort retention analysis.

    Groups records by signup period and tracks what fraction of each
    cohort remains active at each subsequent period offset — the
    classic cohort retention triangle used across every industry this
    suite covers.
    """

    def __init__(self, period: str = "M") -> None:

        self.period = period

    def build_retention_table(
        self,
        dataframe: pd.DataFrame,
        signup_col: str = "signup_date",
        churn_col: str = "churn_date",
        as_of: pd.Timestamp | None = None,
    ) -> CohortRetentionResult:

        df = dataframe.copy()

        df[signup_col] = pd.to_datetime(df[signup_col])

        df[churn_col] = pd.to_datetime(df[churn_col])

        as_of = as_of or pd.Timestamp.today().normalize()

        df["cohort"] = df[signup_col].dt.to_period(self.period)

        cohort_sizes = df.groupby("cohort").size().rename("cohort_size")

        as_of_period = as_of.to_period(self.period)

        rows: list[dict[str, object]] = []

        for cohort, group in df.groupby("cohort"):

            cohort_size = len(group)

            max_offset = (as_of_period - cohort).n

            for offset in range(0, max_offset + 1):

                period_end = (cohort + offset).end_time

                if period_end > as_of:
                    break

                still_active = (
                    group[churn_col].isna()
                    | (group[churn_col] > period_end)
                ).sum()

                rows.append(
                    {
                        "cohort": str(cohort),
                        "month_offset": offset,
                        "retention_rate": round(
                            (still_active / cohort_size) * 100, 1
                        ),
                    }
                )

        long_table = pd.DataFrame(rows)

        retention_table = long_table.pivot(
            index="cohort",
            columns="month_offset",
            values="retention_rate",
        )

        return CohortRetentionResult(
            retention_table=retention_table,
            cohort_sizes=cohort_sizes,
        )
