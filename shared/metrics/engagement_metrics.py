from __future__ import annotations

import pandas as pd


class EngagementMetrics:
    """
    Shared engagement metrics.
    """

    @staticmethod
    def average_session_duration(
        dataframe: pd.DataFrame,
        duration_column: str = "listening_duration",
    ) -> float:

        return round(
            dataframe[duration_column].mean(),
            2,
        )

    @staticmethod
    def skip_rate(
        dataframe: pd.DataFrame,
        skipped_column: str = "skipped",
    ) -> float:

        return round(
            dataframe[skipped_column].mean() * 100,
            2,
        )

    @staticmethod
    def completion_rate(
        dataframe: pd.DataFrame,
        completed_column: str = "completed",
    ) -> float:

        return round(
            dataframe[completed_column].mean() * 100,
            2,
        )

    @staticmethod
    def engagement_score(
        listening_frequency: float,
        playlist_saves: float,
        session_duration: float,
    ) -> float:
        """
        Composite engagement score (0–100).
        """

        score = (
            listening_frequency * 0.45
            + playlist_saves * 0.25
            + session_duration * 0.30
        )

        return round(
            min(score, 100),
            2,
        )