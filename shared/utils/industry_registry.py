from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from analytics.ecommerce.feature_engineering import EcommerceFeatureEngineering
from analytics.education.feature_engineering import EducationFeatureEngineering
from analytics.gaming.feature_engineering import GamingFeatureEngineering
from analytics.healthcare.feature_engineering import HealthcareFeatureEngineering
from analytics.music.feature_engineering import MusicFeatureEngineering


@dataclass(slots=True)
class IndustryConfig:
    """
    Everything the app layer needs to work with one industry, without
    any industry-specific logic living in the app components themselves.
    """

    loader: Callable[[], pd.DataFrame]
    feature_columns: list[str]
    target_column: str
    revenue_column: str
    engagement_column: str
    signup_column: str
    churn_column: str
    segment_names: list[str]
    churn_label: str


def _load(engineering_cls: type) -> pd.DataFrame:
    return engineering_cls().build_features()


INDUSTRY_REGISTRY: dict[str, IndustryConfig] = {
    "Music Entertainment": IndustryConfig(
        loader=lambda: _load(MusicFeatureEngineering),
        feature_columns=[
            "monthly_revenue",
            "listening_hours",
            "playlist_saves",
            "concerts_attended",
            "merchandise_spend",
            "days_since_last_activity",
            "loyalty_score",
            "engagement_score",
        ],
        target_column="churned",
        revenue_column="monthly_revenue",
        engagement_column="engagement_score",
        signup_column="signup_date",
        churn_column="churn_date",
        segment_names=["Casual", "Regular", "Loyal", "Champion"],
        churn_label="Fan Churn",
    ),
    "Education": IndustryConfig(
        loader=lambda: _load(EducationFeatureEngineering),
        feature_columns=[
            "tuition_per_term",
            "lms_hours_weekly",
            "assignments_submitted",
            "advising_sessions_attended",
            "tutoring_hours_used",
            "days_since_last_login",
            "academic_engagement_score",
            "support_utilization",
        ],
        target_column="dropped_out",
        revenue_column="tuition_per_term",
        engagement_column="academic_engagement_score",
        signup_column="enrollment_date",
        churn_column="withdrawal_date",
        segment_names=["At Risk", "Building Momentum", "On Track", "Thriving"],
        churn_label="Student Dropout",
    ),
    "Healthcare": IndustryConfig(
        loader=lambda: _load(HealthcareFeatureEngineering),
        feature_columns=[
            "monthly_care_cost",
            "medication_adherence_rate",
            "followup_appointments_attended",
            "care_plan_checkins",
            "remote_monitoring_hours",
            "days_since_last_contact",
            "care_engagement_score",
            "clinical_complexity_score",
        ],
        target_column="readmitted",
        revenue_column="monthly_care_cost",
        engagement_column="care_engagement_score",
        signup_column="discharge_date",
        churn_column="readmission_date",
        segment_names=["Low Engagement", "Moderate", "Well-Managed", "Optimal"],
        churn_label="Patient Readmission",
    ),
    "E-Commerce": IndustryConfig(
        loader=lambda: _load(EcommerceFeatureEngineering),
        feature_columns=[
            "total_orders",
            "avg_order_value",
            "total_spend",
            "days_since_last_order",
            "cart_abandonment_count",
            "email_open_rate",
            "engagement_score",
        ],
        target_column="churned",
        revenue_column="avg_order_value",
        engagement_column="engagement_score",
        signup_column="first_purchase_date",
        churn_column="churn_date",
        segment_names=["At Risk", "Needs Attention", "Loyal", "Champion"],
        churn_label="Customer Churn",
    ),
    "Gaming": IndustryConfig(
        loader=lambda: _load(GamingFeatureEngineering),
        feature_columns=[
            "sessions_per_week",
            "avg_session_minutes",
            "levels_completed",
            "in_game_purchases",
            "guild_member",
            "days_since_last_session",
            "engagement_score",
        ],
        target_column="churned",
        revenue_column="in_game_purchases",
        engagement_column="engagement_score",
        signup_column="first_session_date",
        churn_column="churn_date",
        segment_names=["Casual", "Regular", "Dedicated", "Hardcore"],
        churn_label="Player Churn",
    ),
}
