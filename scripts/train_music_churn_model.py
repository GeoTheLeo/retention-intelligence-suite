from __future__ import annotations

from analytics.music.feature_engineering import MusicFeatureEngineering
from shared.models.churn_predictor import ChurnPredictor

FEATURE_COLUMNS = [
    "monthly_revenue",
    "listening_hours",
    "playlist_saves",
    "concerts_attended",
    "merchandise_spend",
    "days_since_last_activity",
    "loyalty_score",
    "engagement_score",
]


def main() -> None:

    engineered = MusicFeatureEngineering()

    dataframe = engineered.build_features()

    features = dataframe[FEATURE_COLUMNS]

    target = dataframe["churned"]

    predictor = ChurnPredictor()

    result = predictor.fit(features, target)

    print(f"AUC:       {result.auc}")
    print(f"Precision: {result.precision}")
    print(f"Recall:    {result.recall}")
    print()
    print("Feature importance:")
    print(result.feature_importance)

    print()
    print("Sample risk scores (first 5 users):")

    risk_scores = predictor.predict_risk_score(features.head())

    print(risk_scores)

    print()
    print("Why user 0 got their risk score (SHAP values):")

    explanation = predictor.explain(features.head(1))

    print(explanation.T)


if __name__ == "__main__":

    main()
