from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import shap
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@dataclass(slots=True)
class ChurnModelResult:
    auc: float
    precision: float
    recall: float
    feature_importance: pd.Series


class ChurnPredictor:
    """
    Industry-agnostic churn prediction model.

    Works on any numeric feature matrix and binary churn label. Each
    industry module is responsible for its own feature engineering —
    this class only knows how to fit a model, score rows, and explain
    individual predictions.
    """

    def __init__(self, random_state: int = 42) -> None:

        self.random_state = random_state

        self.model = XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
        )

        self._feature_names: list[str] | None = None

        self._explainer: shap.TreeExplainer | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> ChurnModelResult:

        self._feature_names = list(features.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.25,
            random_state=self.random_state,
            stratify=target,
        )

        self.model.fit(X_train, y_train)

        self._explainer = shap.TreeExplainer(self.model)

        predicted_probability = self.model.predict_proba(X_test)[:, 1]

        predicted_label = self.model.predict(X_test)

        importance = pd.Series(
            self.model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

        return ChurnModelResult(
            auc=round(roc_auc_score(y_test, predicted_probability), 3),
            precision=round(precision_score(y_test, predicted_label), 3),
            recall=round(recall_score(y_test, predicted_label), 3),
            feature_importance=importance,
        )

    def predict_risk_score(self, features: pd.DataFrame) -> pd.Series:
        """
        Returns a continuous churn probability (0-100) per row, replacing
        fixed-threshold risk buckets with an actual model score.
        """

        if self._feature_names is None:
            raise RuntimeError("Call fit() before predict_risk_score().")

        probability = self.model.predict_proba(
            features[self._feature_names]
        )[:, 1]

        return pd.Series(
            (probability * 100).round(1),
            index=features.index,
            name="churn_risk_score",
        )

    def explain(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Returns per-row SHAP values — how much each feature pushed that
        specific customer's risk score up or down. This is what turns
        "risk score is 82" into "risk score is 82 mainly because listening
        hours dropped and no concert was attended in the last 60 days."
        """

        if self._explainer is None:
            raise RuntimeError("Call fit() before explain().")

        shap_values = self._explainer.shap_values(
            features[self._feature_names]
        )

        return pd.DataFrame(
            shap_values,
            columns=self._feature_names,
            index=features.index,
        )
