from __future__ import annotations

from shared.models.churn_predictor import ChurnPredictor
from shared.models.segmentation_engine import SegmentationEngine
from shared.utils.industry_registry import INDUSTRY_REGISTRY

_CHURN_WORDS = ("why", "leav", "churn", "disengag", "losing", "lose")


def _prettify(column: str) -> str:

    return column.replace("_", " ").title()


class AdvisorService:
    """
    Answers free-text executive questions using the same already-fitted
    churn (XGBoost + SHAP) and segmentation (KMeans) models the rest of
    the dashboard uses. No external LLM call - every answer is grounded
    in real, computed model output for the selected industry.
    """

    def answer(self, industry: str, question: str) -> str:

        config = INDUSTRY_REGISTRY[industry]

        dataframe = config.loader()

        features = dataframe[config.feature_columns]

        lowered = question.lower()

        matched_segment = next(
            (
                name
                for name in config.segment_names
                if name.lower() in lowered
            ),
            None,
        )

        if matched_segment is not None:

            return self._segment_answer(
                industry,
                config,
                dataframe,
                features,
                matched_segment,
            )

        if any(word in lowered for word in _CHURN_WORDS):

            return self._driver_answer(
                industry,
                config,
                dataframe,
                features,
            )

        return self._overview_answer(industry, config, dataframe)

    def _driver_answer(
        self,
        industry,
        config,
        dataframe,
        features,
    ) -> str:

        predictor = ChurnPredictor()

        result = predictor.fit(
            features,
            dataframe[config.target_column],
        )

        top = result.feature_importance.head(3)

        drivers = ", ".join(
            f"**{_prettify(name)}** ({share * 100:.0f}% of model signal)"
            for name, share in top.items()
        )

        return (
            f"Based on the churn model (AUC {result.auc:.2f}), the "
            f"strongest drivers of {config.churn_label.lower()} in "
            f"{industry} are: {drivers}. Prioritize outreach triggers "
            "tied to these signals specifically, rather than broad, "
            "untargeted retention campaigns."
        )

    def _segment_answer(
        self,
        industry,
        config,
        dataframe,
        features,
        segment_name,
    ) -> str:

        engine = SegmentationEngine(n_clusters=4)

        labels, result = engine.fit_predict(features)

        name_map = SegmentationEngine.name_clusters_by_rank(
            result.cluster_profile,
            rank_by=config.engagement_column,
            names=config.segment_names,
        )

        named_labels = labels.map(name_map)

        mask = named_labels == segment_name

        population_engagement = round(
            dataframe[config.engagement_column].mean(), 1
        )

        share = round(mask.mean() * 100, 1)

        segment_churn = round(
            dataframe.loc[mask, config.target_column].mean() * 100, 1
        )

        segment_engagement = round(
            dataframe.loc[mask, config.engagement_column].mean(), 1
        )

        comparison = (
            "above" if segment_engagement > population_engagement else "below"
        )

        return (
            f"The **{segment_name}** segment makes up **{share}%** of the "
            f"{industry} population, with a {config.churn_label.lower()} "
            f"rate of **{segment_churn}%** and average engagement of "
            f"**{segment_engagement}** — {comparison} the population "
            f"average of {population_engagement}."
        )

    def _overview_answer(
        self,
        industry,
        config,
        dataframe,
    ) -> str:

        churn_rate = round(
            dataframe[config.target_column].mean() * 100, 1
        )

        engagement = round(
            dataframe[config.engagement_column].mean(), 1
        )

        example_segment = config.segment_names[0]

        return (
            f"{industry} currently has a {config.churn_label.lower()} "
            f"rate of **{churn_rate}%** and average engagement of "
            f"**{engagement}**. Ask about a specific segment (e.g. "
            f'"{example_segment}") or include the word "why" to see the '
            "model's top churn drivers."
        )
