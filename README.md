# Cross-Industry Retention Intelligence Suite

One shared analytical framework — churn prediction, behavioral segmentation, and cohort analysis — applied across five industries that all face the same underlying question: why do customers, students, patients, fans, or players leave?

**Status: complete.** All five industries are built and validated against the same shared code, with zero modification between them. A Streamlit executive dashboard ties them together with real KPIs, retention trends, segments, and generated insights, switchable per industry from the sidebar.

## Architecture

- **`shared/`** — the reusable core, written once and imported by every industry: `ChurnPredictor` (XGBoost + SHAP), `SegmentationEngine` (KMeans), `CohortAnalyzer` (cohort retention curves), `RetentionMetrics` (churn rate, retention rate, ARPU, CLV).
- **`analytics/{industry}/`** — thin, industry-specific adapters: a data loader and feature engineering step per industry, wired into the shared models above. This is deliberately the only part that differs between industries.
- **`shared/utils/industry_registry.py`** — a single config mapping each industry to its feature engineering class, feature columns, and target column, so the app layer (dashboard, metrics service) never contains industry-specific logic itself.
- **`framework/`** — the business/consulting layer (business canvas, stakeholder matrix, success metrics) feeding executive-facing deliverables.
- **`app/`** — the Streamlit executive dashboard.

## The Five Industries

| Industry | Outcome Predicted | Churn/Dropout Rate | Model AUC | Segmentation Silhouette (K=4) | What Makes It Distinct |
|---|---|---|---|---|---|
| Music Entertainment | Fan churn | 24.3% | 0.68 | 0.224 | Reference implementation; loyalty from listening/concerts/merch |
| Education | Student dropout | 31.9% | 0.66 | 0.215 | Support-service usage kept separate from engagement (using tutoring ≠ being disengaged) |
| Healthcare | 30-day+ readmission | 25.2% | 0.69 | 0.184 | Two independent risk factors (care engagement *and* clinical complexity), not one |
| E-Commerce | Customer churn | 27.6% | 0.63 | 0.225 | Built on RFM (Recency/Frequency/Monetary); acquisition-channel risk as an independent modifier |
| Gaming | Player churn | 19.7% | 0.64 | 0.220 | Guild membership as an independent protective factor (13.1% churn for guild members vs. 23.2% for non-members) |

Every dataset is synthetic, generated so that observable features are noisy proxies of a hidden "true engagement" value and the outcome is a *separately* noisy function of that same hidden value — not a deterministic formula reverse-engineerable from the visible features. Rates were checked against commonly-cited real-world benchmarks for each industry and recalibrated where the first pass was unrealistic (see `docs/TALKING_POINTS.md` for the specific corrections made and why).

Notably, all five industries independently land on a silhouette score of 0.18–0.23 at K=4 — five separate confirmations that engagement/loyalty behaves as a continuum rather than forming sharp natural clusters, a real and recurring property of behavioral data rather than an artifact of any one dataset.

## Executive Dashboard

`app/Home.py` — a Streamlit app with an industry selector in the sidebar. Selecting an industry recomputes, live, from that industry's real data:

- KPI cards (retention rate, churn rate, ARPU, CLV, engagement score)
- An executive brief with the actual current numbers, not static text
- A blended retention trend curve (via `CohortAnalyzer`)
- A segment breakdown (via `SegmentationEngine`, auto-labeled by engagement rank)
- Generated executive insights (via `InsightEngine`), calibrated against each industry's real distribution rather than one fixed threshold — including a CLV/ARPU *ratio* check (scale-invariant across industries with very different price points) rather than a raw currency comparison

Run locally:
```bash
cd retention-intelligence-suite
streamlit run app/Home.py
```

## Extensibility

The shared framework is designed to extend to any relationship-based business — finance, telecom, insurance, subscription SaaS — without architectural changes; adding an industry only requires a new data loader and feature engineering module. Five was a deliberate stopping point to prove the pattern thoroughly rather than dilute effort across additional modules that would mostly overlap with mechanics already covered.

## Planned: Downloadable Executive Artifacts

Each industry will get a formal technical report and one-page executive brief, generated from the same underlying models and offered as downloadable artifacts (PDF/Markdown) directly from the dashboard — not separate hand-written documents, but real exports of the same numbers already computed live.

## Further Reading

`docs/TALKING_POINTS.md` — the reasoning behind specific decisions (why K=4 over a statistically-better K=2, the real bugs caught and fixed during Healthcare and Gaming calibration, the InsightEngine recalibration), kept for interview preparation rather than the public repo narrative.
