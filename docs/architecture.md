# System Architecture

The system is divided into five layers.

## 1. Data Layer

The public Bank Marketing dataset is processed into
model-ready features.

## 2. ML Layer

An XGBoost classifier estimates customer conversion
probability.

## 3. Decision Layer

Predictions are combined with campaign economics and
operational constraints.

## 4. API Layer

FastAPI exposes prediction, optimization, and
monitoring endpoints.

## 5. Monitoring Layer

Predictions, outcomes, drift, and production
performance are monitored over time.

## Diagram

```text
┌─────────────────────┐
│  UCI Bank Marketing │
│       Dataset       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Data Processing &   │
│ Feature Engineering │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ML Training       │
│      XGBoost        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Versioned Model     │
│ Artifact            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      FastAPI        │
│ Prediction Service  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Decision Engine    │
│                     │
│ Probability         │
│ Expected Profit     │
│ Budget              │
│ Capacity            │
│ Fatigue             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Streamlit Dashboard │
└──────────┬──────────┘
           │
           ▼
      Campaign
           │
           ▼
      Actual Outcome
           │
           ▼
┌─────────────────────┐
│     Monitoring      │
│                     │
│ Drift               │
│ Predictions         │
│ Performance         │
└──────────┬──────────┘
           │
           ▼
    Model Review /
    Retraining Signal
```
