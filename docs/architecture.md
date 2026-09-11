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

## Flow

Data
→ Feature Engineering
→ Model
→ Prediction API
→ Decision Engine
→ Campaign Recommendation
→ Outcome
→ Monitoring
→ Model Review
