# Campaign Intelligence

### ML-Powered Customer Targeting & Campaign Optimization

An end-to-end machine learning decision system that predicts customer conversion probability and uses campaign economics and operational constraints to prioritize customers for marketing campaigns.

The project goes beyond a traditional classification model by combining **machine learning, decision optimization, explainability, API engineering, and MLOps monitoring**.

---

## Demo

### Campaign Optimization

![Campaign Dashboard](docs/screenshots/dashboard.png)

### Campaign Recommendations

![Campaign Results](docs/screenshots/results.png)

### Model Monitoring

![Model Monitoring](docs/screenshots/monitoring.png)

---

## Problem

A marketing team has a limited campaign budget and cannot contact every customer.

The goal is not simply:

> Who is likely to subscribe?

The practical question is:

> Which customers should we prioritize given their predicted conversion probability, campaign economics, budget, capacity, and contact history?

This project addresses that problem with an end-to-end ML decision system.

---

## Solution

The system performs five major tasks:

1. Predicts customer conversion probability.
2. Estimates expected campaign revenue and profit.
3. Ranks customers using configurable targeting strategies.
4. Applies operational constraints such as budget and contact capacity.
5. Monitors predictions, feature drift, and production performance.

---

## System Architecture

```text
                    ┌──────────────────┐
                    │   Bank Marketing │
                    │      Dataset     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Preparation │
                    │ & Feature Engine │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   ML Training    │
                    │     XGBoost      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Conversion Model │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       Prediction Service             Decision Engine
              │                             │
              └──────────────┬──────────────┘
                             ▼
                    Campaign Recommendation
                             │
                             ▼
                    Actual Customer Outcome
                             │
                             ▼
                    ┌──────────────────┐
                    │    Monitoring    │
                    ├──────────────────┤
                    │ Data Drift       │
                    │ Prediction Drift │
                    │ Performance      │
                    │ Retraining Signal│
                    └──────────────────┘
```

---

## Dataset

The project uses the **UCI Bank Marketing Dataset**, a public research dataset containing customer and campaign information.

The target variable is:

```text
y
```

where the outcome represents whether the customer subscribed to the offered banking product.

The dataset is used for research and portfolio development and does not represent a live production banking environment.

---

## Machine Learning

The project treats the task as binary classification.

The model predicts:

```text
P(customer subscribes)
```

The training pipeline includes:

* Data cleaning
* Feature engineering
* Categorical encoding
* Cross-validation
* Model comparison
* XGBoost training
* Probability prediction
* Calibration analysis
* Model evaluation

### Important leakage prevention

The `duration` feature is excluded from the pre-contact prediction system because it describes the duration of a campaign interaction that occurs after contact.

Using it for pre-contact targeting would introduce target leakage.

---

## Decision Engine

The model probability is not treated as the final business decision.

The decision layer considers:

```text
Conversion Probability
        +
Conversion Value
        -
Contact Cost
        +
Campaign Constraints
        +
Contact Fatigue
```

The system supports three strategies:

### 1. Highest Conversion Probability

Prioritize customers with the highest predicted probability.

### 2. Highest Expected Profit

Prioritize customers based on expected financial value.

```text
Expected Revenue =
Conversion Probability × Conversion Value
```

```text
Expected Profit =
Expected Revenue − Contact Cost
```

### 3. Fatigue-Aware Targeting

Adjust prioritization based on campaign contact frequency.

This is a decision-policy adjustment rather than a causal estimate.

---

## API

FastAPI exposes the model and decision system through endpoints such as:

```text
GET  /health
GET  /ready
GET  /model/info

POST /predict
POST /campaign/optimize
POST /outcomes

GET  /monitoring/drift
GET  /monitoring/predictions
GET  /monitoring/performance
GET  /monitoring/alerts
```

---

## Dashboard

The Streamlit dashboard provides:

* Campaign simulation
* Budget controls
* Contact-cost controls
* Conversion-value controls
* Targeting strategy selection
* Customer recommendations
* Expected revenue
* Expected profit
* ROI
* Prediction monitoring
* Model health
* Drift monitoring

---

## MLOps

The system includes a basic production feedback loop.

```text
Prediction
    ↓
Prediction Log
    ↓
Customer Outcome
    ↓
Performance Evaluation
    ↓
Model Monitoring
    ↓
Retraining Review
```

The system tracks:

* Model version
* Prediction probability
* Prediction timestamp
* Customer identifier
* Actual outcome
* Feature drift
* Prediction distribution
* Production performance

---

## Model Monitoring

Feature drift is monitored by comparing reference and current data distributions.

Numerical features use statistical distribution comparison.

Categorical features are monitored through changes in category distributions.

The monitoring system can flag:

```text
Feature Drift
Prediction Drift
Performance Degradation
```

Drift does not automatically mean that the model is incorrect. It signals that the data-generating environment should be investigated.

---

## Causal Limitation

The Bank Marketing dataset is observational and does not provide randomized treatment assignment.

Therefore, this project does **not** claim to estimate causal uplift or the incremental effect of contacting a customer.

The targeting system should be understood as a predictive and decision-optimization system.

A true uplift model would require suitable treatment/control data or a randomized campaign experiment.

---

## Technology Stack

### Python

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SciPy
* Joblib

### Application

* FastAPI
* Streamlit
* Pydantic

### MLOps

* Evidently
* Prediction logging
* Outcome logging
* Drift monitoring

### Testing

* Pytest

### Deployment

* Docker
* Docker Compose

---

## Project Structure

```text
campaign-intelligence-ml/
│
├── app/
│   ├── api/
│   ├── components/
│   ├── services/
│   └── app.py
│
├── src/
│   ├── models/
│   ├── monitoring/
│   └── utils/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── monitoring/
│
├── artifacts/
│   └── models/
│
├── notebooks/
│
├── tests/
│
├── docs/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Running Locally

Clone the repository and create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
copy .env.example .env
```

Start the API:

```bash
uvicorn app.api.main:app --reload
```

Start Streamlit in another terminal:

```bash
streamlit run app/app.py
```

The API will be available at:

```text
http://127.0.0.1:8000
```

The interactive API documentation is available through the FastAPI documentation interface.

The dashboard runs on:

```text
http://127.0.0.1:8501
```

---

## Run with Docker

The application can be run as two containerized services:

* FastAPI backend
* Streamlit dashboard

### Prerequisites

* Docker Desktop
* Git

### Configuration

Create the local environment configuration from the example file.

```bash
copy .env.example .env
```

For Docker networking, configure the dashboard API URL as:

```text
API_URL=http://api:8000
```

### Build

```bash
docker compose build
```

### Start

```bash
docker compose up
```

### Services

API:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

Dashboard:

```text
http://localhost:8501
```

### Stop

```bash
docker compose down
```

### Rebuild

Use this when dependencies or application code affecting the image have changed:

```bash
docker compose up --build
```

---

## Deployment

The application is containerized into separate backend and dashboard services.

```text
User
  ↓
Streamlit Dashboard
  ↓
FastAPI
  ↓
ML Model
  ↓
Prediction / Optimization
  ↓
Monitoring
```

The same Docker configuration can be used for local development and cloud deployment by changing environment variables.

### Production configuration

The dashboard requires an API URL:

```text
API_URL=https://<api-domain>
```

The API service loads the versioned model artifact and exposes prediction, optimization, and monitoring endpoints.

### Deployment requirements

The deployment environment must provide:

* Python/Docker runtime
* Model artifact
* Processed feature configuration
* Environment variables
* Persistent storage for monitoring data if production logging is enabled

For a production banking application, monitoring data should be stored in a managed database or object store rather than local container storage.

The portfolio deployment is intended as a demonstration of the architecture and should not be treated as a production banking system.

---

## Campaign Strategy Experiment

The system compares multiple targeting policies under the same campaign constraints.

Strategies evaluated:

* Random targeting
* Highest conversion probability
* Expected-profit targeting
* Fatigue-aware targeting

All strategies are evaluated on the same held-out dataset using the same campaign budget and contact cost.

### Evaluation Metrics

* Number of contacts
* Actual conversions
* Conversion rate
* Revenue
* Campaign cost
* Profit
* ROI

### Why This Experiment Matters

A high-probability customer is not necessarily the highest-value customer.

The experiment therefore evaluates the complete decision problem rather than measuring only the ML model's predictive performance.

Results are generated directly from the held-out evaluation data and are not manually specified.

---

## Testing

Run:

```bash
pytest -v
```

Tests cover:

* API health
* API readiness
* Model information
* Prediction behavior
* Monitoring
* Drift detection

---

## Key Engineering Decisions

### Why XGBoost?

The dataset contains mixed numerical and categorical-derived features and nonlinear relationships. Gradient-boosted trees provide a strong baseline for tabular classification while supporting feature importance and probability-based prediction.

### Why separate prediction from decision-making?

A probability model answers:

> How likely is conversion?

The decision engine answers:

> What should we do with that probability?

Separating these layers makes the system easier to test, modify, and explain.

### Why exclude `duration`?

Because it is only known after the interaction. Including it in pre-contact targeting would leak information from the future.

### Why monitor drift?

A model can degrade when the population, campaign strategy, or customer behavior changes. Monitoring provides an early signal for investigation.

---

## Limitations

This project has several deliberate limitations:

* The dataset is historical and public.
* The dataset is not randomized.
* The targeting policy is not a causal uplift model.
* Campaign economics use configurable assumptions.
* Production performance requires real post-campaign outcomes.
* Drift detection indicates distribution changes but does not automatically establish business impact.

---

## Future Improvements

Potential extensions include:

* Randomized campaign experiments
* True uplift modeling
* Treatment-effect estimation
* Feature store integration
* MLflow model registry
* Automated CI/CD
* Cloud deployment
* Real-time monitoring
* Automated model validation
* Champion/challenger model evaluation

---

## Project Goal

The main goal is not simply to maximize classification performance.

The goal is to demonstrate how a machine learning model can be integrated into a complete decision system:

```text
Data
→ ML
→ Prediction
→ Economics
→ Decision
→ API
→ Application
→ Monitoring
→ Feedback
```

This project demonstrates the engineering and analytical thinking required to move from a machine learning experiment toward a deployable ML product.
