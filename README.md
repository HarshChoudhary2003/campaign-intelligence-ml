# Campaign Intelligence

### ML-Powered Customer Targeting & Campaign Optimization

**Predict → Optimize → Explain → Monitor**

Campaign Intelligence is an end-to-end machine learning system that predicts customer conversion probability and prioritizes campaign targets under real-world business constraints such as budget, contact cost, expected conversion value, and campaign fatigue.

The project goes beyond a traditional classification model by combining **machine learning, decision optimization, explainability, API deployment, monitoring, and automated testing**.

---

## 🚀 What It Does

The system answers four business questions:

1. **Who is likely to convert?**
2. **Who should the business contact first?**
3. **Why did the model prioritize a customer?**
4. **Is the model still performing reliably after deployment?**

### System Flow

```text
Bank Marketing Data
        ↓
Data Processing
        ↓
Feature Engineering
        ↓
XGBoost Model
        ↓
Conversion Probability
        ↓
Decision Engine
        ↓
Campaign Optimization
        ↓
Customer Recommendations
        ↓
Prediction + Outcome Logging
        ↓
Monitoring
        ↓
Model Review / Retraining Signal
```

---

## 🎯 Business Problem

Marketing teams often have limited campaign capacity.

For example:

* Budget = ₹5,000
* Contact cost = ₹20
* Maximum contacts = 250
* Conversion value = ₹1,000

A business cannot contact every customer.

Instead of treating every customer equally, Campaign Intelligence estimates the probability of conversion and uses business constraints to prioritize the customers with the highest expected value.

---

## 💡 Key Idea

A prediction alone is not enough.

The system converts:

```text
Probability
     ↓
Expected Revenue
     ↓
Expected Profit
     ↓
Campaign Decision
```

### Expected Revenue

```text
Expected Revenue =
Conversion Probability × Conversion Value
```

### Expected Profit

```text
Expected Profit =
Expected Revenue − Contact Cost
```

The decision engine then ranks customers according to the selected business strategy.

---

## 🧠 Targeting Strategies

The system supports:

### 1. Highest Conversion Probability

Prioritizes customers with the highest predicted probability of conversion.

### 2. Highest Expected Profit

Ranks customers using estimated financial value:

```text
Expected Profit =
P(conversion) × Conversion Value − Contact Cost
```

### 3. Fatigue-Aware Targeting

Adjusts the ranking using campaign-contact history as a configurable business policy.

This is a **policy assumption**, not a causal claim.

---

## 🤖 Machine Learning

### Model

**XGBoost**

XGBoost was selected because the problem is structured/tabular data with nonlinear relationships and mixed customer characteristics.

### Target

Binary classification:

```text
y = 1 → customer subscribed
y = 0 → customer did not subscribe
```

### Probability Output

The model produces:

```text
P(customer converts)
```

rather than only a binary prediction.

This allows the decision engine to rank customers and calculate expected business value.

---

## 🔍 Leakage Prevention

A major modeling decision was excluding `duration` from pre-contact targeting.

`duration` represents the length of the interaction and is only known after the customer has already been contacted.

Using it to decide **who to contact** would introduce future information into the prediction.

Therefore:

```text
duration → excluded
```

This makes the prediction setup more representative of a real pre-contact campaign decision.

---

## 📊 Model Evaluation

The model is evaluated on a held-out test set.

Metrics include:

| Metric      | Result |
| ----------- | -----: |
| ROC-AUC     |    TBD |
| PR-AUC      |    TBD |
| Brier Score |    TBD |
| Precision   |    TBD |
| Recall      |    TBD |

### Why PR-AUC?

The business is particularly interested in identifying customers who actually convert.

PR-AUC therefore provides a useful view of positive-class performance alongside ROC-AUC.

### Why Brier Score?

The system uses predicted probabilities for business decisions.

Brier Score helps evaluate the quality of those probability estimates.

---

## 💰 Business Experiment

The model is evaluated as a decision system, not only as a classifier.

Strategies are compared under the same simulated campaign constraints:

| Strategy            | Contacts | Conversions | Revenue | Cost | Profit | ROI |
| ------------------- | -------: | ----------: | ------: | ---: | -----: | --: |
| Random              |      TBD |         TBD |     TBD |  TBD |    TBD | TBD |
| Highest Probability |      TBD |         TBD |     TBD |  TBD |    TBD | TBD |
| Expected Profit     |      TBD |         TBD |     TBD |  TBD |    TBD | TBD |
| Fatigue-Aware       |      TBD |         TBD |     TBD |  TBD |    TBD | TBD |

These results will be populated from the actual held-out experiment.

The experiment is a **historical simulation**, not a randomized controlled campaign.

---

## 🔎 Explainability

The system uses **SHAP** to explain individual predictions.

For a selected customer, the system can show:

```text
Conversion Probability
        ↓
Expected Value
        ↓
Positive Contributors
        ↓
Negative Contributors
```

SHAP values explain how model features contributed to a prediction.

They **do not establish causality**.

---

## ⚙️ System Architecture

```text
                    ┌───────────────────┐
                    │   Streamlit UI    │
                    │      :8501        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │     FastAPI       │
                    │      :8000        │
                    └─────────┬─────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
        ┌────────────┐ ┌────────────┐ ┌────────────┐
        │ XGBoost    │ │ Decision   │ │   SHAP     │
        │ Model      │ │ Engine     │ │ Explainer  │
        └──────┬─────┘ └──────┬─────┘ └────────────┘
               │              │
               └──────┬───────┘
                      ▼
              ┌───────────────┐
              │  Monitoring   │
              ├───────────────┤
              │ Predictions   │
              │ Outcomes      │
              │ Drift         │
              │ Performance   │
              │ Alerts        │
              └───────────────┘
```

---

## 🛠️ Technology Stack

| Area             | Technology                            |
| ---------------- | ------------------------------------- |
| Language         | Python                                |
| Data Processing  | Pandas, NumPy                         |
| Machine Learning | XGBoost, Scikit-learn                 |
| Explainability   | SHAP                                  |
| API              | FastAPI                               |
| Dashboard        | Streamlit                             |
| Testing          | Pytest                                |
| Monitoring       | Custom drift & performance monitoring |
| Serialization    | Joblib                                |
| Containerization | Docker                                |
| Orchestration    | Docker Compose                        |

---

## 📁 Project Structure

```text
campaign-intelligence/
│
├── app/
│   ├── app.py
│   ├── api/
│   ├── services/
│   └── components/
│
├── src/
│   ├── models/
│   ├── monitoring/
│   ├── explainability/
│   ├── experiments/
│   ├── evaluation/
│   └── utils/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── monitoring/
│   └── experiments/
│
├── artifacts/
│   └── models/
│
├── tests/
│
├── docs/
│   ├── architecture.md
│   ├── model_card.md
│   └── technical_decisions.md
│
├── notebooks/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔌 API

### Health

```http
GET /health
```

### Readiness

```http
GET /ready
```

### Model Information

```http
GET /model/info
```

### Prediction

```http
POST /predict
```

### Campaign Optimization

```http
POST /campaign/optimize
```

### Prediction Explanation

```http
POST /predict/explain
```

### Monitoring

```http
GET /monitoring/performance
GET /monitoring/predictions
GET /monitoring/alerts
```

Interactive API documentation is available through FastAPI Swagger:

```text
http://localhost:8000/docs
```

---

## 📈 Monitoring

The system tracks:

### Data

* Feature distributions
* Numerical drift
* Categorical distribution changes

### Predictions

* Prediction volume
* Conversion probabilities
* Expected values
* Model version

### Outcomes

* Actual campaign outcomes
* Prediction/outcome matching

### Model Performance

* ROC-AUC
* PR-AUC
* Brier Score
* Actual conversion rate

### Alerts

The system can flag significant production performance degradation for model review.

Retraining is treated as a **review signal**, not an automatic replacement of the production model.

---

## 🧪 Testing

Run:

```bash
pytest -q
```

Tests cover areas including:

* API health
* Model artifact loading
* Feature leakage prevention
* Campaign capacity
* Monitoring logic
* Retraining conditions

---

## 🐳 Run with Docker

Build and start the complete system:

```bash
docker compose up --build
```

Dashboard:

```text
http://localhost:8501
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

Stop:

```bash
docker compose down
```

---

## 💻 Run Locally

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.models.train_model
```

Run the API:

```bash
uvicorn app.api.main:app --reload
```

Run Streamlit in another terminal:

```bash
streamlit run app/app.py
```

---

## 📦 Dataset

The project uses the **Bank Marketing dataset** from the UCI Machine Learning Repository.

The original raw dataset is not committed to this repository.

Follow the instructions in:

```text
data/raw/README.md
```

before running the training pipeline.

---

## ⚠️ Limitations

### Historical Data

The model is trained on historical campaign data and may not represent a current organization's customer population.

### Prediction ≠ Causation

The model estimates conversion probability.

It does not estimate:

```text
"What would happen if we contacted this customer
versus if we did not contact them?"
```

Answering that question requires randomized treatment/control data or an appropriate causal inference design.

### Campaign Simulation

The strategy comparison uses historical held-out outcomes and should be interpreted as a simulation rather than proof of future campaign performance.

### Fatigue

The fatigue-aware strategy represents a configurable business policy and should not be interpreted as evidence that repeated contact causes lower conversion.

### Production Monitoring

Reliable production performance metrics require sufficient real prediction and outcome data.

### Responsible Use

Customer-targeting systems should be reviewed for fairness, data quality, business constraints, and applicable policies before real-world deployment.

---

## 🚀 Future Improvements

Potential next steps include:

* Real campaign-history database
* Randomized treatment/control experimentation
* Uplift modeling
* Causal ML
* Automated feature pipelines
* Cloud deployment
* Model registry
* Centralized monitoring
* Automated data validation
* Advanced calibration
* Cost-sensitive optimization
* Human-in-the-loop campaign approval

---

## 👨💻 Project Focus

This project demonstrates practical skills across:

```text
Data Engineering
      ↓
Machine Learning
      ↓
Business Decision Modeling
      ↓
Explainable AI
      ↓
API Engineering
      ↓
Monitoring
      ↓
Testing
      ↓
Containerization
```

The goal was not simply to train a model.

The goal was to build a **complete decision-support system around a machine learning model**.
