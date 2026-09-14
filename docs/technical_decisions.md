# Technical Decisions

## 1. Why XGBoost?

The problem is structured tabular classification with both numerical and categorical-derived features.

XGBoost was selected because gradient-boosted decision trees are well suited to nonlinear relationships in tabular data and provide strong predictive performance without requiring a deep neural network.

The model outputs probabilities rather than only class labels because campaign optimization requires an estimate of conversion likelihood.

---

## 2. Why Probability Instead of a Binary Prediction?

A binary prediction answers:

> Will this customer subscribe?

A probability provides more information:

```text
Customer A → 0.91
Customer B → 0.67
Customer C → 0.23
```

This allows customers to be ranked and allows the business layer to incorporate different conversion values and campaign costs.

---

## 3. Why Exclude `duration`?

`duration` represents the duration of a customer interaction.

For a pre-contact campaign decision, this information is unavailable before contacting the customer.

Including it would allow the model to use information that becomes available only after the campaign interaction.

Therefore it is excluded from the production feature set.

---

## 4. Why Separate Prediction and Optimization?

The prediction model answers:

> How likely is conversion?

The decision engine answers:

> Who should we contact?

Keeping these components separate allows the business policy to change without retraining the ML model.

For example, the same prediction model can support:

* Probability-based targeting
* Expected-profit targeting
* Capacity-constrained targeting
* Fatigue-aware targeting

---

## 5. Why Expected Profit?

Maximizing conversion probability does not necessarily maximize business value.

For example:

```text
Customer A
Probability = 0.90
Value = ₹100
Expected Revenue = ₹90

Customer B
Probability = 0.70
Value = ₹2,000
Expected Revenue = ₹1,400
```

Customer B has lower conversion probability but substantially higher expected revenue under these assumptions.

This demonstrates why prediction and business optimization should be separated.

---

## 6. Why Add Contact Cost?

Campaign resources are limited.

If every contact has a cost, targeting should account for:

```text
Expected Revenue − Contact Cost
```

This prevents the system from treating every high-probability customer as equally valuable.

---

## 7. Why Add Budget Constraints?

A campaign cannot contact an unlimited number of customers.

For a budget `B` and contact cost `C`:

```text
Maximum Contacts = floor(B / C)
```

The optimizer therefore operates under an explicit operational constraint.

---

## 8. Why Monitor Drift?

A model is trained using historical data, but customer behavior can change.

Possible causes include:

* Changes in customer demographics
* Changes in campaign strategy
* Changes in communication channels
* Economic changes
* Changes in data collection

Monitoring feature distributions and prediction distributions provides an early warning that the production environment differs from the training environment.

---

## 9. Why Log Predictions and Outcomes Separately?

Predictions are available immediately.

Actual outcomes become available later.

Separating them allows the system to create a feedback loop:

```text
Prediction
    ↓
Campaign
    ↓
Actual Outcome
    ↓
Performance Evaluation
```

This also makes delayed-label evaluation possible.

---

## 10. Why Not Claim Causal Uplift?

The dataset does not provide randomized treatment assignment.

A high predicted probability does not prove that contacting the customer caused the conversion.

Therefore this project is a predictive decision system, not a causal uplift model.

A future version would require randomized treatment/control data.

---

## 11. Why FastAPI?

FastAPI separates the ML system from the user interface.

This allows:

```text
Streamlit
     ↓
FastAPI
     ↓
Model
```

The same API could later support a web application, internal service, batch process, or another client.

---

## 12. Why Docker?

Docker packages the application and dependencies into a reproducible runtime environment.

This reduces the difference between:

```text
Developer Machine
```

and:

```text
Deployment Environment
```

---

## 13. Why Not Use a Complex Cloud Architecture?

The project is a portfolio demonstration rather than a high-volume production banking system.

Adding Kubernetes, distributed feature stores, or complex streaming infrastructure would increase operational complexity without improving the core demonstration.

The architecture is intentionally simple enough to understand while still demonstrating:

* ML
* APIs
* optimization
* monitoring
* testing
* deployment
