# Campaign Response Model — Model Card

## Model Overview

Model:
Campaign Response Prediction Model

Version:
1.0.0

Algorithm:
XGBoost

Task:
Binary classification

Target:
Customer subscription (`y`)

## Intended Use

The model estimates the probability that a customer
will subscribe to the offered banking product.

The prediction is intended to support campaign
prioritization and resource allocation.

## Dataset

Source:
UCI Bank Marketing Dataset

The dataset is a public research dataset and does
not represent a live production banking environment.

## Features

Customer attributes include:

- Age
- Job
- Marital status
- Education
- Housing loan
- Personal loan
- Balance

Campaign attributes include:

- Contact method
- Campaign count
- Previous contacts
- Previous campaign outcome

## Leakage Prevention

The `duration` variable is excluded from the
production prediction feature set because it is only
known after a campaign interaction has occurred.

Including it would create target leakage for
pre-contact targeting.

## Evaluation

The model is evaluated using:

- ROC-AUC
- PR-AUC
- Brier Score
- Calibration

Cross-validation is used during model development.

## Decision Layer

Predictions are combined with:

- Campaign budget
- Contact cost
- Conversion value
- Maximum contact capacity
- Contact fatigue

The system produces an expected-value-based
campaign ranking.

## Causal Limitation

The dataset does not contain randomized treatment
assignment.

Therefore, the system does not claim to estimate
causal uplift or the incremental effect of contacting
a customer.

## Monitoring

The system supports:

- Feature drift monitoring
- Prediction monitoring
- Prediction logging
- Outcome logging
- Production performance evaluation
- Retraining signals

## Limitations

Model performance may change when customer behavior,
campaign strategy, economic conditions, or data
collection processes change.

The model should be evaluated before being used in a
new operational environment.

## Responsible Use

The system should support human decision-making rather
than automatically determining customer treatment
without appropriate business and compliance review.
