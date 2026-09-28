# Explainable AI for Identity Fraud Detection at Bank Account Opening

MSc Data Analytics dissertation, De Montfort University (2026) — **Sydney Ndabai**

This project builds and stress-tests a fraud detection model for **new bank account applications**. It is harder than normal card-fraud detection because a brand-new applicant has no transaction history to learn from. The model has to work only with what is captured at sign-up, such as device, contact and address details.

I compared a stacking ensemble (XGBoost + LightGBM + Logistic Regression) against strong single models. Then I asked three questions that go beyond accuracy:

1. **Does it work on future data?** The models were trained on earlier months and tested on a later, unseen month, and then on a deliberately shifted version of the data.
2. **Can we trust its explanations?** SHAP explanations were produced and then *tested*, not just plotted.
3. **Is the extra complexity worth it?** Paired bootstrap confidence intervals were used to check whether differences between models are real or just noise.

📄 **Full report:** [`report/dissertation_report.pdf`](Report/Dissertation_report.pdf)

---

## Key findings

- **Gradient boosting clearly beats a linear model.** LightGBM and XGBoost were both significantly better than Logistic Regression.
- **Stacking added very little.** The stack had the best point estimate on the main metric, but it was **not statistically distinguishable from LightGBM alone**. Its base models were highly correlated (0.91–0.98), which leaves little room for an ensemble to help.
- **The explanations held up.** Removing the top 5 SHAP features cut performance by about 25%. The feature rankings agreed strongly with an independent Random Forest (Spearman ρ = 0.96), and they stayed stable across random seeds (ρ ≥ 0.996).
- **The decision threshold broke under distribution shift.** On the shifted dataset, the false positive rate rose from 3.9% to 6.5%, above the 5% target. Real systems need threshold monitoring and recalibration.
- **There is a fairness gap by age.** Applicants over 50 were wrongly flagged far more often (12.7% vs 3.7% false positive rate).

## Results (held-out test month)

| Model | TPR @ 5% FPR | PR-AUC | ROC-AUC |
|---|---|---|---|
| **Stacking Ensemble** | **0.577** | 0.216 | **0.897** |
| Stack + Random Forest | 0.576 | 0.215 | 0.897 |
| LightGBM | 0.573 | **0.217** | 0.897 |
| XGBoost | 0.562 | 0.209 | 0.893 |
| Logistic Regression | 0.534 | 0.185 | 0.887 |
| Random Forest | 0.518 | 0.169 | 0.877 |
| Dummy (always "legitimate") | 0.050 | 0.015 | 0.500 |

*TPR @ 5% FPR = the share of fraud caught when only 5% of genuine applicants are wrongly flagged. With only about 1% fraud, accuracy is misleading: the dummy model scores 98.5% accuracy while catching zero fraud.*

## Method at a glance

- **Data:** [Bank Account Fraud (BAF) dataset, NeurIPS 2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), with 1,000,000 synthetic applications and 1.10% fraud. The *Base* file was used for modelling and *Variant IV* for the distribution-shift test.
- **Time-based split:** train on months 0–5, set the decision threshold on month 6, and test once on month 7. Nothing from the test month was used in training, tuning or threshold setting.
- **Missing data:** BAF hides missing values as negative numbers. These were audited, kept, and given missingness flags. Missing address history turned out to be a strong fraud signal.
- **Class imbalance:** handled with class weighting rather than SMOTE, to avoid creating synthetic fraud cases.
- **Tuning:** walk-forward validation inside the training window.
- **Uncertainty:** 1,000 stratified bootstrap resamples, plus paired bootstrap comparisons between models.
- **Explainability:** Kernel SHAP on the full ensemble, checked with feature removal, cross-model agreement and seed stability.
- **Fairness:** predictive equality (false positive rate) across age groups.

## Repository structure

```
├── notebooks/
│   ├── 01_data_preparation.ipynb                    # data audit, cleaning, time-based split
│   ├── 02_modelling.ipynb                           # tuning, training, thresholds, evaluation
│   └── 03_explainability_fairness_robustness.ipynb  # SHAP, reliability tests, fairness, shift test
├── report/
│   └── dissertation_report.pdf
├── requirements.txt
└── README.md
```

## How to run

The notebooks were built on **Kaggle** and must be run in order (1 → 2 → 3). Each one saves files that the next one loads.

**On Kaggle (easiest):**
1. Create a notebook from `01_data_preparation.ipynb` and add the BAF dataset as input. Run it.
2. Create a notebook from `02_modelling.ipynb` and add notebook 1's output as input. Run it.
3. Create a notebook from `03_explainability_fairness_robustness.ipynb` and add the outputs of notebooks 1 and 2 as input. Run it.

**On your own computer:**
1. Download `Base.csv` and `Variant IV.csv` from the dataset link above.
2. Install the libraries: `pip install -r requirements.txt`
3. In each notebook, change the file paths near the top (`BASE_PATH`, `VARIANT_PATH`, `NB1`, `NB2`) to the folder where your files are saved.

The dataset is **not included** in this repository. Please get it from the original source and check its licence there.

**Reproducibility:** random seed 42 for all main results (seeds 7, 21 and 101 for stability tests). Key library versions: scikit-learn 1.6.1, XGBoost 3.2.0, LightGBM 4.6.0, SHAP 0.51.0.

## Limitations

- BAF is synthetic, so the results still need testing on real banking data.
- The stacking ensemble's internal cross-validation was not time-ordered. It stays inside the training months, so no test data leaks, but a fully time-ordered version would be stricter.
- The SHAP sample sizes were small because of compute limits.
- The fairness audit covered age only.

## Reference

Jesus, S. et al. (2022) *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.* NeurIPS 2022 Datasets and Benchmarks Track.

## Author

**Sydney Ndabai** — MSc Data Analytics, De Montfort University
