# 📅 Biases in Machine Learning for Predicting Appointment No-Shows

**Predicting missed healthcare appointments — and auditing the models for demographic bias.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTETomek-4B8BBE)
![Fairlearn](https://img.shields.io/badge/Fairlearn-fairness-2E7D32)
![AIF360](https://img.shields.io/badge/AIF360-bias%20metrics-0F62FE)
![SHAP](https://img.shields.io/badge/SHAP-explainability-8E44AD)

---

## 📌 Overview

Missed appointments ("no-shows") waste clinical capacity and delay care. Machine-learning models can predict them, but if those models behave differently across patient groups, using them to plan care can deepen existing inequities.

This project asks two research questions:

1. **Which demographic and clinical factors introduce bias into ML models that predict appointment no-shows?**
2. **Which techniques can be used to mitigate these biases?**

## 🗂️ Dataset

[Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (Kaggle, `KaggleV2-May-2016.csv`) — **~110,000 appointments** from public clinics in Vitória, Brazil.

| Feature | Description |
|---|---|
| `Gender`, `Age` → `age_group` | Demographics (age binned into WHO groups: <15, 15–24, 25–44, 45–64, 65+) |
| `Scholarship` | Enrolled in the *Bolsa Família* welfare programme |
| `Hypertension`, `Diabetes`, `Alcoholism`, `Handicap` | Clinical conditions |
| `SMS_received` | Patient received an SMS reminder |
| `No-show` | **Target** — patient missed the appointment (~20% of records) |

## ⚙️ Methodology

1. **Data cleaning** — removed invalid ages (−1, 115), binarised `Handicap`, fixed column names and data types.
2. **EDA** — univariate and bivariate analysis of every feature against no-show, plus a correlation matrix.
3. **Class imbalance** — 80/20 train/test split, then **SMOTETomek** on the training set only (20% → 44% minority class).
4. **Modelling** — 11 classifiers: Logistic Regression, Decision Tree (Gini/Entropy), Random Forest (Gini/Entropy), KNN (k = 3, 5), Gradient Boosting, Bagging, AdaBoost and an MLP neural network.
5. **Bias audit** — 5 models (Decision Tree, Random Forest, KNN, Gradient Boosting, Neural Network) evaluated on **8 sensitive features** with **4 fairness metrics**:
   - Demographic Parity Difference *(Fairlearn)*
   - Equalized Odds Difference *(Fairlearn)*
   - Disparate Impact *(AIF360)*
   - False Negative Rate Difference
6. **Explainability** — SHAP summary plots for Logistic Regression, Decision Tree and Random Forest.

## 🏆 Results

### Model performance (test set, weighted averages)

| Model | Accuracy | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.678 | 0.721 | 0.678 | 0.696 |
| Decision Tree (Gini) | 0.722 | 0.723 | 0.722 | 0.722 |
| Random Forest (Entropy) | 0.724 | 0.722 | 0.724 | 0.723 |
| KNN (k = 5) | 0.795 | 0.704 | 0.795 | 0.714 |
| KNN (k = 3) | 0.778 | 0.710 | 0.778 | **0.727** |
| Gradient Boosting | 0.722 | 0.723 | 0.722 | 0.722 |
| Bagging | 0.714 | 0.721 | 0.714 | 0.717 |

### Bias audit — Decision Tree example

| Sensitive feature | Demographic Parity Diff. | Equalized Odds Diff. | Disparate Impact | Bias? |
|---|:---:|:---:|:---:|:---:|
| Gender | 0.016 | 0.032 | 1.02 | ✅ None |
| Alcoholism | 0.015 | 0.049 | 1.00 | ✅ None |
| Scholarship | 0.123 | 0.155 | 0.83 | ⚠️ Moderate |
| Handicap | 0.128 | 0.193 | 1.12 | ⚠️ Moderate |
| Diabetes | 0.160 | 0.248 | 1.13 | ⚠️ Moderate |
| Hypertension | 0.168 | 0.252 | 1.21 | ⚠️ Moderate |
| Age group | 0.338 | 0.474 | — | 🔴 High |
| SMS received | 0.623 | 0.714 | 0.61 | 🔴 High |

### Key findings

- **Gender** and **alcoholism** introduce no measurable bias.
- **SMS reminders** and **age group** produce the largest disparities — the model almost never predicts a no-show for patients who did *not* receive an SMS, and almost never for patients aged 45–64 and 65+.
- Chronic conditions (**hypertension**, **diabetes**, **handicap**) and welfare status (**scholarship**) show moderate bias.
- High accuracy alone hides these gaps: KNN has the best accuracy but a bias audit is needed before any model is used to plan care.

## 🚀 How to Run

```bash
git clone https://github.com/prabhursingh/No-Show.git
cd No-Show
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn fairlearn aif360 shap
jupyter notebook "no show.ipynb"
```

Update the CSV path in the first data-loading cell to point at `KaggleV2-May-2016.csv` in this folder.

## 📁 Repository Structure

```
No-Show/
├── no show.ipynb            # Full analysis: cleaning → EDA → models → bias audit → SHAP
├── KaggleV2-May-2016.csv    # Dataset (Kaggle)
└── README.md
```

## 🔭 Future Work

- Apply and compare bias-mitigation methods (re-weighing, threshold optimisation, adversarial debiasing).
- Add ROC-AUC and per-group calibration to the evaluation.
- Use appointment lead time (scheduled vs. appointment day) as a feature.

## 👥 Authors

Group project — Health Informatics programme. Analysis and repository maintained by **Prabhu Raj Singh** ([LinkedIn](https://www.linkedin.com/in/prabhu-raj-singh-9a926a20)).
