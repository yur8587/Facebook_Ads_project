#  Facebook Ads Click-Through Prediction

End-to-end machine learning project that predicts whether a user will click on a Facebook ad, based on salary and time spent on site. The project covers the full data science workflow: from exploratory analysis to hyperparameter-optimized ensemble modeling.

##  Dataset

- **499 users** (250 clicked / 249 didn't — a balanced 50/50 split)
- Features: `Salary`, `Time Spent on Site`
- Target: `Clicked` (binary)

##  Methodology

**1. Exploratory Data Analysis**

- Class balance analysis (50.1% vs 49.9%)
- Boxplots and distributions of Salary / Time Spent per class — both features are clearly predictive
- Scatter plot reveals strong class separation in the feature space

**2. Preprocessing (inside pipelines — no data leakage)**

- `Winsorizer` (Gaussian capping, fold=3) — outlier handling
- `StandardScaler` — feature scaling
- All preprocessing is embedded in scikit-learn `Pipeline` and fitted **only on training folds** during cross-validation

**3. Modeling**

- Baselines: KNN, SVC, AdaBoost (with 5-fold CV)
- Hyperparameter tuning: `RandomizedSearchCV` (5-fold CV) for KNN, AdaBoost, Random Forest
- Ensemble: `StackingClassifier` combining tuned pipelines with a Random Forest meta-learner

**4. Evaluation**

- Metrics: Accuracy, F1, Precision, Recall, **ROC-AUC**
- Confusion matrices with correct True/Predicted labels
- ROC curves for all models on a single chart

##  Results


| Model              | Accuracy  | F1        | Precision | Recall    | ROC-AUC  |
| ------------------ | --------- | --------- | --------- | --------- | -------- |
| **RandomForest**   | **0.853** | **0.851** | 0.863     | **0.868** | 0.92     |
| KNN                | 0.847     | 0.846     | 0.851     | 0.840     | **0.93** |
| AdaBoost           | 0.847     | 0.846     | 0.851     | 0.840     | 0.92     |
| Stacking           | 0.847     | 0.846     | 0.851     | 0.840     | 0.91     |


**Best model: Random Forest** — 85.3% accuracy, 0.85 F1, 0.92 ROC-AUC. KNN achieved the highest ROC-AUC (0.93). Stacking provided no gain over the single best model — expected on a dataset of this size.

##  Key Takeaways

- Tuning **inside** the pipeline (rather than on pre-processed data) removed data leakage and slightly changed model rankings — a validation of correct methodology
- Both features are strong predictors; the models capture the click/no-click boundary with \~85% accuracy
- Confusion matrices show balanced error rates across classes, confirming the model is not biased toward the majority class

## 🛠 Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `feature-engine` · `seaborn` · `matplotlib` · `Jupyter`