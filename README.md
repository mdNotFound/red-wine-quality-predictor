# 🍷 Red Wine Quality Predictor

A machine learning project that classifies red wine as **good** (quality ≥ 7) or **not good** using ensemble methods trained on the UCI Wine Quality dataset.

---

## 📊 Dataset

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)
- **Size:** 1,599 samples · 11 physicochemical features
- **Target:** Binary — `1` (quality ≥ 7), `0` (quality < 7)
- **Class distribution:** 1,382 not good / 217 good *(imbalanced)*

---

## 🧪 Features

| Feature | Description |
|---|---|
| Fixed acidity | Tartaric acid concentration |
| Volatile acidity | Acetic acid concentration |
| Citric acid | Freshness/flavor enhancer |
| Residual sugar | Sugar remaining after fermentation |
| Chlorides | Salt content |
| Free sulfur dioxide | Free SO₂ |
| Total sulfur dioxide | Total SO₂ |
| Density | Wine density |
| pH | Acidity level |
| Sulphates | Antimicrobial additive |
| Alcohol | Alcohol percentage |

---

## 📁 Project Structure

```
red-wine-quality-predictor/
│
├── redwine_prediction.ipynb         # Random Forest approach
├── redwine_prediction_adaboost.ipynb  # AdaBoost + SMOTE approach
└── README.md
```

---

## 🔍 Approach 1: Random Forest (`redwine_prediction.ipynb`)

### Baseline Random Forest
- `n_estimators=100`, `max_features='sqrt'`, `oob_score=True`
- OOB Score: **~0.899**
- Test accuracy: **94%**, but class 1 recall: only **63%**

### Handling Class Imbalance
- Tried `class_weight='balanced'` → class 1 recall improved slightly to **67%**

### Threshold Tuning ✅ Best approach
- Plotted Precision-Recall curve to find optimal threshold
- At threshold **0.35**: class 1 recall improved to **79%** with balanced precision

### Feature Importance
Top features by importance:
1. 🥇 **Alcohol** (highest)
2. 🥈 **Sulphates**
3. 🥉 **Volatile acidity**

### Hyperparameter Tuning (GridSearchCV)
```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}
```
- **Best params:** `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}`
- **Best ROC-AUC:** ~0.897

---

## 🔍 Approach 2: AdaBoost + SMOTE (`redwine_prediction_adaboost.ipynb`)

Instead of threshold tuning, this notebook tackles class imbalance directly using **SMOTE** (Synthetic Minority Oversampling Technique) before training an **AdaBoost** classifier.

### SMOTE Resampling
- Balanced both classes to 1,382 samples each
- Eliminates recall bias without adjusting decision threshold

### AdaBoost Classifier
- Base estimator: `DecisionTreeClassifier(max_depth=1)` (decision stump)
- `n_estimators=50`, `learning_rate=1.0`
- AdaBoost iteratively focuses on misclassified examples from previous rounds

### Results
- Test accuracy: **~85%**
- Class 1 recall: **84%** *(consistent across both classes)*
- Well-balanced precision/recall without threshold tuning

---

## 📈 Model Comparison

| Approach | Accuracy | Class 1 Recall | Notes |
|---|---|---|---|
| Baseline Random Forest | 94% | 63% | Default threshold |
| RF + `class_weight='balanced'` | 92% | 67% | Minor improvement |
| RF + threshold = 0.35 | 94% | **79%** | Best RF result |
| AdaBoost + SMOTE | 85% | **84%** | Balanced classes, consistent |

---

## 🚀 Setup

```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn
```

```bash
jupyter notebook redwine_prediction.ipynb
jupyter notebook redwine_prediction_adaboost.ipynb
```

---

## 💡 Key Takeaways

- **Threshold tuning** (RF) boosted minority class recall from 63% → 79% while maintaining high overall accuracy.
- **SMOTE + AdaBoost** achieved even higher minority class recall (84%) with balanced per-class performance, at the cost of slightly lower overall accuracy.
- Choice of approach depends on the tradeoff: **higher accuracy** (RF + threshold) vs **more balanced detection** (SMOTE + AdaBoost).
