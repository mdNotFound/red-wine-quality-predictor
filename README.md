# 🍷 Red Wine Quality Predictor

A machine learning project that classifies red wine as **good** (quality ≥ 7) or **not good** using a Random Forest classifier trained on the UCI Wine Quality dataset.

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

## 🔍 Approach

### 1. Baseline Random Forest
- `n_estimators=100`, `max_features='sqrt'`, `oob_score=True`
- OOB Score: **~0.899**
- Test accuracy: **94%**, but class 1 recall: only **63%**

### 2. Handling Class Imbalance
- Tried `class_weight='balanced'` → class 1 recall improved slightly to **67%**

### 3. Threshold Tuning ✅ Best approach
- Plotted Precision-Recall curve to identify optimal threshold
- At threshold **0.35**: class 1 recall improved to **79%** with balanced precision

### 4. Feature Importance
Top features by importance:
1. 🥇 **Alcohol** (highest)
2. 🥈 **Sulphates**
3. 🥉 **Volatile acidity**

### 5. Hyperparameter Tuning (GridSearchCV)
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

## 📈 Results Summary

| Model | Accuracy | Class 1 Recall |
|---|---|---|
| Baseline Random Forest | 94% | 63% |
| RF + `class_weight='balanced'` | 92% | 67% |
| Baseline + threshold = 0.35 | 94% | **79%** |

---

## 🚀 Setup

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

```bash
jupyter notebook redwine_prediction.ipynb
```

---

## 📁 Project Structure

```
red-wine-quality-predictor/
│
├── redwine_prediction.ipynb   # Main notebook
└── README.md
```

---

## 💡 Key Takeaway

For imbalanced classification, **threshold tuning** outperformed `class_weight` balancing — boosting minority class recall from **63% → 79%** without sacrificing overall accuracy.
