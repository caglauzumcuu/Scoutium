# ⚽ Scoutium — Talent Scouting Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![sklearn](https://img.shields.io/badge/sklearn-1.0+-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-Latest-yellow)
![SMOTE](https://img.shields.io/badge/imbalanced--learn-Latest-green)

## 📌 Problem Definition
Scouts observe footballers during matches and rate them on various attributes.
Based on these ratings, the goal is to predict whether a player is:
- `0` → **average**
- `1` → **highlighted**

This is a **binary classification** problem with class imbalance.

---

## 📁 Dataset
| File | Rows | Columns | Description |
|---|---|---|---|
| `scoutium_attributes.csv` | 10.730 | 8 | Player attribute scores per scout per match |
| `scoutium_potential_labels.csv` | 322 | 5 | Scout's final potential label per player |

**Source:** (https://www.kaggle.com/datasets/ebruiserisobay/scoutium-dataset/data)

### Variables
| Variable | Description |
|---|---|
| `task_response_id` | Set of a scout's evaluations in a match |
| `match_id` | Match ID |
| `evaluator_id` | Scout ID |
| `player_id` | Player ID |
| `position_id` | Player's position (1=GK, 2=CB, ... 10=FW) |
| `attribute_id` | ID of the evaluated attribute |
| `attribute_value` | Score given by scout to the attribute |
| `potential_label` | 🎯 Target variable (average / highlighted) |

---

## 🛣️ Project Pipeline
Read Data → Merge → Preprocessing → Pivot Table →

Label Encoding → Scaling → Modelling → Tuning → SMOTE → Threshold Tuning

### Steps
| Step | Description |
|---|---|
| 1 | Import libraries & read datasets |
| 2 | Merge on 4 keys |
| 3 | Drop goalkeeper (position_id=1) & below_average label |
| 4 | Pivot table — 1 row per player |
| 5 | Label encoding + StandardScaler |
| 6 | Train 5 models & compare |
| 7a | Hyperparameter tuning (ROC-AUC) |
| 7b | Hyperparameter tuning (Recall) |
| 8 | Feature importance plot |
| 9 | SMOTE oversampling |
| 10 | SMOTE + Threshold tuning |

---

## 🏆 Final Model: CatBoost + SMOTE + Threshold (0.35)

| Metric | Default CatBoost | Final Model | Change |
|---|---|---|---|
| Recall | 0.4606 | **0.7152** | +56% 🚀 |
| F1 | 0.5852 | **0.6936** | +18% ✅ |
| Precision | 0.8933 | 0.6887 | -23% ⚠️ |
| ROC-AUC | 0.8924 | **0.8943** | +0.2% ✅ |

### Why CatBoost + SMOTE + Threshold?
- **SMOTE** fixes the 4:1 class imbalance by generating synthetic highlighted players
- **Threshold (0.35)** catches more highlighted players by lowering the decision boundary
- **CatBoost** handles small datasets well with built-in regularization

---

## 📊 Recall Journey

| Model | Default | Tuned (AUC) | Tuned (Recall) | SMOTE | SMOTE+Threshold |
|---|---|---|---|---|---|
| Random Forest | 0.4424 | 0.4258 | 0.4591 | 0.5727 | 0.6273 |
| CatBoost | 0.4606 | 0.5667 | 0.5667 | 0.5894 | **0.7152** |
| XGBoost | 0.5500 | 0.2833 | 0.5879 | 0.6091 | 0.6091 |

---

## 🔥 Feature Importance
- **Attribute 4325** is the most decisive feature
- 3x more important than any other attribute
- Scouts should prioritize this attribute in player evaluations

---

## ⚙️ Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost imbalanced-learn
```

---

## 🚀 Usage

```bash
# Clone the repo
git clone https://github.com/caglauzumcuu/scoutium-classification.git

# Open notebook
jupyter notebook scoutium.ipynb
```

---
## 📂 File Structure

```
scoutium-classification/
│
├── 📓 scoutium.ipynb                   # Main notebook
│
├── scoutium_attributes.csv              # Player attribute scores
│ 
├── sscoutium_potential_labels.csv   # Scout potential labels
│
└── 📄 README.md                        # Project documentation
```

---

## 🧠 Key Lessons Learned
1. **Accuracy is misleading** on imbalanced datasets — use Recall, F1, ROC-AUC
2. **Optimization metric matters** — tuning for AUC vs Recall gives very different results
3. **SMOTE + Threshold** is a powerful combination for imbalanced binary classification
4. **One dominant feature** (4325) suggests potential for a simpler model
5. **Small datasets** make CV scores unstable — collect more data when possible

---

## 📚 References
- [Imbalanced Learn Documentation](https://imbalanced-learn.org)
- [CatBoost Documentation](https://catboost.ai)
- [Miuul — Scoutium Case Study](https://miuul.com)
