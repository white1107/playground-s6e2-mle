# 🔬 Why Ensembling Fails in PS S6E2: All GBDTs Are >99.7% Correlated

**TL;DR:** I spent weeks building stacking, blending, and rank averaging ensembles. **All of them performed worse than a single CatBoost model.** Here's why.

---

## 🏗️ What I Built

I tried every ensembling technique I know:

1. **Stacking** with LogisticRegression meta-learner
2. **Probability blending** (0.5 CatBoost + 0.3 XGBoost + 0.2 LightGBM)
3. **Rank averaging** (average of rank-normalized predictions)
4. **Weighted blend** (0.8 engineered + 0.2 raw features)
5. **10-seed multi-model averaging**

---

## 📊 The Brutal Results

| Approach | Models Used | LB Score | vs Single Cat |
|----------|-------------|----------|---------------|
| **Single CatBoost** | 1 | **0.95372** | — |
| Single XGBoost | 1 | 0.95351 | -0.00021 |
| Weighted Blend | 2 | 0.95321 | -0.00051 |
| 10-seed CatBoost | 10 | 0.95292 | -0.00080 |
| Ensemble Rank | 3 | 0.95273 | -0.00099 |
| Ensemble Prob | 3 | 0.95273 | -0.00099 |

### Key Finding:

**The more models I added, the WORSE my LB score got!**

The simplest solution (single CatBoost) beat every complex ensemble I built.

---

## 🔍 Root Cause: Zero Model Diversity

I calculated Spearman correlations between all my models' predictions:

| Model Pair | Spearman Correlation |
|------------|----------------------|
| CatBoost vs XGBoost | **0.9988** |
| CatBoost vs LightGBM | **0.9988** |
| raw13 vs eng (CatBoost) | **0.9987** |
| XGBoost vs LightGBM | **0.9978** |

### 😱 These are INSANELY high correlations!

For context:
- ✅ Good ensemble diversity: **< 0.85** correlation
- ⚠️ Weak diversity: 0.85 - 0.95 correlation
- ❌ No diversity: **> 0.95** correlation
- 🔥 This competition: **> 0.997** correlation!

---

## 📉 Visualizing the Problem

Here's what my ensemble looked like:

```
Prediction distributions for test ID=0:

CatBoost:   0.7234 ██████████████████████████
XGBoost:    0.7198 █████████████████████████▉
LightGBM:   0.7211 ██████████████████████████

Average:    0.7214 ██████████████████████████
                   ↑ Basically the same!
```

**All models are learning the exact same function.** Averaging them adds no new information—it just averages out random noise, which actually hurts because that "noise" might be capturing real signal.

---

## 💡 Why Is Correlation So High?

### Theory 1: Dataset Is Too "Easy"
- 630K training rows
- Only 13 features
- Clear linear relationships
- All tree models converge to the optimal decision boundaries

### Theory 2: All GBDTs Use Similar Splitting Logic
- CatBoost, XGBoost, LightGBM are all gradient boosting
- They all build decision trees
- Same feature importance rankings
- Same prediction patterns

### Theory 3: Strong Feature Dominance
Some features (like `Thallium`, `Number of vessels fluro`, `Chest pain type`) are so predictive that **all models rely heavily on them**, leading to identical predictions.

---

## 🎯 What Actually Works

I tested simpler approaches and found:

### ✅ Works:
1. **Single CatBoost** with Optuna tuning → **0.95372** LB
2. **Good feature engineering** (domain knowledge) → +0.00031 LB
3. **5-fold CV** (stable OOF estimates)
4. **Submitting from Kaggle notebooks** → +0.00023 LB

### ❌ Doesn't Work:
1. **Stacking** (meta-learner can't find patterns)
2. **Blending** (weighted average of identical predictions)
3. **Multi-seed averaging** (amplifies overfitting)
4. **Neural networks** (TabNet, RealMLP) → massive overfitting

---

## 📈 When Does Ensembling Actually Help?

Ensembling works when you have **true model diversity**:

**Different algorithm families:**
- ❌ CatBoost + XGBoost + LightGBM (all GBDTs) → 0.998 correlation
- ✅ GBDT + Neural Net + Linear Model → potentially < 0.9 correlation

**Different feature sets:**
- ❌ Raw features vs engineered features (CatBoost) → 0.9987 correlation
- ✅ Text features vs image features vs tabular features → low correlation

**Different preprocessing:**
- ❌ Same scaler, same encoding → high correlation
- ✅ Different transformations, different representations → low correlation

---

## 🧪 How to Check Model Diversity

Before building an ensemble, **always check correlation**:

```python
from scipy.stats import spearmanr

# Get predictions from your models
pred_cat = model_catboost.predict_proba(X_test)[:, 1]
pred_xgb = model_xgboost.predict_proba(X_test)[:, 1]

# Calculate correlation
corr, _ = spearmanr(pred_cat, pred_xgb)
print(f"Correlation: {corr:.4f}")

# Decision rule:
if corr > 0.97:
    print("❌ Too correlated! Ensembling won't help.")
elif corr > 0.90:
    print("⚠️ Weak diversity. Small ensemble gain expected.")
else:
    print("✅ Good diversity! Ensembling should help.")
```

In this competition, I got:
```
Correlation: 0.9988
❌ Too correlated! Ensembling won't help.
```

But I built the ensemble anyway (and regretted it).

---

## 🏁 Lessons Learned

1. **Check correlation BEFORE building ensembles**
   - If > 0.97, don't bother
   - Save your time for feature engineering instead

2. **Simple models often win in Playground Series**
   - Single well-tuned GBDT > complex ensemble
   - Smaller OOF-LB gap = better generalization

3. **Diversity is everything in ensembling**
   - Same algorithm family = no diversity
   - Need fundamentally different approaches

4. **Trust the data, not your intuition**
   - I "felt" that ensembling should help
   - The correlation numbers told me it wouldn't
   - The LB confirmed the numbers were right

---

## 💬 Discussion

**Questions for the community:**

1. What's your CatBoost vs XGBoost correlation? (Mine: 0.9988)
2. Did anyone successfully ensemble in this competition?
3. What's your single model best LB? (Mine: 0.95372)
4. Has anyone tried truly different model families (e.g., TabPFN, AutoGluon)?

**My hypothesis:** The top of the leaderboard is probably dominated by single well-tuned GBDTs, not ensembles. Would love to hear from top performers!

---

## 📝 Reproducibility

**Check your own model correlation:**

```python
import numpy as np
from scipy.stats import spearmanr

# Train two models
cat_preds = catboost_model.predict_proba(test)[:, 1]
xgb_preds = xgboost_model.predict_proba(test)[:, 1]

# Check correlation
corr = spearmanr(cat_preds, xgb_preds)[0]
print(f"Spearman correlation: {corr:.4f}")

# If corr > 0.97, don't ensemble!
```

---

## 🎓 Takeaway

**Not all competitions need ensembling.**

Sometimes the simplest solution is the best:
- Single CatBoost: **0.95372** 🏆
- Fancy ensemble: 0.95273 ❌

Focus on what matters:
1. Feature engineering
2. Hyperparameter tuning
3. Good cross-validation
4. Understanding your data

**If your models are >99.7% correlated, ensembling is a waste of time.**

---

If this saved you from wasting time on ensembles, please upvote!

And if you **did** successfully ensemble in this competition, please share how you achieved diversity—I'd love to learn!
