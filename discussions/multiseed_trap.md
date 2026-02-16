# ⚠️ Multi-Seed Averaging is a TRAP in PS S6E2

**TL;DR:** Multi-seed averaging improved my OOF by +0.00007 but **destroyed my LB by -0.00080**. Don't make the same mistake I did!

---

## 🔬 The Experiment

I ran the exact same CatBoost model (Optuna-tuned hyperparameters, 5-fold CV, engineered features) with two approaches:

1. **Single seed** (random_state=42)
2. **10-seed averaging** (random_state=0-9, average predictions)

---

## 📊 Results

| Approach | OOF AUC | LB Score | OOF-LB Gap |
|----------|---------|----------|------------|
| **Single seed** | 0.95549 | **0.95372** | 0.00177 |
| **10-seed averaging** | 0.95556 | 0.95292 | 0.00264 |
| **Δ** | +0.00007 ✅ | **-0.00080** ❌ | +49% wider 😱 |

### Key Observations:

- ✅ Multi-seed **improved OOF** by 0.00007 (seemed promising!)
- ❌ Multi-seed **hurt LB** by 0.00080 (ouch!)
- 😱 OOF-LB gap **widened by 49%** (0.00177 → 0.00264)

---

## 💡 Why Did This Happen?

### Hypothesis: Hyperparameter Bias Amplification

My workflow was:
1. Tune hyperparameters with Optuna on **fold 0, seed 42**
2. Reuse those hyperparameters for **all 10 seeds**

**The problem:** If those hyperparameters slightly overfit fold 0, then averaging 10 models trained with the **same biased hyperparameters** amplifies that overfitting.

It's like taking 10 measurements with a **miscalibrated instrument**—averaging them doesn't fix the systematic bias, it just makes you more confident in the wrong answer!

---

## 📉 Visualizing the Damage

```
OOF Scores:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single:      0.95549 ████████████████████
Multi-seed:  0.95556 ████████████████████ (+0.00007)

LB Scores:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single:      0.95372 ████████████████████████
Multi-seed:  0.95292 ███████████████████      (-0.00080)

Gap (OOF - LB):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single:      0.00177 ███
Multi-seed:  0.00264 █████ (+49% wider!)
```

---

## 🎯 What I Learned

### ❌ Don't Do This:
- Tune hyperparameters on a single fold/seed
- Reuse those hyperparameters across multiple seeds
- Trust OOF improvements that widen the OOF-LB gap

### ✅ Do This Instead:
- Stick with a **single seed** for this competition
- If you must multi-seed, tune hyperparameters **separately for each seed**
- **Trust LB over OOF** when they disagree
- Monitor the **OOF-LB gap** as an overfitting indicator

---

## 🔍 Additional Evidence

I also tested **raw features (13 features, no engineering)** with 10-seed averaging:
- OOF: 0.95537
- LB: 0.95341
- Gap: 0.00196

Still worse than single-seed with engineered features!

---

## 💬 Discussion Questions

1. Has anyone successfully used multi-seed averaging in this competition?
2. What's your OOF-LB gap? (Mine was 0.00177 for single seed)
3. Did you tune hyperparameters separately for each seed?

---

## 📝 Reproducibility

**Single seed model:**
```python
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    # ... other Optuna-tuned params
)
# → LB: 0.95372
```

**Multi-seed averaging:**
```python
preds = []
for seed in range(10):
    model = CatBoostClassifier(
        random_seed=seed,
        # ... SAME hyperparameters as above
    )
    model.fit(X, y)
    preds.append(model.predict_proba(X_test)[:, 1])

final_pred = np.mean(preds, axis=0)
# → LB: 0.95292 (WORSE!)
```

---

## 🏁 Conclusion

**Multi-seed averaging can be a trap!**

In this competition:
- Single seed CatBoost: **0.95372** LB 🏆
- 10-seed averaging: 0.95292 LB ❌

Sometimes simpler is better. Focus on:
- Good feature engineering
- Proper hyperparameter tuning (per-seed if multi-seeding)
- Monitoring OOF-LB gap
- Trusting LB over OOF

**Hope this saves someone from making the same mistake I did!**

If you found this helpful, please upvote! And share your own multi-seed experiences in the comments below.

---

**Related experiments:**
- I also found that all GBDTs (CatBoost, XGBoost, LightGBM) are >99.7% correlated in this competition
- Ensembling/stacking also failed for the same reason
- Single CatBoost beats all complex ensembles

Let me know if you'd like me to share those findings too!
