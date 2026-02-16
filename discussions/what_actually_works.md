# 📝 50+ Experiments: What Actually Works in PS S6E2

**TL;DR:** I spent 3 weeks trying multi-seed averaging, ensembles, neural networks, and feature engineering. Here's what worked and what was a complete waste of time.

**Best LB: 0.95395** (Single CatBoost, engineered features, Kaggle environment)

---

## 🏆 Final Leaderboard of My Approaches

| Rank | Approach | OOF | LB | Gap | Time Spent |
|------|----------|-----|-----|-----|------------|
| 🥇 | Single CatBoost (Kaggle env) | 0.95549 | **0.95395** | 0.00154 | 2h |
| 🥈 | Single CatBoost (Local) | 0.95549 | 0.95372 | 0.00177 | 2h |
| 🥉 | Single XGBoost | 0.95530 | 0.95351 | 0.00179 | 2h |
| 4 | Raw features (10-seed) | 0.95537 | 0.95341 | 0.00196 | 5h |
| 5 | Weighted blend | 0.95557 | 0.95321 | 0.00236 | 8h |
| 6 | 10-seed averaging | 0.95556 | 0.95292 | 0.00264 | 12h ⏰ |
| 7 | Ensemble rank/prob | — | 0.95273 | — | 15h ⏰ |
| 8 | RealMLP (neural net) | 0.95566 | 0.94639 | 0.00927 | 20h ⏰ |

**Key insight:** The top 3 are all **simple single models** that took 2 hours each. Everything below wasted 10+ hours for worse results.

---

## ✅ What Actually Works

### 1. Single CatBoost with Optuna Tuning
- **Best approach by far**
- LB: 0.95372 (local), 0.95395 (Kaggle)
- Time: ~2 hours
- OOF-LB gap: 0.00177 (stable)

**Why it works:**
- CatBoost handles categorical features natively
- Optuna finds good hyperparameters automatically
- Small gap means good generalization
- Simple = less room for mistakes

### 2. Feature Engineering (Domain Knowledge)
- Added 12 engineered features
- **LB improvement: +0.00031** vs raw features
- Based on medical domain knowledge

**Best features:**
- Rate Pressure Product (BP × Max HR)
- MaxHR Relative to Age
- HR Deficit
- Electrical Stress (ST depression × Slope)
- Cardiac Risk Score

**Paradox:** Features hurt OOF (-0.00013) but helped LB (+0.00031). Trust LB!

### 3. 5-Fold Cross-Validation
- More stable than 3-fold
- Better OOF than 10-fold
- Good balance of speed vs accuracy

### 4. Kaggle Environment
- **Same code, +0.00023 LB boost** vs local
- Likely due to library version differences
- Always submit from Kaggle notebooks!

### 5. Trusting LB Over OOF
- OOF can be misleading
- LB is ground truth
- Monitor **OOF-LB gap** as key metric

---

## ❌ What Doesn't Work (Wasted Time)

### 1. Multi-Seed Averaging 🚫
- **Time wasted:** 12+ hours
- **OOF improvement:** +0.00007
- **LB degradation:** -0.00080
- **Gap widening:** +49%

**Why it fails:**
- Reusing Optuna hyperparameters across seeds
- Amplifies systematic bias
- Worse generalization despite better OOF

**Lesson:** Don't multi-seed with fixed hyperparameters!

### 2. Ensembling/Stacking 🚫
- **Time wasted:** 15+ hours
- **Best ensemble LB:** 0.95273
- **Single CatBoost LB:** 0.95372 (-0.00099 better!)

**Why it fails:**
- All GBDTs are >99.7% correlated
- CatBoost vs XGBoost: 0.9988 correlation
- No model diversity = no ensemble benefit

**Lesson:** Check correlation before building ensembles!

### 3. Neural Networks (RealMLP, TabNet) 🚫
- **Time wasted:** 20+ hours
- **Best OOF:** 0.95566 (misleading!)
- **LB:** 0.94639 (disaster!)
- **OOF-LB gap:** 0.00927 (5× larger than CatBoost)

**Why it fails:**
- Massive overfitting on tabular data
- Gap way too large
- OOF improvements don't translate to LB

**Lesson:** Stick to tree-based models for tabular data!

### 4. Target Encoding 🚫
- **OOF degradation:** -0.00019
- CatBoost already handles categoricals well
- Adds complexity for no gain

### 5. 10-Fold CV 🚫
- Slower than 5-fold
- Lower OOF than 5-fold (0.95535 vs 0.95549)
- No LB benefit

---

## 📊 Correlation Analysis: Why Ensembling Failed

| Model Pair | Spearman Correlation |
|------------|---------------------|
| CatBoost vs XGBoost | 0.9988 |
| CatBoost vs LightGBM | 0.9988 |
| raw13 vs engineered (CatBoost) | 0.9987 |
| XGBoost vs LightGBM | 0.9978 |

**All models learn the same function!**

For ensembling to work, you need **< 0.85 correlation**. At >0.997, you're just averaging the same predictions.

---

## 🎯 My Winning Recipe

```python
# 1. Feature Engineering
features = RAW_13_FEATURES + EXPERT_FEATURES + DOMAIN_FEATURES  # 25 total

# 2. Single CatBoost
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    # ... other Optuna-tuned params
)

# 3. 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

# 4. Submit from Kaggle notebook
# → LB: 0.95395
```

**Total time:** ~4 hours (2h Optuna tuning + 2h feature engineering)

---

## 📈 OOF-LB Gap Analysis

The gap is your **most important metric** for detecting overfitting:

| Approach | OOF | LB | Gap | Overfitting? |
|----------|-----|-----|-----|--------------|
| Single CatBoost | 0.95549 | 0.95372 | 0.00177 | ✅ Good |
| XGBoost | 0.95530 | 0.95351 | 0.00179 | ✅ Good |
| Raw features (10-seed) | 0.95537 | 0.95341 | 0.00196 | ✅ Acceptable |
| Weighted blend | 0.95557 | 0.95321 | 0.00236 | ⚠️ Warning |
| 10-seed averaging | 0.95556 | 0.95292 | 0.00264 | ⚠️ Warning |
| RealMLP | 0.95566 | 0.94639 | 0.00927 | ❌ Severe |

**Rule of thumb:**
- Gap < 0.002: Excellent generalization
- Gap 0.002-0.003: Acceptable
- Gap > 0.003: Overfitting detected

---

## 🔬 Detailed Results Table

| Submission | Models | Seeds | Features | OOF | LB | Gap | Hours |
|------------|--------|-------|----------|-----|-----|-----|-------|
| cat_eng_kfold_kaggle | CatBoost | 1 | 25 | 0.95549 | **0.95395** | 0.00154 | 2 |
| cat_eng_kfold | CatBoost | 1 | 25 | 0.95549 | 0.95372 | 0.00177 | 2 |
| xgb_eng_kfold | XGBoost | 1 | 25 | 0.95530 | 0.95351 | 0.00179 | 2 |
| cat_raw13_10seed | CatBoost | 10 | 13 | 0.95537 | 0.95341 | 0.00196 | 5 |
| cat_eng | CatBoost | 1 | 25 | — | 0.95347 | — | 1 |
| blend_catms_raw13 | 2 models | — | mixed | 0.95557 | 0.95321 | 0.00236 | 8 |
| cat_multiseed_eng_kfold | CatBoost | 10 | 25 | 0.95556 | 0.95292 | 0.00264 | 12 |
| ensemble_rank | 3 models | — | 25 | — | 0.95273 | — | 15 |
| ensemble_prob | 3 models | — | 25 | — | 0.95273 | — | 15 |
| realmlp_clean | RealMLP | 1 | 13 | 0.95566 | 0.94639 | 0.00927 | 20 |

---

## 💡 Key Lessons Learned

### Lesson 1: Simple > Complex
- Single model (2h) > Ensemble (15h)
- Lower complexity = smaller gap = better LB

### Lesson 2: Trust LB, Not OOF
- RealMLP: Best OOF (0.95566), worst LB (0.94639)
- Feature engineering: Hurt OOF, helped LB
- OOF-LB gap is the true signal

### Lesson 3: Check Correlation Before Ensembling
- All GBDTs >99.7% correlated
- Ensembling identical predictions = waste of time
- Need < 0.85 correlation for diversity

### Lesson 4: Multi-Seed Can Hurt
- Helped OOF: +0.00007
- Hurt LB: -0.00080
- Don't reuse hyperparameters across seeds

### Lesson 5: Kaggle Environment Matters
- +0.00023 LB boost from Kaggle vs local
- Library versions affect results
- Always submit from Kaggle

---

## 🎯 Recommended Workflow

### Phase 1: Baseline (1 hour)
1. Load data
2. Basic feature engineering
3. Single CatBoost with default params
4. → Get baseline LB score

### Phase 2: Tuning (2 hours)
1. Optuna hyperparameter search
2. 5-fold CV
3. Monitor OOF-LB gap
4. → Optimize for gap, not OOF

### Phase 3: Feature Engineering (2 hours)
1. Domain knowledge features
2. Submit each feature set
3. Trust LB over OOF
4. → Keep features that help LB

### Phase 4: Final Model (1 hour)
1. Train on full data
2. Submit from Kaggle notebook
3. → Get Kaggle environment boost

**Total time: 6 hours → 0.95395 LB**

---

## ❓ FAQ

**Q: Should I try XGBoost or LightGBM instead of CatBoost?**
A: They're all 99.8% correlated. CatBoost is slightly better (0.95372 vs 0.95351). Not worth the effort to switch.

**Q: Is 10-fold CV better than 5-fold?**
A: No. I got worse OOF (0.95535 vs 0.95549) and no LB benefit.

**Q: Should I ensemble different seeds?**
A: No! It hurt my LB by -0.00080 despite helping OOF.

**Q: Why did neural networks fail so badly?**
A: Tabular data + small gap requirement = tree models win. Neural nets overfit (gap = 0.00927 vs 0.00177).

**Q: How do I know if my features are good?**
A: Submit and check LB. OOF can be misleading. My features hurt OOF but helped LB.

**Q: What's the most important metric?**
A: OOF-LB gap. Minimize the gap, not maximize OOF.

---

## 🏁 Final Recommendations

### ✅ Do This:
1. **Start simple** - Single CatBoost beats everything
2. **Use Optuna** - Automated hyperparameter tuning
3. **Engineer features** - Domain knowledge helps
4. **Trust LB** - OOF can mislead
5. **Monitor gap** - < 0.002 is great
6. **Submit from Kaggle** - +0.00023 boost

### ❌ Don't Do This:
1. **Multi-seed averaging** - Hurts LB
2. **Ensembling** - Models too correlated
3. **Neural networks** - Overfit badly
4. **Chasing OOF** - Trust LB instead
5. **Complex solutions** - Simple wins

---

## 💬 Discussion

**What's your experience?**
- What's your best single model LB?
- Did ensembling help or hurt you?
- What's your OOF-LB gap?
- Any other "simple > complex" findings?

I'd especially love to hear from top performers - are you using ensembles or single models?

---

## 📚 Related Posts

I've written detailed analyses on:
1. **Multi-seed averaging trap** (OOF +0.00007 → LB -0.00080)
2. **Why ensembling fails** (>99.7% correlation analysis)
3. **Feature engineering paradox** (Hurts OOF, helps LB)

Let me know if you'd like me to share those too!

---

## 🎓 Takeaway

**Sometimes the best solution is the simplest one.**

I wasted 40+ hours on:
- Multi-seed averaging (hurt LB by -0.00080)
- Ensembling (hurt LB by -0.00099)
- Neural networks (hurt LB by -0.00733)

The winner? **Single CatBoost trained in 2 hours.**

Don't make my mistakes. Start simple, monitor the gap, trust the LB.

---

If this saved you from wasting time, please upvote!

And if you found success with ensembles or multi-seed, please share your approach—I'd love to learn what I missed!

**Best of luck in the competition! 🚀**
