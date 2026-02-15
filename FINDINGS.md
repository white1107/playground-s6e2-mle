# Findings - Playground Series S6E2 (Heart Disease Prediction)

## Competition Info
- Task: Binary classification (Heart Disease: Presence/Absence)
- Metric: AUC-ROC
- Train: 630,000 rows, 14 features + target
- Test: 270,000 rows
- Daily submissions: 5

## LB Scores (All Submissions)

| Submission | OOF AUC | LB Score | OOF-LB Gap |
|---|---|---|---|
| **cat_eng_kfold_kaggle** (Kaggle env, single CatBoost) | 0.95549 | **0.95395** | 0.00154 |
| cat_eng_kfold (single CatBoost, Optuna) | 0.95549 | 0.95372 | 0.00177 |
| cat (old, 3 days ago, additional 303-row data) | - | 0.95354 | - |
| cat_eng_kfold (old, 3 days ago, additional 303-row data) | - | 0.95354 | - |
| xgb_eng_kfold (single XGBoost, Optuna) | 0.95530 | 0.95351 | 0.00179 |
| cat_raw13_10seed (raw 13 features, 10-seed avg) | 0.95537 | 0.95341 | 0.00196 |
| cat_eng (no KFold) | - | 0.95347 | - |
| blend_catms_raw13 (0.8 eng + 0.2 raw) | 0.95557 | 0.95321 | 0.00236 |
| cat_multiseed_eng_kfold (10-seed avg, Optuna) | 0.95556 | 0.95292 | 0.00264 |
| ensemble_rank | - | 0.95273 | - |
| ensemble_prob | - | 0.95273 | - |
| cat_seedavg_eng_kfold (old dataset) | - | 0.95178 | - |
| realmlp_clean | 0.95566 | 0.94639 | 0.00927 |
| realmlp_eng_kfold | 0.95566 | 0.94623 | 0.00943 |
| baseline RF | - | 0.85498 | - |

**Current Best LB: 0.95395** (cat_eng_kfold run in Kaggle environment, single CatBoost with Optuna + engineered features)

---

## Key Findings

### 1. Multi-Seed Averaging HURTS on LB

| Approach | OOF AUC | LB Score |
|---|---|---|
| Single CatBoost (Optuna) | 0.95549 | **0.95372** |
| 10-seed CatBoost (Optuna) | 0.95556 | 0.95292 |

- Multi-seed averaging improved OOF by +0.00007 but **dropped LB by -0.00080**
- The OOF-LB gap widened from 0.00177 to 0.00264
- Hypothesis: Multi-seed averaging amplifies systematic bias in the Optuna-tuned hyperparameters
- **Conclusion: Don't multi-seed average for this competition**

### 2. Feature Engineering Slightly Helps (on LB)

| Features | OOF AUC | LB Score |
|---|---|---|
| Raw 13 features (10-seed) | 0.95537 | 0.95341 |
| Engineered features (single) | 0.95549 | 0.95372 |

- Feature engineering: OOF +0.00012, LB **+0.00031**
- Engineering helps more on LB than OOF - features generalize well
- BUT feature ablation showed expert features **hurt** CatBoost in OOF-only analysis:
  - RAW 13 features (single seed): 0.95528
  - + Expert features: 0.95515 (-0.00013)
  - + Target encoding: 0.95509 (-0.00019)
  - + Domain + orig stats: 0.95535 (+0.00007)
- The contradiction (hurts OOF but helps LB) may be due to regularization effect

### 3. Simple Models Beat Complex Ensembles

Ranking by LB score:
1. Single CatBoost (0.95372)
2. Single XGBoost (0.95351)
3. Single CatBoost raw features (0.95341)
4. Blend (0.95321)
5. Multi-seed CatBoost (0.95292)
6. Ensemble rank/prob (0.95273)

**Simpler models consistently have smaller OOF-LB gap and better LB scores.**

### 4. RealMLP Massively Overfits

| Model | OOF AUC | LB Score | Gap |
|---|---|---|---|
| RealMLP | **0.95566** | 0.94639 | **0.00927** |
| CatBoost | 0.95549 | 0.95372 | 0.00177 |

- RealMLP has the best OOF but worst LB among serious models
- OOF-LB gap is 5x larger than CatBoost
- Neural networks overfit badly on this tabular dataset

### 5. All GBDT Models Are Highly Correlated

| Pair | Spearman Correlation |
|---|---|
| CatBoost vs XGBoost | 0.9988 |
| CatBoost vs LightGBM | 0.9988 |
| raw13 vs eng CatBoost | 0.9987 |
| XGBoost vs LightGBM | 0.9978 |

- Correlations > 0.997 across all GBDTs
- Ensembling provides almost no diversity benefit
- All models are essentially learning the same function

### 6. Correcting Additional Dataset Helped

- Old: Used 303-row original dataset (wrong, had 270 rows)
- New: Used correct 270-row original dataset
- LB improvement: 0.95354 -> 0.95372 (+0.00018)
- Small but real improvement from fixing data quality

### 7. Kaggle Environment Effect

| Environment | Same Code | LB Score |
|---|---|---|
| Local (Ubuntu) | cat_eng_kfold | 0.95372 |
| Kaggle Notebook | cat_eng_kfold | **0.95395** |

- Exact same code, same hyperparameters, same data
- Kaggle environment yields +0.00023 better LB score
- Likely due to different library versions or numerical precision differences
- **Conclusion: Submit from Kaggle notebooks for best results**

---

## What Works

1. **Single CatBoost with Optuna tuning** - best LB score
2. **KFold CV (5-fold)** - stable OOF estimates
3. **Engineered features** (domain knowledge) - helps LB generalization
4. **GPU training** - fast iterations
5. **Kaggle environment** - submit from Kaggle notebooks for +0.00023 LB

## What Doesn't Work

1. **Multi-seed averaging** - overfits, worse LB
2. **Stacking / LR meta-learner** - models too correlated
3. **Rank blending** - no diversity to exploit
4. **Neural networks (RealMLP, TabNet, FT-Transformer)** - massive overfitting
5. **Target encoding** - hurts CatBoost (it already handles categoricals well)
6. **10-fold CV** - lower OOF than 5-fold, no LB benefit

## Hyperparameters (Best CatBoost - Optuna Tuned)

From the OOF generator with Optuna tuning on first fold.
Uses engineered features including original 270-row dataset statistics.

## Next Steps to Try

- [ ] Post-processing: calibration, threshold tuning
- [ ] Pseudo-labeling with high-confidence test predictions
- [ ] Different CatBoost hyperparameter ranges (deeper trees, lower LR)
- [ ] Feature selection (drop weakest features)
- [ ] Investigate why multi-seed hurts LB (is it the fixed hyperparams across seeds?)
- [ ] Try single-seed XGBoost with Optuna (currently 0.95351, close to CatBoost)
