# Comparison of Training Models

1. **Lite Model**
- Uses basic ensemble voting with fixed weights [2,1,2]
- Pre-configured hyperparameters (no optimization)
- Lemmatization for text preprocessing
- Uses 3 algorithms: SVC, MultinomialNB, ExtraTreesClassifier
- Fastest because no parameter tuning/optimization

2. **Legacy Model**
- Uses simple voting ensemble without weight optimization
- Porter Stemming for text preprocessing (simpler than lemmatization)
- Slightly different hyperparameters:
  - SVC with 'sigmoid' kernel
  - Fewer trees in ExtraTreesClassifier (50 vs 200)
- Medium speed due to simpler preprocessing

3. **Monarch Butterfly Optimization (MBO) Model**
- Uses nature-inspired optimization algorithm
- Optimizes 7 parameters simultaneously:
  - SVC parameters (C, gamma)
  - MultinomialNB alpha
  - Number of trees
  - Ensemble weights (w1, w2, w3)
- Population-based search with:
  - 20 butterflies
  - 30 iterations
  - Cross-validation for each evaluation
- Slowest because:
  - Runs multiple training cycles (20 butterflies × 30 iterations)
  - Each evaluation requires 5-fold cross-validation
  - Total of ~3000 model evaluations

**Summary**:
- Lite: Quick, fixed parameters
- Legacy: Traditional, basic ensemble
- MBO: Advanced optimization, but computationally intensive