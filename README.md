# PISA Math Score Prediction - Hackathon Solution

A comprehensive machine learning pipeline for predicting PISA math scores with maximum accuracy using ensemble methods and advanced feature engineering.

## 🎯 Goal

Predict PISA math scores for students based on ~300 features including parent income, reading scores, socioeconomic indicators, etc.

## 📊 Dataset

- **X_train.csv**: ~1 million rows × 300 columns of features
- **y_train.csv**: Math scores (target variable)
- Features include: PISA reading/science scores, socioeconomic status, parent education, wealth indicators, etc.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data (ONE TIME ONLY)

**Important**: Run this first to preprocess your data once:

```bash
cd ML
python preprocessing.py
```

This will create `X_train_processed.csv` that all models will use.

### 3. Train Models

Now you can train any model individually or all at once:

```bash
# Train individual models
python XGBoost.py
python Random_forest.py
python advanced_models.py

# Or train all models at once
python master_training.py
```

This will:
- ✅ Preprocess the data (handle missing values, scale features, engineer new features)
- ✅ Train XGBoost with hyperparameter tuning
- ✅ Train Random Forest and ExtraTrees
- ✅ Train LightGBM and CatBoost
- ✅ Create ensemble models (weighted average, median, stacking)
- ✅ Save all models to `models/` directory

### 3. Train Models

Now you can train any model individually or all at once:

```bash
# Train individual models
python XGBoost.py
python Random_forest.py
python advanced_models.py

# Or train all models at once
python master_training.py
```

### 4. Check for Overfitting

### 4. Check for Overfitting

After training, visualize train/validation metrics:

```bash
cd ML
python check_overfitting.py
```

This will generate plots for each model showing training vs validation performance.

### 5. Generate Predictions

```bash
cd ML
python predict.py ../data/X_test.csv
```

Predictions will be saved to `results/predictions.csv`

## 📁 Project Structure

```
Hackaton/
├── data/
│   ├── X_train.csv              # Training features
│   ├── y_train.csv              # Training targets
│   ├── X_test.csv               # Test features (for prediction)
│   ├── X_train_processed.csv   # Preprocessed training data (generated)
│   └── preprocessor.pkl         # Fitted preprocessor (generated)
├── ML/
│   ├── data_exploration.py      # Exploratory data analysis
│   ├── preprocessing.py         # Data preprocessing & feature engineering
│   ├── XGBoost.py              # XGBoost model with Optuna tuning
│   ├── Random_forest.py        # Random Forest & ExtraTrees models
│   ├── advanced_models.py      # LightGBM & CatBoost models
│   ├── ensemble.py             # Ensemble & stacking models
│   ├── master_training.py      # Main training pipeline
│   └── predict.py              # Prediction script
├── models/                      # Saved models (generated)
├── results/                     # Predictions (generated)
└── requirements.txt             # Python dependencies
```

## 🔧 Key Features

### 1. Advanced Preprocessing
- **Missing value imputation**: KNN imputer for numeric features
- **Feature scaling**: Robust scaling (resilient to outliers)
- **Correlation removal**: Drops highly correlated features (>0.95)
- **Categorical encoding**: Label encoding for categorical features
- **Feature selection**: Optional mutual information-based selection

### 2. Feature Engineering
- **PISA score aggregations**: Mean, std, max, min of reading/science scores
- **Socioeconomic indicators**: Aggregations of income, education, wealth features
- **Polynomial features**: Squared and sqrt transformations of key predictors
- **Interaction features**: Multiplicative interactions between top features

### 3. Multiple Models
- **XGBoost**: Gradient boosting with GPU support and Optuna tuning
- **Random Forest**: Classic ensemble with randomized search
- **ExtraTrees**: More randomized trees for diversity
- **LightGBM**: Fast gradient boosting with leaf-wise growth
- **CatBoost**: Gradient boosting with built-in categorical handling

### 4. Ensemble Methods
- **Weighted Average**: Optimally weighted combination of models
- **Median Ensemble**: Robust to outlier predictions
- **Stacking**: Meta-learner (Ridge) on top of base model predictions

## 🎛️ Hyperparameter Tuning

All models use automated hyperparameter optimization:
- **XGBoost & LightGBM & CatBoost**: Optuna with 30-50 trials
- **Random Forest**: RandomizedSearchCV with 20-50 iterations
- **Cross-validation**: 5-fold CV for robust evaluation

## 📈 Expected Performance

The ensemble models typically achieve the best performance by combining strengths of different algorithms. Expected metrics:
- **RMSE**: Target metric for competition
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination

## 🛠️ Advanced Usage

### Train Individual Models

```bash
cd ML

# Preprocess data only
python preprocessing.py

# Train specific models
python XGBoost.py
python Random_forest.py
python advanced_models.py

# Create ensembles
python ensemble.py
```

### Generate Predictions with All Models

```python
from predict import predict_with_all_models
predict_with_all_models('../data/X_test.csv')
```

This creates separate prediction files for each model.

### Data Exploration

```bash
cd ML
python data_exploration.py
```

Generates detailed statistics about the dataset.

## 🔍 Model Selection Strategy

1. **For maximum accuracy**: Use `stacking_model.pkl` or `ensemble_weighted.pkl`
2. **For speed**: Use `lightgbm_model.pkl` or `xgboost_model.pkl`
3. **For interpretability**: Use `random_forest_model.pkl` and check feature importance

## 💡 Tips for Best Results

1. **GPU Acceleration**: If you have a CUDA-capable GPU, XGBoost, LightGBM, and CatBoost will automatically use it
2. **More Tuning**: Increase `n_trials` in model files for better hyperparameters (takes longer)
3. **Feature Engineering**: Add domain-specific features in `preprocessing.py`
4. **Ensemble Tuning**: Adjust ensemble weights in `ensemble.py`

## 📊 Monitoring Training

The training pipeline will print:
- Data shapes at each step
- Cross-validation scores
- Feature importance rankings
- Training/validation metrics
- Best hyperparameters found

## 🐛 Troubleshooting

### Out of Memory
- Reduce `use_knn_imputer=False` in preprocessing
- Use smaller datasets for tuning, then retrain on full data
- Reduce `n_trials` in hyperparameter tuning

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### GPU Not Detected
- Install CUDA toolkit and cuDNN
- Install GPU versions: `pip install xgboost[gpu] lightgbm[gpu]`

## 📝 Next Steps

1. Run `master_training.py` to train all models
2. Check `models/` directory for saved models
3. Use `predict.py` with your test data
4. Submit `results/predictions.csv` to the competition

## 🏆 Competition Strategy

The stacking ensemble typically performs best because it:
- Combines diverse models (tree-based + gradient boosting)
- Learns optimal weighting through meta-model
- Reduces variance through ensemble averaging
- Captures different patterns with different algorithms

Good luck with your hackathon! 🚀

