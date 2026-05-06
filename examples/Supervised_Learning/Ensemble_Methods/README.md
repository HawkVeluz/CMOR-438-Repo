# Ensemble Methods - Bagging and Random Forest
Ensemble methods are machine learning techniques that combine multiple models to produce a single, stronger predictor than any individual model alone. This project implements both bagging and random forest algorithms, with a classifier and regressor for each.

## Bagging Algorithm

1. Create multiple bootstrap samples from training data (sampling with replacement)

2. Train a decision tree on each sample

3. Store all trained models

4. Make predictions
    Classification: Return most common class across all trees  
    Regression: Return average prediction across all trees

## Random Forest Algorithm

1. Create multiple bootstrap samples from training data

2. For each tree, randomly select a subset of features

3. Train a decision tree on each sampled dataset and feature subset

4. Store all trained trees

5. Make predictions
    Classification: Majority vote across trees  
    Regression: Average prediction across trees

## UCI Datasets

## Files
- `Ensemble_Methods.ipynb` — notebook explaining KNN, as well as data preprocessing methods and results analysis
- `winequality-red.csv` — dataset
