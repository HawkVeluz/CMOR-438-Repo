# rice_ml Examples

This directory contains Jupyter Notebooks (.ipynb) and documentation that demonstrate the use of the various algorithms written in the rice_ml package.

## Overview

There are two types of examples:

# Supervised_Learning

Models that learn from labeled data (X, y) to predict targets (classification or regression).
- Decision Trees
- Ensemble Methods (Bagging, Random Forest)
- K-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Multilayer Perceptron
- Perceptron

# Unsupervised_Learning

Models that find hidden patterns or structures in unlabeled data (X).
- PCA (Principal Component Analysis)
- DBSCAN (Density-based)
- K_Means_Clustering 
- Community_Detection (Label Propagation)

---

## Preprocessing and Evaluation
The notebooks make frequent use of scikit-learn for both preprocessing (standardization, training/test data splitting) and evaluation(scores and errors), as well as pyplot for various graphs showcasing functionality. The preprocessing is required for the rice_ml algorithms to work properly.

## Example Use

1. Install rice_ml package and dependencies.
2. Choose an algorithm.
3. Run all cells in sequential order.