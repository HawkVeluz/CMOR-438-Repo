# rice_ml

A from-scratch machine learning package written using in NumPy.  
This package provides implementations of machine learning algorithms for classification, regression, clustering, dimensionality reduction, and neural networks.

---

# Overview

**rice_ml** is a machine learning package designed to demonstrate how common algorithms work internally without relying on external ML libraries like scikit-learn.
 
---

# Implemented Algorithms

## Supervised Learning

- Linear Regression (Gradient Descent)
- Logistic Regression
- Perceptron
- Multilayer Perceptron (MLP)
- K-Nearest Neighbors (Classifier and Regressor)
- Decision Tree Classifier (Gini impurity)
- Decision Tree Regressor (MSE)
- Bagging (Classifier & Regressor)
- Random Forest (Classifier & Regressor)

## Unsupervised Learning

- K-Means Clustering (with inertia calculation)
- DBSCAN (density-based clustering)
- Principal Component Analysis (PCA)
- Community Detection (Weighted Label Propagation)

---

# Example Usage

```python
from rice_ml.knn import KNN

model = KNN(k=3)
model.train(X_train, y_train)

predictions = model.predict(X_test)