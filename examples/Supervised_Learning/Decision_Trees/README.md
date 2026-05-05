# Decision Trees

A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It models decisions as a tree-like structure, where each internal node represents a test on a feature, each branch represents the outcome of that test, and each leaf node represents a final prediction.


## Algorithm

1. Training data is placed at root node.

2. For each feature and possible threshold:
    Split the dataset into two groups
    Compute impurity (Gini for classification, MSE for regression)
    Select the split that gives the best improvement (highest information gain)

3. Split data
    Divide dataset into left and right child nodes

4. Repeat recursively till stopping condition:
    Maximum depth reached  
    Minimum number of samples reached

5. Make predictions
Classification: Leaf node stores the majority class
Regression: Leaf node stores the mean value of targets

## Dataset

**Red Wine Quality** — 1599 Instances, 12 Features
**Classification Label:** 1: Quality >= 7, 0 : Quality < 7  
**Regression Label:** Quality: 0-10 range

## Files
- `Decision_Trees.ipynb` — notebook explaining the decision tree, as well as data preprocessing methods and results analysis
- `winequality-red.csv` — the dataset