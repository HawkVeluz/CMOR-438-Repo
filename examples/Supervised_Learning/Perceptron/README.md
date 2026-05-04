# Perceptron

The Perceptron is a binary classification algorithm invented by Frank Rosenblatt in 1958. Based on neurons, the Perceptron weighs input signals and sends a signal when the weighted sum is high enough. This results in a linear decision boundary between two potential results.


## Algorithm

**1: Net Input**

z = wᵀx + b

**2: Activation**

Φ(z) = 1  if z > 0
Φ(z) = -1 if z ≤ 0

**3: Update Rule:**

w ← w - α(ŷ - y) · x
b ← b - α(ŷ - y)

Where:
- α = learning rate (eta)
- ŷ = predicted label
- y = true label
- x = input

## Dataset

**Breast Cance Wisconsin (Diagnostic)** — 569 Instances, 30 Features
**Label:** 1 = Malignant, -1 = Benign

## Files
- `Perceptron.ipynb` — notebook explaining the perceptron, as well as data preprocessing methods and result analysis
- `wdbc.data` — the dataset