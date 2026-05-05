# Linear Regression
Linear Regression models the relationship between input features and a continuous target variable by learning weights that minimize prediction error using gradient descent. This project uses 2 different methods to perform linear regression: calculating the closed-form solution using the Normal Equation, and using Stochastic Gradient Descent in an iterative solution.

## Algorithm

This implementation of Linear Regression uses Stochastic Gradient Descent (SGD), where model parameters are updated after each training example.

---

### 1: Initialize parameters

We initialize the weight vector:

$$
w = [w_1, w_2, ..., w_d, b]
$$

where:

* ( w_1, ..., w_d ) are feature weights
* ( b ) is the bias term (stored as the last element of the weight vector)

---

### 2: Make prediction

For each training sample ( x_i ):

$$
\hat{y}_i = w^T x_i
$$

(where the bias is included in ( w ))

---

### 3: Compute error

For each sample:

$$
\text{error}_i = \hat{y}_i - y_i
$$

---

### 4: Update weights (SGD)

Feature weights are updated as:

$$
w_j := w_j - \eta (\hat{y}*i - y_i) x*{ij}
$$

Bias term is updated as:

$$
b := b - \eta (\hat{y}_i - y_i)
$$

where:

* ( eta ) is the learning rate

---

### 5: Compute loss

Per-sample squared error:

$$
L_i = \frac{1}{2} (\hat{y}_i - y_i)^2
$$

---

### Step 6: Repeat

Repeat steps 2–5 for all training samples over multiple epochs until convergence.

## UCI Datasets

**Red and White Wine Quality** — 6497 Total Instances, 12 Features  
**Classification Label:** 1: Quality >= 7, 0 : Quality < 7  
**Regression Label:** Quality: 0-10 range

## Files
- `KNN.ipynb` — notebook explaining KNN, as well as data preprocessing methods and results analysis
- `winequality-red.csv` — dataset
- `winequality-white.csv` — dataset