# Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for binary classification problems. Despite its name, it is a classification method that models the probability that an input belongs to a particular class.

It works by applying a linear function to the input features and passing the result through a sigmoid function, which maps values to the range ([0, 1]).

## Algorithm

Logistic Regression learns a decision boundary by optimizing weights that minimize classification error using gradient descent.

---

### 1: Initialize parameters

We initialize the weight vector:

$$
w = [w_1, w_2, ..., w_d, b]
$$

where:

* ( w_1, ..., w_d ) are feature weights
* ( b ) is the bias term (stored as the last weight)

---

### 2: Compute linear output

For each input sample ( x_i ):

$$
z_i = w^T x_i
$$

---

### 3: Apply sigmoid function

The sigmoid function converts the linear output into a probability:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

So the predicted probability is:

$$
\hat{y}_i = \sigma(z_i)
$$

---

### 4: Compute prediction error

For each sample:

$$
\text{error}_i = \hat{y}_i - y_i
$$

---

### 5: Update weights (SGD)

Feature weights are updated as:

$$
w_j := w_j - \eta (\hat{y}*i - y_i) x*{ij}
$$

Bias term is updated as:

$$
b := b - \eta (\hat{y}_i - y_i)
$$

where:

* ( \eta ) is the learning rate

---

### 6: Compute loss

Logistic loss (cross-entropy) is:

$$
L = -\sum \left[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})\right]
$$

---

### Step 7: Repeat

Repeat steps 2–6 for multiple epochs until convergence.

---

### Final prediction

A class label is assigned using a threshold:

$$
\hat{y} =
\begin{cases}
1 & \text{if } \sigma(w^T x) \geq 0.5 \
0 & \text{otherwise}
\end{cases}
$$

## Dataset

**Breast Cance Wisconsin (Diagnostic)** — 569 Instances, 30 Features  
**Label:** 1 = Malignant, 0 = Benign

## Files
- `Logistic_Regression.ipynb` — notebook explaining the algorithm, as well as data preprocessing methods and result analysis
- `wdbc.data` — the dataset
