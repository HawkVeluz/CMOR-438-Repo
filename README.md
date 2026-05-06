# CMOR-438-Repo

Author: Hawk Veluz (CMOR 438, Spring 2026)
License: This project is licensed under the MIT License. See the LICENSE file for details. 

## Overview

This repository contains a machine learning package rice_ml for the CMOR 438 course at Rice University. It implements various supervised and unsupervised learning algorithms, coded from scratch using NumPy.

Project features:

- Implementations of common machine learning algorithms built from scratch using NumPy only.
- Example Jupyter notebooks for every algorithm.
- 81 unit tests covering algorithms, preprocessing, and metrics.
- CI via GitHub Actions on every push and pull request.

---

## Project Structure

```
CMOR438/

├── examples/
│   ├── Supervised_Learning/
│   │   ├── Decision Trees/
│   │   ├── Ensembles_Methods/
│   │   ├── K_Nearest_Neighbors/
│   │   ├── Linear_Regression/
│   │   ├── Logistic_Regression/
│   │   ├── Multilayer_Perceptron
│   │   ├── Perceptron/
│   └── Unsupervised_Learning/
│       ├── Community_Detection/
│       ├── DBSCAN/
│       ├── K_Means_Clustering/
│       ├── PCA/
├── src/
│   └── rice_ml/
│       ├── __init__.py
│       ├── supervised_learning/
│       └── unsupervised_learning/
├── tests/
│   ├── test_community_detection.py
│   ├── test_dbscan.py
│   ├── test_decision_trees.py
│   ├── test_ensemble_methods.py
│   ├── test_kmc.py
│   ├── test_knn.py
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   ├── test_mlp.py
│   ├── test_pca.py
│   ├── test_perceptron.py
├── pyproject.toml
├── pytest.ini
├── requirements.txt
└── README.md
```

## Testing
There are 81 different unit tests for the various algorithms in the rice_ml package. In order to run the tests, ensure that all requirements are installed, and run

```bash
pytest -v
```

## Installation

```bash
git clone https://github.com/HawkVeluz/CMOR-438-Repo.git
cd rice_ml
pip install -e .
```

