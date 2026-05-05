# K-Nearest-Neighbors

K-Nearest-Neighbors (KNN) is a simple machine learning algorithm used for both classification and regression. It does not use parameters when training, and only calculates the distance between data points during prediction. This project uses Euclidian distance in particular.


## Algorithm

1. Store training data in model

2. Select K - number of neighbors

3. Compute all distances between nodes

4. Select K nearest neighbors

5. Make predictions
    Classification: Return most common class among neighbors
    Regression: Return average value of neighbors

## UCI Datasets

**Red and White Wine Quality** — 6497 Total Instances, 12 Features  
**Classification Label:** 1: Quality >= 7, 0 : Quality < 7  
**Regression Label:** Quality: 0-10 range

## Files
- `KNN.ipynb` — notebook explaining KNN, as well as data preprocessing methods and results analysis
- `winequality-red.csv` — dataset
- `winequality-white.csv` — dataset