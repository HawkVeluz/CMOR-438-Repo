# K-Means Clustering

K-Means Clustering is an unsupervised machine learning algorithm used to group data into **K** distinct clusters based on similarity.  
It is different from supervised learning in how it finds structure in the data by minimizing the distance between points and cluster centers instead of using labels.

---

## Algorithm

K-Means works by iteratively updating cluster assignments and centroids until convergence.


1. Choose number of clusters \( K \)
2. Initialize \( K \) centroids randomly from the dataset
3. Repeat until convergence or max iterations:

   - Assign each point to the nearest centroid:
   $$
   c_i = \arg\min_j \|x_i - \mu_j\|^2
   $$

   - Update each centroid as the mean of assigned points:
   $$
   \mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
   $$

4. Stop when centroids no longer change significantly (or max iterations reached)

### Objective Function (Inertia)

K-Means minimizes the within-cluster sum of squares:

$$
J = \sum_{j=1}^{K} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
$$

---

## Kaggle Dataset

- `Mall_customers.csv` - — 200 Instances, 5 Features 

---

## Files
- `K_Means_Clustering.ipynb` — notebook explaining K-Means-Clustering, as well as data preprocessing methods and results analysis
- `Mall_customers.csv` — dataset 