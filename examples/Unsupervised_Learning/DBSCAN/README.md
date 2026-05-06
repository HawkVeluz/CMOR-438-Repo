# DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised clustering algorithm that groups points based on density rather than predefined cluster counts. It is capable of detecting arbitrary-shaped clusters and identifying outliers as noise.


## Algorithm

1. Select parameters:
   - ε (eps): radius for neighborhood search
   - min_samples: minimum points required to form a dense region

2. For each unvisited point:
   - Find all points within ε distance

3. Classify point:
   - Core point → enough neighbors
   - Border point → reachable from a core point
   - Noise → insufficient density

4. Expand clusters using BFS-style propagation from core points

5. Repeat until all points are visited

## UCI Dataset

## Files

- `DBSCAN.ipynb` — notebook