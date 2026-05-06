# Community Detection

Community Detection identifies groups of nodes in a graph that are more densely connected internally than externally. This implementation uses a label propagation approach.

## Algorithm

1. Initialize each node with a unique label

2. Iteratively update node labels:
   - Collect neighbor labels
   - Perform weighted voting based on edge strengths
   - Assign most influential label to node

3. Repeat until convergence or max iterations reached

4. Count number of unique communities

## UCI Dataset


## Files

- `CommunityDetection.ipynb` — graph experiments and visualization