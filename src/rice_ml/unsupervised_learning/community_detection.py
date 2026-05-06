import numpy as np

class CommunityDetection:
    """
    Label Propagation Algorithm for community detection in graphs.
    Parameters
    ----------
    max_iters: int, default=100
        Maximum number of iterations to run the algorithm.
    Attributes
    ----------
    labels_: array, shape (n_samples,)
        The community label assigned to each node after training.
    """

    def __init__(self, max_iters=100):
        self.max_iters = max_iters
        self.labels_ = None
        self.n_communities_ = None

    def train(self, A):
        """
         Train Label Propagation algorithm on given data.
        Parameters
        ----------
        A: adjacency matrix (n x n), assumed unweighted or weighted graph
            The input data to train the model on.
        Attributes
        ----------
        labels_: ndarray of shape (n_samples,)
            The community labels for each node in the graph.
        """
        A = np.array(A)
        n = A.shape[0]

        # initialize each node with its own label
        labels = np.arange(n)

        for _ in range(self.max_iters):
            prev_labels = labels.copy()

            for i in range(n):
                neighbors = np.where(A[i] > 0)[0]

                if len(neighbors) == 0:
                    continue

                neighbor_labels = labels[neighbors]
                weights = A[i, neighbors]

                label_scores = {}

                for lbl, w in zip(neighbor_labels, weights):
                    label_scores[lbl] = label_scores.get(lbl, 0) + w

                # choose label with highest total weight
                if len(label_scores) > 0:
                    labels[i] = max(label_scores, key=label_scores.get)

            # stop if converged
            if np.array_equal(labels, prev_labels):
                break

        self.labels_ = labels
        self.n_communities_ = len(np.unique(labels))
        return self

    def predict(self, A):
        self.train(A)
        return self.labels_