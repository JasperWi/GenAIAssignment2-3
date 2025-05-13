from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv

class BinaryCLT:

    def __init__(self, data, root: int = None, alpha: float = 0.01):
        """
        Initialize and learn the Chow-Liu Tree structure and parameters.
        """
        self.data = data
        self.alpha = alpha
        self.n, self.d = data.shape
        self.root = np.random.randint(self.d) if root is None else root
        self.tree = None
        self.log_params = None
        self._learn_structure()
        self._learn_parameters()

    def _learn_structure(self):
        """
        Learn tree structure using the Chow-Liu algorithm.
        """
        # TODO: Compute mutual information matrix
        # TODO: Build MST from negative mutual information
        # TODO: Use breadth-first order to set parents
        pass

    def _learn_parameters(self):
        """
        Learn CPTs using Laplace smoothing and convert to log domain.
        """
        # TODO: Count co-occurrences with Laplace correction
        # TODO: Store log CPTs in self.log_params
        self.log_params = []
        for i in range(self.d):
            if i == self.root:
                # Root node has no parent 
                counts = np.zeros((2, 1))
                for j in range(self.n):
                    counts[self.data[j, i]] += 1
                # Laplace smoothing
                counts += 2 * self.alpha
                # Change shape to (2, 2) for consistency
                counts = np.repeat(counts[:, np.newaxis], 2, axis=1)
            else:
                parent = self.tree[i]
                # Count co-occurrences
                counts = np.zeros((2, 2))
                for j in range(self.n):
                    counts[self.data[j, parent], self.data[j, i]] += 1
                # Laplace smoothing
                counts += self.alpha
            # Convert to log domain
            log_prob = np.log(counts / (4 * self.alpha + counts.sum(axis=1, keepdims=True)))
            self.log_params.append(log_prob)
        return self.log_params

    def get_tree(self):
        """
        Return the list of parents for each variable in the tree.
        """
        return self.tree

    def get_log_params(self):
        """
        Return the log CPTs.
        """
        return self.log_params

    def log_prob(self, x, exhaustive: bool = False):
        """
        Compute log-probability of observed/marginalized samples.
        """
        lp = []
        for query in x:
            if exhaustive:
                # TODO: Exhaustive inference over missing values
                pass
            else:
                # TODO: Efficient inference using variable elimination
                pass
        return np.array(lp)

    def sample(self, n_samples: int):
        """
        Generate i.i.d. samples from the CLT distribution using ancestral sampling.
        """

        samples = []
        for i in range(n_samples):
            sample = np.zeros(self.d, dtype=int) * -1
            sample[self.root] = 1 if np.random.rand() < np.exp(self.log_params[self.root][1][1]) else 0
            num_variables_set = 1
            while num_variables_set < self.d:
                for j in range(self.d):
                    parent = self.tree[j]
                    if j != self.root and sample[j] == -1 and sample[parent] != -1:
                        # Sample from the conditional distribution
                        prob = np.exp(self.log_params[j][sample[parent]][1])
                        sample[j] = 1 if np.random.rand() < prob else 0
                        num_variables_set += 1
            samples.append(sample)

        return np.array(samples)

# === Utility for loading datasets ===
def load_csv_dataset(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        dataset = np.array(list(reader)).astype(np.float_)
    return dataset
