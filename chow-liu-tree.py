from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv
import networkx as nx
import matplotlib.pyplot as plt

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
        Learn tree structure using the Chow-Liu algorithm:
        1. Compute mutual information matrix
        2. Build MST from negative mutual information
        3. Use breadth-first order to determine parent relationships
        """
        mi_matrix = np.zeros((self.d, self.d))

        # Compute pairwise mutual information
        for i in range(self.d):
            for j in range(i + 1, self.d):
                counts = np.zeros((2, 2))
                for k in range(self.n):
                    xi = int(self.data[k, i])
                    xj = int(self.data[k, j])
                    counts[xi, xj] += 1
                counts += self.alpha  # Laplace smoothing
                probs = counts / counts.sum()

                mi = 0.0
                for xi in [0, 1]:
                    for xj in [0, 1]:
                        p_ij = probs[xi, xj]
                        p_i = probs[xi, :].sum()
                        p_j = probs[:, xj].sum()
                        if p_ij > 0:
                            mi += p_ij * np.log(p_ij / (p_i * p_j))

                mi_matrix[i, j] = mi_matrix[j, i] = mi

        # Build MST from negative mutual information
        neg_mi = -mi_matrix
        mst = minimum_spanning_tree(neg_mi).toarray()

        # Use BFS to determine parents in tree
        _, parents = breadth_first_order(mst + mst.T, directed=False, i_start=self.root, return_predecessors=True)
        self.tree = [-1 if i == self.root else parents[i] for i in range(self.d)]

    def _learn_parameters(self):
        self.log_params = []
        for i in range(self.d):
            if i == self.root:
                counts = np.zeros((2,))
                for j in range(self.n):
                    xi = int(self.data[j, i])
                    counts[xi] += 1
                counts += 2 * self.alpha
                probs = counts / counts.sum()
                log_prob = np.log(probs)
                log_prob = np.tile(log_prob[None, :], (2, 1))  # (2,2)
            else:
                parent = self.tree[i]
                counts = np.zeros((2, 2))
                for j in range(self.n):
                    pi = int(self.data[j, parent])
                    xi = int(self.data[j, i])
                    counts[pi, xi] += 1
                counts += self.alpha
                log_prob = np.log(counts / counts.sum(axis=1, keepdims=True))
            self.log_params.append(log_prob)
        return self.log_params

    def get_tree(self):
        """
        Return the list of parents for each variable in the tree.
        """
        return self.tree

    def get_log_params(self):
        """
        Efficiently return the learned log CPTs as a (d, 2, 2) NumPy array.
        Assumes _learn_parameters has already populated self.log_params.
        """
        return np.stack(self.log_params, axis=0)


    def log_prob(self, x, exhaustive: bool = False):
        """
        Compute log-probability of observed/marginalized samples.
        
        Parameters:
            x (ndarray): N x D matrix where rows are binary vectors (0/1) with optional np.nan
            exhaustive (bool): Whether to perform exhaustive marginalization over missing values
        
        Returns:
            log_probs (ndarray): N x 1 vector of log-probabilities
        """
        lp = []
        for query in x:
            if exhaustive:
                # Identify missing indices
                missing_indices = np.where(np.isnan(query))[0]

                # List to collect log-probs of all completions
                log_probs = []

                # Try all combinations of values for missing variables
                for values in itertools.product([0, 1], repeat=len(missing_indices)):
                    filled = query.copy()
                    filled[missing_indices] = values
                    logp = 0.0
                    for i in range(self.d):
                        xi = int(filled[i])
                        parent = self.tree[i]
                        if parent == -1:
                            logp += self.log_params[i][0, xi]  # root: marginal prob
                        else:
                            pi = int(filled[parent])
                            logp += self.log_params[i][pi, xi]
                    log_probs.append(logp)

                lp.append(logsumexp(log_probs))

            else:
                # Assume all values observed
                logp = 0.0
                for i in range(self.d):
                    xi = int(query[i])
                    parent = self.tree[i]
                    if parent == -1:
                        logp += self.log_params[i][0, xi]
                    else:
                        pi = int(query[parent])
                        logp += self.log_params[i][pi, xi]
                lp.append(logp)

        return np.array(lp).reshape(-1, 1)
    
    def sample(self, n_samples: int):
        """
        Generate i.i.d. samples from the CLT distribution using ancestral sampling.
        """

        samples = []
        for i in range(n_samples):
            sample = -1 * np.ones(self.d, dtype=int)
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


def plot_tree(tree):
    G = nx.DiGraph()
    for child, parent in enumerate(tree):
        if parent != -1:
            G.add_edge(parent, child)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", arrows=True)
    plt.title("Chow-Liu Tree Structure")
    plt.show()


nltcs_data = load_csv_dataset("nltcs.train.data")

model_nltcs = BinaryCLT(nltcs_data, root=0, alpha=0.01)
# Get tree and CPTs
tree_structure = model_nltcs.get_tree()
log_cpts = model_nltcs.get_log_params()

print("\n--- Tree Structure (Parent List) ---")
print(tree_structure)

print("\n--- Log CPTs (shape:", log_cpts.shape, ") ---")
print(log_cpts)