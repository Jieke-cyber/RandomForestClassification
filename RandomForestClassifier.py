import numpy as np
from collections import Counter
from DecisionTreeNode import DecisionTreeNode

class RandomForestClassifier:
    def __init__(self, n_trees=1000, max_depth=10, max_features=None, categorical_features=None, rng=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.categorical_features = list(categorical_features) if categorical_features is not None else []
        self.trees = []
        self.bootstraps_indices = []
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.trees = []
        self.bootstraps_indices = []

        for _ in range(self.n_trees):
            indices = self.rng.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeNode(
                depth=0,
                max_depth=self.max_depth,
                categorical_features=self.categorical_features,
                max_features=self.max_features,
                rng=self.rng
            )
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)
            self.bootstraps_indices.append(indices)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        votes = [tree.predict(x) for tree in self.trees]
        return Counter(votes).most_common(1)[0][0]
