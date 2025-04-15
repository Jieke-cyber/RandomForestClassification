import numpy as np

from DecisionTreeNode import DecisionTreeNode


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)  # Bootstrap
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeNode(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([self._predict_tree(x) for x in X])
        return predictions

    def _predict_tree(self, x):
        votes = [tree.predict(x) for tree in self.trees]
        return np.bincount(votes).argmax()