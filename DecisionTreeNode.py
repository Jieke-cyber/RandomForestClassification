import numpy as np
from collections import Counter

class DecisionTreeNode:
    def __init__(self, depth=0, max_depth=5, categorical_features=None, max_features=None, rng = None):
        self.depth = depth
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.category = None
        self.label = None
        self.categorical_features = list(categorical_features) if categorical_features is not None else []
        self.n_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng()

    def most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        counts = np.array(list(Counter(y).values()))
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_value = None
        split_type = None
        current_gini = self.gini_impurity(y)

        n_total_features = X.shape[1]
        features_to_consider = self.rng.choice(n_total_features, self.n_features, replace=False) if self.n_features else range(n_total_features)

        for feature in features_to_consider:
            column = X[:, feature]
            unique_values = np.unique(column)

            for value in unique_values:
                if feature in self.categorical_features:
                    left_indices = column == value
                    right_indices = column != value
                else:
                    left_indices = column <= value
                    right_indices = column > value

                y_left = y[left_indices]
                y_right = y[right_indices]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self.gini_impurity(y_left)
                gini_right = self.gini_impurity(y_right)
                weighted_gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right
                gain = current_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = value
                    split_type = "categorical" if feature in self.categorical_features else "numeric"

        return best_feature, best_value, best_gain, split_type

    def fit(self, X, y):
        if len(y) == 0 or self.depth >= self.max_depth or len(set(y)) == 1:
            self.label = self.most_common_label(y) if len(y) > 0 else None
            return

        feature, value, gain, split_type = self.best_split(X, y)
        if gain == 0 or feature is None:
            self.label = self.most_common_label(y)
            return

        self.feature_index = feature
        if split_type == "categorical":
            self.category = value
            left_indices = X[:, feature] == value
            right_indices = X[:, feature] != value
        else:
            self.threshold = value
            left_indices = X[:, feature] <= value
            right_indices = X[:, feature] > value

        self.left = DecisionTreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            categorical_features=self.categorical_features,
            max_features=self.n_features
        )
        self.left.fit(X[left_indices], y[left_indices])

        self.right = DecisionTreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            categorical_features=self.categorical_features,
            max_features=self.n_features
        )
        self.right.fit(X[right_indices], y[right_indices])

    def predict(self, x):
        if self.label is not None or self.feature_index is None:
            return self.label

        if self.feature_index in self.categorical_features:
            return self.left.predict(x) if x[self.feature_index] == self.category else self.right.predict(x)
        else:
            return self.left.predict(x) if x[self.feature_index] <= self.threshold else self.right.predict(x)

    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])
