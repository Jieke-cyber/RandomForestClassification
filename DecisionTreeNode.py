import numpy as np

class DecisionTreeNode:
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.label = None

    def gini_impurity(self, y):
        """ Calcola la Gini impurity di un array di etichette """
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def best_split(self, X, y):
        """ Trova lo split con la massima riduzione di Gini """
        best_gain = 0
        best_feature = None
        best_threshold = None
        current_gini = self.gini_impurity(y)

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_indices = X[:, feature] <= t
                right_indices = X[:, feature] > t

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
                    best_threshold = t

        return best_feature, best_threshold, best_gain

    def fit(self, X, y):
        """ Costruisce l'albero ricorsivamente """
        if self.depth >= self.max_depth or len(set(y)) == 1:
            self.label = np.bincount(y).argmax()  # Classe più frequente
            return

        feature, threshold, gain = self.best_split(X, y)

        if gain == 0 or feature is None:
            self.label = np.bincount(y).argmax()
            return

        self.feature_index = feature
        self.threshold = threshold

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        self.left = DecisionTreeNode(depth=self.depth + 1, max_depth=self.max_depth)
        self.left.fit(X[left_indices], y[left_indices])

        self.right = DecisionTreeNode(depth=self.depth + 1, max_depth=self.max_depth)
        self.right.fit(X[right_indices], y[right_indices])

    def predict(self, x):
        """ Predice la classe per un singolo esempio """
        if self.label is not None:
            return self.label
        if x[self.feature_index] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def predict_batch(self, X):
        """ Predice la classe per un array di esempi """
        return np.array([self.predict(x) for x in X])
