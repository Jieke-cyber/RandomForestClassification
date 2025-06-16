import csv
import numpy as np

class DatasetLoader:
    def __init__(self, path, label_index=0):
        """
        path: percorso del file CSV
        label_index: indice della colonna da usare come etichetta (y)
        threshold: numero massimo di valori unici per considerare una feature categorica
        """

        self.path = path
        self.label_index = label_index

        self.X = None
        self.y = None
        self.categorical_features = []

    def load_data(self):
        cleaned_data = []
        with open(self.path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                row = [val.strip() for val in row]
                if '?' in row:
                    continue
                if len(cleaned_data) > 0 and len(row) != len(cleaned_data[0]):
                    continue
                cleaned_data.append(row)

        data = np.array(cleaned_data, dtype=object)

        self.y = data[:, self.label_index]
        self.X = np.delete(data, self.label_index, axis=1)

        for i in range(self.X.shape[1]):
            col = self.X[:, i]
            if all(self.is_float(val) for val in col):
                self.X[:, i] = col.astype(float)

        self.categorical_features = [i for i in range(self.X.shape[1]) if
                                     not all(self.is_float(val) for val in self.X[:, i])]

        return self.X, self.y, self.categorical_features

    def is_float(self, val):
        try:
            float(val)
            return True
        except ValueError:
            return False
