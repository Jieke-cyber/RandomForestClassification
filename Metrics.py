import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from DatasetLoader import DatasetLoader  # quella versione che abbiamo fatto
from RandomForest import RandomForestClassifier  # la tua classe RF

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

def run_on_dataset(path, target_index):
    print(f"\n=== Dataset: {path.split('/')[-1]} ===")
    loader = DatasetLoader(path, target_index)
    X, y, categorical_features= loader.load_data()
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Categorical features indices: {loader.categorical_features}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_trees=50, max_depth=10, categorical_features = categorical_features)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc, prec, rec, f1, cm = evaluate_model(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    # Inserisci i path ai tuoi dataset e l'indice della colonna target (label)
    datasets = [
        ("datasets/wine/wine.data", 0),      # Wine: label è la prima colonna
        ("datasets/mushroom/agaricus-lepiota.data", 0),  # Mushroom: label è la prima colonna
        ("datasets/hepatitis/hepatitis.data", 0), # Hepatitis: label è la prima colonna spiegazione nella forto ma anche per il fatto che si hanno due tipi di feature che complica la distinzione
        ("datasets/glass+identification/glass.data", 10), # Glass: label è la
    ]

    for path, target_index in datasets:
        run_on_dataset(path, target_index)
