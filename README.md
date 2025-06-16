# 2025 Artificial Intelligence - Ingegneria Informatica Unifi



## Introduzione

Il progetto si basa sull'implementazione da zero dell'algoritmo **Random Forest** per la **classificazione supervisionata**, seguendo le linee guida e concetti presentati nel libro *Artificial Intelligence: A Modern Approach* (R&N 2020, Capitolo 19).

## Struttura del Progetto
Il programma è strutturato in più file:

- **DecisionTreeNode.py**:

      file contenente la classe per l'albero di decisione, in grado di gestire sia attributi numerici che categorici. 
      Include metodi per la costruzione dell'albero in modo ricorsiva, la selezione degli split ottimali e la predizione su nuovi esempi.

- **RandomForest.py**:

      file contenente la classe RandomForestClassifier, con metodi per l'addestramento di più alberi su 
      sottoinsiemi casuali del dataset (bagging) e predizione tramite majority voting.

- **DatasetLoader.py**:

      modulo per il caricamento dei dataset UCI, con gestione automatica di attributi categorici e numerici.

- **Metrics.py**:

      file principale in cui vengono caricati i dataset, suddivisi in Train/Test, e vengono eseguiti i test 
      sull'algoritmo Random Forest. Include valutazioni con accuratezza, precisione, recall, F1-score e matrice di confusione.

- **/datasets**:

      directory contenente i file `.csv` dei dataset scelti dal [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

## Flusso di Esecuzione
Il programma carica automaticamente uno o più dataset `.csv`, li divide in features (`X`) e label (`Y`), e procede alla loro suddivisione in set di **addestramento (Train)** e **test (Test)**.  
Successivamente:
- Viene addestrata una foresta composta da più alberi su subset casuali del train set.
- Ogni albero viene costruito con un numero casuale di feature (**feature bagging**).
- La classificazione finale su nuovi esempi è decisa tramite **voto di maggioranza**.

Per ciascun dataset vengono calcolate le seguenti metriche:
- **Accuracy**
- **Precision / Recall / F1-score**
- **Confusion Matrix** (non normalizzata)

## Come aggiungere un Dataset
Per aggiungere un nuovo dataset, è sufficiente:
1. Inserire il file `.csv` nella cartella `/datasets`.
2. Specificarne il nome nel modulo `Metrics.py`.
3. Il `DatasetLoader` si occuperà del resto (codifica, parsing, suddivisione in `X` e `Y`).

⚠️ È importante conoscere la struttura e la semantica dei dati per specificare correttamente le colonne target e le feature.

## Output
Per ciascun dataset verranno mostrate le seguenti informazioni:

- Accuracy sul test set
- Confusion Matrix (non normalizzata)
- Classification Report:
    - Precision
    - Recall
    - F1-score
    - Support

## Librerie esterne utilizzate

- **numpy**: utilizzata per operazioni numeriche efficienti, come calcoli vettoriali e gestione di array multidimensionali. Fondamentale per l’implementazione degli alberi decisionali e per il campionamento casuale dei dati.

- **scikit-learn (sklearn)**:
  - `train_test_split`: usato per suddividere i dataset in sottoinsiemi di training e test in modo casuale, garantendo una valutazione più affidabile del modello.
  - `metrics`: usato per calcolare le metriche di valutazione (accuracy, precision, recall, F1-score, confusion matrix) al fine di confrontare le prestazioni del classificatore implementato con metriche standard.

- **collections.Counter**: utilizzato per contare in modo semplice e veloce le occorrenze delle classi nei voti degli alberi, utile per determinare la classe finale predetta dalla foresta.

- **csv**: utilizzato per la lettura diretta dei file CSV contenenti i dataset, in alternativa a librerie più complesse come `pandas`, per mantenere il codice più semplice e leggibile.

## Riferimenti
Ringrazio le seguenti fonti per i dati, ispirazioni ed esempi:

- [UCI Repository](https://archive.ics.uci.edu/ml/index.php)
- *Artificial Intelligence: A Modern Approach*, Russell & Norvig (2020)
- [scikit-learn API Reference](https://scikit-learn.org/stable/modules/classes.html)
- [Wikipedia](https://en.wikipedia.org/wiki/Random_forest)

---
