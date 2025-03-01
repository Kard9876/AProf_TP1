import pandas as pd
import numpy as np

from Code.SVM.svm.svm import SVM
from Code.SVM.utils.metrics import accuracy
from Code.SVM.utils.data import read_csv


# Muito lento
def get_bow_matrix(df, vocab):
    # Inicializar a matriz BoW
    bow_matrix = np.zeros((len(df), len(vocab)))

    # Preencher a matriz BoW
    for i, text in enumerate(df['text']):
        for word in text.split():
            if word in vocab:
                bow_matrix[i, vocab.index(word)] += 1

    return bow_matrix

def main():
    # training data

    train = pd.read_csv('../../Dataset/dataset_training_small.csv', sep=';')
    test = pd.read_csv('../../Dataset/dataset_test_small.csv', sep=';')

    X_train = train.drop('ai_generator', inplace=False, axis=1)
    n_features = X_train.shape[1]

    X_train['text'] = X_train['text'].apply(hash)
    X_train = X_train.to_numpy()
    y_train = train['ai_generator']
    y_train = y_train.to_numpy()

    X_test = test.drop('ai_generator', inplace=False, axis=1)
    X_test['text'] = X_test['text'].apply(hash)
    X_test = X_test.to_numpy()
    y_test = test['ai_generator']
    y_test = y_test.to_numpy()

    """
    train = pd.read_csv('../../Dataset/dataset_training_small.csv', sep=';')
    test = pd.read_csv('../../Dataset/dataset_test_small.csv', sep=';')

    X_train = train.drop('ai_generator', inplace=False, axis=1)

    vocab = list(set(" ".join(X_train['text']).split()))
    X_train = get_bow_matrix(X_train, vocab)

    y_train = train['ai_generator']
    y_train = y_train.to_numpy()

    X_test = test.drop('ai_generator', inplace=False, axis=1)
    X_test = get_bow_matrix(X_test, vocab)

    y_test = test['ai_generator']
    y_test = y_test.to_numpy()

    n_features = X_train.shape[1]
    """

    """
    dataset = read_csv('breast-bin.csv', sep=',', features=True, label=True)
    X = dataset.X
    y = dataset.y

    n_features = X.shape[1]
    """

    learning_rate = 0.001
    epochs = 1000
    lambda_val = 0.01

    svm = SVM(n_features, learning_rate, epochs, lambda_val)

    # test
    svm.fit(X_train, y_train)
    out = svm.predict(X_test)
    print(out)
    print(accuracy(y_test, out))


if __name__ == '__main__':
    main()
