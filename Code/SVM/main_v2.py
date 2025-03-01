import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from Code.SVM.svm.svm import SVM
from Code.SVM.utils.metrics import accuracy
from Code.SVM.utils.data import read_csv


def main():
    # training data

    train = pd.read_csv('../../Dataset/dataset_training_small.csv', sep=';')
    test = pd.read_csv('../../Dataset/dataset_test_small.csv', sep=';')

    vectorizer = CountVectorizer()

    X_train = train.drop('ai_generator', inplace=False, axis=1)
    vectorizer.fit(X_train['text'])
    X_train = vectorizer.transform(X_train['text']).toarray()

    y_train = train['ai_generator']
    y_train = y_train.to_numpy()

    X_test = test.drop('ai_generator', inplace=False, axis=1)
    X_test = vectorizer.transform(X_test['text']).toarray()

    y_test = test['ai_generator']
    y_test = y_test.to_numpy()

    n_features = X_train.shape[1]

    learning_rate = 0.0001
    epochs = 10
    lambda_val = 0.01

    svm = SVM(n_features, learning_rate, epochs, lambda_val)

    # test
    svm.fit(X_train, y_train)
    out = svm.predict(X_test)
    print(out)
    print(accuracy(y_test, out))


if __name__ == '__main__':
    main()
