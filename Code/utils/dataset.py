import string
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class Dataset:
    def __init__(self, train_path, validation_path, test_path):
        self._train_path = train_path
        self._validation_path = validation_path
        self._test_path = test_path
        self._vectorizer = CountVectorizer()
        self._vectorizer_fitted = False

    def get_train_dataset(self, fit_column, target, sep=';', rem_punctuation=False):
        train = pd.read_csv(self._train_path, sep=sep)
        X_train = train.drop(target, inplace=False, axis=1)

        if rem_punctuation:
            X_train[fit_column] = X_train[fit_column].apply(remove_ponctuation)

        self._vectorizer.fit(X_train[fit_column])
        self._vectorizer_fitted = True

        X_train = self._vectorizer.transform(X_train[fit_column]).toarray()

        y_train = train[target]
        y_train = y_train.to_numpy()

        return X_train, y_train

    def get_validation_dataset(self, fit_column, target, sep=';', rem_punctuation=False):
        if not self._vectorizer_fitted:
            raise Exception("Please obtain train dataset first")

        validation = pd.read_csv(self._validation_path, sep=sep)
        X_validation = validation.drop(target, inplace=False, axis=1)

        if rem_punctuation:
            X_validation[fit_column] = X_validation[fit_column].apply(remove_ponctuation)

        X_validation = self._vectorizer.transform(X_validation[fit_column]).toarray()

        y_validation = validation[target]
        y_validation = y_validation.to_numpy()

        return X_validation, y_validation

    def get_test_dataset(self, fit_column, target, sep=';', rem_punctuation=False):
        if not self._vectorizer_fitted:
            raise Exception("Please obtain train dataset first")

        test = pd.read_csv(self._test_path, sep=sep)
        X_test = test.drop(target, inplace=False, axis=1)

        if rem_punctuation:
            X_test[fit_column] = X_test[fit_column].apply(remove_ponctuation)

        X_test = self._vectorizer.transform(X_test[fit_column]).toarray()

        y_test = test[target]
        y_test = y_test.to_numpy()

        return X_test, y_test

    def get_datasets(self, fit_column, target, sep=';', rem_punctuation=False):
        X_train, y_train = self.get_train_dataset(fit_column, target, sep=sep, rem_punctuation=rem_punctuation)
        X_validation, y_validation = self.get_validation_dataset(fit_column, target, sep=sep, rem_punctuation=rem_punctuation)
        X_test, y_test = self.get_test_dataset(fit_column, target, sep=sep, rem_punctuation=rem_punctuation)

        return X_train, y_train, X_validation, y_validation, X_test, y_test


def remove_ponctuation(row):
    for s in string.punctuation:
        row = row.replace(s, "")

    return row
