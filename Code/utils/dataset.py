import string
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class Dataset:
    def __init__(self, train_input, train_output, validation_input, validation_output, test_input, test_output):
        self._train_input = train_input
        self._train_output = train_output
        self._validation_input = validation_input
        self._validation_output = validation_output
        self._test_input = test_input
        self._test_output = test_output
        self._vectorizer = CountVectorizer()
        self._vectorizer_fitted = False
        self._tfidf_vectorizer = TfidfVectorizer()
        self._tfidf_vectorizer_fitted = False
        self._label_encoder = LabelEncoder()
        self._label_encoder_fitted = False

    def set_dataset_test(self, test_input, test_output):
        self._test_input = test_input
        self._test_output = test_output

    def get_train_dataset(self, fit_column, target, sep=';', rem_punctuation=False, id_column='ID'):
        X_train = pd.read_csv(self._train_input, sep=sep)
        X_train = X_train.drop(id_column, inplace=False, axis=1)

        if rem_punctuation:
            X_train[fit_column] = X_train[fit_column].apply(remove_ponctuation)

        self._vectorizer.fit(X_train[fit_column])
        self._vectorizer_fitted = True

        X_train = self._vectorizer.transform(X_train[fit_column]).toarray()

        y_train = pd.read_csv(self._train_output, sep=sep)
        y_train = y_train.drop(id_column, inplace=False, axis=1)
        y_train = y_train.to_numpy().reshape(y_train.shape[0], )

        self._label_encoder_fitted = True
        y_train = self._label_encoder.fit_transform(y_train)

        return X_train, y_train

    def get_validation_dataset(self, fit_column, target, sep=';', rem_punctuation=False, id_column='ID'):
        if not self._vectorizer_fitted or not self._label_encoder_fitted:
            raise Exception("Please obtain train dataset first")

        X_validation = pd.read_csv(self._validation_input, sep=sep)
        X_validation = X_validation.drop(id_column, inplace=False, axis=1)

        if rem_punctuation:
            X_validation[fit_column] = X_validation[fit_column].apply(remove_ponctuation)

        X_validation = self._vectorizer.transform(X_validation[fit_column]).toarray()

        y_validation = pd.read_csv(self._validation_output, sep=sep)
        y_validation = y_validation.drop(id_column, inplace=False, axis=1)
        y_validation = y_validation.to_numpy().reshape(y_validation.shape[0], )

        y_validation = self._label_encoder.transform(y_validation)

        return X_validation, y_validation

    def get_test_dataset(self, fit_column, target, sep=';', rem_punctuation=False, id_column='ID'):
        if not self._vectorizer_fitted or not self._label_encoder_fitted:
            raise Exception("Please obtain train dataset first")

        X_test = pd.read_csv(self._test_input, sep=sep)
        ids = X_test[id_column]
        X_test = X_test.drop(id_column, inplace=False, axis=1)

        if rem_punctuation:
            X_test[fit_column] = X_test[fit_column].apply(remove_ponctuation)

        X_test = self._vectorizer.transform(X_test[fit_column]).toarray()

        y_test = None

        if self._test_output is not None:
            y_test = pd.read_csv(self._test_output, sep=sep)

            y_test = y_test.drop(id_column, inplace=False, axis=1)
            y_test = y_test.to_numpy().reshape(y_test.shape[0], )

            y_test = self._label_encoder.transform(y_test)

        return X_test, y_test, ids

    def get_datasets(self, fit_column, target, sep=';', rem_punctuation=False, id_column='ID'):
        X_train, y_train = self.get_train_dataset(fit_column, target, sep=sep, rem_punctuation=rem_punctuation, id_column=id_column)
        X_validation, y_validation = self.get_validation_dataset(fit_column, target, sep=sep, rem_punctuation=rem_punctuation, id_column=id_column)
        X_test, y_test, ids = self.get_test_dataset(fit_column, target, sep=sep, rem_punctuation=rem_punctuation, id_column=id_column)

        return X_train, y_train, X_validation, y_validation, X_test, y_test, ids

    def get_train_dataset_embedding(self, fit_column, target, sep=';', rem_punctuation=False, id_column='ID'):
        X_train = pd.read_csv(self._train_input, sep=sep)
        X_train = X_train.drop(id_column, inplace=False, axis=1)

        if rem_punctuation:
            X_train[fit_column] = X_train[fit_column].apply(remove_ponctuation)

        y_train = pd.read_csv(self._train_output, sep=sep)
        y_train = y_train.drop(id_column, inplace=False, axis=1)
        y_train = y_train.to_numpy().reshape(y_train.shape[0], )

        self._label_encoder_fitted = True
        y_train = self._label_encoder.fit_transform(y_train)

        return X_train, y_train

    def get_validation_dataset_embedding(self, fit_column, target, sep=';', rem_punctuation=False, id_column='ID'):
        if not self._label_encoder_fitted:
            raise Exception("Please obtain train dataset first")

        X_validation = pd.read_csv(self._validation_input, sep=sep)
        X_validation = X_validation.drop(id_column, inplace=False, axis=1)

        if rem_punctuation:
            X_validation[fit_column] = X_validation[fit_column].apply(remove_ponctuation)

        y_validation = pd.read_csv(self._validation_output, sep=sep)
        y_validation = y_validation.drop(id_column, inplace=False, axis=1)
        y_validation = y_validation.to_numpy().reshape(y_validation.shape[0], )

        y_validation = self._label_encoder.transform(y_validation)

        return X_validation, y_validation

    def get_test_dataset_embedding(self, fit_column, target, sep=';', rem_punctuation=False, id_column='ID'):
        if not self._label_encoder_fitted:
            raise Exception("Please obtain train dataset first")

        X_test = pd.read_csv(self._test_input, sep=sep)
        ids = X_test[id_column]
        X_test = X_test.drop(id_column, inplace=False, axis=1)

        if rem_punctuation:
            X_test[fit_column] = X_test[fit_column].apply(remove_ponctuation)

        y_test = None

        if self._test_output is not None:
            y_test = pd.read_csv(self._test_output, sep=sep)

            y_test = y_test.drop(id_column, inplace=False, axis=1)
            y_test = y_test.to_numpy().reshape(y_test.shape[0], )

            y_test = self._label_encoder.transform(y_test)

        return X_test, y_test, ids

    def get_datasets_embedding(self, fit_column, target, sep=';', rem_punctuation=False, id_column='ID'):
        X_train, y_train = self.get_train_dataset_embedding(fit_column, target, sep=sep, rem_punctuation=rem_punctuation, id_column=id_column)
        X_validation, y_validation = self.get_validation_dataset_embedding(fit_column, target, sep=sep, rem_punctuation=rem_punctuation, id_column=id_column)
        X_test, y_test, ids = self.get_test_dataset_embedding(fit_column, target, sep=sep, rem_punctuation=rem_punctuation, id_column=id_column)

        return X_train, y_train, X_validation, y_validation, X_test, y_test, ids

    def merge_results(self, id, results):
        ans = pd.DataFrame({})

        results = [np.round(results[i][0]).astype(int) for i in range(len(results))]
        results = self._label_encoder.inverse_transform(results)

        ans['ID'] = id
        ans['Label'] = results

        return ans


def remove_ponctuation(row):
    for s in string.punctuation:
        row = row.replace(s, "")

    return row
