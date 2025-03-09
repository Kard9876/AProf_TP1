import sys
import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from Code.SVM.svm.svm import SVM
from Code.SVM.utils.metrics import accuracy
from Code.SVM.utils.data import read_csv
from Code.utils.dataset import Dataset


def main(args):
    np.random.seed(42)

    dataset = Dataset('../../Dataset/DatasetsGerados/dataset_training_input.csv',
                      '../../Dataset/DatasetsGerados/dataset_training_output.csv',
                      '../../Dataset/DatasetsGerados/dataset_validation_input.csv',
                      '../../Dataset/DatasetsGerados/dataset_validation_output.csv',
                      '../../Dataset/DatasetsGerados/dataset_test_input.csv',
                      '../../Dataset/DatasetsGerados/dataset_test_output.csv')

    # Remover pontuação deu pior resultado
    X_train, y_train, X_validation, y_validation, X_test, y_test, ids = dataset.get_datasets('Text', 'Label', sep='\t',
                                                                                             rem_punctuation=False)

    n_features = X_train.shape[1]

    learning_rate = 0.0001
    epochs = 10
    lambda_val = 0.01

    svm = SVM(n_features, learning_rate, epochs, lambda_val)

    # test
    svm.fit(X_train, y_train)
    out = svm.predict(X_test)
    print(accuracy(y_test, out))

    store_results = args[1] if len(args) > 1 else './Results/dnn_results.csv'
    out = out.reshape(out.shape[0], 1)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(store_results), exist_ok=True)

    results = dataset.merge_results(ids, out)
    results.to_csv(store_results, sep='\t', index=False)


if __name__ == '__main__':
    main(sys.argv)
