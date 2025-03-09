import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from Code.SVM.svm.svm import SVM
from Code.SVM.utils.metrics import accuracy
from Code.SVM.utils.data import read_csv
from Code.utils.dataset import Dataset


def main():
    np.random.seed(42)

    dataset = Dataset('../../Dataset/dataset_training_small.csv', '../../Dataset/dataset_validation_small.csv',
                      '../../Dataset/dataset_test_small.csv')

    X_train, y_train, X_validation, y_validation, X_test, y_test = dataset.get_datasets('text', 'ai_generator', sep=';',
                                                                                        rem_punctuation=False)

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
