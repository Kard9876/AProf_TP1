import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from Code.LogisticRegression.logisticReg.logisticReg import LogisticRegression
from Code.utils.dataset import Dataset


def main():
    np.random.seed(42)

    dataset = Dataset('../../Dataset/dataset_training_small.csv', '../../Dataset/dataset_validation_small.csv',
                      '../../Dataset/dataset_test_small.csv')

    X_train, y_train, X_validation, y_validation, X_test, y_test = dataset.get_datasets('text', 'ai_generator', sep=';',
                                                                                        rem_punctuation=False)

    n_features = X_train.shape[1]

    model = LogisticRegression(n_features)

    maxiter = 10
    maxfun = 10
    model.optim_model(X_train, y_train, maxiter, maxfun)

    print("Final cost:", model.cost_function(X_test, y_test))


if __name__ == '__main__':
    main()
