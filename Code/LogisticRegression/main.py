import sys
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from Code.LogisticRegression.logisticReg.logisticReg import LogisticRegression
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

    model = LogisticRegression(n_features)

    # maxiter = 10
    # maxfun = 10
    # model.optim_model(X_train, y_train, maxiter, maxfun)
    model.gradient_descent(X_train, y_train, alpha=0.01, iters=50)

    if y_test is not None:
        print("Final cost:", model.cost_function(X_test, y_test))

    out = model.predict_many(X_test)
    out = out.reshape(out.shape[0], 1)
    store_results = args[1] if len(args) > 1 else './Results/dnn_results.csv'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(store_results), exist_ok=True)

    results = dataset.merge_results(ids, out)
    results.to_csv(store_results, sep='\t', index=False)


if __name__ == '__main__':
    main(sys.argv)
