import numpy as np
import sys
import os

from layers.sigmoid import SigmoidActivation
from functions.metrics import mse, accuracy
from networks.neuralnet import NeuralNetwork
from functions.mse import MeanSquaredError
from layers.dense import DenseLayer
from layers.dropout import DropOutLayer
from optimizations.retained_gradient import RetGradient
from Code.DNN.optimizations.l1_reg import L1Reg
from Code.DNN.optimizations.l2_reg import L2Reg
from Code.DNN.functions.bce import BinaryCrossEntropy

from Code.utils.dataset import Dataset
import Code.utils.store_model as store_model


def main(args):
    np.random.seed(42)

    dataset = Dataset('../../Dataset/DatasetsGerados/dataset_training_input.csv',
                      '../../Dataset/DatasetsGerados/dataset_training_output.csv',
                      '../../Dataset/DatasetsGerados/dataset_validation_input.csv',
                      '../../Dataset/DatasetsGerados/dataset_validation_output.csv',
                      '../../Dataset/DatasetsGerados/dataset1_inputs.csv',
                      '../../Dataset/DatasetsGerados/dataset1_outputs.csv')

    # Remover pontuação deu pior resultado
    X_train, y_train, X_validation, y_validation, X_test, y_test, ids = dataset.get_datasets('Text', 'Label', sep='\t',
                                                                                             rem_punctuation=False)

    # network
    optimizer = RetGradient(learning_rate=0.0034, momentum=0.65)
    loss = BinaryCrossEntropy()

    regulator = L2Reg(l2_val=0.05)
    net = NeuralNetwork(epochs=60, batch_size=30, optimizer=optimizer, regulator=regulator, verbose=True, loss=loss,
                        metric=accuracy, patience=50, min_delta=0.001)

    n_features = X_train.shape[1]
    net.add(DenseLayer(32, (n_features,)))
    net.add(SigmoidActivation())
    net.add(DenseLayer(16, (n_features)))
    net.add(SigmoidActivation())
    net.add(DropOutLayer(8, 0.8, (n_features,)))
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    # train
    net.fit(X_train, y_train, X_val=X_validation, y_val=y_validation)

    # test
    out = net.predict(X_test)
    net.plot_train_curves()

    if y_test is not None:
        print(net.score(y_test, out))

    store_results = args[1] if len(args) > 1 else './Results/dnn_results.csv'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(store_results), exist_ok=True)

    results = dataset.merge_results(ids, out)
    results.to_csv(store_results, sep='\t', index=False)

    store_model.store_model('./Model/dnn', 'dnn', net)


def main2(args):
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

    net = store_model.retrieve_model('./Model/dnn', 'dnn')

    # test
    out = net.predict(X_test)
    net.plot_train_curves()

    if y_test is not None:
        print(net.score(y_test, out))

    store_results = args[1] if len(args) > 1 else './Results/dnn_results2.csv'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(store_results), exist_ok=True)

    results = dataset.merge_results(ids, out)
    results.to_csv(store_results, sep='\t', index=False)


if __name__ == '__main__':
    main(sys.argv)
    main2(sys.argv)
