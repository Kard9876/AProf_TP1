import numpy as np
import sys
import os

from layers.sigmoid import SigmoidActivation
from functions.metrics import mse, accuracy
from networks.recorrent_neural_network import RecorrentNeuralNetwork
from functions.mse import MeanSquaredError
from layers.rnn import RNN
from layers.dense import DenseLayer
from optimizations.retained_gradient import RetGradient
from Code.DNN.optimizations.l2_reg import L2Reg
from Code.RNN.functions.bce import BinaryCrossEntropy
from Code.RNN.layers.relu import ReLUActivation

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

    timestep = 2
    batch_size = 8

    # network
    optimizer = RetGradient(learning_rate=0.005, momentum=0.90)
    loss = BinaryCrossEntropy()

    regulator = L2Reg(l2_val=0.001)
    net = RecorrentNeuralNetwork(epochs=3, batch_size=batch_size, optimizer=optimizer, regulator=regulator, verbose=True, loss=loss,
                        metric=accuracy, patience=2, min_delta=0.001, timestep=timestep)

    net.add(RNN(5, input_shape=(timestep, X_train.shape[1])))
    net.add(SigmoidActivation())
    net.add(DenseLayer(1, timestep))
    net.add(SigmoidActivation())

    # train
    net.fit(X_train, y_train, X_val=X_validation, y_val=y_validation)

    # test
    out = net.predict(X_test)
    out = out.reshape(out.shape[1], 1)

    if y_test is not None:
        print(net.score(y_test, out))

    net.plot_train_curves()

    store_results = args[1] if len(args) > 1 else './Results/dnn_results.csv'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(store_results), exist_ok=True)

    results = dataset.merge_results(ids, out)
    results.to_csv(store_results, sep='\t', index=False)


if __name__ == '__main__':
    main(sys.argv)
