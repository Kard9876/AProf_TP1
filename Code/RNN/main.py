from layers.sigmoid import SigmoidActivation
from functions.metrics import mse, accuracy
from networks.recorrent_neural_network import RecorrentNeuralNetwork
from functions.mse import MeanSquaredError
from layers.rnn import RNN
from layers.dense import DenseLayer
from optimizations.retained_gradient import RetGradient
from Code.DNN.optimizations.l2_reg import L2Reg

from Code.utils.dataset import Dataset


def main():
    dataset = Dataset('../../Dataset/dataset_training_small.csv', '../../Dataset/dataset_validation_small.csv',
                      '../../Dataset/dataset_test_small.csv')

    # Remover pontuação deu pior resultado
    X_train, y_train, X_validation, y_validation, X_test, y_test = dataset.get_datasets('text', 'ai_generator', sep=';',
                                                                                        rem_punctuation=False)

    timestep = 2
    batch_size = 6

    # network
    optimizer = RetGradient(learning_rate=0.01, momentum=0.90)
    loss = MeanSquaredError()

    regulator = L2Reg(l2_val=0.001)
    net = RecorrentNeuralNetwork(epochs=3, batch_size=batch_size, optimizer=optimizer, regulator=regulator, verbose=True, loss=loss,
                        metric=accuracy, patience=2, min_delta=0.001, timestep=timestep)

    net.add(RNN(5, input_shape=(timestep, X_train.shape[1])))
    net.add(SigmoidActivation())
    net.add(DenseLayer(1, timestep))
    net.add(SigmoidActivation())

    # train
    net.fit(X_train, y_train)

    # test
    out = net.predict(X_test)
    out = out.reshape(out.shape[1], 1)
    print(net.score(y_test, out))


if __name__ == '__main__':
    main()
