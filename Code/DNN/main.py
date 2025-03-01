from layers.sigmoid import SigmoidActivation
from functions.metrics import mse, accuracy
from networks.neuralnet import NeuralNetwork
from functions.mse import MeanSquaredError
from layers.dense import DenseLayer
from optimizations.optimizer import Optimizer
from utils.data import read_csv


def main():
    # training data
    dataset = read_csv('breast-bin.csv', sep=',', features=True, label=True)

    # network
    optimizer = Optimizer(learning_rate=0.01, momentum=0.90)
    loss = MeanSquaredError()
    net = NeuralNetwork(epochs=1000, batch_size=16, optimizer=optimizer, verbose=True, loss=loss, metric=accuracy)

    n_features = dataset.X.shape[1]
    net.add(DenseLayer(6, (n_features,)))
    net.add(SigmoidActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    # train
    net.fit(dataset)

    # test
    out = net.predict(dataset)
    print(net.score(dataset, out))


if __name__ == '__main__':
    main()
