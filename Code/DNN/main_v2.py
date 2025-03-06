from layers.sigmoid import SigmoidActivation
from functions.metrics import mse, accuracy
from networks.neuralnet import NeuralNetwork
from functions.mse import MeanSquaredError
from layers.dense import DenseLayer
from layers.dropout import DropOutLayer
from optimizations.retained_gradient import RetGradient
from Code.DNN.optimizations.l1_reg import L1Reg
from Code.DNN.optimizations.l2_reg import L2Reg

from Code.utils.dataset import Dataset


def main():
    dataset = Dataset('../../Dataset/dataset_training_small.csv', '../../Dataset/dataset_validation_small.csv',
                      '../../Dataset/dataset_test_small.csv')

    # Remover pontuação deu pior resultado
    X_train, y_train, X_validation, y_validation, X_test, y_test = dataset.get_datasets('text', 'ai_generator', sep=';',
                                                                                        rem_punctuation=False)

    # network
    optimizer = RetGradient(learning_rate=0.01, momentum=0.90)
    loss = MeanSquaredError()

    regulator = L2Reg(l2_val=0.001)
    net = NeuralNetwork(epochs=15, batch_size=16, optimizer=optimizer, regulator=regulator, verbose=True, loss=loss,
                        metric=accuracy, patience=2, min_delta=0.001)

    n_features = X_train.shape[1]
    net.add(DenseLayer(6, (n_features,)))
    net.add(SigmoidActivation())
    net.add(DropOutLayer(3, 0.5, (n_features,)))
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    # train
    net.fit(X_train, y_train)

    # test
    out = net.predict(X_test)
    print(net.score(y_test, out))


if __name__ == '__main__':
    main()
