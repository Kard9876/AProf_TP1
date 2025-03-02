import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from layers.sigmoid import SigmoidActivation
from functions.metrics import mse, accuracy
from networks.neuralnet import NeuralNetwork
from functions.mse import MeanSquaredError
from layers.dense import DenseLayer
from layers.dropout import DropOutLayer
from optimizations.retained_gradient import RetGradient
from utils.data import read_csv
from Code.DNN.optimizations.l1_reg import L1Reg
from Code.DNN.optimizations.l2_reg import L2Reg


def main():
    # training data

    train = pd.read_csv('../../Dataset/dataset_training_small.csv', sep=';')
    test = pd.read_csv('../../Dataset/dataset_test_small.csv', sep=';')

    vectorizer = CountVectorizer()

    X_train = train.drop('ai_generator', inplace=False, axis=1)
    vectorizer.fit(X_train['text'])
    X_train = vectorizer.transform(X_train['text']).toarray()

    y_train = train['ai_generator']
    y_train = y_train.to_numpy()

    X_test = test.drop('ai_generator', inplace=False, axis=1)
    X_test = vectorizer.transform(X_test['text']).toarray()

    y_test = test['ai_generator']
    y_test = y_test.to_numpy()

    # network
    optimizer = RetGradient(learning_rate=0.01, momentum=0.90)
    loss = MeanSquaredError()
    # net = NeuralNetwork(epochs=1000, batch_size=16, optimizer=optimizer, verbose=True, loss=loss, metric=accuracy)

    regulator = L2Reg(l2_val=0.001)
    net = NeuralNetwork(epochs=15, batch_size=16, optimizer=optimizer, regulator=regulator, verbose=True, loss=loss,
                        metric=accuracy, patience=2, min_delta=0.015)

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
