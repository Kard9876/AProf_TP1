# Se for necessário instalar punkt_tab
# import nltk
# nltk.download('punkt_tab')

import numpy as np
import sys
import os
from itertools import product

from Code.RNN.networks.recorrent_neural_network import RecorrentNeuralNetwork
from layers.sigmoid import SigmoidActivation
from functions.metrics import mse, accuracy
from Code.RNN.networks import recorrent_neural_network
from functions.mse import MeanSquaredError
from layers.rnn import RNN
from layers.dense import DenseLayer
from optimizations.retained_gradient import RetGradient
from Code.DNN.optimizations.l2_reg import L2Reg
from Code.RNN.functions.bce import BinaryCrossEntropy
from Code.RNN.layers.relu import ReLUActivation
from Code.utils.dataset import Dataset

def build_and_train_network(X_train, y_train, X_val, y_val, X_test, y_test, params):
    np.random.seed(42)

    optimizer = RetGradient(learning_rate=params['learning_rate'], momentum=params['momentum'])
    loss = BinaryCrossEntropy()
    regulator = L2Reg(l2_val=params['l2_val']) if params['l2_val'] > 0 else None

    net = RecorrentNeuralNetwork(
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        optimizer=optimizer,
        regulator=regulator,
        verbose=False,
        loss=loss,
        metric=accuracy,
        patience=-1,
        min_delta=0.001
    )

    # Topologia
    net.add(RNN(params['rnn_units'], input_shape=(X_train.shape[1], X_train.shape[2])))
    net.add(ReLUActivation())  # Apenas ReLU

    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    # Treino
    net.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    out = net.predict(X_test)
    score = net.score(y_test, out)

    return score, net

def grid_search(X_train, y_train, X_val, y_val, X_test, y_test):
    # Hiperparamentros para alterar
    param_grid = {
        'rnn_units': [8, 10],              # melhor 10
        'batch_size': [8, 6],              # melhor 8
        'learning_rate': [0.001, 0.005],     # melhor 0.01
        'momentum': [0.9, 0.95],            # melhor 0,9
        'l2_val': [0.01, 0.015],              # melhor 0.01
        'epochs': [8, 10]                   # melhor 10
    }

    # Fazer as combinações e testar combinações
    keys = param_grid.keys()
    combinations = [dict(zip(keys, values)) for values in product(*param_grid.values())]

    best_score = -float('inf')
    best_params = None
    best_net = None

    for params in combinations:
        print(f"Testando combinação: {params}")
        score, net = build_and_train_network(X_train, y_train, X_val, y_val, X_test, y_test, params)  # Correção aqui
        print(f"Score obtido: {score}")

        if score > best_score:
            best_score = score
            best_params = params
            best_net = net

    print("\nResultados finais do GridSearch:")
    print(f"Melhor score: {best_score}")
    print(f"Melhores parâmetros: {best_params}")

    return best_net, best_score, best_params

def main(args):
    np.random.seed(42)

    dataset = Dataset('../../Dataset/DatasetsGerados/dataset_training_input.csv',
                      '../../Dataset/DatasetsGerados/dataset_training_output.csv',
                      '../../Dataset/DatasetsGerados/dataset_validation_input.csv',
                      '../../Dataset/DatasetsGerados/dataset_validation_output.csv',
                      '../../Dataset/DatasetsGerados/dataset_test_input.csv',
                      '../../Dataset/DatasetsGerados/dataset_test_output.csv')

    X_train, y_train, X_validation, y_validation, X_test, y_test, ids = dataset.get_dataset_embedding(
        'Text', 'Label', sep='\t', rem_punctuation=False, vector_size=100, words_phrase=100
    )

    best_net, best_score, best_params = grid_search(
        X_train, y_train, X_validation, y_validation, X_test, y_test
    )

    out = best_net.predict(X_test)
    print(f"Score final no conjunto de teste com a melhor configuração: {best_score}")

    best_net.plot_train_curves()

    # Salvar resultados
    store_results = args[1] if len(args) > 1 else './Results/dnn_results.csv'
    os.makedirs(os.path.dirname(store_results), exist_ok=True)
    results = dataset.merge_results(ids, out)
    results.to_csv(store_results, sep='\t', index=False)

if __name__ == '__main__':
    main(sys.argv)