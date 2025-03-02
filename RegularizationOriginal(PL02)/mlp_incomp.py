#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import numpy as np
from dataset import Dataset


class MLP:
    
    def __init__(self, dataset, hidden_nodes = 2, normalize = False):
        self.X, self.y = dataset.getXy()
        self.X = np.hstack ( (np.ones([self.X.shape[0],1]), self.X ) )
        
        self.h = hidden_nodes
        self.W1 = np.zeros([hidden_nodes, self.X.shape[1]])
        self.W2 = np.zeros([1, hidden_nodes+1])
        
        if normalize:
            self.normalize()
        else:
            self.normalized = False


    def setWeights(self, w1, w2):
        self.W1 = w1
        self.W2 = w2
        

    def predict(self, instance):
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        
        if self.normalized:
            if np.all(self.sigma!= 0): 
                x[1:] = (x[1:] - self.mu) / self.sigma
            else: x[1:] = (x[1:] - self.mu)

        # Predict First Layer
        # Since we are doing it for each node, we are multiplying a matrix and a node, so we need to invert the order to allow matrix multiplication
        tmp = np.dot(self.W1, x)

        ans = np.empty(tmp.shape[0] + 1)
        ans[0] = 1
        ans[1:] = sigmoid(tmp)

        # Predict Second Layer
        # Since we are doing it for each node, we are multiplying a matrix and a node, so we need to invert the order to allow matrix multiplication
        ans = np.dot(self.W2, ans)
        
        return sigmoid(ans)


    def predictMany(self, Xpred = None):
        
        if Xpred is None: ## use training set
            Xp = self.X
        else:
            Xp = Xpred

        # Weights are matrix of n+1 * m, where n is the number of nodes in the i-th layer, m is the number of nodes in the i-th + 1 layer
        # We transpose it to allow matrix multiplication
        ans = np.dot(Xp, self.W1.T)

        # Stacks arrays horizontally, in this case, adds the number one to each row resulting from sigmoid(ans)
        tmp = np.hstack((np.ones([ans.shape[0], 1]), sigmoid(ans)))

        # Weights are matrix of n+1 * m, where n is the number of nodes in the i-th layer, m is the number of nodes in the i-th + 1 layer
        # We transpose it to allow matrix multiplication
        ans = np.dot(tmp, self.W2.T)

        return sigmoid(ans)


    def costFunction(self, weights = None, loss = "mse"):
        if weights is not None:
            self.W1 = weights[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
            self.W2 = weights[self.h * self.X.shape[1]:].reshape([1, self.h+1])
        
        predictions = self.predictMany()
        m = self.X.shape[0]

        ans = math.inf

        if loss == "mse":
            # Y is a column, so we need to reshape it into a line to perform the difference operator
            ans = (predictions - self.y.reshape(m, 1)) ** 2
            ans = np.sum(ans) / (2 * m)

        if loss == "entropy":
            p = np.clip(predictions, 1e-15, 1 - 1e-15)
            cost = (-self.y.dot(np.log(p)) - (1 - self.y).dot(np.log(1 - p)))
            ans = np.sum(cost) / m

        return ans



    def build_model(self):
        from scipy import optimize

        size = self.h * self.X.shape[1] + self.h+1
        
        initial_w = np.random.rand(size)        
        result = optimize.minimize(lambda w: self.costFunction(w), initial_w, method='BFGS', 
                                    options={"maxiter":1000, "disp":False} )
        weights = result.x
        self.W1 = weights[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
        self.W2 = weights[self.h * self.X.shape[1]:].reshape([1, self.h+1])

    def normalize(self):
        self.mu = np.mean(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] - self.mu
        self.sigma = np.std(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] / self.sigma
        self.normalized = True


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

    

def test():
    ds= Dataset("xnor.data")
    nn = MLP(ds, 2)
    w1 = np.array([[-30,20,20],[10,-20,-20]])
    w2 = np.array([[-10,20,20]])
    nn.setWeights(w1, w2)
    print( nn.predict(np.array([0,0]) ) )
    print( nn.predict(np.array([0,1]) ) )
    print( nn.predict(np.array([1,0]) ) )
    print( nn.predict(np.array([1,1]) ) )
    print(nn.costFunction())

def test2():
    ds= Dataset("xnor.data")
    nn = MLP(ds, 5, normalize = False)
    nn.build_model()
    print( nn.predict(np.array([0,0]) ) )
    print( nn.predict(np.array([0,1]) ) )
    print( nn.predict(np.array([1,0]) ) )
    print( nn.predict(np.array([1,1]) ) )
    print(nn.costFunction())

if __name__ == "__main__":
    test()
    #test2()
