import numpy as np


def sigmoid(eta):
    return 1. / (1. + np.exp(-eta))


def draw_bernoulli(m):
    """ Returns array A of same shape as m with A[x] = True with probability m[x] """
    return np.random.rand(*m.shape) < m


class Layer:
    def __init__(self):
        pass

    def expectation(self, activations):
        pass

    def sample(self, activations):
        pass


def BinaryLayer(Layer):
    def expectation(self, activations):
        return sigmoid(activations)

    def sample_activations(self, activations):
        return draw_bernoulli(self.expectation(activations))

    def sample_exp(self, exp):
        return draw_bernoulli(exp)
