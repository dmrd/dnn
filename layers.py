import numpy as np


def sigmoid(eta):
    return 1. / (1. + np.exp(-eta))


def draw_bernoulli(m):
    """ Returns array A of same shape as m with A[x] = True with probability m[x] """
    return np.random.rand(*m.shape) < m


class Layer(object):
    def __init__(self, size, initial_bias=None):
        self.size = size
        if initial_bias is not None:
            assert(initial_bias.size == size)
            self.bias = initial_bias
        else:
            self.bias = np.zeros(size)

    def expectation(self, activations):
        raise NotImplementedError("No expectation method defined")

    def sample(self, activations):
        raise NotImplementedError("No sampling method defined")

    def gradient(self, positive, negative):
        return positive.sum(axis=0) - negative.sum(axis=0)

    def gradient_update(self, update):
        self.bias += update

    def __repr__(self):
        return str(self.bias)

    def __getitem__(self, key):
        return self.bias[key]


class BinaryLayer(Layer):
    def expectation(self, activations):
        return sigmoid(activations + self.bias)

    def sample(self, activations):
        return draw_bernoulli(self.expectation(activations))

    def sample_exp(self, exp):
        return draw_bernoulli(exp)
