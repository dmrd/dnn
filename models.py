import numpy as np
import layers
import connections


class Model:
    def __init__(self, layers, connections, statistics):
        self.layers = layers
        for i in range(len(connections) - 1):
            assert(connections[i].dim_t == connections[i + 1].dim_b)
        self.connections = connections
        self.statistics = statistics

    def activation(self, index, states):
        """
        Get activation of the given layer
        index: index (0, 1, ...) of layer to get activation for
        states: current states of each layer
        """
        if index == 0:
            return self.connections[0].prop_down(states[1])
        if index == len(self.layers) - 1:
            return self.connections[-1].prop_up(states[-2])
        return (self.connections[index].prop_down(states[index + 1])
                + self.connections[index - 1].prop_up(states[index - 1]))

    def expectation(self, index, states):
        return self.layers[index].expectation(self.activation(index, states))

    def sample(self, index, states):
        return self.layers[index].sample(self.activation(index, states))

    def train(self, epoch, minibatch_size, data):
        nbatches = int(np.ceil(data.shape[0] / float(self.batch_size)))
        for e in epoch:
            np.random.shuffle(data)
            for batch in range(nbatches):
                pass


class BinaryRBM:
    def __init__(self, num_v, num_h, trainers):
        l1 = layers.BinaryLayer()
        l2 = layers.BinaryLayer()
        c1 = connections.FullConnection(num_v, num_h)
        super(self)([l1, l2], [c1], trainers)


class ShapeBM:
    def __init__(self, num_v, num_h1, num_h2, patches, trainers):
        l = layers.BinaryLayer()
        c1 = connections.ShapeBMConnection(num_v, num_h1, patches)
        c2 = connections.FullConnection(num_h1, num_h2)
        super(self)([l] * 3, [c1, c2], trainers)
