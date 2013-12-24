import numpy as np
import layers
import connections
import trainers


class Model(object):
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

    def train(self, lr, epoch, batch_size, data, checkpoint=None):
        data = np.reshape(data, (-1, self.layers[0].size))  # Ensure input is 2d array
        nbatches = int(np.ceil(data.shape[0] / float(batch_size)))
        for e in range(epoch):
            np.random.shuffle(data)
            err = 0.0
            for batch in range(nbatches):
                stats = dict()
                v_pos = data[batch * batch_size:(batch+1) * batch_size]
                for statistic in self.statistics:
                    statistic(self, v_pos, stats)

                dd = stats['data']  # Data dependent
                md = stats['model']  # Model dependent
                examples = v_pos.shape[0]
                for i, connection in enumerate(self.connections):
                    gradient = connection.gradient(dd[i], dd[i+1], md[i], md[i+1])
                    gradient = lr * (gradient / examples)
                    connection.gradient_update(gradient)

                for i, layer in enumerate(self.layers):
                    gradient = layer.gradient(dd[i], md[i])
                    gradient = lr * (gradient / examples)
                    layer.gradient_update(gradient)
                err += np.sum((v_pos - md[0]) ** 2)
            if checkpoint is not None and ((e + 1) % checkpoint) == 0:
                print("Epoch {}: {}".format(e+1, np.sqrt(err / data.size)))

    def dream(self, data, steps=1):
        """
        Generator that returns samples separated by #steps
        Returns probabilities
        """
        data = np.reshape(data, (-1, self.connections[0].dim_b))  # Ensure input is 2d array
        states = trainers.initialize_states(self, data)

        while True:
            for s in range(steps):
                # Up to hidden
                for i in range(1, len(self.layers)):
                    states[i] = self.sample(i, states)

                # Back down to visible
                for i in range(len(self.layers)-1, 0, -1):
                    states[i] = self.sample(i, states)
                visible = self.expectation(0, states)
                states[0] = self.layers[0].sample_exp(visible)
            yield visible


class BinaryRBM(Model):
    def __init__(self, num_v, num_h):
        l1 = layers.BinaryLayer(num_v)
        l2 = layers.BinaryLayer(num_h)
        c1 = connections.FullConnection(num_v, num_h)
        #stats = [trainers.CD_data(), trainers.PCD_model()]
        stats = [trainers.CD_data(), trainers.CD_model()]
        Model.__init__(self, [l1, l2], [c1], stats)


class ShapeBM(Model):
    def __init__(self, num_v, num_h1, num_h2, patches, trainers):
        blayers = [layers.BinaryLayer(s) for s in [num_v, num_h1, num_h2]]
        c1 = connections.ShapeBMConnection(num_v, num_h1, patches)
        c2 = connections.FullConnection(num_h1, num_h2)
        stats = [trainers.CD_data(), trainers.PCD_model()]
        Model.__init__(self, blayers, [c1, c2], stats)
