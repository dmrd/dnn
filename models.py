import numpy as np
import layers
import connections
import trainers
import utils
import time


class Model(object):
    def __init__(self, layers, connections, statistics, ordered_trainers=None):
        self.layers = layers
        for i in range(len(connections) - 1):
            assert(connections[i].dim_t == connections[i + 1].dim_b)
        self.connections = connections
        self.statistics = statistics
        # Default to basic gradient trainer alone
        self.trainers = ordered_trainers or [trainers.Gradient()]
        self.err = []  # Track rmse across training

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
        """ Returns expectation, sample tuple """
        exp = self.expectation(index, states)
        return exp, self.layers[index].sample_exp(exp)

    def train(self, lr, epoch, batch_size, data, lr_schedule=trainers.lr_linear, checkpoint=None):
        data = np.reshape(data, (-1, self.layers[0].size))  # Ensure input is 2d array
        nbatches = int(np.ceil(data.shape[0] / float(batch_size)))
        start = time.time()
        for e in range(epoch):
            np.random.shuffle(data)
            err = 0.0
            epoch_lr = lr_schedule(lr, e, epoch)  # Get lr for current epoch
            for batch in range(nbatches):
                stats = dict()
                v_pos = data[batch * batch_size:(batch+1) * batch_size]
                for statistic in self.statistics:
                    statistic(self, v_pos, stats)

                examples = v_pos.shape[0]
                batch_lr = epoch_lr / examples
                for i, connection in enumerate(self.connections):
                    # Get first update
                    gradient = self.trainers[0].get_update(self, stats, i, batch_lr, e)
                    for trainer in self.trainers[1:]:  # Run any additional updates (momentum etc.)
                        gradient += trainer.get_update(self, stats, i, batch_lr, e)
                    for trainer in self.trainers:
                        trainer.update_state(self, i, gradient)
                    connection.gradient_update(gradient)

                dd = stats['data']  # Data dependent
                md = stats['model']  # Model dependent
                for i, layer in enumerate(self.layers):
                    gradient = layer.gradient(dd[i], md[i])
                    gradient = batch_lr * gradient
                    layer.gradient_update(gradient)

                if checkpoint is not None and ((e + 1) % checkpoint) == 0:
                    if 'reconstruction' not in stats:
                        stats['reconstruction'] = self.reconstruct(v_pos)
                    err += np.sum((v_pos - stats['reconstruction']) ** 2)

            if checkpoint is not None and ((e + 1) % checkpoint) == 0:
                err = np.sqrt(err / data.size)
                self.err.append(err)
                print("Epoch {}: {} | {}".format(e+1,
                                                 time.time() - start,
                                                 err))
                start = time.time()

    def reconstruct(self, data):
        return self.dream(data).next()

    def dream(self, data, steps=1, known_mask=None, known_values=None):
        """
        Generator that returns samples separated by #steps
        Returns probabilities
        """
        data = np.reshape(data, (-1, self.connections[0].dim_b))  # Ensure input is 2d array
        states = trainers.initialize_states(self, data)

        if known_mask is not None and known_values is not None:
            known_mask = np.reshape(known_mask, (1, self.connections[0].dim_b))
            known_values = np.reshape(known_values, (1, self.connections[0].dim_b))

        while True:
            for s in range(steps):
                # Up to hidden
                for i in range(1, len(self.layers)):
                    _, states[i] = self.sample(i, states)

                # Back down to visible
                for i in range(len(self.layers)-1, -1, -1):
                    exp, states[i] = self.sample(i, states)

                # Mask visible layer with known values
                if known_mask is not None:
                    exp[:, known_mask] = known_values[known_mask]
                    states[0][:, known_mask] = known_values[known_mask]
            yield exp


class BinaryRBM(Model):
    def __init__(self, num_v, num_h, model_stat=None):
        l1 = layers.BinaryLayer(num_v)
        l2 = layers.BinaryLayer(num_h)
        c1 = connections.FullConnection(num_v, num_h)
        if model_stat is None:
            model_stat = trainers.CD_model()
        stats = [trainers.CD_data(), model_stat]
        Model.__init__(self, [l1, l2], [c1], stats)


class ShapeRBM(Model):
    def __init__(self, num_v, num_h, patches, model_stat=None, data=None,
                 v_damping=0.3, w_init=0.1, double_up=False, double_down=False):
        if data is not None:
            mean_v = v_damping + (1-2*v_damping)*np.mean(data, axis=0)
            bias_v = np.log(mean_v / (1.0 - mean_v))
            l1 = layers.BinaryLayer(num_v, initial_bias=bias_v)
        else:
            l1 = layers.BinaryLayer(num_v)
        l2 = layers.BinaryLayer(num_h)
        c1 = connections.ShapeBMConnection(num_v, num_h, patches, w_init=w_init,
                                           double_down=double_down, double_up=double_up)
        if model_stat is None:
            model_stat = trainers.PCD_model()
        stats = [trainers.CD_data(), model_stat]
        Model.__init__(self, [l1, l2], [c1], stats)


class ShapeBM(Model):
    def __init__(self, num_v, num_h1, num_h2, patches):
        self.num_v = num_v
        self.num_h1 = num_h1
        self.num_h2 = num_h2
        self.patches = patches
        blayers = [layers.BinaryLayer(s) for s in [num_v, num_h1, num_h2]]
        c1 = connections.ShapeBMConnection(num_v, num_h1, patches)
        c2 = connections.FullConnection(num_h1, num_h2)
        stats = [trainers.MF_data(10), trainers.PCD_model(5)]
        Model.__init__(self, blayers, [c1, c2], stats)

    def pretrain(self, data,
                 epoch=[3000, 3000],
                 v_damping=[0.3, 1e-10],
                 w_init=[0.01, 0.1],
                 lr=[0.001, 0.002],
                 batch_size=64,
                 lr_schedule=trainers.lr_slow_start,
                 rbml1=None, rbml2=None,
                 rbml1_path=None, rbml2_path=None,
                 checkpoint=None):
        if rbml1 is None:
            rbml1 = ShapeRBM(self.num_v, self.num_h1, self.patches,
                             model_stat=trainers.CD_model(),
                             data=data,
                             v_damping=v_damping[0],
                             w_init=w_init[0],
                             double_up=True)
            rbml1.train(lr=lr[0], epoch=epoch[0], batch_size=batch_size,
                        data=data, lr_schedule=lr_schedule,
                        checkpoint=checkpoint)
            if rbml1_path:
                utils.save_model(rbml1, rbml1_path)
        self.rbml1 = rbml1

        if rbml2 is None:
            data_l2 = rbml1.expectation(1, [data, None])
            rbml2 = ShapeRBM(self.num_h1, self.num_h2,
                             patches=[slice(None, None, None)],
                             model_stat=trainers.CD_model(),
                             data=data_l2,
                             v_damping=v_damping[1],
                             w_init=w_init[1],
                             double_down=True)
            rbml2.train(lr=lr[1], epoch=epoch[1], batch_size=batch_size,
                        data=data_l2, lr_schedule=lr_schedule,
                        checkpoint=checkpoint)
            if rbml2_path:
                utils.save_model(rbml2, rbml2_path)
        self.rbml2 = rbml2

        # Combine parameters to full dbm
        self.connections[0].W = rbml1.connections[0].W.copy()
        self.connections[1].W = rbml2.connections[0].W.copy()
        self.layers[0].bias = rbml1.layers[0].bias.copy()
        self.layers[1].bias = rbml1.layers[1].bias + rbml2.layers[0].bias
        self.layers[2].bias = rbml2.layers[1].bias.copy()
        return
