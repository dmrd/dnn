import gnumpy as gp


def initialize_states(model, data):
    """
    Initialize states for a deep boltzmann machine given data.
    Double weights up to account for lack of top down input
    """
    states = [data]
    for i, connection in enumerate(model.connections[:-1]):
        states.append(model.layers[i + 1].expectation(connection.prop_up(states[-1] * 2)))

    # Last layer has only input from below, so no doubling
    states.append(model.layers[-1].expectation(model.connections[-1].prop_up(states[-1])))
    return states


# Data dependent statistics
class CD_data:
    """ Data dependent contrastive divergence """
    def __call__(self, model, data, stats):
        """
            Sets stats: 'data'
        """
        assert(len(model.connections) == 1)  # Only use for RBMs for now
        h_pos_exp = model.expectation(1, [data, None])
        stats['data'] = [data, h_pos_exp]


class MF_data:
    """ Data dependent mean field inference """
    def __init__(self, steps=10, convergence=0.00001):
        self.steps = steps
        self.convergence = convergence

    def __call__(self, model, data, stats):
        # Initialize states for mean-field inference
        # Use doubled up weights (equivalent to doubling input) to account for lack
        # of top-down input for inside layers
        layers = initialize_states(model, data)

        for step in range(self.steps):
            # Sample even layers
            for layer in range(2, len(layers), 2):
                layers[layer] = model.expectation(layer, layers)

            # Sample odd layers
            for layer in range(1, len(layers), 2):
                layers[layer] = model.expectation(layer, layers)

        stats['data'] = layers


# Model dependent statistics
class CD_model:
    """
        Sets stats: 'model'

        We sample (instead of using expectations) for each step to avoid
        letting any unit convey more information than it should (e.g. binary
        units should not be able to pass decimal values)

        Use expectation for final step
    """
    def __init__(self, steps=1):
        self.steps = steps

    def __call__(self, model, data, stats):
        assert(len(model.connections) == 1)  # Only use for RBMs
        h_states = model.layers[1].sample_exp(stats['data'][1])
        for i in range(self.steps):
            v_neg_prob, v_states = model.sample(0, [None, h_states])
            h_neg_prob, h_states = model.sample(1, [v_states, None])

        stats['model'] = [v_neg_prob, h_neg_prob]
        stats['reconstruction'] = v_neg_prob


class PCD_model:
    def __init__(self, steps=1):
        self.steps = steps
        self.chains = None

    def __call__(self, model, data, stats):
        # Chains correspond to states from last round

        if self.chains is None:
            self.chains = initialize_states(model, data)

        for step in range(self.steps - 1):
            # Sample even layers
            for layer in range(0, len(self.chains), 2):
                _, self.chains[layer] = model.sample(layer, self.chains)

            # Sample odd layers
            for layer in range(1, len(self.chains), 2):
                _, self.chains[layer] = model.sample(layer, self.chains)

        # Save expectations in last step to return
        result = [None] * len(model.layers)
        for layer in range(0, len(self.chains), 2):
            result[layer], self.chains[layer] = model.sample(layer, self.chains)
        for layer in range(1, len(self.chains), 2):
            result[layer], self.chains[layer] = model.sample(layer, self.chains)

        stats['model'] = result


# Learning rate schedules
def lr_constant(lr, epoch, epochs):
    return lr


def lr_linear(lr, epoch, epochs):
    return lr * (1.0 - (epoch / float(epochs)))


def lr_slow_start(lr, epoch, epochs):
    lr_kink1 = epochs/10
    lr_kink2 = epochs/5

     #Adjust learning rate between epochs
    if epoch < lr_kink1:
        return lr * epoch/lr_kink1/10
    elif epoch < lr_kink2:
        ledge = lr/10
        return ledge + (lr - ledge) * (epoch-lr_kink1)/(lr_kink2-lr_kink1)
    return lr


class Trainer(object):
    def __init__(self):
        pass

    def get_update(self, model, stats, i, learning_rate, epoch, **kwargs):
        raise NotImplemented("No update method defined")

    def update_state(self, model, i, update):
        pass


class Gradient(Trainer):
    def get_update(self, model, stats, i, learning_rate, epoch):
        dd = stats['data']  # Data dependent
        md = stats['model']  # Model dependent
        return learning_rate * model.connections[i].gradient(dd[i], dd[i+1], md[i], md[i+1])


class L1WeightDecay(Trainer):
    def __init__(self, strength):
        self.strength = strength

    def get_update(self, model, stats, i, learning_rate, *args):
        return -self.strength * gp.sign(model.connections[i].W)


class L2WeightDecay(Trainer):
    def __init__(self, strength):
        self.strength = strength

    def get_update(self, model, stats, i, learning_rate, *args):
        return -self.strength * learning_rate * model.connections[i].W


class Momentum(Trainer):
    def __init__(self, strength):
        self.strength = strength
        self.last = [None]

    def get_update(self, model, stats, i, *args):
        while len(self.last) <= i:
            self.last.append(None)
        if self.last[i] is None:
            return 0
        return self.strength * self.last[i]

    def update_state(self, model, i, update):
        self.last[i] = update
        return
