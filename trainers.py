def initialize_states(model, data):
    """
    Initialize states for a deep boltzmann machine given data.
    Double weights up to account for lack of top down input
    """
    layers = [data]
    for connection in model.connections[:-1]:
        layers.append(connection.prop_up(layers[-1] * 2))
    layers.append(model.connections[-1].prop_up(layers[-1]))


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
        self.steps = 1

    def __call__(self, model, data, stats):
        assert(len(model.connections) == 1)  # Only use for RBMs
        h_pos_states = model.layers[1].sample_exp(stats['data'][1])
        v_neg_prob = model.expectation(0, [None, h_pos_states])
        v_neg_states = model.layers[0].sample_exp(v_neg_prob)
        h_neg_prob = model.expectation(1, [v_neg_states, None])

        stats['model'] = [v_neg_prob, h_neg_prob]


class PCD_model:
    def __init__(self, steps=5):
        self.steps = 5
        self.chains = None

    def __call__(self, model, data, stats):
        # Chains correspond to states from last round
        if self.chains is None:
            self.chains = [data]
            for connection in model.connections[:-1]:
                self.chains.append(connection.prop_up(self.chains[-1] * 2))
            self.chains.append(model.connections[-1].prop_up(self.chains[-1]))

        for step in range(self.steps):
            # Sample even layers
            for layer in range(0, len(self.chains), 2):
                self.chains[layer] = model.sample(layer, self.chains)

            # Sample odd layers
            for layer in range(1, len(self.chains), 2):
                self.chains[layer] = model.sample(layer, self.chains)

        stats['model'] = self.chains
