class Updater:
    def __init__():
        pass


# IMPLEMENT AS CLASSES WITH __call__ METHOD

# Data dependent statistics
def CD_data(model, data, stats, *args):
    """
        Sets stats: 'data'
    """
    assert(len(model.connections) == 1)  # Only use for RBMs
    h_pos_exp = model.expectation(1, [data, None])
    stats['data'] = [data, h_pos_exp]


def mean_field(model, data, stats, *args, steps=10, convergence=0.00001):
    # Initialize states for mean-field inference
    # Use doubled up weights (equivalent to doubling input) to account for lack
    # of top-down input for inside layers
    layers = [data]
    for connection in model.connections[:-1]:
        layers.append(connection.prop_up(layers[-1] * 2))
    layers.append(model.connections[-1].prop_up(layers[-1]))

    for step in range(steps):
        # Sample even layers
        for layer in range(2, len(layers), 2):
            layers[layer] = model.expectation(layer, layers)

        # Sample odd layers
        for layer in range(1, len(layers), 2):
            layers[layer] = model.expectation(layer, layers)

    stats['data'] = layers


# Model dependent statistics
def CD_model(model, data, stats, *args, steps=1):
    """
        Sets stats: 'model'

        We sample for everything to avoid letting any unit convey more
        information than it should (e.g. binary units should not be able to
        pass through a none binary value)

        Use expectation for final step
    """
    assert(len(model.connections) == 1)  # Only use for RBMs
    h_pos_states = model.layers[1].sample_exp(stats['data'][1])
    v_neg = model.sample(0, [None, h_pos_states])
    for step in range(steps - 1):
        h_neg = model.sample(1, [v_neg, None])
        v_neg = model.sample(0, [None, h_neg])
    h_neg = model.expectation(1, [v_neg, None])

    stats['model'] = [v_neg, h_neg]


def PCD_model(model, data, stats, state, steps=5):
    # Chains correspond to states from last round
    chains = state['chains']

    for step in range(steps):
        # Sample even layers
        for layer in range(0, len(chains), 2):
            chains[layer] = model.sample(layer, chains)

        # Sample odd layers
        for layer in range(1, len(chains), 2):
            chains[layer] = model.sample(layer, chains)

    stats['model'] = chains
    state['chains'] = chains
