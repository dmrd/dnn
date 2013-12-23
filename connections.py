"""
Connections for constructing layered Boltzmann machines

A connection connects two layers and calculates activations between them

Deeper layers are "top", and lower layers (close to visible units) are "bottom"
"""
import numpy as np


class Connection:
    def __init__(self, dim_bottom, dim_top):
        self.dim_b = dim_bottom
        self.dim_t = dim_top

    def prop_up(self, states):
        pass

    def prop_down(self, states):
        pass

    def gradient(self, b_pos, t_pos, b_neg, t_neg):
        pass

    def gradient_update(self, update):
        self.W += update

    def __repr__(self):
        return str(self.W)

    def __getitem__(self, key):
        return self.W[key]


class FullConnection(Connection):
    def __init__(self, dim_bottom, dim_top):
        self.dim_b = dim_bottom
        self.dim_t = dim_top
        self.W = 0.1 * np.random.randn(dim_top, dim_bottom)

    def prop_up(self, bottom):
        return np.dot(bottom, self.W.T)

    def prop_down(self, top):
        return np.dot(top, self.W)

    def gradient(self, b_pos, t_pos, b_neg, t_neg):
        return (np.dot(b_pos.T, t_pos) - np.dot(b_neg.T, t_neg)).T


class ShapeBMConnection(Connection):
    def __init__(self, dim_bottom, dim_top, patches):
        # Each patch gets same number of units
        assert(dim_top % len(patches) == 0)
        assert(all(patch.size == patches[0].size for patch in patches))
        self.dim_b = dim_bottom
        self.dim_t = dim_top
        self.patches = patches
        self.W = np.zeros((dim_top / len(patches), dim_bottom))

    def prop_up(self, bottom):
        raw = np.zeros(bottom.shape[0], self.dim_t)
        stride = self.dim_t / len(self.patches)
        for i, patch in enumerate(self.patches):
            raw[:, i * stride:(i + 1) * stride] += np.dot(bottom[:, patch], self.W.T)
        return raw

    def prop_down(self, top):
        raw = np.zeros(top.shape[0], self.dim_b)
        stride = self.dim_t / len(self.patches)
        for i, patch in enumerate(self.patches):
            raw[:, patch] += np.dot(top[:, i * stride:(i + 1) * stride], self.W)
        return raw

    def gradient(self, b_pos, t_pos, b_neg, t_neg):
        w_grad = np.zeros(self.W.shape)
        stride = self.num_h1 / len(self.patches)
        for i, patch in enumerate(self.patches):
            h_slice = slice(i*stride, (i+1)*stride)
            w_grad += (np.dot(b_pos[:, patch].T, t_pos[:, h_slice])
                       - np.dot(b_neg[:, patch].T, t_neg[:, h_slice])).T
        return w_grad
