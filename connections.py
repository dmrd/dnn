"""
Connections for constructing layered Boltzmann machines

A connection connects two layers and calculates activations between them

Deeper layers are "top", and lower layers (close to visible units) are "bottom"
"""
import gnumpy as gp


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
    def __init__(self, dim_bottom, dim_top, double_up=False, double_down=False):
        self.dim_b = dim_bottom
        self.dim_t = dim_top
        self.W = 0.1 * gp.randn(dim_top, dim_bottom)
        self.double_up = double_up
        self.double_down = double_down

    def prop_up(self, bottom):
        if self.double_up:
            bottom = bottom * 2
        return gp.dot(bottom, self.W.T)

    def prop_down(self, top):
        if self.double_down:
            top = top * 2
        return gp.dot(top, self.W)

    def gradient(self, b_pos, t_pos, b_neg, t_neg):
        return (gp.dot(b_pos.T, t_pos) - gp.dot(b_neg.T, t_neg)).T


class ShapeBMConnection(Connection):
    def __init__(self, dim_bottom, dim_top, patches, w_init=0.1,
                 double_up=False, double_down=False):
        # Each patch gets same number of units
        assert(dim_top % len(patches) == 0)
        if len(patches) > 1:
            assert(all(patch.size == patches[0].size for patch in patches))
        self.dim_b = dim_bottom
        self.dim_t = dim_top
        self.patches = patches
        self.patch_size = gp.zeros(dim_bottom)[patches[0]].size
        self.W = w_init * gp.randn(dim_top / len(patches), self.patch_size)
        self.double_up = double_up
        self.double_down = double_down

    def prop_up(self, bottom):
        if self.double_up:
            bottom = bottom * 2
        raw = gp.zeros((bottom.shape[0], self.dim_t))
        stride = self.dim_t / len(self.patches)
        for i, patch in enumerate(self.patches):
            raw[:, i * stride:(i + 1) * stride] += gp.dot(bottom[:, patch], self.W.T)
        return raw

    def prop_down(self, top):
        if self.double_down:
            top = top * 2
        raw = gp.zeros((top.shape[0], self.dim_b))
        stride = self.dim_t / len(self.patches)
        for i, patch in enumerate(self.patches):
            raw[:, patch] += gp.dot(top[:, i * stride:(i + 1) * stride], self.W)
        return raw

    def gradient(self, b_pos, t_pos, b_neg, t_neg):
        w_grad = gp.zeros(self.W.shape)
        stride = self.dim_t / len(self.patches)
        for i, patch in enumerate(self.patches):
            h_slice = slice(i*stride, (i+1)*stride)
            w_grad += (gp.dot(b_pos[:, patch].T, t_pos[:, h_slice])
                       - gp.dot(b_neg[:, patch].T, t_neg[:, h_slice])).T
        return w_grad
