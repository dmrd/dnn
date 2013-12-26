import os
import pickle
import numpy as np


def create_patches(num_v, patches, dimensions, overlap):
    """
    patches: (x, y, z) tuple of # patches to split each dimension into
    dimensions: (x, y, z) tuple of size of each dimension
    overlap: size of overlap between adjacent patches

    Return an array of patch arrays, each of which is a set of indices into the
    visible layer.  Can also be replaced with logical indexing (e.g. boolean mask).
    """
    if sum(patches) == 3:
        return [slice(None, None, None)]
    final = []
    # Patches for division of visible layer
    if patches is None:
        final = np.ones((1, num_v))  # All units included
    else:
        assert(dimensions)
        assert(num_v == dimensions[0] * dimensions[1] * dimensions[2])
        assert(dimensions[0] % patches[0] == 0 and
               dimensions[1] % patches[1] == 0 and
               dimensions[2] % patches[2] == 0)  # Check even split of dimensions
        assert(all(1 <= d <= 2 for d in patches))  # Can split each dim into 1 or 2 patches
        xp = dimensions[0] / patches[0]
        yp = dimensions[1] / patches[1]
        zp = dimensions[2] / patches[2]
        for x in range(patches[0]):
            for y in range(patches[1]):
                for z in range(patches[2]):
                    patch = np.zeros(dimensions)
                    patch[max(0, x*xp-overlap):min(dimensions[0], (x + 1)*xp + overlap),
                          max(0, y*yp-overlap):min(dimensions[1], (y + 1)*yp + overlap),
                          max(0, z*zp-overlap):min(dimensions[2], (z + 1)*zp + overlap)] = True
                    final.append(np.where(patch.flatten())[0])
    # Check that all patches have same number of on units
    assert(all(x.size == final[0].size for x in final))
    return np.asanyarray(final)


def save_model(m, name):
    # Save as .model file if no extension specified
    if not os.path.splitext(name)[1]:
        name += ".model"
    with open(name, 'wb') as f:
        pickle.dump(m, f)


def load_model(name):
    if os.path.isfile(name):
        return pickle.load(open(name, "rb"))
    return None
