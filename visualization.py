import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


class Sampler:
    pass


def animate2d(sampler):
    return animation.FuncAnimation(plt.figure(), sampler)


def reshape_image(im, dim=None):
    if dim is None:
        dim = (-1, int(im.size**(0.5) + 0.5))
    return np.reshape(im, dim)


def show_image(im, dim=None):
    plt.imshow(reshape_image(im, dim), interpolation='nearest', cmap=plt.gray())
    plt.show()


def plot_images(images, image_dim, grid_dim, space=5, size=None):
    rows, cols = grid_dim
    height, width = image_dim
    height += space
    width += space
    result = np.zeros((rows * height, cols * width))
    for y in range(rows):
        for x in range(cols):
            im = reshape_image(images[y*cols + x], image_dim)
            # Normalize image to [0, 1] to maintain same intensity throughout grid
            normalized = (im - im.min())
            normalized /= (normalized.max() or 1.0)
            result[y * height:(y+1)*height - space, x * width: (x+1)*width - space] = normalized
    plt.imshow(result, interpolation='nearest', cmap=plt.gray())
    f = plt.gcf()
    if size:
        f.set_size_inches(size[0], size[1])
    plt.axis('off')
    plt.show()


def mean_activations(rbm, data, size):
    expectations = rbm.expectation(1, [data, None])
    average = expectations.mean(axis=0)

    plt.imshow(np.reshape(average, size),
               interpolation="nearest", cmap=plt.gray())
    plt.show()
