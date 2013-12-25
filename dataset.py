import cPickle
import gzip
import os
import numpy as np
import Image


def mnist():
    # Load the dataset
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    #train_set, valid_set, test_set = cPickle.load(f)
    train_set, _, _ = cPickle.load(f)
    f.close()
    return train_set[0]


def image_to_np(image, size):
    """
    Takes binary image array, crops to content, returns as nparray.
    (x, y) final size tuple
    """
    # Autocrop image
    image = image.crop(image.getbbox())
    return np.array(image.resize(size).getdata()) > 127


def load_dataset(directory, size=(32, 32)):
    """Load an image dataset from the given directory"""
    print("Loading dataset...")
    data = []
    for name in os.listdir(directory):
        image = Image.open(os.path.join(directory, name))
        data.append(image_to_np(image, size))
    print("Dataset loaded")
    return np.array(data)
