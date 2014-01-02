import cPickle
import gzip
import os
from urllib2 import urlopen, URLError, HTTPError
import numpy as np
import Image


def download_file(url, filename):
    try:
        f = urlopen(url)
        with open(filename, 'wb') as target:
            data = f.readlines()
            target.writelines(data)
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url


def mnist():
    """ Load the pickled MNIST file from deeplearning.net """
    filename = './data/mnist.pkl.gz'
    if not os.path.isfile(filename):
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz (16MB)...'
        print("Downloading mnist dataset from {}".format(url))
        download_file(url, filename)

    # Load the dataset
    f = gzip.open(filename, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # Only using training set (50,000 examples) for now
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
