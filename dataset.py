import cPickle
import gzip


def mnist():
    # Load the dataset
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    #train_set, valid_set, test_set = cPickle.load(f)
    train_set, _, _ = cPickle.load(f)
    f.close()
    return train_set[0]
