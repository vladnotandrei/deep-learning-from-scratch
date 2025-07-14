import pickle
import numpy as np


def load_batch(filename):
    """
    Read a single CIFAR-10 batch file. Return image data and labels.

    Args:
        filename (str): Path to the CIFAR-10 batch file.
    Returns:
        X (numpy.ndarray, dtype=float64) (n_imgs, img_dim): Image pixel data with entries in [0, 1].
        Y (numpy.ndarray, dtype=float64) (n_labels, n_imgs): One-hot encoded labels.
        y (numpy.ndarray, dtype=int32) (n_imgs,): Class labels as integers between 0 and 9, will be used for indexing.
    """
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        fo.close()
    
    X = dict[b'data'].astype(np.float64) / 255.0
    y = np.array(dict[b'labels'])  # (n_imgs,)

    # Create one-hot encoded labels
    n_imgs = X.shape[0]
    n_labels = 10
    Y = np.zeros((n_labels, n_imgs), dtype=np.float64)
    for i in range(n_imgs):
        Y[y[i], i] = 1.0
    
    return X, Y, y