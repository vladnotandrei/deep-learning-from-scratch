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
    
    X = dict[b'data'].astype(np.float64) / 255.0  # (n_imgs, img_dim), normalize to [0, 1]
    X = X.transpose()  # (img_dim, n_imgs)  
    y = np.array(dict[b'labels'])  # (n_imgs,)

    # Create one-hot encoded labels
    n_imgs = X.shape[1]
    n_labels = 10
    Y = np.zeros((n_labels, n_imgs), dtype=np.float64)
    for i in range(n_imgs):
        Y[y[i], i] = 1.0
    
    return X, Y, y


def preprocess_data(X):
    """
    Transform training data to have zero mean.
    Args:
        X (numpy.ndarray, dtype=float64) (img_dim, n_imgs): Data.
    Returns:
        X (numpy.ndarray, dtype=float64) (img_dim, n_imgs): Preprocessed data.
    """
    img_dim = X.shape[0]
    mean_X = np.mean(X, axis=1).reshape(img_dim, 1)  # (img_dim, 1)
    std_X = np.std(X, axis=1).reshape(img_dim, 1)  # (img_dim, 1)

    # Standardization (with numpy broadcasting)
    X = X - mean_X  # (img_dim, n_imgs)  
    X = X / std_X  # (img_dim, n_imgs)
    return X


def initialize_model_parameters():
    """
    Initialize parameters of model W and b in a dict, with each entry initialized
    with Gaussian random values with mean=0 and std=0.01.

    Returns:
        model (dict): Model parameters W and b.
            W (numpy.ndarray, dtype=float64) (n_labels, img_dim): Weight matrix.
            b (numpy.ndarray, dtype=float64) (n_labels, 1): Bias vector. 
    """
    rng = np.random.default_rng(seed=42)
    # bit_gen = type(rng.bit_generator)  # get the BitGenerator used by default rng
    # seed = 42  # use the state from a fresh bit generator
    # rng.bit_generator.state = bit_gen(seed=seed).state

    init_net = {}

    # NOTE: c * N(0, 1) = N(0, c^2)
    init_net['W'] = 0.01 * rng.standard_normal(size=(10, 3072))  # (n_labels, img_dim)
    init_net['b'] = 0.01 * rng.standard_normal(size=(10, 1))  # (n_labels, 1)
    return init_net


def apply_network(X, network):
    """
    Apply network to data X.

    Args:
        X (numpy.ndarray, dtype=float64) (img_dim, n_imgs): Data.
        network (dict): Model parameters W and b.
            W (numpy.ndarray, dtype=float64) (n_labels, img_dim): Weight matrix.
            b (numpy.ndarray, dtype=float64) (n_labels, 1): Bias vector.
    Returns:
        P (numpy.ndarray, dtype=float64) (n_labels, n_imgs): Probability for each label for each image.
    """
    W = network['W']  # (n_labels, img_dim)
    b = network['b']  # (n_labels, 1)
    s = W @ X + b  # (n_labels, n_imgs)
    exp_s = np.exp(s)  # (n_labels, n_imgs)
    P = exp_s / np.ones((1, exp_s.shape[0])) @ exp_s  # Softmax via broadcasting, (n_labels, n_imgs)
    return P







