import os
from typing import Tuple

import cv2
import numpy as np


def load_mnist_dataset(dataset: str, path: str) -> Tuple[np.array, np.array]:
    """
    load an MNIST data set
    :param dataset: the type of data set (either "test" or "train")
    :param path: the path to the parent directory of the data set
    :return:
    """
    labels = os.listdir(os.path.join(path, dataset))

    # this will be the image data
    X = []
    # this will be the "true" label of the image data (0-9)
    y = []

    # going over all the images in the MNIST dataset, get their "true" labels (0-9)
    for label in labels:
        # deal with annoying macOS indexing feature
        if label != ".DS_Store":
            for file in os.listdir(os.path.join(path, dataset, label)):
                # read in the image as an array of grayscale values
                image_data = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
                X.append(image_data)
                y.append(label)

    # convert the data to numpy arrays for ingestion into the neural network
    return np.array(X), np.array(y).astype("uint8")


def create_mnist_data(path: str) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    loads the MNIST data set
    :param path: the path to the Fashion MNIST directory
    :return:
    """
    # load the training set
    X, y = load_mnist_dataset("train", path)

    # load the test set
    X_test, y_test = load_mnist_dataset("test", path)

    return X, y, X_test, y_test


def process_mnist_data(dataset: np.array, labels: np.array) -> Tuple[np.array, np.array]:
    """
    do some pre-processing so that the data is optimized for ingestion and testing/training by the network
    :param dataset: the data set to scale
    :param labels: the "true" values for all feature sets
    :return:
    """
    # scale the data set so that all array values are between -1 and 1
    scaled_dataset = (dataset.astype(np.float32) - 127.5) / 127.5

    # the neural network takes in a 1D vector. the input data sets need to be reshaped to this shape so that
    # the neural network can handle the data
    flattened_dataset = scaled_dataset.reshape(scaled_dataset.shape[0], -1)

    # the data set needs to be shuffled
    # since the data was loaded from the path in order, all labels of one type are batched
    # this would cause the network to overfit as it would only see one class of image for a while
    # before rapidly switching to the next
    indices = np.array(range(flattened_dataset.shape[0]))
    np.random.shuffle(indices)
    shuffled_dataset = flattened_dataset[indices]
    shuffled_labels = labels[indices]

    return shuffled_dataset, shuffled_labels
