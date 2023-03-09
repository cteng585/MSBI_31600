import cv2
import numpy as np


def load_image(image_path: str, light_bg: bool):
    """
    load in an image (in PNG format) and pre-process it for use with the neural network

    :param image_path: the path to the image
    :param light_bg: a bool for whether the background of the image is lighter colored.
        this is important since the model was trained on images with a dark background
    :return:
    """
    # force grayscale since that's what the training data for the neural network used
    image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # if the background of the image is light coloured, invert to match the Fashion MNIST data
    if light_bg:
        inverted_image_data = 255. - image_data
        # downscale the image to the resolution that is compatible with the neural network
        downscaled_image_data = cv2.resize(inverted_image_data, (28, 28))
    else:
        downscaled_image_data = cv2.resize(image_data, (28, 28))


    # flatten and scale the pixel brightness values
    processed_image_data = (downscaled_image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    return processed_image_data
