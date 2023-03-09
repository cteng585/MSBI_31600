from data_set_processing import *
from image_processing import *
from model import *
from layers import *

import pickle

import typer

def make_new_fashion_model(fashion_mnist_path: str, train=True, batch_size: int = 60, epochs: int = 100) -> Model:
    """
    make a new model to predict clothing types using the Fashion MNIST data set

    :param fashion_mnist_path: path to the Fashion MNIST directory
    :param train: a bool for whether the model should be trained on the Fashion MNIST data
    :return:
    """
    if train:
        # make the training and testing data sets

        X, y, X_test, y_test = create_mnist_data(fashion_mnist_path)

        # the data needs to be pre-processed so that it can be used to train the neural network
        # this includes:
        # 1. scaling the data
        # 2. flattening the arrays for input
        # 3. shuffling the data to avoid batch bias
        X, y = process_mnist_data(X, y)
        X_test, y_test = process_mnist_data(X_test, y_test)

    # instantiate the network
    model = Model()

    # add layers
    model.add(LayerDense(784, 64))
    model.add(ActivationReLU())
    model.add(LayerDense(64, 64))
    model.add(ActivationReLU())
    model.add(LayerDense(64, 10))
    model.add(ActivationSoftmax())

    # set training objects
    model.set(
        loss=LossCategoricalCrossEntropy(),
        optimizer=OptimizerSGD(decay=0.001),
        accuracy_calculator=Accuracy(),
    )

    # link layers
    model.finalize()

    # only train if not planning on passing saved parameters later
    if train:
        # train the model (fingers-crossed)
        model.train(X, y, validation_data=(X_test, y_test), batch_size=batch_size, summary_freq_batch=10, epochs=epochs,
                    summary_freq_epoch=5)

    return model


def save_model(model: Model, save_str: str) -> None:
    """
    saves a previously trained model to a pickle file
    :param model: the model to save the parameters of
    :param save_str: the name of the file to save the parameters to
    :return:
    """
    parameters = model.get_parameters()

    with open(save_str, "wb") as outfile:
        pickle.dump(parameters, outfile)


def load_model(model: Model, saved_parameters: str) -> None:
    """
    loads the parameters from a previously trained model to the current model

    :param model: the model to load saved parameters into
    :return:
    """
    with open(saved_parameters, "rb") as infile:
        parameters = pickle.load(infile)

    model.set_parameters(parameters)


def predict(model: Model, X: np.array) -> Tuple[str, float]:
    """
    use the model to predict the classification of a novel input

    :param model: the model to use for the classification task
    :param X: the novel input to try to classify
    :return: the string corresponding to the article of clothing that is predicted
    """
    # from the fashion MNIST repo: https://github.com/zalandoresearch/fashion-mnist
    clothing_dict = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    # do the prediction
    model.predict(X)
    prediction = int(model.output_layer_activation.predictions())
    confidence = model.output_layer_activation.output[0, prediction]

    return clothing_dict[prediction], confidence


def main(
        mnist_path: str = typer.Option(
            "fashion_mnist_images", help="path to the Fashion MNIST data set, doesn't need to be passed if loading a "
                                         "model"
        ),
        predict_image: str = typer.Option(
            None, help="the image to predict"
        ),
        light_bg: bool = typer.Option(
            True, help="whether the image to label has a light background (True) or not (False)"
        ),
        build: bool = typer.Option(
            False, help="train a new model from scratch"
        ),
        batch_size: int = typer.Option(
            30, help="how large of batches to step through the input data set"
        ),
        epochs: int = typer.Option(
            10, help="number of training epochs"
        ),
        load_params: str = typer.Option(
            "", help="use the parameters from a saved model"
        ),
        save: str = typer.Option(
            None, help="save the parameters of the model to file"
        )
):
    try:
        if build:
            # make a new model
            print("Not loading any saved model parameters")
            model = make_new_fashion_model(fashion_mnist_path=mnist_path, batch_size=batch_size, epochs=epochs)
        elif load_params:
            model = make_new_fashion_model(fashion_mnist_path=mnist_path, train=False)
            load_model(model, load_params)
        else:
            print("Pass a valid runtime option")
            return

    except FileNotFoundError:
        raise FileNotFoundError(f"The passed path: {mnist_path} to the training data set couldn't be found")

    if save:
        save_model(model, save)

    if predict_image:
        image_data = load_image(predict_image, light_bg)
        prediction, confidence = predict(model, image_data)
        print(f"That looks like a {prediction} with {confidence * 100:.2f}% certainty")


if __name__ == "__main__":
    typer.run(main)
