from typing import Tuple

import numpy as np


class Layer:
    def __init__(self):
        # for keeping track of layer order in the model. important for connectedness
        self.previous = None
        self.next = None


class LayerInput(Layer):
    """
    a layer of the network to handle the input data
    """

    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, inputs) -> np.array:
        """
        take the input data in the forward pass and pass it along to the next layer
        :return: the outputs (really just the input array since this layer doesn't do anything but make it
            traverse the network during training)
        """
        self.output = inputs


class LayerDense(Layer):
    """
    a "hidden layer" in the neural network
    these are called "dense" because all neurons of adjacent layers are connected

    """

    def __init__(self, n_inputs: int, n_neurons: int):
        super().__init__()
        # we need to know the size of the input coming in, and we want to know how many neurons are in
        #   the layer
        # the size of the input coming in is length of inputs per set in a batch of inputs
        # randn is a Gaussian distribution bounded around 0
        # scale the matrix by 0.1 to ensure that elements of the weights matrix are not larger than abs(1)
        # the way this weights matrix is created here, the neurons of the layer are the COLUMNS of the
        #   weights matrix, and each element of a column are the weights a SINGLE neuron gives to an
        #   element of the input; in the event that there is a preceding layer, each element of a column
        #   is the weight a SINGLE neuron of the CURRENT layer gives the output of a SINGLE neuron from
        #   the PREVIOUS layer.
        # this way we don't need to do the transpose operation each forward pass

        # initialize weights between -0.1 and 0.1
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # these will be used by the optimizer to prevent being stuck in local minima
        self.weight_momentums = np.zeros_like(self.weights)

        # gradient on values, to be passed backwards
        self.dinputs = None

        # gradients on parameters, to be used to nudge this layer towards correctness
        self.dweights = None
        self.dbiases = None

        # initialize biases at 0 (kind of the default)
        # this can be problematic in the case that if the neurons aren't firing initially,
        # 0's might propagate through the network (0 * weight is always 0, and if biases of future layers are also 0...)
        # causing a dead network
        self.biases = np.zeros((1, n_neurons))

        # these will be used by the optimizer to prevent being stuck in local minima
        self.bias_momentums = np.zeros_like(self.biases)

        # inputs feed into the layer
        self.inputs = None

        # outputs proceed from the layer
        self.output = None

    def get_parameters(self) -> Tuple[np.array, np.array]:
        """
        accessor method to return the weights and biases of this layer
        :return:
        """
        return self.weights, self.biases

    def set_parameters(self, weights, biases) -> None:
        """
        setter method to set the weights and biases. used if a model was previous saved

        :param weights: an array of weights to use for this layer
        :param biases: an array of biases to use for this layer
        :return:
        """
        self.weights = weights
        self.biases = biases

    def forward(self, inputs: np.array) -> None:
        """
        standard dot product to produce outputs of the layer
        the intuition here being that for each set (row) of inputs, we calculate a set (row) of outputs
        using the weights of neurons of this layer.

        :param inputs: a batch of inputs/feature sets
        :return: a matrix of outputs resulting from passing each set of inputs from a batch through
            this layer
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.array) -> None:
        """
        moving backwards through the neural network as the training step

        :param dvalues: gradients of activation output with respect to the layer input
        :return:
        """
        # gradient on values, to be passed backwards
        self.dinputs = np.dot(dvalues, self.weights.T)
        # gradients on parameters, to be used to nudge this layer towards correctness
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


class ActivationReLU:
    """
    implementation of the Rectified Linear activation function
    basically determines if a neuron will "fire" or not
    """

    def __init__(self):
        # start uninitialized, but `dinputs` will hold the neuron function gradients for passing backward
        self.dinputs = None
        # inputs feed into the layer
        self.inputs = None

        # outputs proceed from the layer
        self.output = None

    def forward(self, inputs: np.array) -> None:
        """
        if the weighted sum input into a layer doesn't exceed 0 the neuron will not fire

        :param inputs: the inputs to be evaluated for neuron "firing"
        :return:
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: np.array) -> None:
        """
        moving backward through the layer as a training step

        :param dvalues: neuron output gradients with respect to the input received from the "next" layer
        :return:
        """
        # the incoming gradients. copying to prevent unpredictable array object modification
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    """
    implementation of the Softmax Activation Function which should yield a probability distribution in the very
    last layer, ideal for classification problems
    """

    def __init__(self):
        # the inputs into the function
        self.inputs = None
        # the output of the function
        self.output = None
        # start uninitialized, but `dinputs` will hold the neuron function gradients for passing backward
        self.dinputs = None

    def forward(self, inputs: np.array) -> None:
        """
        moving forward through the layer, this should produce a probability distribution
        :param inputs: the neurons that are firing into this layer
        :return:
        """
        # implement the Softmax Activation Function
        # recall that one row of the inputs = one processed feature set from the PREVIOUS layer
        # therefore in the normalization step, we want to sum one ROW at a time to get the normalization for a
        #   feature set at a time
        # numpy functions can be passed an axis argument, where axis=0 operates over the columns and axis=1 operates
        #   over the rows
        # numpy functions can be passed a keepdims=True value which will maintain the "row" structure of the output
        #   result
        self.inputs = inputs
        bounded_values = inputs - np.max(np.array(inputs), axis=1, keepdims=True)
        exp_values = np.exp(bounded_values)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues: np.array) -> None:
        """
        passing the activation function gradient backward for training
        :param dvalues: numerical derivative values to calculate gradients from to be passed backward
        :return:
        """
        # create uninitialized array to hold the sample gradients
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            # flatten output array
            single_output = single_output.reshape(-1, 1)
            # calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            # calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self) -> np.array:
        """
        give the prediction (the highest likelihood outcome) of the forward pass for a particular sample
        :return: an array of a probability distribution for classification likelihoods
        """
        return np.argmax(self.output, axis=1)


# noinspection PyMethodMayBeStatic
class LossCategoricalCrossEntropy:
    """
    implementation of Categorical Cross Entropy loss, the standard for calculating loss in classification
    problems
    """

    def __init__(self):
        # start uninitialized, but `dinputs` will hold the neuron function gradients for passing backward
        self.dinputs = None

        # use these to keep track of loss across a batch in a training epoch. that way the model can calculate
        # the average loss for a training epoch and use batches to enhance the training step
        self.accumulated_loss = 0
        self.accumulated_count = 0
        self.trainable_layers = None

    def remember_trainable(self, trainable_layers: list[LayerDense]) -> None:
        """
        remember the layers that are actually trainable so that un-trainable layers are not
        iterated over during the loss calculation

        :param trainable_layers: a list of LayerDense object
        :return:
        """
        self.trainable_layers = trainable_layers

    def calculate(self, y_predictions: np.array, y_true: np.array) -> np.array:
        """
        calculate the loss associated with a batch of data
        also keep track of the accumulated loss over the entire training epoch

        :param y_predictions: np array of predicted values
        :param y_true: np array of true values
        :return: an array of the calculated loss comparing predicted labels vs true labels for
            a batch of data
        """
        sample_losses = self.forward(y_predictions, y_true)
        batch_loss = np.mean(sample_losses)

        # keeping track of loss across a training epoch
        self.accumulated_loss += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        return batch_loss

    def calculate_accumulated(self) -> float:
        """
        used to calculated average loss across a training epoch
        :return: the average loss across a training epoch
        """
        mean_loss = self.accumulated_loss / self.accumulated_count

        return mean_loss

    def forward(self, y_predictions: np.array, y_true: np.array) -> np.array:
        """
        the forward pass for the loss step
        :param y_predictions: predicted values for the given inputs
        :param y_true: the true labels for the given inputs
        :return: an array of the losses for a single pass
        """
        n_samples = len(y_predictions)
        y_predictions_clipped = np.clip(y_predictions, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_predictions_clipped[range(n_samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_predictions_clipped * y_true, axis=1)

        else:
            raise IndexError(
                "ShapeError: The shape of the true values isn't compatible"
            )

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    def backward(self, dvalues: np.array, y_true: np.array) -> None:
        """
        moving backwards through the loss layer
        :param dvalues: here, these are the predicted values
        :param y_true: the true labels for the samples being classified
        :return:
        """
        # get the number of samples in the batch
        n_samples = len(dvalues)

        # use the first sample in the batch to count the number of labels
        n_labels = len(dvalues[0])

        # if using a vector of scalar values to denote the true labels, turn into a one-hot encoded array
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]

        # calculate the derivative/gradient of the loss function with respect to the inputs
        self.dinputs = -y_true / dvalues

        # normalize the calculated gradient
        self.dinputs = self.dinputs / n_samples

    def new_pass(self) -> None:
        """
        used to reset the accumulated loss and count when starting a new training epoch
        :return:
        """
        self.accumulated_loss = 0
        self.accumulated_count = 0


class ActivationSoftmaxLossCategoricalCrossEntropy:
    """
    backwards method of the activation and the loss steps can be implemented as a single step
    to speed up the backward pass (~7-fold increase over implementing separately)
    """

    def __init__(self):
        self.dinputs = None

    # no forward pass is included here since those layers will be instantiated separately during model
    # construction

    # backward pass
    def backward(self, dvalues: np.array, y_true: np.array) -> None:
        """
        combining the softmax activation function's backwards step with the loss function's backwards step
        :param dvalues: neuron output gradient with respect to the input
        :param y_true: true labels for the input
        :return:
        """
        # number of samples
        samples = len(dvalues)
        # if labels are one-hot encoded turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy so we can safely modify
        self.dinputs = dvalues.copy()

        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        # normalize gradient
        self.dinputs = self.dinputs / samples


class OptimizerSGD:
    """
    implementation of Stochastic Gradient Descent with momentum
    this will speed up training AND prevent the neural network from getting stuck in local minima
    during the training step
    """

    def __init__(
        self, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0
    ):
        # the learning rate is a scalar multiple that adjusts how quickly the weights/biases will be
        # adjusted during the training step based on the gradient
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        # decay lets the model start with a large learning rate, and then decreases it as the number of
        # training iterations increases
        self.decay = decay
        # how many iterations have been trained
        self.iterations = 0
        # momentum is a rolling average of the most recent learning rates. this helps stop the neural
        # network from getting stuck in a local minima during training
        self.momentum = momentum

    def pre_update_params(self) -> None:
        """
        called once before any parameters are updated. helps speed up the first few iterations of training
        by introducing a decay to the learning rate. this way the model can start with a large learning rate
        with lower risk of bouncing over a minimum
        :return:
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer: LayerDense) -> None:
        """
        updates a layers weights and biases
        :param layer: the layer to be updated
        :return:
        """
        # handle the case where momentum is used
        if self.momentum:
            # build weight updates with momentum: take previous updates multiplied by retain factor
            # and update with current gradients
            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * layer.dweights
            )
            layer.weight_momentums = weight_updates

            # build bias updates similarly
            bias_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates

        # handle the case where momentum is not used
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # use the updates to adjust the layer's parameters
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """
        simply increment the number of iterations the optimizer has gone through to keep track of training epochs
        :return:
        """
        self.iterations += 1


class Accuracy:
    """
    wrapper object to handle finding the accuracy of predictions
    """
    def __init__(self):
        # parameters for keeping track of accuracy in a single batch
        self.comparisons = None
        self.accuracy = None

        # parameters to support doing epoch-wise accuracy introspection
        self.accumulated_sum = 0
        self.accumulated_count = 0
        self.epoch_accuracy = None

    def compare(self, predictions: np.array, y: np.array) -> None:
        """
        create an array where the predictions were correct
        :param predictions: predicted values for the input
        :param y: true values for the input
        :return:
        """
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        self.comparisons = predictions == y

    def calculate(self, predictions: np.array, y: np.array) -> None:
        """
        calculate the accuracy (how close to the true value the vector of predictions was)
        :param predictions: predicted values for the input
        :param y: true values for the input
        :return:
        """
        # get comparison results
        self.compare(predictions, y)

        # get the accuracy of the calculations
        self.accuracy = np.mean(self.comparisons)

        # add accumulated sum of matching values for epoch-wise accuracy
        self.accumulated_sum += np.sum(self.comparisons)
        self.accumulated_count += len(self.comparisons)

    def calculate_accumulated(self) -> None:
        """
        calculate the epoch accuracy measures
        :return:
        """
        epoch_accuracy = self.accumulated_sum / self.accumulated_count
        self.epoch_accuracy = epoch_accuracy

    def new_pass(self) -> None:
        """
        reset epoch variables for accumulated accuracy on a new training epoch
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0
