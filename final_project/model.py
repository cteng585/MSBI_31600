from layers import *


class Model:
    """
    wrapper object to manage the layers of the network
    """

    def __init__(self):
        # the layers of the neural network
        self.input_layer = LayerInput()
        self.layers = []
        # keep track of the last layer for easy access to see what the model actually outputs
        self.output_layer_activation = None
        # keep track of which layers are actually trainable for the backward pass of nudging parameters
        self.trainable_layers = []
        # training elements of the neural network
        self.loss = None
        self.softmax_loss_combined = None
        self.optimizer = None
        self.accuracy_calculator = None

    def add(self, layer: LayerDense | ActivationReLU | ActivationSoftmax) -> None:
        """
        adds a "layer" (not in the strict neural network sense, more in the operation sense) to the networkk
        :param layer: a LayerDense or Activation object
        :return:
        """
        self.layers.append(layer)

    def set(self, loss: LossCategoricalCrossEntropy, optimizer: OptimizerSGD, accuracy_calculator: Accuracy) -> None:
        """
        adds loss and optimizer objects to the model for training purposes
        :param loss: the loss object to use. can be the combined loss/cross-entropy object
        :param optimizer: the optimizer to use for the model
        :param accuracy_calculator: the accuracy object to use for calculating epoch prediction accuracy
        :return:
        """
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy_calculator = accuracy_calculator

    def get_parameters(self) -> list[Tuple[np.array, np.array]]:
        """
        get the parameters (weights and biases) for all the hidden layers in the model

        :return: a list of arrays, where each array are the weights and biases of a hidden layer
        """
        hidden_layer_parameters = []

        for layer in self.trainable_layers:
            hidden_layer_parameters.append(layer.get_parameters())

        return hidden_layer_parameters

    def set_parameters(self, parameters) -> None:
        """
        set the weights and biases for the trainable layers in this model

        :return:
        """
        for parameter_tuple, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_tuple)

    def finalize(self) -> None:
        """
        make each layer aware of the adjacent layers to "connect" the model
        :return:
        """
        layer_count = len(self.layers)

        for idx in range(layer_count):
            # handle connecting each hidden layer "backwards" in the model
            if idx == 0:
                # the first hidden layer connects to the input layer
                self.layers[idx].previous = self.input_layer
            else:
                self.layers[idx].previous = self.layers[idx - 1]

            # handle connecting each hidden layer "forwards" in the model
            if idx < layer_count - 1:
                self.layers[idx].next = self.layers[idx + 1]
            else:
                # the last hidden layer passes to the loss object
                self.layers[idx].next = self.loss
                # remember the last layer since this will be the model's output
                self.output_layer_activation = self.layers[idx]

            if hasattr(self.layers[idx], "weights"):
                self.trainable_layers.append(self.layers[idx])

        # create a combined Softmax + Loss object to optimize the backward pass
        self.softmax_loss_combined = ActivationSoftmaxLossCategoricalCrossEntropy()

    def forward(self, X: np.array) -> np.array:
        """
        get all layers of the model to do the forward pass

        :param X: this is a batch of inputs into the model
        :return: the output (predictions) of the model as a whole
        """
        # call the forward method on the input layer to set the .output attribute
        # that the first hidden layer is expecting
        self.input_layer.forward(X)

        # iterate over each layer of the network and do a forward pass
        # move a pointer along the iteration to keep track of where we are in the layer
        last_layer = self.input_layer
        for layer in self.layers:
            last_layer = layer
            layer.forward(layer.previous.output)

        # return the output of the last layer
        return last_layer.output

    def backward(self, output: np.array, y: np.array) -> None:
        """
        move backwards through the model to propagate neuron output gradients (respective of the inputs) to
        adjust neuron parameters (weights and biases)

        :param output: the model output, used to calculate the first neuron output gradient to pass backwards
        :param y: the true labels of the input
        :return:
        """
        # the backwards pass needs to be initialized by the loss object
        # since the combined ActivationSoftmax + LossCategoricalCrossEntropy object implements backward
        #   pass differently, call that first
        self.softmax_loss_combined.backward(output, y)

        # pass in the inputs gradient from the ActivationSoftmax + LossCategoricalCrossEntropy object's
        # to use as the inputs gradient instead of what is generated by the last layer (for runtime optimization)
        self.layers[-1].dinputs = self.softmax_loss_combined.dinputs

        # now move backward through the layers, propagating the neuron output gradients
        # with respect to the inputs. do not call the backward method from the Softmax Activation function
        # since the combined backward method was already called
        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)

    def train(
        self,
        X: np.array,
        y: np.array,
        batch_size: int = 30,
        summary_freq_batch: int = 1,
        epochs: int = 1,
        summary_freq_epoch: int = 1,
        validation_data: Tuple[np.array, np.array] = None,
        validation_batch_size: int = None,
    ):
        """
        the main training function

        :param X: training inputs
        :param y: true labels for the training inputs
        :param batch_size: how many batches to split the training input into. the default of 30 is a good starting point
        :param summary_freq_batch: how frequently test statistics should be printed for batches
        :param epochs: number of training epochs
        :param summary_freq_epoch: how frequently test statistics should be printed for training epochs
        :param validation_data: data used to spot-check the model
        :param validation_batch_size: how many batches to split the validation data into. the default of None means to see
            how the validation data set performs as a whole
        :return:
        """
        # split the training data into batches. this optimizes the training steps
        training_steps = len(X) // batch_size

        # handle the case where the last "batch" isn't the full size
        # this is also the case if the batch size exceeds the size of the input data
        if training_steps * batch_size < len(X):
            training_steps += 1

        # handle validation separately in the event that the validation step should be
        # batched differently
        if validation_data is not None:
            X_val, y_val = validation_data

            # if no validation batch size is passed, try validating the test set as a whole
            if validation_batch_size is None:
                validation_batch_size = len(X_val)

            validation_steps = len(X_val) // validation_batch_size
            if validation_steps * validation_batch_size < len(X_val):
                validation_steps += 1

        # the main training loop
        for epoch in range(1, epochs + 1):
            if not epoch % summary_freq_epoch:
                print(f"epoch {epoch}")
            # reset the accumulated loss for each epoch
            self.accuracy_calculator.new_pass()
            self.loss.new_pass()

            # iterate over the batches of data within a training epoch
            for step in range(training_steps):
                batch_X = X[step * batch_size:(step + 1) * batch_size]
                batch_y = y[step * batch_size:(step + 1) * batch_size]
                # move through the layer to get the predictions
                output = self.forward(batch_X)

                # calculate the loss of the predictions
                data_loss = self.loss.calculate(output, batch_y)

                # get the predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions()

                # calculate an accuracy
                self.accuracy_calculator.calculate(predictions, batch_y)
                accuracy = self.accuracy_calculator.accuracy

                # do a full backward pass through the model
                self.backward(output, batch_y)

                # update the hidden layer parameters after discovering the weight/bias gradients
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # print some summary statistics to see how the model is doing
                # every couple of batches (specifically every `summary_freq_batch` number of batches)
                if not epoch % summary_freq_epoch and (not step % summary_freq_batch or step == training_steps - 1):
                    print(
                        f"step: {step + 1}",
                        f"accuracy: {accuracy:.3f}",
                        f"loss: {data_loss:.3f}",
                        f"learning rate: {self.optimizer.current_learning_rate}",
                    )

            # get the accuracy and loss metrics over the epoch as a whole
            # do this every `summary_freq_epoch` number of epochs
            if not epoch % summary_freq_epoch:
                epoch_loss = self.loss.calculate_accumulated()
                self.accuracy_calculator.calculate_accumulated()
                epoch_accuracy = self.accuracy_calculator.epoch_accuracy

                print(f"epoch accuracy: {epoch_accuracy}",
                      f"epoch loss: {epoch_loss}",
                      f"learning rate: {self.optimizer.current_learning_rate}")

        # if there is validation data used to spot-check the model, handle it here
        if validation_data is not None:
            # reset accumulated accuracy and loss metrics to calculate anew for the test step
            self.accuracy_calculator.new_pass()
            self.loss.new_pass()

            for step in range(validation_steps):
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

                # do a forward pass to get the outputs
                output = self.forward(batch_X)

                # calculate the loss using the validation data
                # don't need to keep the batch loss here, but still call the method to get the accumulated
                #   loss rolling
                self.loss.calculate(output, batch_y)

                # get predictions and calculate an accuracy for the validation data
                predictions = self.output_layer_activation.predictions()
                self.accuracy_calculator.calculate(predictions, batch_y)

            total_validation_loss = self.loss.calculate_accumulated()
            self.accuracy_calculator.calculate_accumulated()
            total_validation_accuracy = self.accuracy_calculator.epoch_accuracy

            # no backward pass is needed for validation data since we're just spot-checking the model
            # also don't need to print the validation metrics per batch since we're only interested in the
            #   validation as a whole
            print(
                f"\n---\nvalidation set\n---\n",
                f"total validation accuracy: {total_validation_accuracy:.3f}",
                f"total validation loss: {total_validation_loss:.3f}",
            )

    def predict(self, X) -> np.array:
        """
        use the forward method to see what the neural network outputs when fed in a novel input

        :param X: the novel input (should be a 1D array)
        :return: an array of confidence values for the input
        """
        output = self.forward(X)
        return output
