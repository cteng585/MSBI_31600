# Final Project

## Description
Using the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) data set, train a neural network to
try to classify images of clothing. On the backend, the neural network should have an object-oriented implementation
with NO reliance on existing ML libraries (e.g. PyTorch, scikit learn). 

## Motivation
I chose to pursue this project because I'm deeply interested in the precise mechanics of how deep learning works.
Having taken a few courses on the basics of machine learning (e.g. learning about loss functions, gradient descent,
and optimization), it's difficult for me to imagine how such simple concepts are able to build such a powerful tool.
This project will give me an opportunity to explore how different, very complex objects can interact with one
another, exercise good coding practice to keep the code clean and tight, and different methods of storing data.

Managing many multi-referential objects reinforced the principles of good object design, and made me deeply consider
how methods were implemented such that they could be easily accessed by other objects in the network. I also attempted
utilizing several different data storage methods to manage the parameters of the network, and actually found a good
use case for pickling.

## Relevance
The two major class themes I decided to leverage in this project were object-oriented programming and data storage
methodologies.

The process of repeatedly passing forwards and backwards through a neural network is mind-bending to keep track of 
state. Even when taking full advantage of the ease of array-based operations that numpy offers, attempting to keep
track of the network's state through a functional programming approach became rapidly untenable. By utilizing an
object-oriented programming approach, I was able to develop the atomic element of the network in a (relatively)
straightforward method, and the ultimate prediction/training/testing methods became very intuitive to implement.

Given the time to train the network, from a usability perspective it's highly desirable to be able to load the
parameters from a saved file to skip the initial training step. I explored using JSON, XML, and databases to store
the training step, but keeping track of index of each weight, bias, and layer was very tedious and difficult. I
actually used pickling as a very straightforward way of dumping a multidimensional array into a binary file, and
retrieving it at runtime when necessary.

## Next Steps
If I were to do this in the future, there are a number of things that I would do:
1. Implement a GUI for the application. It would be very fun to be able to actually see the image that is being predicted on
2. Implement some other training optimization methodologies. There are several approaches to speed up and enhance training that I was not able to implement (e.g. drop out, regularization, etc.)
3. Do it for cats vs non-cats. I like cats and wish I was able to do it for cats (also would give me a chance to try out convolutional neural networks for higher-res inputs)

## Steps to run
### Install Dependencies
```
python -m pip install -r requirements.txt
```

### Building a new model from scratch
Use the command:
```
python main.py --mnist-path [PATH TO THE FASHION MNIST DIRECTORY] --build --epochs [NUMBER OF TRAINING EPOCHS] --batch-size [BATCH SIZE FOR TRAINING] --save model.pickle.save
```
- The `--mnist-path` argument can be omitted. If so, the script will look for a `fashion_mnist_images` directory in the current working directory
- `--epochs` defaults to 10 if no value is provided
- `--batch-size` defaults to 30 if no value is provided
- If `--save` isn't provided, the parameters for the model won't be saved to disk

### Predicting on an image
```
python main.py --predict_image [PATH TO IMAGE TO CLASSIFY] --load-params [PATH TO PICKLED MODEL PARAMS] --light-bg
```
- Since the model was trained on dark-background images, if the grayscale version of an image has a light background, it needs to be inverted
  - Use the `--light-bg/--no-light-bg` option for this
- `--predict` to predict the label for an image. The image should be in **.png** format.
- `--load-params` will load a model saved on disk, skipping the training step

### Building and predicting together
```
python main.py --mnist-path [PATH TO THE FASHION MNIST DIRECTORY] --build --epochs [NUMBER OF TRAINING EPOCHS] \
    --batch-size [BATCH SIZE FOR TRAINING] --save model.pickle.save --predict_image [PATH TO IMAGE TO CLASSIFY]
```
- Builds a model, saves the parameters to disk, and then makes a prediction

## Challenges
1. The math was very challenging to wrap my head around
2. Management of objects and methods was as hard as I imagined, and then some
3. Did not anticipate how hard it would be to find an appropriate way of saving the model

## Sources
1. [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
2. [Neural Networks from Scratch](https://www.youtube.com/watch?v=Wo5dMEP_BbI)
3. [Batching vs Epochs](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
4. [Backpropagation Intuition](https://www.youtube.com/watch?v=tIeHLnjs5U8)