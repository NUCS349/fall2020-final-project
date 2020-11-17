import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture (see problems.md for more information):
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    There should be a maxpool after each convolution.

    The sequence of operations looks like this:

        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2

    Inputs:
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()

    def forward(self, inputs):
        # Note that the ordering of dimensions in the input may not be what you
        # need for the convolutional layers.  The permute() function can help.
        raise NotImplementedError()


class Synth_Classifier(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying
    synthesized images.

    Network architecture (see problems.md for more information):
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Third hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 2)

    Activation function: ReLU for both hidden layers

    There should be a maxpool after each convolution.

    The sequence of operations looks like this:

        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2

    Inputs:
    kernel_size: list of length 3 containing kernel sizes for the three convolutional layers
                 e.g., kernel_size = [(5,5), (3,3),(3,3)]
    stride: list of length 3 containing strides for the three convolutional layers
            e.g., stride = [(1,1), (1,1),(1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Synth_Classifier, self).__init__()

    def forward(self, inputs):
        # Note that the ordering of dimensions in the input may not be what you
        # need for the convolutional layers.  The permute() function can help.
        raise NotImplementedError()

