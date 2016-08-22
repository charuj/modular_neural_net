'''

A modular, fully-connected neural network that can be scaled up to have an arbitrary number of hidden layers.

Dependencies: numpy, tensorflow

Features:
- ReLU non-linearities (can easily be swapped out)
- dropout (optional)
- batch normalization (optional but nifty)
- L2 regularization

The architecture of this network will be:
{affine - [batch_norm] - relu - [dropout]} x (L-1) - affine - softmax_loss_function

    L= number of layers
    batch norm and dropout are optional
    {...} is repeated (L-1) times


'''

import numpy as np
import tensorflow as tf

class config(object):

    '''

    Parameters of the neural network.

    num_layers: number of hidden layers in the neural network
    hidden_dims: list of integers giving the size of each hidden layer
    input_dims: integer that is the size of the input
    num_classes: number of classes to be classified (i.e. MNIST has 10)
    keep_prob : probabiliy of keeping a neuron; scalar between 0 - 1, if 1 then there is no dropout and you keep all neurons
    use_batch_norm: boolean; whether to use batch normalization

    '''

    num_layers= 5
    hidden_dims=
    input_dims = 28 * 28 # MNIST data is 28 x 28 pixels
    num_classes = 10 # 10 classes in MNIST
    keep_prob=
    use_batch_norm = True


class modular_net():
    def __init__(self, config):
        hidden_dims= config.hidden_dims
        num_layers= config.num_layers

        # Create placeholders for inputs and targets
        self.inputs= tf.placeholder(tf.float32, [None, config.input_dims])
        self.targets= tf.placeholder(tf.int32, [None, config.num_classes])

# Todo: variable scope to scale up ??

