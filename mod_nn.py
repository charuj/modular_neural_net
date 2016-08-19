'''

A modular, fully-connected neural network that can be scaled up to have an arbitrary number of hidden layers.

Dependencies: numpy, tensorflow

Features:
- ReLU non-linearities (can easily be swapped out)
- dropout (optional)
- batch normalization (optional but nifty)

The architecture of this network will be:
{affine - [batch_norm] - relu - [dropout]} x (L-1) - affine - softmax_loss_function

    L= number of layers
    batch norm and dropout are optional
    {...} is repeated (L-1) times


'''


class config(object):

    '''

    Parameters of the neural network.

    num_layers: number of hidden layers in the neural network
    hidden_dims: list of integers giving the size of each hidden layer
    input_dims: integer that is the size of the input
    num_classes: number of classes to be classified (i.e. MNIST has 10)
    keep_prob : probabiliy of keeping a neuron; a decimal between 0 - 1, if 1 then there is no dropout and you keep all neurons
    batch_norm: boolean; whether to use batch normalization

    '''

    num_layers= 5
    hidden_dims=
    input_dims =
    num_classes =
    keep_prob=
    batch_norm = True


class modular_net(object):
    def __init__(self, ):
