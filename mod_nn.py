'''

A modular, fully-connected neural network that can be scaled up to have an arbitrary number of hidden layers.

Dependencies: Tensorflow

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

* Affine: no non-linear activation, i.e. just the dot product between input and weights

'''

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
    hidden_size=
    input_dims = 28 * 28 # MNIST data is 28 x 28 pixels
    num_classes = 10 # 10 classes in MNIST
    keep_prob=
    use_batch_norm = True


class modular_net():
    def __init__(self, config):
        hidden_size= config.hidden_size
        num_layers= config.num_layers
        num_classes= config.num_classes

        # Create placeholders for inputs and targets
        self.inputs= tf.placeholder(tf.float32, [None, config.input_dims])
        self.targets= tf.placeholder(tf.int32, [None, config.num_classes])
        self.keep_prob= config.keep_prob

#TODO: Figure out how to make scalable; for now I will create a unit neural net cell and then pass the variables (created in variable scope) through it

    def nn_cell(self, config, weights, biases, keep_prob):
        '''
        I'm trying to mimic BasicLSTMCell and multirnncell by creating a basic neural network cell that can be layered.
        It uses batch normalization for the inputs and dropout.

        Batch normalization is applied to the input.
        Dropout is applied after the activation function (Relu)


        :param inputs: matrix of inputs
        :param weights: matrix of weighst
        :param biases: matrix of biases
                keep_prob: float, probability of keeping a neuron
        :return: output matrix (pre-softmax), after dropout has been applied
        '''


        # Batch Normalization applied to inputs
        batch_mean, batch_variance= tf.nn.moments(self.inputs, axes=[0,1,2])
        gamma = tf.get_variable(tf.ones([config.hidden_size]))
        beta = tf.get_variable(tf.zeros([config.hidden_size]))
        bn = tf.nn.batch_norm_with_global_normalization(self.inputs, batch_mean, batch_variance, beta=beta,gamma=gamma,variance_epsilon=1e-3)

        # create variable named "weights" and "biases"

        weights = tf.get_variable("weights", [config.hidden_size, config.num_classes],
                                  initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", [config.num_classes], initializer=tf.constant_initializer(0.0))

        # Dropout applied after Relu
        hidden_layer = tf.nn.relu((tf.matmul(bn, weights) + biases))
        h_dropout = tf.nn.dropout(hidden_layer, self.keep_prob)


        




        #APPLY L2 REG TO LOSS




        with tf.variable_scope('neuralnet-'):
            ''' I'm using variable_scope to make it easier to scale up the num_layers
            (which really means scaling up the weights and biases)
            '''

            # Create weight variables, using random_normal initializer (this can be changed!)
            weights= tf.get_variable('weights',[config.hidden_size, config.input_dims], dtype= tf.float32, initializer=tf.random_normal_initializer())

            # Create biases, initalize to value of 0
            biases = tf.get_variable('biases',config.input_dims, dtype=tf.float32, initializer=tf.constant_initializer(0.0))

