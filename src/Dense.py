from __future__ import absolute_import
from __future__ import division

import numpy as np

from collections import OrderedDict
import copy
from six.moves import zip

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.regularizers import ActivityRegularizer

import marshal
import types
import sys
class Dense():
    '''Apply a same Dense layer for each dimension[1] (time_dimension) input.
    Especially useful after a recurrent network with 'return_sequence=True'.

    # Input shape
        3D tensor with shape `(nb_sample, time_dimension, input_dim)`.

    # Output shape
        3D tensor with shape `(nb_sample, time_dimension, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 3

    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        '''
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        '''
        
        #super(TimeDistributedDense, self).__init__(**kwargs)

    #def build(self):
        

        self.W = self.init((self.input_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))

        self.params = [self.W, self.b]
        '''
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        '''

    def get_output(self, X):
        output = self.activation(K.dot(X, self.W) + self.b)
        return output
    '''
    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        #base_config = super(TimeDistributedDense, self).get_config()
        return dict(list(config.items()))
    '''