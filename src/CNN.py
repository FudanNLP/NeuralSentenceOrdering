# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.regularizers import ActivityRegularizer
from theano.tensor.signal import downsample


def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride


class Convolution1D():
    '''Convolution operator for filtering neighborhoods of one-dimensional inputs.
    When using this layer as the first layer in a model,
    either provide the keyword argument `input_dim`
    (int, e.g. 128 for sequences of 128-dimensional vectors),
    or `input_shape` (tuple of integers, e.g. (10, 128) for sequences
    of 10 vectors of 128-dimensional vectors).

    # Input shape
        3D tensor with shape: `(samples, steps, input_dim)`.

    # Output shape
        3D tensor with shape: `(samples, new_steps, nb_filter)`.
        `steps` value might have changed due to padding.

    # Arguments
        nb_filter: Number of convolution kernels to use
            (dimensionality of the output).
        filter_length: The extension (spatial or temporal) of each filter.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample_length: factor by which to subsample output.
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
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
    '''
    input_ndim = 3

    def __init__(self, nb_filter, filter_length,
                 init='uniform', activation='linear', weights=None,
                 border_mode='valid', subsample_length=1,
                 input_dim=None):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution1D:', border_mode)
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample_length = subsample_length

        self.subsample = (subsample_length, 1)

        
        self.input_dim = input_dim
        
 
        input_dim = self.input_dim
        self.W_shape = (self.nb_filter, input_dim, self.filter_length, 1)
        self.W = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
        self.params = [self.W, self.b]
 
    def get_output(self, train=False):
        X = train
        X = K.expand_dims(X, -1)  # add a dimension of the right
        X = K.permute_dimensions(X, (0, 2, 1, 3))
        conv_out = K.conv2d(X, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering='th')

        output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        output = self.activation(output)
        output = K.squeeze(output, 3)  # remove the dummy 3rd dimension
        output = K.permute_dimensions(output, (0, 2, 1))
        return output
    
class MaxPooling1D():
    '''Max pooling operation for temporal data.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        3D tensor with shape: `(samples, downsampled_steps, features)`.

    # Arguments
        pool_length: factor by which to downscale. 2 will halve the input.
        stride: integer or None. Stride value.
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
    '''
    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid'):
        self.pool_length = pool_length = 2
        self.border_mode = border_mode
        self.params = []

    def get_output(self, train=False):
        #output = K.pool2d(x = train, pool_size = (self.pool_length,1), 
        #                  border_mode = self.border_mode, pool_mode='max')
        pool_size = (self.pool_length, 1)
        strides = (self.pool_length, 1)
        ignore_border = True
        padding = (0, 0)
        output = downsample.max_pool_2d(train, ds=pool_size, st=strides,
                                          ignore_border=ignore_border,
                                          padding=padding,
                                          mode='max')
        return output
