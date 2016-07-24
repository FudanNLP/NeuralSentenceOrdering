from __future__ import absolute_import
from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.regularizers import ActivityRegularizer
import numpy as np
import theano
import theano.tensor as T

class Embedding():
    '''Turn positive integers (indexes) into dense vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

    This layer can only be used as the first layer in a model.

    # Input shape
        2D tensor with shape: `(nb_samples, sequence_length)`.

    # Output shape
        3D tensor with shape: `(nb_samples, sequence_length, output_dim)`.

    # Arguments
      input_dim: int >= 0. Size of the vocabulary, ie.
          1 + maximum integer index occurring in the input data.
      output_dim: int >= 0. Dimension of the dense embedding.
      init: name of initialization function for the weights
          of the layer (see: [initializations](../initializations.md)),
          or alternatively, Theano function to use for weights initialization.
          This parameter is only relevant if you don't pass a `weights` argument.
      weights: list of numpy arrays to set as initial weights.
          The list should have 1 element, of shape `(input_dim, output_dim)`.
      W_regularizer: instance of the [regularizers](../regularizers.md) module
        (eg. L1 or L2 regularization), applied to the embedding matrix.
      W_constraint: instance of the [constraints](../constraints.md) module
          (eg. maxnorm, nonneg), applied to the embedding matrix.
      mask_zero: Whether or not the input value 0 is a special "padding"
          value that should be masked out.
          This is useful for [recurrent layers](recurrent.md) which may take
          variable length input. If this is `True` then all subsequent layers
          in the model need to support masking or an exception will be raised.
      input_length: Length of input sequences, when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).
    '''
    input_ndim = 2

    def __init__(self, input_dim, output_dim, init='uniform', 
                 weights=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        
        self.W = self.init((self.input_dim, self.output_dim))
        #self.W_Tag = self.init((self.size_label_set, self.output_dim_tag))
        
        
        if weights != None:
            self.W = theano.shared(value = np.asarray(weights, dtype = theano.config.floatX), borrow=True)
        self.params = [self.W]
        
    def get_output(self, train=False):
        X = train
        out = K.gather(self.W, X)
        return out

    
    
