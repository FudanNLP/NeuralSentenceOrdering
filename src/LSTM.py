# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations
import theano
import theano.tensor as T

class LSTM():
    '''Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
    '''
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid'):
        #self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        
        self.input_dim = input_dim
        #self.input = K.placeholder(input_shape)

        # initial states: 2 all-zero tensor of shape (output_dim)
        self.states = [None, None]

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim,))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))

        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_c, self.U_c, self.b_c,
                       self.W_f, self.U_f, self.b_f,
                       self.W_o, self.U_o, self.b_o]

        #if self.initial_weights is not None:
        #    self.set_weights(self.initial_weights)
        #    del self.initial_weights

    def numpy_floatX(self,data):
        return np.asarray(data, dtype=np.float32)
    def reset_states(self,batch_size):
        #self.states = [K.zeros((batch_size, self.output_dim)),
        #               K.zeros((batch_size, self.output_dim))]
        
        self.states = [T.alloc(self.numpy_floatX(0.),batch_size,self.output_dim),
                       T.alloc(self.numpy_floatX(0.),batch_size,self.output_dim)]

    def step(self, x, h_tm1, c_tm1):
        #assert len(states) == 2
        #h_tm1 = states[0]
        #c_tm1 = states[1]

        x_i = K.dot(x, self.W_i) + self.b_i
        x_f = K.dot(x, self.W_f) + self.b_f
        x_c = K.dot(x, self.W_c) + self.b_c
        x_o = K.dot(x, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.U_o))
        h = o * self.activation(c)
        return h, c

    def get_output(self, go_backwards = False, train = False):
        self.reset_states(train.shape[0])
        inputs = train.dimshuffle((1, 0, 2))
        results, _ = theano.scan(
            self.step,
            sequences=inputs,
            outputs_info=[self.states[0],self.states[1]],
            go_backwards=go_backwards)
        '''
        # deal with Theano API inconsistency
        if type(results) is list:
            outputs = results[0]
            states = results[1:]
        else:
            outputs = results
            states = []
    
        outputs = T.squeeze(outputs)
        last_output = outputs[-1]
        '''
        
        #outputs = np.asarray(results)[:,0]
        #outputs = T.squeeze(outputs)
        #outputs = outputs.dimshuffle((1, 0, 2))
        
        #states = [T.squeeze(state[-1]) for state in states]
        #return last_output, outputs, states
        
        outputs = results[0]
        outputs = T.squeeze(outputs)
        outputs = outputs.dimshuffle((1, 0, 2))
        return outputs
    
    
class BLSTM():
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid'):
        #self.input_dim = input_dim
        self.output_dim = int(output_dim / 2)
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        
        self.input_dim = input_dim
        #self.input = K.placeholder(input_shape)

        # initial states: 2 all-zero tensor of shape (output_dim)
        self.forward_lstm = LSTM(input_dim = input_dim, output_dim = self.output_dim)
        self.backward_lstm = LSTM(input_dim = input_dim, output_dim = self.output_dim)
        
        self.params = self.forward_lstm.params + self.backward_lstm.params

        #if self.initial_weights is not None:
        #    self.set_weights(self.initial_weights)
        #    del self.initial_weights

    

    def get_output(self, train = False):
        res_forward = self.forward_lstm.get_output(train)
        res_backward = self.backward_lstm.get_output(train[:,::-1,:])
        outputs = T.concatenate([res_forward, res_backward[:,::-1,:]], axis = -1)
        return outputs
