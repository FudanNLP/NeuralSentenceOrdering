from collections import OrderedDict
import cPickle as pkl
import sys
import time
import argparse
import copy

import random
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)
def sgd(lr, tparams, grads, sentence1,sentence1_mask,sentence2,sentence2_mask,y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(v.get_value() * 0., name='%s_grad' % k)
               for k, v in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([sentence1,sentence1_mask,sentence2,sentence2_mask,y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(v, v - lr * g) for v, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, sentence1,sentence1_mask,sentence2,sentence2_mask,y, cost):
    zipped_grads = [theano.shared(q.get_value() * numpy_floatX(0.), name='%s_grad' % k)
                    for k, q in tparams.iteritems()]
    running_grads = [theano.shared(q.get_value() * numpy_floatX(0.), name='%s_rgrad' % k)
                     for k, q in tparams.iteritems()]
    running_grads2 = [theano.shared(q.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k)
                      for k, q in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([sentence1,sentence1_mask,sentence2,sentence2_mask,y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(q.get_value() * numpy_floatX(0.), name='%s_updir' % k)
             for k, q in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(q, q + udn[1])
                for q, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, sentence1,sentence1_mask,sentence2,sentence2_mask,y, cost):
    '''
    zipped_grads = [theano.shared(q.get_value() * numpy_floatX(0.), name='%s_grad' % k)
                    for k, q in tparams.iteritems()]
    running_up2 = [theano.shared(q.get_value() * numpy_floatX(0.),name='%s_rup2' % k)
                   for k, q in tparams.iteritems()]
    running_grads2 = [theano.shared(q.get_value() * numpy_floatX(0.),name='%s_rgrad2' % k)
                      for k, q in tparams.iteritems()]
    '''
    zipped_grads = [theano.shared(q.get_value() * numpy_floatX(0.))
                    for q in tparams]
    running_up2 = [theano.shared(q.get_value() * numpy_floatX(0.))
                   for q in tparams]
    running_grads2 = [theano.shared(q.get_value() * numpy_floatX(0.))
                      for q in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([sentence1,sentence1_mask,sentence2,sentence2_mask,y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared', allow_input_downcast=True)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(q, q + ud) for q, ud in zip(tparams, updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update', allow_input_downcast=True)

    return f_grad_shared, f_update