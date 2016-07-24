import numpy as np
import time
import sys
from embeddings import Embedding
from Dropout import dropout
import theano
import theano.tensor as T
import pickle
from collections import OrderedDict
from Dense import Dense
from keras import backend as K

from CNN import Convolution1D
from LSTM import LSTM
def CNN_embed(embed_s,s_mask,sentence_encode_layer):
    s_mask = s_mask.reshape((s_mask.shape[0],s_mask.shape[1],1)) # n_samples * len_sentence * 1
    s_mask = s_mask.repeat(embed_s.shape[2],axis = 2) # n_samples * len_sentence * embed_dim
    embed_s = embed_s * s_mask # n_samples * len_sentence * embed_dim
    
    embed_s = sentence_encode_layer.get_output(embed_s) # n_samples * len_sentence- * embed_dim
    embed_s = T.max(embed_s,axis = 1) # n_samples * embed_dim
    return embed_s # n_samples * embed_dim

def LSTM_embed(embed_s,s_mask,sentence_encode_layer, options):
    s_mask = s_mask.reshape((s_mask.shape[0],s_mask.shape[1],1)) # n_samples * len_sentence * 1
    s_mask = s_mask.repeat(embed_s.shape[2],axis = 2) # n_samples * len_sentence * embed_dim
    embed_s = embed_s * s_mask # n_samples * len_sentence * embed_dim
    
    embed_s = sentence_encode_layer.get_output(go_backwards = options['LSTM_go_backwards'], train = embed_s) # n_samples * len_sentence * embed_dim
    return embed_s[:,-1,:] # n_samples * embed_dim
    
def ave_embed(embed_s,s_mask):
    n = s_mask.sum(axis = 1) # n_samples
    n = n.reshape((s_mask.shape[0],1)) # n_samples * 1
    n = n.repeat(embed_s.shape[2],axis = 1) # n_samples * embed_dim
    s_mask = s_mask.reshape((s_mask.shape[0],s_mask.shape[1],1)) # n_samples * len_sentence * 1
    s_mask = s_mask.repeat(embed_s.shape[2],axis = 2) # n_samples * len_sentence * embed_dim
    embed_s = embed_s * s_mask # n_samples * len_sentence * embed_dim
    return embed_s.sum(axis = 1) /n  # n_samples * embed_dim
def build_model(options):
    print('Build model...')
    sys.stdout.flush()
    weights = None
    if options['flag_random_lookup_table'] == False: weights = options['embedding']
    embed_layer = Embedding(input_dim = options['embedding'].shape[0], 
                            output_dim = options['embedding'].shape[1], 
                            weights = weights)
    dense_layers = []
    dense_layers.append(Dense(input_dim = options['embedding'].shape[1] * 2, output_dim = options['size_hidden_layer'], activation = 'tanh'))
    dense_layers.append(Dense(input_dim = options['size_hidden_layer'], output_dim = 1, activation = 'sigmoid'))
    
    # for training
    sentence1 = T.imatrix('s1')  # sentence1, n_samples * len_sentence
    sentence1_mask = T.matrix('s1_mask')
    sentence2 = T.imatrix('s2')  # sentence2, n_samples * len_sentence
    sentence2_mask = T.matrix('s2_mask')
    y = T.ivector('y1')  # n_samples
    
    embed_s1 = embed_layer.get_output(sentence1) # n_samples * len_sentence * embed_dim
    embed_s2 = embed_layer.get_output(sentence2) # n_samples * len_sentence * embed_dim
    if options['sentence_modeling'] == 'CBoW':
        embed_s1 = ave_embed(embed_s1,sentence1_mask) # n_samples * embed_dim
        embed_s2 = ave_embed(embed_s2,sentence2_mask) # n_samples * embed_dim
    elif options['sentence_modeling'] == 'CNN':
        sentence_encode_layer = Convolution1D(input_dim = options['embedding'].shape[1], activation = 'tanh',
                                nb_filter = options['embedding'].shape[1], filter_length = options['CNN_filter_length'],
                                border_mode = 'same')
        embed_s1 = CNN_embed(embed_s1,sentence1_mask,sentence_encode_layer) # n_samples * embed_dim
        embed_s2 = CNN_embed(embed_s2,sentence2_mask,sentence_encode_layer) # n_samples * embed_dim
    elif options['sentence_modeling'] == 'LSTM':
        sentence_encode_layer = LSTM(input_dim = options['embedding'].shape[1], output_dim = options['embedding'].shape[1])
        embed_s1 = LSTM_embed(embed_s1,sentence1_mask,sentence_encode_layer,options) # n_samples * embed_dim
        embed_s2 = LSTM_embed(embed_s2,sentence2_mask,sentence_encode_layer,options) # n_samples * embed_dim
    else:
        print 'Error: No model called %s available!' % options['sentence_modeling']
        return
    
    output = T.concatenate([embed_s1,embed_s2],axis = -1) # n_samples * (embed_dim * 2)
    
    if options['flag_dropout'] == True:
        output = dropout(output, level=options['dropoutRates'])
    for dense_layer in dense_layers:
        output = dense_layer.get_output(output)
    f_pred = theano.function([sentence1,sentence1_mask,sentence2,sentence2_mask],output, allow_input_downcast=True)
    
    output = output.reshape((output.shape[0],))
    #y = y.reshape((output.shape[0],1))
    cost = T.nnet.binary_crossentropy(output, y).mean()
    f_debug = theano.function([sentence1,sentence1_mask,sentence2,sentence2_mask,y],[output,y,T.nnet.binary_crossentropy(output, y),cost], allow_input_downcast=True)
    tparams = []
    tparams += embed_layer.params
    if options['sentence_modeling'] != 'CBoW':
        tparams += sentence_encode_layer.params
    for dense_layer in dense_layers: tparams += dense_layer.params
    return sentence1,sentence1_mask,sentence2,sentence2_mask,y,cost,f_pred,tparams,f_debug
    