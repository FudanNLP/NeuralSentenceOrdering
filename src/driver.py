import numpy as np
import sys
import time
from pairwise import Pairwise
from model import build_model
import cPickle as pkl
#from Activation import *
if __name__ == '__main__':
    flag_toy_data = 0.1
    random_seed = 1234
    alpha = 0.2
    batch_size = 128
    dispFreq = 2048
    n_epochs = 600
    wordVecLen = 25 # useless
    flag_dropout = False
    size_hidden_layer = 100
    dropoutRates = 0.2 # for output of the embedding layer
    optimizer = 'adadelta'
    beam_size = 128
    dataset = 'all'
    datapath = '../data/%s.pkl.gz'%dataset
    result_path = './result/'
    sentence_modeling = 'CNN' # available: 'CBoW' 'LSTM' 'CNN'
    CNN_filter_length = 3
    LSTM_go_backwards = True
    
    flag_random_lookup_table = False
    
    pair_score = Pairwise(alpha = alpha,
             batch_size=batch_size,
             n_epochs=n_epochs,
             wordVecLen = wordVecLen,
             flag_dropout = flag_dropout,
             datapath=datapath,
             random_seed=random_seed,
             dropoutRates = dropoutRates,
             optimizer = optimizer,
             dispFreq = dispFreq,
             beam_size = beam_size,
             flag_random_lookup_table = flag_random_lookup_table,
             flag_toy_data = flag_toy_data,
             size_hidden_layer = size_hidden_layer,
             dataset = dataset,
             result_path = result_path,
             sentence_modeling = sentence_modeling,
             CNN_filter_length = CNN_filter_length,
             LSTM_go_backwards = LSTM_go_backwards
             )
    
    
