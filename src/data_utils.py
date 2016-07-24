import cPickle
import gzip
import os
import sys

import numpy as np
import theano

def get_max_length(sentences):
    n = 0
    for sentence in sentences:
        l = len(sentence)
        if n < l: n = l
    return n
def padding(sentences, max_len):
    res = np.zeros((len(sentences),max_len),dtype = np.int32)
    mask = np.zeros((len(sentences),max_len))
    for s_id, sentence in enumerate(sentences):
        for w_id, word in enumerate(sentence):
            res[s_id][w_id] = word
            mask[s_id][w_id] = 1
    return res, mask
    
def data_padding(batch_samples):
    s1 = []
    s2 = []
    y = []
    for fir, sec, label in batch_samples:
        s1.append(fir)
        s2.append(sec)
        y.append(label)
    max_len1 = get_max_length(s1)
    max_len2 = get_max_length(s2)
    # s: 2d_array n_samples * max_len
    # mask: 2d_array n_samples * max_len
    s1, s1_mask = padding(s1, max_len1)
    s2, s2_mask = padding(s2, max_len2)
    y = np.asarray(y)
    return s1, s1_mask, s2, s2_mask, y

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def prepare_data(examples):
    data = [] # (s1, s2, y)
    pairdict = {}
    n_sentences = 0
    for paragraph, cur_categories in examples:
        for s1_id,s1 in enumerate(paragraph):
            for s2_id,s2 in enumerate(paragraph):
                if s1_id == s2_id: continue
                if s1_id < s2_id: 
                    data.append((s1, s2, 1))
                else:
                    data.append((s1, s2, 0))
                pairdict[(n_sentences + s1_id, n_sentences + s2_id)] = len(data) - 1
        n_sentences += len(paragraph)
    return data, pairdict


def load_data(path='tsp_test.pkl.gz'):
    data_dir, data_file = os.path.split(path)
    if data_dir == "" and not os.path.isfile(path):
        path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            path
        )

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    src_train,src_valid,src_test,dic_w2idx, dic_idx2w, dic_w2embed, dic_idx2embed, embedding = cPickle.load(f)
    f.close()
    return src_train,src_valid,src_test,dic_w2idx, dic_idx2w, dic_w2embed, dic_idx2embed, embedding


if __name__ == '__main__':
    pass