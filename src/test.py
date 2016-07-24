import numpy as np
np.random.seed(1234)
n =10
idx_list = np.arange(n, dtype="int32")
def shuffle(idx_list):
    np.random.shuffle(idx_list)
    return idx_list
n =10
idx_list = np.arange(n, dtype="int32")
idx_list = shuffle(idx_list)
print idx_list
