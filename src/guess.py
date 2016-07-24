import numpy as np
from data_utils import load_data

def score_rank(sentence):
    n_total = 0
    n_correct = 0
    for i in range(len(sentence)):
        for j in range(i+1, len(sentence)):
            n_total += 1
            if sentence[i] < sentence[j]: n_correct += 1
    patial_correct = n_correct * 1.0 / n_total
    total_correct = 0.0
    if n_correct == n_total: total_correct = 1.0
    return patial_correct, total_correct
def save_result(path,top1_res):
    fw = open(path,'w')
    for paragraph, cur_categories in top1_res:
        paragraph = np.asarray(paragraph) - np.min(paragraph)
        paragraph = list(paragraph)
        for sentence in paragraph:
            fw.write(str(sentence))
            fw.write(' ')
        fw.write('#')
        for category in cur_categories:
            fw.write(category)
            fw.write(' ')
        fw.write('\n')
    fw.close()
    
    
dataset = 'cs'
datapath = '../data/%s.pkl.gz'%dataset
src_train,src_valid,src_test,dic_w2idx, dic_idx2w, dic_w2embed, dic_idx2embed, embedding = load_data(path=datapath)

res_order = []
res_eva = []
for paragraph, cur_categories in src_test:
    n = len(paragraph)
    candidates = [x for x in xrange(n)]
    guess_order = []
    for i in xrange(n):
        idx = np.random.randint(n - i)
        guess_order.append(candidates[idx])
        candidates.remove(candidates[idx])
    res_order.append((guess_order, cur_categories))
    patial_correct, total_correct = score_rank(guess_order)
    res_eva.append(np.asarray([patial_correct, total_correct]))
res_eva = np.asarray(res_eva)

print 'Guess result# patial_correct, total_correct: ', np.average(res_eva, axis = 0)
result_path = './result/guess_%s'%dataset

save_result(result_path, res_order)