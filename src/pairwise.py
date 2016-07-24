import numpy as np
import sys
import time
from collections import OrderedDict
import theano
import copy
from data_utils import load_data, prepare_data, get_minibatches_idx, data_padding
from model import build_model
from optimizer import sgd,rmsprop,adadelta
import theano.tensor as tensor
class Pairwise(object):
    def get_score(self, pre_sentence_list, cur_sentence, preds, pairdict):
        score = 0.0
        for pre_sentence in pre_sentence_list:
            idx = pairdict[(pre_sentence, cur_sentence)]
            score += np.log(preds[idx])
        return score
    def score_rank(self, sentence):
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
    def eva(self,
            f_pred, src_data, data, pairdict, kf, model_options):
        preds = []
        for _, data_index in kf:
            batch_samples = [data[t] for t in data_index]
            sentence1,sentence1_mask,sentence2,sentence2_mask,y = data_padding(batch_samples)
            preds.append(f_pred(sentence1,sentence1_mask,sentence2,sentence2_mask))
        preds = np.concatenate(preds, axis = 0) # 1d_array n_samples
        
        categories = []
        n_sentences = 0
        data_beams = [] # n_paragraph * n_sentences (sentence, score)
        for paragraph, cur_categories in src_data:
            categories.append(cur_categories)
            beam = []
            for s_id in xrange(len(paragraph)):
                beam.append(([s_id + n_sentences],0.0))
            for nid in xrange(len(paragraph)-1):
                new_beam = []
                for item in beam:
                    for s_id in xrange(len(paragraph)):
                        new_sentence = item[0] + [s_id + n_sentences]
                        if len(set(new_sentence)) < nid + 2: continue # repeated elements occur
                        new_score = item[1] + self.get_score(item[0], s_id + n_sentences, preds, pairdict)
                        new_beam.append((new_sentence, new_score))
                new_beam = sorted(new_beam, key=lambda item : -item[1]) #from high score to lower ones
                beam = new_beam[:model_options['beam_size']]
            data_beams.append(beam)
            n_sentences += len(paragraph)
        
        top1_res = [] # sentence_rank, paragraph_categories
        eva_res = np.zeros((len(src_data),model_options['beam_size'],2)) # n_paragraph * beam_size * 2  patial_correct, total_correct
        for id_paragraph, beam in enumerate(data_beams):
            top1_res.append((beam[0][0], categories[id_paragraph]))
            for idx, (sentence, _) in enumerate(beam):
                patial_correct, total_correct = self.score_rank(sentence)
                eva_res[id_paragraph][idx] = np.asarray([patial_correct, total_correct])
        
        top = 1
        while top <= model_options['beam_size']:
            eva_res_top = np.max(eva_res[:,:top,:], axis = 1) # n_paragraph * 2  patial_correct, total_correct
            print 'Top %d beam ' % top
            average = np.average(eva_res_top, axis = 0)
            print 'patial_correct_rate: ', average[0]
            print 'total_correct_rate: ', average[1]
            top *= 2
        print ''
                 
        return top1_res
    def save_result(self,path,top1_res):
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
            
    def __init__(self,
                 alpha,
             batch_size,
             n_epochs,
             wordVecLen,
             flag_dropout,
             datapath,
             random_seed,
             dropoutRates,
             optimizer,
             dispFreq,
             beam_size,
             flag_random_lookup_table,
             flag_toy_data,
             size_hidden_layer,
             dataset,
             result_path,
             sentence_modeling,
             CNN_filter_length,
             LSTM_go_backwards
             ):
        model_options = locals().copy()
        model_options['rng'] = np.random.RandomState(random_seed)
        print 'Loading data'
        src_train,src_valid,src_test,dic_w2idx, dic_idx2w, dic_w2embed, dic_idx2embed, embedding = load_data(path=datapath)
        if flag_toy_data == True:
            src_valid = src_valid[:10]
            src_test = src_test[:10] 
            #src_train = copy.copy(src_valid)
            src_train = src_train[:10]
        elif flag_toy_data != False:
            valid_l = len(src_valid) * flag_toy_data
            test_l = len(src_test) * flag_toy_data
            train_l = len(src_train) * flag_toy_data
            src_valid = src_valid[:int(valid_l)]
            src_test = src_test[:int(test_l)] 
            src_train = src_train[:int(train_l)]
            
        train,pairdict_train = prepare_data(src_train)
        valid,pairdict_valid = prepare_data(src_valid)
        test,pairdict_test = prepare_data(src_test)
        model_options['embedding'] = embedding
        
        (sentence1,sentence1_mask,sentence2,sentence2_mask,y,cost,f_pred,tparams,f_debug) = build_model(model_options)
        #f_cost = theano.function([sentence1,sentence1_mask,sentence2,sentence2_mask,y], cost, name='f_cost')
    
        #grads = tensor.grad(theano.gradient.grad_clip(cost, -2.0, 2.0), wrt=tparams.values())
        grads = tensor.grad(theano.gradient.grad_clip(cost, -2.0, 2.0), wrt=tparams)
        # grads = tensor.grad(cost, wrt=tparams.values())
        #f_grad = theano.function([sentence1,sentence1_mask,sentence2,sentence2_mask,y], grads, name='f_grad')
    
        lr = tensor.scalar(name='lr')
        if model_options['optimizer'] == 'sgd': optimizer = sgd
        elif model_options['optimizer'] == 'rmsprop': optimizer = rmsprop
        else: optimizer = adadelta
        f_grad_shared, f_update = optimizer(lr, tparams, grads, sentence1,sentence1_mask,sentence2,sentence2_mask,y, cost)
        
        
        print 'Optimization'

        kf_valid = get_minibatches_idx(len(valid), model_options['batch_size'])
        kf_test = get_minibatches_idx(len(test), model_options['batch_size'])
    
        print "%d train examples" % len(train)
        print "%d valid examples" % len(valid)
        print "%d test examples" % len(test)
        sys.stdout.flush()
        
        
        best_validation_score = -np.inf
        best_iter = 0
        uidx = 0  # the number of update done
        for epoch in xrange(model_options['n_epochs']):
            print ('Training on %d epoch' % epoch)
            sys.stdout.flush()
            kf = get_minibatches_idx(len(train), batch_size, shuffle=True)
            start_time = time.time()
            samples_seen = 0
            for _, train_index in kf:
                uidx += 1
                batch_samples = [train[t] for t in train_index]
                samples_seen += len(batch_samples)
                #print batch_samples
                sentence1,sentence1_mask,sentence2,sentence2_mask,y = data_padding(batch_samples)
                #print sentence1,sentence1_mask,sentence2,sentence2_mask,y
                #print sentence1.shape,sentence1_mask.shape,sentence2.shape,sentence2_mask.shape,y.shape
                #o = f_debug(sentence1,sentence1_mask,sentence2,sentence2_mask,y)
                #print o
                #print o[0].shape,o[1].shape,o[2].shape,o[3].shape
                cost = f_grad_shared(sentence1,sentence1_mask,sentence2,sentence2_mask,y)
                f_update(model_options['alpha'])
                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', epoch, 'Update ', uidx, 'Cost ', cost, 'Samples_seen ', samples_seen
                    sys.stdout.flush()
            print 'Epoch ', epoch, 'Update ', uidx, 'Cost ', cost, 'Samples_seen ', samples_seen
            sys.stdout.flush()
            '''
            if epoch % 5 == 0:
                kf_train = get_minibatches_idx(len(train), batch_size)
                print ('Train_score:')
                self.eva(f_pred, src_train, train, pairdict_train, kf_train, model_options)
                sys.stdout.flush()
            '''
            print ('Valid_score:')
            top1_res = self.eva(f_pred, src_valid, valid, pairdict_valid, kf_valid, model_options)
            self.save_result(model_options['result_path'] + 'dev.on.' + str(epoch) +'th_epoch_' + model_options['dataset'],top1_res)
            sys.stdout.flush()
            print ('Test_score:')
            top1_res = self.eva(f_pred, src_test, test, pairdict_test, kf_test, model_options)
            self.save_result(model_options['result_path'] + 'test.on.' + str(epoch) +'th_epoch_' + model_options['dataset'],top1_res)
            sys.stdout.flush()
            
            print ('%d epoch completed.' % epoch)
            sys.stdout.flush()
            '''
            if(best_validation_score < valid_score):
                best_iter = epoch
                best_validation_score = valid_score
            print ('Current best_dev_F is %.2f, at %d epoch'%(best_validation_score,best_iter))
            '''
        
            end_time = time.time()
            minu = int((end_time - start_time)/60)
            sec = (end_time - start_time) - 60 * minu
            print ('Time: %d min %.2f sec' % (minu, sec))
            sys.stdout.flush()
        print('Training completed!')
        sys.stdout.flush()
       
        