from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.lib.io import file_io
from collections import Counter
from itertools import chain
from os import path

import numpy as np
import random
import pickle
import time

## preprocessing
def mk_lookup_table(list_2d):
    word_2d = list_2d.tolist()
    word_1d = list(chain.from_iterable(word_2d))
    word_counts = Counter(word_1d)
    sorted_word = sorted(word_counts.items(), key=lambda i: i[1])
    filtered_word = ['zeros','uncommon'] + [x[0] for x in sorted_word if x[1] >= 10]
    int_to_word = {ii: word for ii, word in enumerate(filtered_word)}
    word_to_int = {word: ii for ii, word in int_to_word.items()}
    print("Total word: {}".format(len(word_1d)))
    print("Unique word: {}".format(len(set(word_1d))))
    return int_to_word, word_to_int

def raw_word_to_int(list_2d, word_to_int):
    int_word = []
    for wd in list_2d:
        if len(wd)==1:
            wd_1d=[word_to_int.get(wd[0],1)] 
        else:
            wd_1d=[word_to_int.get(e,1) for e in wd]
        int_word.append(wd_1d)
    return int_word

def zero_pad(x, size):
    tmp = np.zeros(size, dtype='int')
    tmp[(size-len(x)):] = x
    return tmp

## word2vec
def word_drop_prob(int_word, threshold=1e-4):
    tic = time.time()
    int_word_1d = list(chain.from_iterable(int_word))
    word_counts = Counter(int_word_1d)
    total_count = sum(list(word_counts.values())[1:])
    freqs = {word: count/total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
    print("ELAPSED TIME:", round(time.time() - tic,2), "sec")
    return p_drop

def subsampling(int_word, p_drop):
    tic = time.time()
    train_word = []
    for dm_list in int_word:
        dm_list = [dm for dm in dm_list if random.random() > p_drop[dm]]
        if len(dm_list) > 1:
            train_word.append(dm_list)
    print("size of data:", len(train_word))
    print("ELAPSED TIME:", round(time.time() - tic,2), "sec")
    return train_word

def print_topk(sim, valid_examples, int_to_word, top_k=5):
    for i in range(len(valid_examples)):
        valid_word = int_to_word[valid_examples[i]]
        top_k = top_k
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = int_to_word[nearest[k]]
            log = '%s %s,' % (log, close_word)
        print(log)
        
def get_batches(word, wsize):
    for batch in word:
        x, y = [], []
        for i in range(len(batch)):
            batch_x = batch[i]
            if i<=wsize:
                batch_y = batch[:i] + batch[i+1:i+(wsize+1)]
            else:
                batch_y = batch[i-(wsize+1):i] + batch[i+1:i+wsize]
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y