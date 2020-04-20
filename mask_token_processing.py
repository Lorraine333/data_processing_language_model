import re
import pdb
import nltk
import pickle
import random
import argparse
import numpy as np
import unicodedata
from tqdm import tqdm
from collections import defaultdict
from nltk.corpus import wordnet as wn

def read_required_file(mode='train'):
    with open('../semcor/aug_'+mode+'_bpe_tokens.pkl', 'rb') as f:
        bpe_tokens =  pickle.load(f)
    with open('../semcor/aug_'+mode+'_bpe_idx.pkl', 'rb') as f:
        bpe_idx =  pickle.load(f)
    with open('../semcor/marked_'+mode+'_idx.pkl', 'rb') as f:
        marked_tokens =  pickle.load(f)
    assert bpe_tokens.shape == bpe_idx.shape == marked_tokens.shape
    print('Number of tokens %d' %bpe_tokens.shape)
    return bpe_tokens, bpe_idx, marked_tokens

# get max number of sub-word sequence
# max number of continuous 2 following up either 0 or 1.
def get_max_sub(marked_token_list):
    max_num=0
    max_idx = 0
    i=0
    while i < len(marked_token_list):
        if marked_token_list[i] ==0:
            i+=1
            continue
        elif marked_token_list[i] == 1:
            i+=1
            continue
        else:
            curr_num=0
            while marked_token_list[i] == 2:
                curr_num+=1
                i+=1
            if curr_num>max_num:
                max_num = curr_num
                max_idx = i
    # print(max_idx)
    print('Max length of following bpe tokens', max_num)
    return max_num

def get_bert_vocab(bert_token2idx, bert_token, bert_idx):
    for i in range(len(bert_token)):
        if bert_token[i] not in bert_token2idx:
            bert_token2idx[bert_token[i]] = bert_idx[i]
    print('Number of bert tokens in this corpus', len(bert_token2idx))
    return bert_token2idx

# combine all marked list
# 0 for normal word
# 1 for hypernym
# 2 for word-piece weird word
def combine_data_tokens(normal_word, hyper_word, sub_word, bert_tokens, marked_list):
    for i in range(len(bert_tokens)):
        if marked_list[i] == 0:
            if bert_tokens[i] not in normal_word:
                normal_word.append(bert_tokens[i])
        elif marked_list[i] == 1:
            if bert_tokens[i] not in hyper_word:
                hyper_word.append(bert_tokens[i])
        elif marked_list[i] == 2:
            if bert_tokens[i] not in sub_word:
                sub_word.append(bert_tokens[i])
        else:
            raise ValueError('Wrong marked list identifier')
    print('*'*20)
    print('Number of normal tokens %d' %len(normal_word))
    print('Number of hyper tokens %d' %len(hyper_word))
    print('Number of sub word tokens %d' %len(sub_word))
    return normal_word, hyper_word, sub_word

def get_marked_token_list(token_list, mark_list, normal_word2idx, hyper_word2idx, sub_word2idx):
    marked_token_idx_list = []
    for i in range(len(mark_list)):
        if mark_list[i] == 0:
            marked_token_idx_list.append(normal_word2idx[token_list[i]])
        elif mark_list[i] == 1:
            marked_token_idx_list.append(hyper_word2idx[token_list[i]])
        elif mark_list[i] == 2:
            marked_token_idx_list.append(sub_word2idx[token_list[i]])
    return marked_token_idx_list

def main():
	# train_bert_tokens, train_bert_idx, train_marked_list = read_required_file('train')
	# get_max_sub(list(train_marked_list))
	dev_bert_tokens, dev_bert_idx, dev_marked_list = read_required_file('dev')
	test_bert_tokens, test_bert_idx, test_marked_list = read_required_file('test')
	
	bert_token2idx = {}
	# bert_token2idx = get_bert_vocab(bert_token2idx, train_bert_tokens, train_bert_idx)
	bert_token2idx = get_bert_vocab(bert_token2idx, dev_bert_tokens, dev_bert_idx)
	bert_token2idx = get_bert_vocab(bert_token2idx, test_bert_tokens, test_bert_idx)

	normal_word, hyper_word, sub_word = [], [], []
	# normal_word, hyper_word, sub_word = combine_data_tokens(normal_word,
		# hyper_word, sub_word, train_bert_tokens, train_marked_list)
	normal_word, hyper_word, sub_word = combine_data_tokens(
		normal_word, hyper_word, sub_word, dev_bert_tokens, dev_marked_list)
	normal_word, hyper_word, sub_word = combine_data_tokens(
		normal_word, hyper_word, sub_word, test_bert_tokens, test_marked_list)

	NORMAL_WORD_NUM=len(normal_word)
	HYPER_WORD_NUM=len(normal_word)+len(hyper_word)
	SUB_WORD_NUM=len(normal_word)+len(hyper_word)+len(sub_word)
	normal_word2idx = {normal_word[i]:i for i in range(len(normal_word))}
	hyper_word2idx = {hyper_word[i]:NORMAL_WORD_NUM+i for i  in range(len(hyper_word))}
	sub_word2idx = {sub_word[i]:HYPER_WORD_NUM+i for i in range(len(sub_word))}
	assert len(normal_word2idx)+len(hyper_word2idx)+len(sub_word2idx) == SUB_WORD_NUM
	

	myidx2bertidx = {}
	for word in normal_word2idx:
	    if normal_word2idx[word] not in myidx2bertidx:
	        myidx2bertidx[normal_word2idx[word]] = bert_token2idx[word]
	for word in hyper_word2idx:
	    if hyper_word2idx[word] not in myidx2bertidx:
	        myidx2bertidx[hyper_word2idx[word]] = bert_token2idx[word]
	for word in sub_word2idx:
	    if sub_word2idx[word] not in myidx2bertidx:
	        myidx2bertidx[sub_word2idx[word]] = bert_token2idx[word]
	assert len(myidx2bertidx) == SUB_WORD_NUM

	# train_marked_token_idx_list = get_marked_token_list(train_bert_tokens, train_marked_list,
 #                                                    normal_word2idx, hyper_word2idx, sub_word2idx)
	dev_marked_token_idx_list = get_marked_token_list(dev_bert_tokens, dev_marked_list,
	                                                  normal_word2idx, hyper_word2idx, sub_word2idx)
	test_marked_token_idx_list = get_marked_token_list(test_bert_tokens, test_marked_list, 
	                                                  normal_word2idx, hyper_word2idx, sub_word2idx)

	# assert len(train_marked_list) == len(train_bert_idx)
	assert len(dev_marked_list) == len(dev_bert_idx)
	assert len(test_marked_list) == len(test_bert_idx)

	# with open('./WSD_Training_Corpora/bert_train/train.pkl', 'wb') as f:
	    # pickle.dump(np.asarray(train_marked_token_idx_list), f)
	with open('../semcor/train_bert/dev.pkl', 'wb') as f:
	    pickle.dump(np.asarray(dev_marked_token_idx_list), f)
	with open('../semcor/train_bert/test.pkl', 'wb') as f:
	    pickle.dump(np.asarray(test_marked_token_idx_list), f)

	idx_mapping_list = [0]*len(myidx2bertidx)
	for k in myidx2bertidx:
	    idx_mapping_list[k] = myidx2bertidx[k]
	with open('../semcor/train_bert/myidx2bertidx.pkl', 'wb') as f:
	    pickle.dump(np.asarray(idx_mapping_list), f)



if __name__ == "__main__":
    main()