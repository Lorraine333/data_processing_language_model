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

import synsets
import main_func

def get_idx2sense():
    idx2sense = defaultdict()
    with open('../semcor/SemCor/semcor.gold.key.txt') as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(' ')
            temp = parts[1][:parts[1].find('%')]
            Punc = False
            for c in temp:
                if synsets._is_punctuation(c):
                    Punc = True
            if not Punc:
                idx2sense[parts[0]] = parts[1]
    return idx2sense

def get_idx2synsets(idx2sense):
    # In order to get the level information, we start from depth=1, then depth=2 (the final augumented version used 2 levels)
    hyp = lambda s:s.hypernyms()
    idx2synsets = defaultdict()
    for idx in idx2sense:
        try:
            if ('%1:') in idx2sense[idx]:
                synset = synsets.synset_from_sense_key(idx2sense[idx])
                for i in list(synset.closure(hyp, depth=1)):
                    lemma_list = i.lemmas()
                    for j in lemma_list:
                        if '_' not in j._name and '.' not in j._name and '-' not in j._name and '\'' not in j._name:
                            if idx not in idx2synsets:
                                idx2synsets[idx] = [j._name+'%%d_1']
                            else:
                                idx2synsets[idx] += [j._name+'%%d_1']
                for i in list(synset.closure(hyp, depth=2)):
                    lemma_list = i.lemmas()
                    for j in lemma_list:
                        if '_' not in j._name and '.' not in j._name and '-' not in j._name and '\'' not in j._name:
                            if idx not in idx2synsets:
                                idx2synsets[idx] = [j._name+'%%d_2']
                            elif j._name+'%%d_1' not in idx2synsets[idx]:
                                idx2synsets[idx] += [j._name+'%%d_2']
                                
                                
        except:
            pass
    print('Number of words with noun senses that we will replace in our dataset %d' %len(idx2synsets))
    return idx2synsets

def get_sequence_info(filename, idx2synsets):
    print('*'*50)
    # idx is sentence index which is represented by the first location of the sentece.
    idxword_to_replacedword_dict, idx_to_original_word_sentence_dict = main_func.get_sentence_and_replaced_word_for_sentence(
        filename, idx2synsets)
    idx_to_replacedword_dict = main_func.get_idx_to_replaceword_dict(idxword_to_replacedword_dict)
    # this token list is the repeted sentences
    token_list, level_list, hypernym_set_list, hyper_token_idx_list = main_func.construct_token_list_and_level_list(
        idx_to_original_word_sentence_dict, idx_to_replacedword_dict)
    return token_list, level_list, hypernym_set_list, hyper_token_idx_list

def combine_hyperset(train_hypernym_set_list, dev_hypernym_set_list, test_hypernym_set_list):
    print('*'*50)
    print('combining hyperset to create hyperset index')
    hypernym_set_list = train_hypernym_set_list+dev_hypernym_set_list+test_hypernym_set_list
    hyperset_set = set()
    for hyper in hypernym_set_list:
        if tuple(hyper) not in hyperset_set:
            hyperset_set.add(tuple(hyper))
    print('Number of unique hyperset', len(hyperset_set))
    idx2hyperset = list(hyperset_set)
    hyperset2idx = {idx2hyperset[i]: i+1 for i in range(len(idx2hyperset))}
    assert len(idx2hyperset) == len(hyperset2idx)
    return idx2hyperset, hyperset2idx

def get_hyperset_idx(hyperset2idx, hypernym_set_list):
    hyperset_idx_list = []
    for hyper in hypernym_set_list:
        if hyper!="None":
            hyperset_idx_list.append(hyperset2idx[tuple(hyper)])
        else:
            hyperset_idx_list.append(0)
    return hyperset_idx_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    args = parser.parse_args()

    idx2sense = get_idx2sense()
    idx2synsets = get_idx2synsets(idx2sense)

    train_file = '../semcor/SemCor/semcor.train_data.xml'
    dev_file = '../semcor/SemCor/semcor.dev_data.xml'
    test_file = '../semcor/SemCor/semcor.test_data.xml'
    filename = '../semcor/SemCor/semcor.'+args.data_file+'_data.xml'
    # idx is sentence index which is represented by the first location of the sentece.
    dev_token_list, dev_level_list, dev_hypernym_set_list, dev_hyper_token_idx_list = get_sequence_info(
        dev_file, idx2synsets)
    test_token_list, test_level_list, test_hypernym_set_list, test_hyper_token_idx_list = get_sequence_info(
        test_file, idx2synsets)
    train_token_list, train_level_list, train_hypernym_set_list, train_hyper_token_idx_list = get_sequence_info(
        train_file, idx2synsets)

    idx2hyperset, hyperset2idx = combine_hyperset(train_hypernym_set_list, dev_hypernym_set_list, test_hypernym_set_list)
    if args.data_file == 'dev':
        dev_hyperset_idx_list = get_hyperset_idx(hyperset2idx, dev_hypernym_set_list)
        assert len(dev_hyperset_idx_list) == len(dev_hypernym_set_list)
        with open('../semcor/aug_dev_word_tokens.pkl', 'rb') as f:
            word_tokens =  pickle.load(f)
        print('Number of Bert word tokens', len(word_tokens))
        token_list = dev_token_list
        hyper_token_idx_list = dev_hyper_token_idx_list
        level_list = dev_level_list
        hyperset_idx_list = dev_hyperset_idx_list
        assert len(token_list) == 734152
    elif args.data_file == 'test':
        test_hyperset_idx_list = get_hyperset_idx(hyperset2idx, test_hypernym_set_list)
        assert len(test_hyperset_idx_list) == len(test_hypernym_set_list)
        with open('../semcor/aug_test_word_tokens.pkl', 'rb') as f:
            word_tokens =  pickle.load(f)
        print('Number of Bert word tokens', len(word_tokens))
        token_list = test_token_list
        hyper_token_idx_list = test_hyper_token_idx_list
        level_list = test_level_list
        hyperset_idx_list = test_hyperset_idx_list
        assert len(token_list) == 881918
    
    elif args.data_file == 'train':
        train_hyperset_idx_list = get_hyperset_idx(hyperset2idx, train_hypernym_set_list)
        assert len(train_hyperset_idx_list) == len(train_hypernym_set_list)
        with open('../semcor/aug_train_word_tokens.pkl', 'rb') as f:
            word_tokens = pickle.load(f)
        print('Number of Bert word tokens', len(word_tokens))
        token_list = train_token_list
        hyper_token_idx_list = train_hyper_token_idx_list
        level_list = train_level_list
        hyperset_idx_list = train_hyperset_idx_list
        assert len(token_list) == 6664040
    else:
        raise ValueError('Invalid data_file input, choose from dev, test or train')

    updated_hyper_token_idx_list, updated_level_list, updated_hyperset_list = main_func.update_level_hyper_set_list(token_list,
        level_list, hyperset_idx_list, hyper_token_idx_list, word_tokens)
    assert len(updated_hyper_token_idx_list) == len(hyper_token_idx_list)
    with open('../semcor/updated_'+args.data_file+'_idx.pkl', 'wb') as f:
        pickle.dump(np.asarray(updated_hyper_token_idx_list), f)

    assert len(updated_level_list) == len(word_tokens)
    with open('../semcor/updated_'+args.data_file+'_level_idx.pkl', 'wb') as f:
        pickle.dump(np.asarray(updated_level_list), f)

    assert len(updated_hyperset_list) == len(word_tokens)
    with open('../semcor/updated_'+args.data_file+'_hyperset_idx.pkl', 'wb') as f:
        pickle.dump(np.asarray(updated_hyperset_list), f)




