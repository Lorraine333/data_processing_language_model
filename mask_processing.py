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

def read_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    args = parser.parse_args()

    idx2sense = get_idx2sense()
    idx2synsets = get_idx2synsets(idx2sense)

    filename = '../semcor/SemCor/semcor.'+args.data_file+'_data.xml'
    bert_filename = 'aug_'+args.data_file
    token_list, level_list, hypernym_set_list, hyper_token_idx_list = get_sequence_info(
        filename, idx2synsets)
    if args.data_file == 'dev':
        assert len(token_list) == 734152
    elif args.data_file == 'test':
        assert len(token_list) == 881918
    
    elif args.data_file == 'train':
        assert len(token_list) == 6664307
    else:
        raise ValueError('Invalid data_file input, choose from dev, test or train')

    # read file
    updated_hyper_token_idx_list = read_file('../semcor/updated_'+args.data_file+'_idx.pkl')
    word_tokens = read_file('../semcor/'+bert_filename+'_word_tokens.pkl')
    bpe_tokens = read_file('../semcor/'+bert_filename+'_bpe_tokens.pkl')
    bpe_idx = read_file('../semcor/'+bert_filename+'_bpe_idx.pkl')
    
    assert bpe_tokens.shape == bpe_idx.shape
    bpe_tokens = list(bpe_tokens)
    # check stats
    sub_token_size = 0
    for k in bpe_tokens[:]:
        if k.startswith('#'):
            sub_token_size+=1
    print('Number of word tokens %d' %len(word_tokens))
    print('Number of sub token size %d' %sub_token_size)
    print('Number of bpe size %d' %len(bpe_tokens))

    assert sub_token_size+len(word_tokens) == len(bpe_tokens)
    # check stats
    # 0 for normal word
    # 1 for hypernym
    # 2 for word-piece weird word
    print('Getting masked_list...')
    marked_list = []
    offset = 0
    for original_idx in tqdm(range(len(word_tokens))):
        bpe_idx = original_idx+offset
        if original_idx in updated_hyper_token_idx_list:
            marked_list.append(1)
        else:
            marked_list.append(0)
        while bpe_idx+1 < len(bpe_tokens) and str(bpe_tokens[bpe_idx+1]).startswith('#'):
            bpe_idx+=1
            offset+=1
            marked_list.append(2)
    assert len(marked_list) == len(bpe_tokens)
    with open('../semcor/marked_'+args.data_file+'_idx.pkl', 'wb') as f:
        pickle.dump(np.asarray(marked_list), f)



if __name__ == "__main__":
    main()