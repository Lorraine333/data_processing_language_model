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

import utils
import synsets

def get_sentence_and_replaced_word_for_sentence(filename, idx2synsets):
    first_pos, last_pos, current_idx = 0,0,0
    sentence_firstloc_to_replacedword_dict = defaultdict(list)
    sentence_firstloc_to_original_word_sentence_dict = defaultdict(str)
    with open(filename) as inputfile:
        lines = inputfile.readlines()
        while current_idx < len(lines):
            if lines[current_idx].startswith('<sentence') and first_pos!=current_idx:
                first_pos = current_idx
            elif 'lemma' in lines[current_idx]:
                word = lines[current_idx][lines[current_idx].find('lemma=\"')+7:lines[current_idx].find(' pos')-1]
                if lines[current_idx].startswith('<instance') and 'pos=\"NOUN' in lines[current_idx]:
                    idx = lines[current_idx][lines[current_idx].find('id=\"')+4:lines[current_idx].find(' lemma')-1]
                    if idx not in idx2synsets:
                        pass
                    else:
                        word += '%%'+idx
                        for j in idx2synsets[idx]:
                            if (str(first_pos)+' '+word) not in sentence_firstloc_to_replacedword_dict:
                                sentence_firstloc_to_replacedword_dict[str(first_pos)+' '+word] = [j]
                            else:
                                sentence_firstloc_to_replacedword_dict[str(first_pos)+' '+word].append(j)
                            
                if first_pos not in sentence_firstloc_to_original_word_sentence_dict:
                    sentence_firstloc_to_original_word_sentence_dict[first_pos] = word
                else:
                    sentence_firstloc_to_original_word_sentence_dict[first_pos]+=' '+word
                
            current_idx += 1
    print('number of words get replaced in the corpus', len(sentence_firstloc_to_replacedword_dict))
    print('number of sentence in the corpus', len(sentence_firstloc_to_original_word_sentence_dict))
    return sentence_firstloc_to_replacedword_dict, sentence_firstloc_to_original_word_sentence_dict

def construct_token_list_and_level_list(sentence_dict, replaced_word_dict):
    token_list = []
    level_list = []
    hypernym_set_list = []
    hyper_token_idx_list = []
    for key in sentence_dict:
        original_sentence = sentence_dict[key]
        if key in replaced_word_dict:
            replaced_wordlist = replaced_word_dict[key]
            original_word_list = [i[0] for i in replaced_wordlist]
            for word in original_sentence.split(' '):
                if word in original_word_list:
                    matched = False
                    for m in range(len(original_word_list)):
                        if word == original_word_list[m]:
                            if matched:
                                raise ValueError('wrong')
                            else:
                                matched = True
                                level_list.append(1)
                                hyper_token_idx_list.append(len(token_list))
                                hypernym_set_list.append(replaced_wordlist[m])
                                break
                else:
                    hypernym_set_list.append('None')
                    level_list.append(0)
                if '%%d' in word:
                    word = word[:word.find('%%d')]
                token_list.append(word.lower())
                      
            for m in range(len(replaced_wordlist)):
                single_replaced_wordlist = replaced_wordlist[m]
                original_word = single_replaced_wordlist[0]
                assert len(utils.get_matches(original_word, original_sentence))<=1
                
                for i in range(1, len(single_replaced_wordlist)):
                    new_word = single_replaced_wordlist[i]
                    # we have to do this, otherwise, previous easy word can be replaced in this sentece as well. 
                    new_sentence = original_sentence.replace(" "+original_word+" ", " "+new_word+" ")
#                     new_sentence = original_sentence.replace(original_word, new_word)
                    for word in new_sentence.split(' '):
                        if word == new_word:
                            hypernym_set_list.append(single_replaced_wordlist)
                            hyper_token_idx_list.append(len(token_list))
                            level_list.append(int(word[word.find('%%d_')+4:])+1)
                            word = word[:word.find('%%d')]
                        elif '%%d' in word:
                            for temp_wordlist in replaced_wordlist:
                                if temp_wordlist[0] == word:
                                    hypernym_set_list.append(temp_wordlist)
#                             hypernym_set_list.append()
                            hyper_token_idx_list.append(len(token_list))
                            level_list.append(1)
                            word = word[:word.find('%%d')]
                        else:
                            hypernym_set_list.append('None')
                            level_list.append(0)
                        token_list.append(word.lower())
        else:
            for word in original_sentence.split(' '):
                token_list.append(word.lower())
                level_list.append(0)
                hypernym_set_list.append('None')
    print('Number of hypernym set %d' %len(hypernym_set_list))
    print('Number of token level info %d' %len(level_list))
    print('Number of tokens %d' %len(token_list))
    print('Number of hypernym tokens %d' %len(hyper_token_idx_list))
    assert max(hyper_token_idx_list)<len(token_list)
    assert len(token_list) == len(level_list) == len(hypernym_set_list)
    return token_list, level_list, hypernym_set_list, hyper_token_idx_list


def get_idx_to_replaceword_dict(idxword_to_replacedword_dict):
    idx_to_replacedword_dict = defaultdict()
    for key in idxword_to_replacedword_dict:
        idx = key.split(' ')[0]
        word = key.split(' ')[1]
        if int(idx) not in idx_to_replacedword_dict:
            idx_to_replacedword_dict[int(idx)] = [[word]+idxword_to_replacedword_dict[key]]
        else:
            idx_to_replacedword_dict[int(idx)] += [[word]+idxword_to_replacedword_dict[key]]
    print('Number of sentence have replaced word %d' %len(idx_to_replacedword_dict))
    return idx_to_replacedword_dict


def update_level_hyper_set_list(token_list, level_list, hyperset_idx_list, hyper_token_idx_list, word_tokens):
    updated_hyper_token_idx_list = []
    updated_level_list = []
    updated_hyperset_list = []
    offset = 0
    offset_for_level = 0
    for i in tqdm(range(len(token_list))):
    # for i in tqdm(range(100)):
        if i in hyper_token_idx_list:
    #         pdb.set_trace()
            if token_list[i] != word_tokens[i+offset]:
                print(i)
                print(offset)
                print(token_list[i])
                print(word_tokens[i+offset])
                import pdb;
                pdb.set_trace()
                raise ValueError('hypernym split by bert')
            else:
                updated_hyper_token_idx_list.append(i+offset)
        if token_list[i] == word_tokens[i+offset]:
            updated_level_list.append(level_list[i])
            updated_hyperset_list.append(hyperset_idx_list[i])
            pass
        else:
            chars = list(token_list[i])
            i = 0
            start_new_word = True
            output = []
            while i < len(chars):
                char = chars[i]
                if synsets._is_punctuation(char):
                    output.append([char])
                    start_new_word = True
                else:
                    if start_new_word:
                        output.append([])
                    start_new_word = False
                    output[-1].append(char)
                i += 1
            # print(output)
            offset += len(["".join(x) for x in output])-1
            offset_for_level += len(["".join(x) for x in output])
            for m in range(len(["".join(x) for x in output])):
                updated_level_list.append(level_list[i])
                updated_hyperset_list.append(hyperset_idx_list[i])
    return updated_hyper_token_idx_list, updated_level_list, updated_hyperset_list

