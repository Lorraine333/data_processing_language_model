import re
import pdb
import nltk
import pickle
import random
import numpy as np
import unicodedata
from tqdm import tqdm
from collections import defaultdict
from nltk.corpus import wordnet as wn

def synset_from_sense_key(sense_key):
    ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    sense_key_regex = re.compile(r"(.*)\%(.*):(.*):(.*):(.*):(.*)")
    synset_types = {1: NOUN, 2: VERB, 3: ADJ, 4: ADV, 5: ADJ_SAT}
    lemma, ss_type, _, lex_id, _, _ = sense_key_regex.match(sense_key).groups()
    
    # check that information extracted from sense_key is valid
    error = None
    if not lemma:
        error = "lemma"
    elif int(ss_type) not in synset_types:
        error = "ss_type"
    elif int(lex_id) < 0 or int(lex_id) > 99:
        error = "lex_id"
    if error:
        raise WordNetError(
            "valid {} could not be extracted from the sense key".format(error))
    synset_id = '.'.join([lemma, synset_types[int(ss_type)], lex_id])
    return wn.synset(synset_id)
def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False