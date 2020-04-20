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

def get_matches(sub_string, string):
    matches = list(re.finditer(' '+sub_string+' ', string))
    matches_positions = [match.start() for match in matches]
    return matches_positions