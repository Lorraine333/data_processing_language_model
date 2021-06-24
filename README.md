# Introduction
This repo is used for pre-processing hypernym based language model learning. It served as both augumenting the original raw text with the wordnet annotaition hypernym information, but also pre-idx the bpe tokens using bert tokenizations. It will produce the token index and the corresponding level information.

# Code Structure:
* data_processing.py: update hypernym index according to bert word tokenization.
* mask_processing.py: generate mask mark list.
* mask_token_processing.py: using the generate marked list, generated masked-idxed tokens.

# Usage:
1) ran hyper_processing finish for dev/test/train. 3 job
2) ran masked tokens for dev/test/train. 3 jobs
3) ran run re-generate masked tokens. 1 combination job

# Each file readme.
`data_processing.py`

**Input**: 
- SemCor xml file (train/dev/test)
- Augumented word tokens seperated by Bert Tokenizer(train/dev/test)

**Output**:
- updated hyper_idx/level/hyperset pkl file.

**Function**:
1. Takes SemCor xml file, augument the sentence using Wordnet synsets, repeating each sentence to by replacing the hyponym to its hypernym.
2. Compare the resulting repeated sentence to the Bert tokenized version. Since Bert use its own word tokenizer, the tokenization is different. 
3. Sync both tokenizations, save the updated tokenization to the new pickle file. 
The resulting pickle include: the index of the hypernym words in the list, the level information of the hypernym words in the list, the idx of the hypernym set information in the list. 

**Input file location**:

`gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/SemCor/semcor/.\*.xml`
`gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/aug_*_word_tokens.pkl`

**Output file location**: 
gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/updated_*_idx.pkl
gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/updated_*_level_idx.pkl
gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/updated_*_hyperset_idx.pkl

`mask_processing.py`
**Input**: 
- SemCor xml file (train/dev/test)
- Updated hypernym token idx list (output from data_processing.py)
- Updated hyper idx list (output from data_processing.py)
- Updated hyperset idx list (output from data_processing.py)
- Augumented word tokens seperated by Bert Tokenizer(train/dev/test)
- Augumented bpe tokens seperated by Bert Tokenizer(train/dev/test)
- Augumented bpe idx seperated by Bert Tokenizer(train/dev/test)
**Output**:
- Marked bpe idx: with 0(normal), 1(hypernym), 2(subword)
**Function**:
1. Takes SemCor xml file, bert tokenized results, updated word tokenization results.
2. Compare the updated word tokenization with the bpe tokenization, mark each bpe token with the special annotation. 0 for normal words, 1 for hypernym and 2 for subwords.
3. Write the marked list, and the bpe level/hyperset idx. 

**Input file location**: 
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/SemCor/semcor.*.xml
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/aug_*_word_tokens.pkl
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/aug_*_bpe_tokens.pkl
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/aug_*_bpe_idx.pkl
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/updated_*_idx.pkl
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/updated_*_level_idx.pkl
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/updated_*_hyperset_idx.pkl

**Output file location**: 
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/marked_*_idx.pkl
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/marked_bpe_*_level.pkl
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/marked_bpe_*_hyperset.pkl

`mask_token_processing.py`
Combine al index from train/dev/test together, and re-idx the whole file.
Input files are taken from the previously output files. 

**Output file location:**
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/train_bert/train_level_hyperset.pkl
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/train_bert/dev_level_hyperset.pkl
- gypsum:/mnt/nfs/work1/mccallum/xiangl/concept_lm/semcor/train_bert/test_level_hyperset.pkl
