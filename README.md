# Todo:
* data_processing.py: update hypernym index according to bert word tokenization.
* mask_processing.py: generate mask mark list.
* mask_token_processing.py: using the generate marked list, generated masked tokens.

1) wait for hyper_processing finish for training. 1 job
2) mark masked tokens for dev/test/train. 3 jobs
3) re-generate masked tokens. 1 combination job

