from pathlib import Path
import os

LOC = {
    'csqa_train': 'data/csqa_raw/train_rand_split.jsonl',
    'csqa_test': 'data/csqa_raw/dev_rand_split.jsonl',
    'cose_huggingface': 'src/data/huggingface_cose.py',
    'cose': 'data/eraser/cose/',
    'cose_docs': 'data/eraser/cose/docs.jsonl',
    'cose_train': 'data/eraser/cose/train.jsonl',
    'cose_test': 'data/eraser/cose/test.jsonl',
    'cose_val': 'data/eraser/cose/val.jsonl',
    'models_dir': 'data/models/',
    'experiments_dir': 'data/experiments/',
    'preprocessed_data': 'data/csqa_preprocessed/',
    'predictions': 'data/predicted/',
    'viz_data_dir': 'data/viz/data/',
    'viz_dir': 'data/viz/',
    'bert_checkpoints': 'data/experiments/bert',
    'ud_parses': 'data/ud_parses/',
    '4lang_parses': 'data/4lang_parses/',
    'glove_dir': 'data/glove',
    'glove_cache': 'data/glove/cached.json',
    'glove_embedding': 'data/glove/glove.840B.300d.txt'
}

# Creates all folders of the LOC dictionary
def init_locations():
    for key, path in LOC.items():
        if path.endswith('/'):
            Path(path).mkdir(parents=True, exist_ok=True)
        else:
            if not os.path.exists(path):
                print('FILE: ' + path + "MISSING!")

    return True
