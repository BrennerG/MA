from experiments.bert_experiment import BERTExperiment
from experiments.random_experiment import RandomClassifierExperiment
import torch

'''
# ***************************** MAIN FILE ******************************** #

It's the main file!
Currently this is for handling the Experiment Class, which is the main unit of this repo.
An Experiment is run from start to finish using the given parameter settings.
(Parameter search, generation and reading from .yaml will be supported later)

'''

PARAMS = {
    'model_type': 'BERT',
    'debug': True,
    'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5],
    # TODO load from, but continue training from that checkpoint ...
    'load_from': 'data/experiments/22_7/checkpoint-5470', # use a checkpoint (currently only for inference) train from pretrained base if empty
    'skip_training': True, # skip training - mostly for inference with saved checkpoints
    'rnd_seed': 69, # Random obligatory 
    # new params
    'bert_base': 'albert-base-v2', # choose the pretrained bert-base
    'attention': 'lime', # choose how the model generates the attention weights (relevant for explainability metrics)
    'lime_num_features': 10, 
    'lime_num_permutations': 3,
    'learning_rate': 5e-5,
    'batch_size': 16,
    'epochs': 3,
    'save_strategy': 'epoch', # {'no', 'epoch', 'steps'}
    'save_loc': 'data/experiments/22_7_ctd/',
    'overwrite_output_dir': False,
}


if __name__ == "__main__":
    torch.manual_seed(PARAMS['rnd_seed'])
    exp = BERTExperiment(PARAMS)
    exp(PARAMS)
    print('done')