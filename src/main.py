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
    'model_type': 'BERT', # determines the type of model for the pipeline (used by Experiment.model_factory())
    'debug': True, # only use a tiny number of samples for testing purposes
    'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5], # ERASER thresholds to determine k
    # TODO load from, but continue training from that checkpoint ...
    'load_from': 'data/experiments/22_7/checkpoint-5470', # use a checkpoint (currently only for inference) train from pretrained base if empty
    'skip_training': True, # skip training - mostly for inference with saved checkpoints
    'rnd_seed': 69, # Random obligatory 
    'bert_base': 'albert-base-v2', # choose the pretrained bert-base
    'attention': 'lime', # choose how the model generates the attention weights (relevant for explainability metrics)
    'lime_num_features': 10, # number of tokens that lime assigns a weight to
    'lime_num_permutations': 3, # number of input permutations per sample; default=5k :O
    'learning_rate': 5e-5, # Optimizer learning rate
    'batch_size': 16, # batch size for training and evaluation
    'epochs': 3, # epochs of training
    'save_strategy': 'epoch', # determines rules for saving artifacts {'no', 'epoch', 'steps'}
    'save_loc': 'data/experiments/22_7_ctd/', # location where the above are saved
    'overwrite_output_dir': False, # overwrites the given output directory above
}


if __name__ == "__main__":
    torch.manual_seed(PARAMS['rnd_seed'])
    exp = BERTExperiment(PARAMS)
    exp(PARAMS)
    print('done')