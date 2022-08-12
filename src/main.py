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
    # 'debug': True, # deprecated: manipulate _LIMIT variable in huggingface_cose.py manually # TODO fix this when param space is read from .yaml files!
    'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5], # ERASER thresholds to determine k
    'load_from': 'data/experiments/bert5/checkpoint-2735', # use a checkpoint (currently only for inference) train from pretrained base if empty
    'skip_training': True, # skip training - mostly for inference with saved checkpoints
    #'skip_evaluation': True, # skip evaluation (prediction still happens...) TODO change that?
    'use_cuda': True, # use cuda
    'rnd_seed': 69, # Random obligatory 
    'bert_base': 'albert-base-v2', # choose the pretrained bert-base
    'attention': 'lime', # how to generate token weights (relevant for explainability metrics) {'lime', 'zeros', 'random', None}
    'lime_num_features': 10, # number of tokens that lime assigns a weight to
    'lime_num_permutations': 20, # number of input permutations per sample; default=5k :O
    'learning_rate': 5e-5, # Optimizer learning rate
    'batch_size': 16, # batch size for training and evaluation
    'epochs': 5, # epochs of training
    'save_strategy': 'epoch', # determines rules for saving artifacts {'no', 'epoch', 'steps'}
    'save_loc': 'data/experiments/default/', # location where the above are saved
    'overwrite_output_dir': True, # overwrites the given output directory above
}


if __name__ == "__main__":
    torch.manual_seed(PARAMS['rnd_seed'])
    exp = BERTExperiment(PARAMS)
    exp(PARAMS)
    print('done')
