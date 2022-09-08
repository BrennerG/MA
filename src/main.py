#from experiments.bert_experiment import BERTExperiment
#from experiments.random_experiment import RandomClassifierExperiment
from experiments.ud_gcn_experiment import UD_GCN_Experiment
import torch

'''
# ***************************** MAIN FILE ******************************** #

It's the main file!
Currently this is for handling the Experiment Class, which is the main unit of this repo.
An Experiment is run from start to finish using the given parameter settings.
(Parameter search, generation and reading from .yaml will be supported later)

'''

# TODO what to do with params that are only used by a single model/pipeline
PARAMS = {
    'model_type': 'UD_GCN', # determines the type of model for the pipeline (used by Experiment.model_factory())
    # 'debug': True, # deprecated: manipulate _LIMIT variable in huggingface_cose.py manually # TODO fix this when param space is read from .yaml files!
    #'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5], # ERASER thresholds to determine k
    #'load_from': 'data/experiments/gcn/gcn.pt', # use a checkpoint (currently only for inference) train from pretrained base if empty
    #'skip_training': True, # skip training - mostly for inference with saved checkpoints
    #'skip_evaluation': True, # skip evaluation (prediction still happens...)
    'use_cuda': True, # use cuda
    'rnd_seed': 69, # Random obligatory 
    #'bert_base': 'albert-base-v2', # choose the pretrained bert-base # TODO rename this so it also fits glove (e.g. pretrained_embedding or sth.)
    #'softmax_logits': True, # TODO check what this actually does for BERT Baseline
    #'attention': 'lime', # how to generate token weights (relevant for explainability metrics) {'lime', 'zeros', 'random', None}
    #'lime_num_features': 7, # number of tokens that lime assigns a weight to
    #'lime_num_permutations': 5000, # number of input permutations per sample; default=5k :O
    #'lime_scaling': 'none', # decides how the limeweights should be scaled (per sample) # TODO add abs (apply np.abs(attn_weights))
    'learning_rate': 0.01, # Optimizer learning rate
    'weight_decay': 5e-4, # TODO currently only for GCN
    #'batch_size': 16, # batch size for training and evaluation
    'epochs': 10, # epochs of training
    #'save_strategy': 'epoch', # determines rules for saving artifacts {'no', 'epoch', 'steps'} # TODO only for BERT
    # 'save_loc': 'data/experiments/gcn/', # location where the above are saved
    #'overwrite_output_dir': True, # overwrites the given output directory above
    #'save_predictions': True,
    #'save_eraser_input': True
    'gcn_hidden_dim': 16, # hidden dim for the pure GCN Baseline
    'qa_join': 'to-root'
}


if __name__ == "__main__":
    torch.manual_seed(PARAMS['rnd_seed'])
    exp = UD_GCN_Experiment(PARAMS) # TODO create Experiment Factory! (w. default params?)
    exp(PARAMS)
    print('done')
