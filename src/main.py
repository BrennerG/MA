import sys
import torch

#from experiments.bert_experiment import BERTExperiment
#from experiments.random_experiment import RandomClassifierExperiment
from experiments.ud_gcn_experiment import UD_GCN_Experiment

'''
# ***************************** MAIN FILE ******************************** #

It's the main file!
Currently this is for handling the Experiment Class, which is the main unit of this repo.
An Experiment is run from start to finish using the given parameter settings.
(Parameter search, generation and reading from .yaml will be supported later)

'''

# TODO what to do with params that are only used by a single model/pipeline
PARAMS = {
    'model_type': 'UD_GAT', # determines the type of model for the pipeline (used by Experiment.model_factory())
    # 'debug': True, # deprecated: manipulate _LIMIT variable in huggingface_cose.py manually # TODO fix this when param space is read from .yaml files!
    'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5], # ERASER thresholds to determine k
    #'load_from': 'data/experiments/gcn/gcn.pt', # use a checkpoint (currently only for inference) train from pretrained base if empty
    #'skip_training': True, # skip training - mostly for inference with saved checkpoints
    #'skip_evaluation': True, # skip evaluation (prediction still happens...)
    'use_cuda': True, # use cuda
    'rnd_seed': 69, # Random obligatory 
    #'bert_base': 'albert-base-v2', # choose the pretrained bert-base # TODO rename this so it also fits glove (e.g. pretrained_embedding or sth.) (see 'embedding' below)
    #'softmax_logits': True, # TODO check what this actually does for BERT Baseline
    #'attention': 'lime', # how to generate token weights (relevant for explainability metrics) {'lime', 'zeros', 'random', None}
    #'lime_num_features': 7, # number of tokens that lime assigns a weight to
    #'lime_num_permutations': 5000, # number of input permutations per sample; default=5k :O
    #'lime_scaling': 'none', # decides how the limeweights should be scaled (per sample) # TODO add abs (apply np.abs(attn_weights))
    'learning_rate': 0.003, # Optimizer learning rate
    'weight_decay': 0, # TODO currently only for GCN
    #'batch_size': 16, # batch size for training and evaluation
    'epochs': 100, # epochs of training
    #'save_strategy': 'epoch', # determines rules for saving artifacts {'no', 'epoch', 'steps'} # TODO only for BERT
    'save_loc': 'data/experiments/gat/', # location where the above are saved
    #'overwrite_output_dir': True, # overwrites the given output directory above
    #'save_predictions': True,
    #'save_eraser_input': True

    # GNN STUFF STARTS HERE (there is general stuff that is used for GNNs above!)
    #'gcn_hidden_dim': 1024, # hidden dim for the pure GCN Baseline # TODO not needed if 'input_dims' param is here
    'qa_join': 'to-root', # gnn only
    'wandb_logging': True, # gnn only
    'inter_training_expl_eval': False, # gnn only
    'dropout': 0.1, # gnn only
    'num_heads': 8, # gnn only
    'num_layers': 7, # gnn only
    'input_dims': [300, 1024, 512, 256, 64, 32, 16],
    'embedding': 'albert-base-v2'
}


if __name__ == "__main__":
    torch.manual_seed(PARAMS['rnd_seed'])

    # get & set overwritten params
    args = [a.replace('--','').split('=') for a in sys.argv][1:]
    overwrites = [a for a in args if a[0] in PARAMS.keys()]
    for o in overwrites:
        print(f"WARNING: overwriting {o[0]} parameter: {PARAMS[o[0]]} -> {o[1]} !")
        # parse value
        if o[1]=='True':
            value = True
        elif o[1]=='False':
            value = False
        elif o[1].isdecimal():
            value = int(o[1])
        elif o[1].replace('.','',1).isdigit():
            value = float(o[1])
        else:
            value = o[1]
            print(f"WARNING: couldn't find a proper parse for parameter '{o[0]}'={value}. keeping it as string")
        PARAMS[o[0]] = value

    # run the experiment
    print("\nSTARTING EXPERIMENT")
    for k,v in PARAMS.items(): print(f"\t{k}={v}")
    exp = UD_GCN_Experiment(PARAMS) # TODO create Experiment Factory! (w. default params?)
    exp(PARAMS)
    print('done')