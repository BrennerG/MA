from experiments.bert_experiment import BERTExperiment
from experiments.random_experiment import RandomClassifierExperiment
from experiments.ud_gcn_experiment import UD_GCN_Experiment


DEFAULT_PARAMS = {
    'general': {
        # 'debug': True, # deprecated: manipulate _LIMIT variable in huggingface_cose.py manually # TODO fix this when param space is read from .yaml files!
        #'load_from': 'data/experiments/gcn/gcn.pt', # use a checkpoint (currently only for inference) train from pretrained base if empty
        'skip_training': False, # skip training - mostly for inference with saved checkpoints
        'skip_evaluation': False, # skip evaluation (prediction still happens...)
        'use_cuda': True, # use cuda
        'rnd_seed': 69, # Random obligatory 
        'learning_rate': 0.005, # Optimizer learning rate
        'epochs': 30, # epochs of training
        'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5], # ERASER thresholds to determine k
        'save_loc': 'data/experiments/default/', # location where the above are saved
        'wandb_logging': True, 
    },
    'BERT': {
        'batch_size': 16, # batch size for training and evaluation
        'bert_base': 'albert-base-v2', # choose the pretrained bert-base # TODO rename this so it also fits glove (e.g. pretrained_embedding or sth.) (see 'embedding' below)
        'weight_decay': 0.01,
        # 'softmax_logits': False, # wo don't need this? default val is False anyway
        'attention': 'lime', # how to generate token weights (relevant for explainability metrics) {'lime', 'zeros', 'random', None}
        'lime_num_features': 7, # number of tokens that lime assigns a weight to
        'lime_num_permutations': 5000, # number of input permutations per sample; default=5k :O
        'lime_scaling': 'none', # decides how the limeweights should be scaled (per sample) # TODO add abs (apply np.abs(attn_weights))
        # TODO bring these into other experiments?
        'save_strategy': 'epoch', # determines rules for saving artifacts {'no', 'epoch', 'steps'} # TODO only for BERT
        'overwrite_output_dir': True, # overwrites the given output directory above
        'save_predictions': True,
        'save_eraser_input': True
    },
    'GNN': {
        'weight_decay': 0,
        'qa_join': 'to-root',
        'inter_training_expl_eval': False, 
        'dropout': 0.1, 
        'num_heads': 2, 
        'num_layers': 7, 
        'input_dims': [512, 256, 64],
        'embedding': 'glove'
    }
}

# TODO make it so load_from can be disabled in default params, but passed per CLI in order to load!
def experiment_factory(exp_type:str, args:{}):

    def overwrite_params(params):
        overwrites = [a for a in args if a[0] in params.keys()]
        for o in overwrites:
            print(f"WARNING: overwriting {o[0]} parameter: {params[o[0]]} -> {o[1]} !")
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
                print(f"WARNING: couldn't find a proper parse for parameter '{o[0]}'={value}. parsed to string")
            params[o[0]] = value
        return params
    
    if exp_type == 'Random':
        params = overwrite_params(DEFAULT_PARAMS['general'])
        params['model_type'] = exp_type
        exp = RandomClassifierExperiment(params)
    elif exp_type == 'BERT':
        params = overwrite_params({**DEFAULT_PARAMS['general'], **DEFAULT_PARAMS['BERT']})
        params['model_type'] = exp_type
        exp = BERTExperiment(params)
    elif exp_type == 'UD_GCN' or exp_type == 'UD_GAT':
        params = overwrite_params({**DEFAULT_PARAMS['general'], **DEFAULT_PARAMS['GNN']})
        params['model_type'] = exp_type
        exp = UD_GCN_Experiment(params)
    else:
        raise AttributeError(f"experiment_type '{exp_type}' unknown!")
    
    # print start
    print("\INITIALIZING EXPERIMENT")
    for k,v in params.items(): print(f"\t{k}={v}")

    return exp