from experiments.bert_experiment import BERTExperiment
from experiments.random_experiment import RandomClassifierExperiment
from experiments.ud_gcn_experiment import UD_GCN_Experiment
from experiments.final_experiment import FinalExperiment
from experiments.qagnn_experiment import QagnnExperiment
from data.locations import LOC


DEFAULT_PARAMS = {
    'general': {
        'load_from': None, # 'data/experiments/gcn/gcn.pt', # use a checkpoint (currently only for inference) train from pretrained base if empty
        'skip_training': False, # skip training - mostly for inference with saved checkpoints
        'skip_evaluation': False, # skip evaluation (prediction still happens...)
        'use_cuda': True, # use cuda
        'rnd_seed': 69, # Random obligatory 
        'learning_rate': 0.005, # Optimizer learning rate
        'epochs': 30, # epochs of training
        'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5], # ERASER thresholds to determine k
        'save_loc': 'data/experiments/default/', # location where the above are saved
        'wandb_logging': True, 
        'use_cache': True
    },
    'BERT': {
        'batch_size': 16, # batch size for training and evaluation
        'bert_base': 'albert-base-v2', 
        'weight_decay': 0.01,
        # 'softmax_logits': False, # wo don't need this? default val is False anyway
        'attention': 'lime', # how to generate token weights (relevant for explainability metrics) {'lime', 'zeros', 'random', None}
        'lime_num_features': 7, # number of tokens that lime assigns a weight to
        'lime_num_permutations': 5000, # number of input permutations per sample; default=5k :O
        'lime_scaling': 'none', # decides how the limeweights should be scaled (per sample) 
        # TODO bring these into other experiments? (currently mostly GCN - if needed)
        'save_strategy': 'epoch', # determines rules for saving artifacts {'no', 'epoch', 'steps'}
        'overwrite_output_dir': True, # overwrites the given output directory above
        'save_predictions': True,
        'save_eraser_input': True
    },
    'GNN': {
        'weight_decay': 0,
        'qa_join': 'to-root',
        'inter_training_expl_eval': False, 
        'dropout': 0.2,
        'num_heads': 2, 
        'embedding': 'albert-base-v2',
        'graph_form': '4lang',
        'gat_hidden_dim': 200,
        'bert_dim': 768,
        'max_num_nodes': None,
        'expand': None,
        'bidirectional': True
    },
    'QAGNN': {
        # FROM GNN
        'qa_join': 'statements',
        'graph_form': '4lang',
        'bert_dim': 768, # TODO needed
        # 'max_num_nodes': 200,
        'max_node_num': 200, # TODO same as above, but we're using this one bc qagnn does
        'expand': None,
        # TRAIN
        'k':5, # num of gat layers
        'gat_hidden_dim': 100,
        'concept_dim': 1024,
        'num_heads': 2, 
        'clf_layer_dim': 200,
        'clf_layer_depth': 0, # +1
        'dropout': 0.2,
        'embedding': 'bert-large-uncased', # TODO same as 'encoder' ...
        'num_node_types': 4,
        'num_relation': 1,
        'weight_decay': 0.01,
        'encoder_lr': 2e-05,
        'decoder_lr': 0.001,
        'optim': 'radam',
        'lr_schedule': 'fixed',
        'loss': 'cross_entropy',
        'unfreeze_epoch': 4,
        'refreeze_epoch': 10000,
        'mini_batch_size': 1,
        'encoder_layer': -1,
        'fp16': False,
        'max_grad_norm': 1.0,
        'log_interval': 10,
        'save_model': True,
        'save_dir': './saved_models/qagnn/', # in params
        'max_epochs_before_stop': 10, # in params
        # DATALOADER
        'train_statements': LOC['qagnn_statements_train'],
        'dev_statements': LOC['qagnn_statements_dev'],
        'test_statements': LOC['qagnn_statements_test'],
        'batch_size': 32,
        'eval_batch_size': 16,
        'encoder': 'bert-large-uncased', # TODO same as 'embedding' ...
        'drop_partial_batch': False,
        'fill_partial_batch': False,
        # NEW
        'node_relevance': True
    }
}

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
    
    # clean None params
    def clean_non_params(params:{}):
        non_params = [k for k in params if params[k] == None]
        for n in non_params:
            params.pop(n)
        return params
    
    # the actual experiment factory
    if exp_type == 'Random':
        params = overwrite_params(DEFAULT_PARAMS['general'])
        params = clean_non_params(params)
        params['model_type'] = exp_type
        exp = RandomClassifierExperiment(params)
    elif exp_type == 'BERT':
        params = overwrite_params({**DEFAULT_PARAMS['general'], **DEFAULT_PARAMS['BERT']})
        params = clean_non_params(params)
        params['model_type'] = exp_type
        exp = BERTExperiment(params)
    elif exp_type == 'UD_GCN' or exp_type == 'UD_GAT':
        params = overwrite_params({**DEFAULT_PARAMS['general'], **DEFAULT_PARAMS['GNN']})
        params = clean_non_params(params)
        params['model_type'] = exp_type
        exp = UD_GCN_Experiment(params)
    elif exp_type == 'final':
        params = overwrite_params({**DEFAULT_PARAMS['general'], **DEFAULT_PARAMS['GNN']})
        params = clean_non_params(params)
        params['model_type'] = exp_type
        exp = FinalExperiment(params)
    elif exp_type == 'qagnn':
        params = overwrite_params({**DEFAULT_PARAMS['general'], **DEFAULT_PARAMS['QAGNN']})
        params = clean_non_params(params)
        params['model_type'] = exp_type
        exp = QagnnExperiment(params)
    else:
        raise AttributeError(f"experiment_type '{exp_type}' unknown!")
    
    # print start
    print("INITIALIZING EXPERIMENT")
    for k,v in params.items(): print(f"\t{k}={v}")

    return exp