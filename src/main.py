from experiments.bert_experiment import BERTExperiment
from experiments.random_experiment import RandomClassifierExperiment


PARAMS = {
    'model_type': 'BERT',
    'debug': True,
    'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5],
    #'load_from': 'data/experiments/22_7/checkpoint-5470', # continue from a checkpoint
    # TODO load from, but continue training from that checkpoint ...
    'rnd_seed': 69, # Random obligatory # TODO also use this for BERT!
    # new params
    'bert_base': 'albert-base-v2', # choose the pretrained bert-base
    # TODO expand hyper_param space
    # current
    #'attention_mode': 'lime',
    #'lime_num_features': 30,
    #'lime_num_permutations': 3,
    #'learning_rate': 5e-5,
    #'batch_size': 16,
    #'epochs': 3,
    #'save_strategy': 'epoch',
    #'overwrite_output_dir':False,
}


if __name__ == "__main__":
    exp = BERTExperiment(PARAMS)
    exp(PARAMS)
    print('done')