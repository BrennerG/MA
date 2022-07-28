from experiments.bert_experiment import BERTExperiment
from experiments.random_experiment import RandomClassifierExperiment


PARAMS = {
    'model_type': 'Random',
    'debug': True,
    'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5],
    'load_from': 'data/experiments/22_7/checkpoint-5470', # BERT optional
    'rnd_seed': 69 # Random obligatory # TODO also use this for BERT!
}


if __name__ == "__main__":
    exp = RandomClassifierExperiment(PARAMS)
    exp(PARAMS)
    print('done')