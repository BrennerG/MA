from experiments.bert_experiment import BERTExperiment


PARAMS = {
    'model_type': 'BERT',
    'debug': True,
    'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5]
}


if __name__ == "__main__":
    exp = BERTExperiment(PARAMS)
    exp(PARAMS)
    print('done')