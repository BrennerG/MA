from experiments.bert_experiment import BERTExperiment
from experiments.random_experiment import RandomClassifierExperiment

'''
# ***************************** MAIN FILE ******************************** #

It's the main file!
Currently this is for handling the Experiment Class, which is the main unit of this repo.
An Experiment is run from start to finish using the given parameter settings.
(Parameter search, generation and reading from .yaml will be supported later)

'''

PARAMS = {
    'model_type': 'Random', # {'Random', 'BERT'} # defines the kind of model
    'debug': True, # only uses 11 samples of the data for training and testing
    'aopc_thresholds':[0.01, 0.05, 0.1, 0.2, 0.5], # the aopc thresholds for the eraser benchmark
    'load_from': 'data/experiments/22_7/checkpoint-5470', # loads a BERT model from the given checkpoint
    'rnd_seed': 69 # sets the random seed for the Random Classifier
}


if __name__ == "__main__":
    exp = RandomClassifierExperiment(PARAMS)
    # exp = BERTExperiment(PARAMS)
    exp(PARAMS) # find all the data stored in the Experiment Class