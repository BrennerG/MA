from data_access.locations import LOC
from data_access.csqa_dataset import CsqaDataset
from models.random_clf import RandomClassifier
from data_access.experiment import Experiment

parameters = {
    'limit': -1,
    'epochs': 1,
    'print_every': 1000,
    'batch_size': 1,
    'lr': 0.001,
    'momentum': 0.9,
    'random_seed':69
}

exp = Experiment(
    parameters=parameters,
    dataset = CsqaDataset(LOC['csqa_train'], limit=parameters['limit']),
    testset = CsqaDataset(LOC['csqa_test'], limit=-1),
    model = RandomClassifier(parameters['random_seed']),
    evaluation_mode='competence'
)

exp.train()
exp.evaluate()

#exp.save()
print('done')