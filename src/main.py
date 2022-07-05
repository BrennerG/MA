from models.random_attn_clf import RandomAttentionClassifier
from data_access.experiment import Experiment


# TODO s 
# TEST LOADING AGAIN
# initiate all folders if not present in init! (besides data folders!)

parameters = {
    'limit': -1,
    'epochs': 1,
    'print_every': 1000,
    'batch_size': 1,
    'lr': 0.001,
    'momentum': 0.9,
    'random_seed': 69,
    'eraser_k': -1,
    'aopc_thresholds' : [0.01, 0.05, 0.1, 0.2, 0.5],
}

exp = Experiment(
    eid='default',
    #NOWRITE=True,
    parameters=parameters,
    dataset = 'cose_train',
    testset = 'cose_test',
    model = RandomAttentionClassifier(parameters['random_seed']),
    evaluation_mode = ['competence', 'explainability'], # ['competence', 'explainability']
    viz_mode=['loss']
)

exp.train()
exp.evaluate()
exp.visualize()
exp.save()
print('saved')

loaded = exp.load('default')
evaluation = loaded.evaluate()
print('loaded')