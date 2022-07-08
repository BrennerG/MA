from models.random_attn_clf import RandomAttentionClassifier
from data_access.experiment import Experiment
import data_access.locations as LOC


'''
# ***************************** MAIN FILE ******************************** #
# Grand Description incoming! ^.~
#      ************************** TODO *****************************       #

UNITE PARAMETERS AND EXPERIMENT INPUT!
- Experiment could take only parameters dict
- do more sophisticated structure
- read and write it as .yaml - from the experiment folder? (do a folder with plans or sth?)

DOCUMENT PARAMETERS, MAIN AND EXPERIMENT CLASS
- create reference.yaml

SHARE WITH GABOR

FIND OUT WHY AUPCR_AGREEMENT GIVES NAN FOR TESTSET!

ADD EFFICIENCY EVALUATION METRICS!
# ************************************************************************ #
'''

# initialize relevant folders
LOC.init_locations()

# set parameters for model, training and evaluation
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
    'NOWRITE': False,
    'dataset': 'cose_train',
    'testset': 'cose_test',
    'model': 'RandomAttentionClassifier',
    'evaluation_mode': ['competence', 'explainability'],
    'viz_mode': ['loss']
}

# set the parameters of the experiment
exp = Experiment(
    eid='default',
    parameters = parameters,
    NOWRITE = parameters['NOWRITE'],
    dataset = parameters['dataset'],
    testset = parameters['testset'],
    model = parameters['model'],
    evaluation_mode = ['competence', 'explainability'], # ['competence', 'explainability']
    viz_mode=['loss']
)

# run the experiment
exp.train()
exp.evaluate()
exp.visualize()
exp.save()
print('saved')

# load the experiment
loaded = exp.load(exp.eid)
print('loaded')
evaluation = loaded.evaluate()
print('done')