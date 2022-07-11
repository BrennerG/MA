from models.random_attn_clf import RandomAttentionClassifier
from data_access.experiment import Experiment
import data_access.locations as LOC


'''
# ***************************** MAIN FILE ******************************** #

It's the main file!
Currently this is for handling the Experiment Class, which is the main unit of this repo.
Here you can set the parameters of the experiment and algorithm and train, evaluate, visualize.
You can also save and load experiments.

continue reading src/data_acess/experiment.py

#      ************************** TODO *****************************       #

* add efficiency evaluation metrics!
* import params from .yaml
* implement CLI

# ************************************************************************ #
'''

# initialize relevant folders
LOC.init_locations()

# these are all the parameters for a single experiment (=a single pass through the pipeline)
parameters = {
    # DATASETS PARAMS
    'dataset': 'cose_train', # which datasets should be used for training
    'testset': 'cose_test', # which datasets should be used for evaluation
    'limit': -1, # how many samples should be used for the experiment (-1 means all)

    # MODEL PARAMS
    'model': 'RandomAttentionClassifier', # currently: RandomClassifier, RandomAttentionClassifier
    'random_seed': 69, # random seed for reproducibility
    'epochs': 1, # train for x epochs
    'batch_size': 1, # have batches of size x (currently only bs=1) # TODO
    'lr': 0.001, # learning rate of the optimization alg
    'momentum': 0.9, # adam optimizer momentum

    # EVALUATION PARAMS
    'evaluation_mode': ['competence', 'explainability', 'efficiency'], # the modes of evaluation that should be calculated and saved in the .yaml (['competence', 'explainability', 'efficiency'])
    'print_every': 1000, # print loss, metrics, etc. every x samples
    'eraser_k': None, # set the k parameter for the eraser benchmark manually # TODO not implemented yet
    'aopc_thresholds' : [0.01, 0.05, 0.1, 0.2, 0.5], # parameter for the eraser benchmark (see src/train.py: predict_aopc_thresholded())
    'viz_mode': ['loss'], # the visualizations that should be produced and saved during the run

    # META PARAMS
    'NOWRITE': False, # do not produce any file artefacts (disables any sort of reproduciblity)
}

# the main unit
exp = Experiment(
    eid='default', # experiment identifier - None for automatic name
    parameters = parameters,
    NOWRITE = parameters['NOWRITE'],
    dataset = parameters['dataset'],
    testset = parameters['testset'],
    model = parameters['model'],
    evaluation_mode = parameters['evaluation_mode'],
    viz_mode = parameters['viz_mode']
)

# run the experiment
exp.train()
exp.evaluate()
exp.visualize()
exp.save()
print('saved')

# load the experiment & evaluate again
loaded = exp.load(exp.eid)
evaluation = loaded.evaluate()
print('done')