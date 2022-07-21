import data_access.locations as LOC
import data_access.persist as P

from models.random_attn_clf import RandomAttentionClassifier
from data_access.experiment import Experiment


'''
# ***************************** MAIN FILE ******************************** #

It's the main file!
Currently this is for handling the Experiment Class, which is the main unit of this repo.
This means training, evaluation, visualization.
You can also save and load experiments.
see src/data_access/experiment.py

The input to the Experiment Class is a .yaml file with all the experiment settings.
(model_parameters, evaluation_parameters, ...)
This file can either be manually written or generated automatically (TODO).
see data/experiments/reference.yaml

In the future this file will be a CLI interface, with the following arguments:
- EID (experiment id - load a saved experiment or create new experiment with new id)
- NOWRITE (do a pass through the pipeline without saving the data to disk)

'''

# script arguments
EID = 'new' # this is the rnd classifier
NOWRITE = False

# initialize relevant folders
LOC.init_locations()

# the main unit
exp = Experiment(
    eid=EID, # experiment identifier - None for automatic name
    NOWRITE=NOWRITE
)

print('training...')
exp.train()

print('evaluating...')
exp.evaluate()

print('visualizing...')
exp.visualize()

print('saving...')
exp.save()

print('done')