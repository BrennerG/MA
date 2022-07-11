import data_access.locations as LOC
import data_access.persist as P

from models.random_attn_clf import RandomAttentionClassifier
from data_access.experiment import Experiment


'''
# ***************************** MAIN FILE ******************************** #

It's the main file!
Currently this is for handling the Experiment Class, which is the main unit of this repo.
Here you can set the parameters of the experiment and algorithm and train, evaluate, visualize.
You can also save and load experiments.

continue reading src/data_acess/experiment.py

#      ************************** TODO *****************************       #

* ~~import params from .yaml~~
* BUG: experiment can have high lvl, but if it's not saved, the actual state may be lower! - is this really relevant? unlikely scenario!
* remove redundancies in the .yaml!
* revisit saving / loading!
* BOW Model
    * Bagging
    * Logistic Regression
* revisit efficiency calculations for BOW
* implement CLI (low prio, coz it's only fancy)

# ************************************************************************ #
'''

# initialize relevant folders
LOC.init_locations()

# load configuration file
yaml = P.read_experiment_yaml(LOC.LOC['experiments_dir'] + 'init.yaml')

# the main unit
exp = Experiment(
    eid='init', # experiment identifier - None for automatic name
    parameters = yaml['parameters'],
    dataset = yaml['parameters']['dataset'],
    testset = yaml['parameters']['testset'],
    model = yaml['model_type'],
    evaluation_mode = yaml['parameters']['evaluation_mode'],
    viz_mode = yaml['parameters']['viz_mode']
)

# run the experiment
exp.train()
exp.evaluate()
#exp.visualize()
#exp.save()
#print('saved')

# load the experiment & evaluate again
loaded = exp.load(exp.eid)
print('done')