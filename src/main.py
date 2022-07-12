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

* CURRENT rework saving and loading
    * start with saving and loading then insert all the steps between and debug/develop!!!
* BUG: experiment can have high lvl, but if it's not saved, the actual state may be lower! - is this really relevant? unlikely scenario!
* BOW Model
    * Bagging
    * Logistic Regression
* revisit efficiency calculations for BOW
* implement module for dynamic .yaml creation!
* implement CLI (low prio, coz it's only fancy)
    * loading new scripts vs creating new scripts
* implement dynamic re-use of already saved data (e.g. preprocessed data!)

# ************************************************************************ #
'''

# script arguments
EID = 'new'
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

print('saving...')
exp.save()

print('done')