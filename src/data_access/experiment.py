import os
import yaml
import json
import torch
import torch.nn as nn

from locations import LOC

# An Experiment is stored as a yaml, linking all essential paths:
#   parameters, model_path, preproc_path, viz_path
class Experiment():

    def __init__(self, eid=None, parameters=None, preprocessed=None, model=None, predictions=None, viz=None):
        self.eid = None
        self.date = None # TODO init date string + time, also start/end date?
        self.parameters = None # TODO define default parameters somewhere... (probably this file)
        self.preprocessed = None
        self.model = None
        self.predictions = None
        self.viz = None
    
    def hash_id(self, as_str=True):
        dirs = next(os.walk(LOC['experiments_dir']))[1]
        hash_id = hash(self)
        if hash_id in dirs: return False
        else: 
            if as_str: return str(hash_id)
            else: return hash_id

    # TODO do proper checks (for len, types, etc.)
    # TODO make this return the progress (how far was the model progressed?)
    #       init(main) -> preprocessing(preproc) -> model(training) -> predictions(eval) -> visualizations(viz)
    def check(self):
        if self.eid == None: print('no eid'); return False
        if self.date == None: print('no date'); return False
        if self.parameters == None: print('no params'); return False
        if self.preprocessed == None: print('no preprocessed'); return False
        if self.model == None: print('no model'); return False
        if self.viz == None: print('no viz'); return False
        if self.predictions == None: print('no predictions'); return False


    def save(self):
        dic = {}

        # save date
        dic['date'] = self.date

        # save parameters
        dic['parameters'] = self.parameters

        # save preprocessed
        preproc_save_loc = LOC['preprocessed_data'] + str(hash(self.preprocessed)) + '.jsonl'
        with open(preproc_save_loc, 'w') as outfile:
            json.dump(self.preprocessed, outfile)
        dic['preprocessed'] = preproc_save_loc
    
        # save model
        model_save_loc = LOC['models_dir'] + str(hash(self.model)) + '.pth'
        if type(self.model) == nn.Module: torch.save(clf.state_dict(), model_save_loc)
        else: pass # TODO use other method of saving the model (prob pickle or sth)
        dic['model'] = model_save_loc

        # save viz
        dic['viz'] = 'Not Implemented'

        # save predictions
        predictions_save_loc = LOC['predictions'] + str(hash(self.model)) + '.jsonl'
        with open(predictions_save_loc, 'w') as outfile:
            json.dump(self.predictions, outfile)
        dic['predictions'] = predictions_save_loc

        # save all paths in yaml!
        #with open(r'data\.yaml', 'w') as file:
        with open(LOC['experiments_dir'] + self.hash_id() + '.yaml', 'w') as file:
            yaml_file = yaml.dump(dic, file)

        return yaml_file

    @staticmethod
    def load(self, obj):
        return None
    
    def copy(self):
        return None
    
    def run(self):
        return None


if __name__ == '__main__':
    exp = Experiment()
    exp.parameters = {'this':0, 'is':1, 'a':2, 'test':3}
    exp.save()
    print('done')