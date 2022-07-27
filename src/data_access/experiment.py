import os
import yaml
import json
import torch
import pickle

from os import walk
from datetime import datetime
from yaml import CLoader as Loader

import train as T
import eval as E
import data_access.persist as P
import evaluation.visualizations.viz as viz
import torch.nn as nn

from data_access.locations import LOC
from data_access.csqa_dataset import CsqaDataset
from data_access.cose_dataset import CoseDataset


''' 
# ***************************** EXPERIMENT CLASS ******************************** #

This class represents a single pass through the pipeline and provides means to reproducing
such passes/runs by saving data such as:
- the trained model
- predictions
- data for visualizations
- calculated evaluation metrics 
- ...

An Experiment can be:
- initialized
- trained
- evaluated
- visualized
- saved
- loaded (with constructor)

The central piece of information from which an experiment can be reloaded is its .yaml file,
which is located at data/experiments/ and named after the eid (Experiment ID).
Due to the nature of yaml, we will not persist the objects themselves (e.g. the model) in the .yaml,
but rather ids or locations of the actual persisted objects (e.g. pickle file of the model).

to see an example for the .yaml see data/experiments/reference.yaml

# ***************************** **************** ******************************** # 
'''

class Experiment():

    # init function also loads an experiment from disk
    def __init__(self, 
            eid:str,  # 'auto' for automatic name
            NOWRITE = False,
            state:{}=None  # the state dictionary for the experiment, keeps everything
            ):
        
        # meta
        self.eid = eid
        # self.eid = P.hash_id(str(model) + str(parameters)) # automatic name - TODO prbly put this in CLI
        if state == None: 
            if P.check_yaml(eid): self.state = state = P.load_yaml(eid)
            else: # self.state = state = P.create_barebones_yaml(eid)
                raise NotImplementedError('Please (for now manually) create a proper .yaml experiment file!')
        else:
            self.state = state
        self.NOWRITE = NOWRITE
        self.lvl = state['lvl']
        # write date
        if 'date' in state and state['date']: self.date = state['date']
        else: state['date'] = self.date = str(datetime.now())[:19]
        # initializing model
        self.model_params = state['model_params']
        if 'model_loc' in state and os.path.exists(state['model_loc']): # old
            self.model = P.model_factory(type=state['model_type'], parameters=state['model_params'], path=state['model_loc'])
        else: # new model
            self.model = P.model_factory(type=state['model_type'], parameters=state['model_params'], path=None)
        # data
        if 'limit' in state and state['limit'] > 0:
            if state['dataset'] == 'cose': self.dataset = CoseDataset(mode='train', limit=state['limit'])
            if state['testset'] == 'cose': self.testset = CoseDataset(mode='test', limit=state['limit'])
        else:
            if state['dataset'] == 'cose': self.dataset = CoseDataset(mode='train')
            if state['testset'] == 'cose': self.testset = CoseDataset(mode='test')
        # get preprocessed data
        if 'preprocessed' in state: self.preprocessed = state['preprocessed']
        else: self.preprocessed = None
        if 'train_predictions' in state: self.train_predictions = state['train_predictions']
        else: self.train_predictions = None
        if 'test_predictions' in state: self.test_predictions = state['test_predictions']
        else: self.test_predictions = None
        # eval
        self.evaluation_params = state['evaluation_params']
        self.evaluation_mode = state['evaluation_mode']
        # evaluation results
        if 'evaluation_results' in state: self.evaluation_results = state['evaluation_results']
        else: state['evaluation_results'] = self.state['evaluation_results'] = None
        # viz
        self.viz_mode = state['viz_mode']
        if 'viz_data' in state: self.viz_data = state['viz_data'] # semi_dir
        else: self.viz_data = {}
        if 'viz_dir' in state: self.viz_dir = state['viz_dir'] # semi_dir

        # hide state for consistency
        self.hidden_state = state.copy()
        self.state = None
    

    # switch objects vs locations and pop object entries!
    # TODO add option to copy and rename the experiment!
    def save(self):
        self.state = self.hidden_state.copy() # reactivate hidden state
        self.hidden_state = None

        # model
        if self.lvl>0: # only save model if it has been trained!
            self.state['model_loc'] = P.save_model(self)
            if 'model' in self.state: self.state.pop('model', None)

        # data
        if self.preprocessed:
            self.state['preprocessed_loc'] = P.save_json(self, self.preprocessed, type='preprocessed_data')
            if 'preprocessed' in self.state: self.state.pop('preprocessed')

        if self.train_predictions:
            self.state['train_predictions_loc'] = P.save_json(self, [P.parse_tensor_list(x) for x in self.train_predictions], type='train_predictions')
            if 'train_predictions' in self.state: self.state.pop('train_predictions')

        if self.test_predictions:
            self.state['test_predictions_loc'] = P.save_json(self, [P.parse_tensor_list(x) for x in self.test_predictions], type='test_predictions')
            if 'test_predictions_loc' in self.state: self.state.pop('test_predictions_loc')

        if len(self.viz_data.keys()) != 0: 
            self.state['viz_data_loc'] = P.save_pickle(self, self.viz_data)
            if 'viz_data' in self.state: self.state.pop('viz_data')

        # UPDATE state!
        self.state['lvl'] = self.lvl
        self.state['evaluation_results'] = self.evaluation_results # TODO if evaluation_results already exist, make a new save?
        self.state['viz_dir'] = self.viz_dir

        # then simply save state as yaml
        P.save_yaml(self, self.state)

    # Trains the algorithms of the experiment
    # actually a wrapper for the training module src/train.py
    # also saves training predictions and data for training visualization
    def train(self): 
        t_out = T.train(self.model_params, self.dataset, self.model, proba=True)
        self.model = t_out['model'] # updates the model
        if t_out['losses']:
            self.viz_data['train_loss'] = [float(round(x,4)) for x in t_out['losses']] # save training relevant vis data
        self.train_predictions = t_out['outputs'], t_out['attentions'] # keep the preds & attentions
        self.lvl = 1 # increase progress level of experiment

        return self.train_predictions
    
    # evaluates the algorithm trained on the testset for the given evaluation modes
    # results are saved in the main .yaml
    # real evaluation logic at src/evaluation/ and src/eval.py
    def evaluate(self):
        assert self.train_predictions != None
        assert self.lvl > 0
        result = {'train':{}, 'test':{}}
        for mode in result.keys():
            if mode == 'train':
                dataset = self.dataset
                pred, attn = self.train_predictions

            elif mode == 'test':
                dataset = self.testset
                pred, attn = self.test_predictions = T.predict(self.model_params, self.model, self.testset, proba=True)
            
            gold = dataset.labels
            doc_ids = dataset.docids

            if 'explainability' in self.evaluation_mode: 
                result[mode]['agreement_aupcr'], result[mode]['classification_scores'] = E.explainability_metrics(self.model, dataset, self.model_params, self.evaluation_params, mode=mode)

            if 'efficiency' in self.evaluation_mode:
                # TODO does this actually need the train/test mode? or is eff beyond train / test
                result[mode]['efficiency'] = E.efficiency_metrics(self.model.lin, (1000, 3, 50)) # TODO this only works for RandomAttentionClassifier

            if 'competence' in self.evaluation_mode:
                result[mode]['accuracy'], result[mode]['precision'], result[mode]['recall'] = E.competence_metrics(gold, pred)

        self.evaluation_results = result
        self.lvl = 2
        return result

    # visualizes pretty much anything from the saved data for visualization (data/viz/data)
    # actual vizualizations go into (data/viz/<eid>)
    def visualize(self, show=False):
        # create viz_directory
        if not self.NOWRITE:
            newpath = LOC['viz_dir'] + str(self.eid) + "/"
            if not os.path.exists(newpath):
                os.makedirs(newpath)

            for mode in self.viz_mode:
                if mode == 'loss' and 'train_loss' in self.viz_data:
                    viz.loss_plot(self.viz_data['train_loss'], save_loc=newpath, show=show)
                    self.lvl = 3
                else:
                    'VIZ_MODE:' + '"' + mode + '"' + " failed!"
            
            self.viz_dir = newpath
        else:
            print('NOWRITE param set: not even pictures will be exported!')