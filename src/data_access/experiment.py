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

from data_access.locations import LOC
from data_access.csqa_dataset import CsqaDataset
from data_access.cose_dataset import CoseDataset


''' 
# ***************************** EXPERIMENT CLASS ******************************** #

This class represents a single pass through the pipeline and provides means to reproducing
such passes/runs by saving data such as:
- the trained model
- data for visualizations
- calculated evaluation metrics 
- ...

An Experiment can be:
- initialized
- trained
- evaluated
- visualized
- saved
- loaded

The central piece of information from which an experiment can be reloaded is its .yaml file,
which is located at data/experiments/ and named after the eid (Experiment ID).
Due to the nature of yaml, we will not persist the objects themselves (e.g. the model) in the .yaml,
but rather ids or locations of the actual persisted objects (e.g. pickle file of the model).

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
            state = P.load_yaml(eid)
            self.state = state
        else:
            self.state = state
        self.NOWRITE = NOWRITE
        self.lvl = state['lvl']
        self.date = state['date']
        # model
        self.model_params = state['model_params']
        if 'model_loc' in state and os.path.exists(state['model_loc']): # load old model
            self.model = P.model_factory(type=state['model_type'], parameters=state['model_params'], path=state['model_loc'])
        elif 'model_loc' in state: # load new model
            self.model = P.model_factory(type=state['model_type'], parameters=state['model_params'], path=None)
        # data
        if state['dataset'] == 'cose': self.dataset = CoseDataset(mode='train')
        if state['testset'] == 'cose': self.testset = CoseDataset(mode='test')
        if 'preprocessed' in state: self.preprocessed = state['preprocessed']
        else: self.preprocessed = None
        if 'train_predictions' in state: self.train_predictions = state['train_predictions']
        else: self.train_predictions = None
        if 'test_predictions' in state: self.test_predictions = state['test_predictions']
        else: self.test_predictions = None
        # eval
        self.evaluation_params = state['evaluation_params']
        self.evaluation_mode = state['evaluation_mode']
        self.evaluation_results = state['evaluation_results']
        # viz
        self.viz_mode = state['viz_mode']
        if 'viz_data' in state: self.viz_data = state['viz_data'] # semi_dir
        else: self.viz_data = {}
        if 'viz_dir' in state: self.viz_dir = state['viz_dir'] # semi_dir

        # TODO
        # hidden_state = state.copy()
        # state = None
    

    # switch objects vs locations and pop object entries!
    # TODO add option to copy and rename the experiment!
    def save(self):
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

        # then simply save state as yaml
        P.save_yaml(self, self.state)

    # Trains the algorithms of the experiment
    # actually a wrapper for the training module src/train.py
    # also saves training predictions and data for training visualization
    def train(self, output_softmax=False): 
        t_out = T.train(self.model_params, self.dataset, self.model)
        self.model = t_out['model'] # updates the model
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
                pred, attn = self.test_predictions = T.predict(self.model_params, self.model, self.testset)
            
            gold = dataset.labels
            doc_ids = dataset.docids

            if 'explainability' in self.evaluation_mode:
                attn_detached = [x.detach().numpy() for x in attn] # TODO model could return detached attn and predictions?
                # retrain for comp and suff:
                comp_data = dataset.erase(attn, mode='comprehensiveness')
                suff_data = dataset.erase(attn, mode='sufficiency')
                comp_predictions, _ = T.predict(self.model_params, self.model, comp_data) # _ is attn vector
                suff_predictions, _ = T.predict(self.model_params, self.model, suff_data) # _ is attn vector
                aopc_predictions = T.predict_aopc_thresholded(self.model_params, self.evaluation_params, self.model, attn, dataset)
                er_results = E.create_results(doc_ids, pred, comp_predictions, suff_predictions, attn_detached, aopc_thresholded_scores=aopc_predictions)
                result[mode]['agreement_auprc'] = E.soft_scores(er_results, docids=doc_ids, ds=f'cose_{mode}') # TODO avg_precision and roc_auc_score NaN, but only for testset!
                result[mode]['classification_scores'] = E.classification_scores(results=er_results, mode=mode, aopc_thresholds=self.evaluation_params['aopc_thresholds'])

            if 'efficiency' in self.evaluation_mode:
                # TODO does this actually need the train/test mode? or is eff beyond train / test
                result[mode]['efficiency'] = E.efficiency_metrics(self.model.lin, (1000, 3, 50)) # TODO input correct input sizes, but where do we get them from?

            if 'competence' in self.evaluation_mode:
                result[mode]['accuracy'], result[mode]['precision'], result[mode]['recall'] = E.competence(gold, pred)

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
                if mode == 'loss':
                    viz.loss_plot(self.viz_data['train_loss'], save_loc=newpath, show=show)
                else:
                    'VIZ_MODE:' + '"' + mode + '"' + " is unknown!"
            
            self.viz_dir = newpath
            self.lvl = 3
        else:
            print('NOWRITE param set: not even pictures will be exported!')