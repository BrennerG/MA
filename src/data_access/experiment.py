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

    def __init__(self, 
            eid=None, 
            date=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
            edited=None,
            NOWRITE=False,
            parameters=None, 
            model=None, 
            dataset='csqa_train', 
            testset='csqa_test',
            preprocessed=None, 
            train_predictions=None, 
            test_predictions=None, 
            evaluation_mode=['accuracy', 'precision', 'recall'],
            evaluation_results = None,
            viz_mode=['loss'],
            viz_data={},
            viz=None):

        # meta
        if eid != None: self.eid = eid
        else: self.eid = P.hash_id(str(model) + str(parameters))
        self.date = date
        self.edited = None # TODO use this!
        self.NOWRITE = NOWRITE
        self.lvl = 0 # progress level of the experiment 0=initiated, 1=trained, 2=evaluated, 3=visualized
        # model relevant
        self.parameters = parameters 
        self.model = P.model_factory(model, self.parameters)
        # data
        self.dataset = CoseDataset(mode='train')
        self.testset = CoseDataset(mode='test')
        self.preprocessed = preprocessed
        self.train_predictions = train_predictions
        self.test_predictions = test_predictions
        # learnings
        self.evaluation_mode = evaluation_mode
        self.evaluation_results = evaluation_results
        self.viz_mode = viz_mode
        self.viz_data = viz_data
        self.viz = viz
    
    def save(self):
        dic = {}
        # meta
        dic['date'] = self.date
        dic['edited'] = datetime.now().strftime("%d/%m/%Y, %H:%M:%S") 
        dic['lvl'] = self.lvl
        # model
        dic['parameters'] = self.parameters
        if not self.NOWRITE and self.lvl>1:
            dic['model'] = P.save_model(self)
        dic['model_type'] = self.model.TYPE
        # save data
        dic['dataset'] = self.dataset.location 
        dic['testset'] = self.testset.location
        if not self.NOWRITE:
            dic['preprocessed'] = P.save_json(self, self.preprocessed, type='preprocessed_data')
            if self.train_predictions: dic['train_predictions'] = P.save_json(self, [P.parse_tensor_list(x) for x in self.train_predictions], type='train_predictions')
            if self.test_predictions: dic['test_predictions'] = P.save_json(self, [P.parse_tensor_list(x) for x in self.test_predictions], type='test_predictions')
        # save learnings
        dic['evaluation_mode'] = self.evaluation_mode
        dic['evaluation_results'] = self.evaluation_results
        dic['viz_mode'] = self.viz_mode
        if not self.NOWRITE and len(self.viz_data.keys()) != 0: 
            dic['viz_data'] = P.save_pickle(self, self.viz_data) # TODO shift second check into save_pickle()
        dic['viz'] = self.viz

        # save all paths in yaml!
        if not self.NOWRITE:
            return P.save_yaml(self, dic)
        else:
            return dic
    
    def load(self, eid:str):
        filename = str(eid) + '.yaml'
        path = LOC['experiments_dir'] + filename
        files_in_dir = next(walk(LOC['experiments_dir']), (None, None, []))[2]
        if filename in files_in_dir:
            with open(path, 'r') as file:
                exp_yaml = yaml.load(file, Loader=Loader)
        else:
            exp_yaml = None

        # init the experiment
        new = Experiment()
        # meta
        new.eid = filename
        new.date = (exp_yaml['date'])
        new.edited = exp_yaml['edited']
        new.lvl = exp_yaml['lvl']
        # model
        new.parameters = exp_yaml['parameters']
        if exp_yaml['lvl'] > 0: # == the experiment was trained already 
            new.model = P.model_factory(type=exp_yaml['model_type'], parameters=self.parameters, path=exp_yaml['model'])
        # data
        new.dataset = CoseDataset(mode='train')
        new.testset = CoseDataset(mode='test')
        new.preprocessed = P.load_json(exp_yaml['preprocessed'])
        if 'train_predictions' in exp_yaml.keys():
            train_predictions, train_attention = P.load_json(exp_yaml['train_predictions'])[0] # squeeze due to json
            new.train_predictions = ([torch.tensor(x) for x in train_predictions], [torch.tensor(x) for x in train_attention]) # reconstruct tensors
        if 'test_predictions' in exp_yaml.keys():
            test_predictions, test_attention = P.load_json(exp_yaml['test_predictions'])[0] # squeeze due to json
            new.test_predictions = ([torch.tensor(x) for x in test_predictions], [torch.tensor(x) for x in test_attention]) # reconstruct tensors
        # learnings
        new.evaluation_mode = exp_yaml['evaluation_mode']
        new.evaluation_results = exp_yaml['evaluation_results']
        new.viz_mode = exp_yaml['viz_mode']
        if 'viz_data' in exp_yaml:
            new.viz_data = P.load_pickle(exp_yaml['viz_data'])
        if 'viz' in exp_yaml: # TODO rename to viz_dir
            new.viz = exp_yaml['viz']
        return new
    
    # Trains the algorithms of the experiment
    # actually a wrapper for the training module src/train.py
    # also saves training predictions and data for training visualization
    def train(self, output_softmax=False): 
        t_out = T.train(self.parameters, self.dataset, self.model)
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
                pred, attn = self.test_predictions = T.predict(self.parameters, self.model, self.testset)
            
            gold = dataset.labels
            doc_ids = dataset.docids

            if 'explainability' in self.evaluation_mode:
                attn_detached = [x.detach().numpy() for x in attn] # TODO model could return detached attn and predictions?
                # retrain for comp and suff:
                comp_data = dataset.erase(attn, mode='comprehensiveness')
                suff_data = dataset.erase(attn, mode='sufficiency')
                comp_predictions, _ = T.predict(self.parameters, self.model, comp_data) # _ is attn vector
                suff_predictions, _ = T.predict(self.parameters, self.model, suff_data) # _ is attn vector
                aopc_predictions = T.predict_aopc_thresholded(self.parameters, self.model, attn, dataset)
                er_results = E.create_results(doc_ids, pred, comp_predictions, suff_predictions, attn_detached, aopc_thresholded_scores=aopc_predictions)
                result[mode]['agreement_auprc'] = E.soft_scores(er_results, docids=doc_ids, ds=f'cose_{mode}') # TODO avg_precision and roc_auc_score NaN, but only for testset!
                result[mode]['classification_scores'] = E.classification_scores(results=er_results, mode=mode, aopc_thresholds=self.parameters['aopc_thresholds'])

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
            
            self.viz = newpath
            self.lvl = 3
        else:
            print('NOWRITE param set: not even pictures will be exported!')