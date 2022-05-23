import os
import yaml
from yaml import CLoader as Loader
import json
import torch
import torch.nn as nn
from os import walk
from datetime import datetime
from data_access.locations import LOC
from train import train, predict
from data_access.csqa_dataset import CsqaDataset
from data_access.cose_dataset import CoseDataset
from models.random_clf import RandomClassifier
import eval
import pickle
import evaluation.visualizations.viz as viz
import evaluation.eraserbenchmark.rationale_benchmark.metrics as EM
import evaluation.eraserbenchmark.rationale_benchmark.utils as EU

# An Experiment is stored as a yaml, linking all essential paths:
#   parameters, model_path, preproc_path, viz_path
# TODO
#   - implement synonym searching/loading
#   - state id of previous saves, if new id is made...
#   - NO_SAVE option - never write to disk, but only return objects
class Experiment():

    def __init__(self, 
            eid=None, 
            level=0,
            date=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
            edited=None,
            NOWRITE=False,
            parameters=None, 
            model=None, 
            dataset='csqa_train', 
            testset='csqa_test',
            rationales='cose_train',
            test_rationales='cose_test',
            preprocessed=None, 
            train_predictions=None, 
            test_predictions=None, 
            evaluation_mode=['accuracy', 'precision', 'recall'],
            evaluation_results = None,
            viz_mode=['loss'],
            viz_data={},
            viz=None):

        assert model != None
        assert parameters != None
        assert dataset == 'csqa_train' # TODO reminder to change loading!
        assert testset == 'csqa_test'
        assert rationales == 'cose_train'
        assert test_rationales == 'cose_test'

        # meta
        if eid != None: self.eid = eid
        else: self.eid = self.hash_id(str(model) + str(parameters))
        self.level = level
        self.date = date
        self.edited = None
        self.NOWRITE = NOWRITE
        # model relevant
        self.parameters = parameters 
        self.model = model # model type is in dic, but not here!
        # data
        self.dataset = CsqaDataset(LOC[dataset], limit=self.parameters['limit']),
        self.testset = CsqaDataset(LOC[testset], limit=-1),
        # TODO UNNECESSARY BULLSHIT - WHY DOES THE CONSTRUCTOR RETURN A 1 ELEMENT TUPLE
        self.dataset = self.dataset[0]
        self.testset = self.testset[0]
        # END UNNECESSARY BULLSHIT

        #self.rationales = CoseDataset(ids=self.dataset.get_ids(), path_to_raw=LOC[rationales], path_to_docs=LOC['cose_docs'], return_original_form=True),
        #self.test_rationales = CoseDataset(ids=self.testset.get_ids(), path_to_raw=LOC[test_rationales], path_to_docs=LOC['cose_docs'], return_original_form=True),
        self.rationales = EU.annotations_from_jsonl(LOC['cose_train'])
        self.test_rationales = EU.annotations_from_jsonl(LOC['cose_test'])

        self.preprocessed = preprocessed
        self.train_predictions = train_predictions
        self.test_predictions = test_predictions
        # learnings
        self.evaluation_mode = evaluation_mode
        self.evaluation_results = evaluation_results
        self.viz_mode = viz_mode
        self.viz_data = viz_data
        self.viz = viz
    
    def hash_id(self, as_str=True):
        dirs = next(os.walk(LOC['experiments_dir']))[1]
        hash_id = abs(hash(self))
        if hash_id in dirs: return False
        else: 
            if as_str: return str(hash_id)
            else: return hash_id
        
    # TODO check & save level (progress in the experiment)
    def calc_level(self):
        return '?'

    def save(self):
        dic = {}
        # meta
        dic['level'] = self.calc_level()
        dic['date'] = self.date
        dic['edited'] = datetime.now().strftime("%d/%m/%Y, %H:%M:%S") 
        # model
        dic['parameters'] = self.parameters
        if not self.NOWRITE:
            dic['model'] = self.save_model()
        dic['model_type'] = self.model.TYPE
        # save data
        dic['dataset'] = self.dataset.location 
        dic['testset'] = self.testset.location
        dic['rationales'] = 'cose_train' # TODO de-hardcode
        dic['test_rationales'] = 'cose_test' # TODO de-hardcode
        if not self.NOWRITE:
            dic['preprocessed'] = self.save_json(self.preprocessed, type='preprocessed_data')
            dic['train_predictions'] = self.save_json(self.train_predictions, type='train_predictions')
            dic['test_predictions'] = self.save_json(self.test_predictions, type='test_predictions')
        # save learnings
        dic['evaluation_mode'] = self.evaluation_mode
        dic['evaluation_results'] = self.evaluation_results
        dic['viz_mode'] = self.viz_mode
        if not self.NOWRITE and len(self.viz_data.keys()) != 0: 
            dic['viz_data'] = self.save_pickle(self.viz_data) # TODO shift second check into save_pickle()
        dic['viz'] = self.viz

        # save all paths in yaml!
        if not self.NOWRITE:
            return self.save_yaml(dic)
        else:
            return dic
    
    def save_model(self):
        assert self.model != None
        model_save_loc = LOC['models_dir'] + str(self.eid) + '.pth'

        torch.save(self.model.state_dict(), model_save_loc)
        return model_save_loc

    # only for predictions
    def save_json(self, obj, type=None): # type = 'preprocessed_data' | 'train_predictions' | 'predictions'
        if obj == None: return None
        if type == 'train_predictions':
            save_loc = LOC['predictions'] + str(self.eid) + '_train.jsonl'
        elif type == 'test_predictions':
            save_loc = LOC['predictions'] + str(self.eid) + '_test.jsonl'
        else:
            print('json type "' + type + '" unknown')
            assert False

        with open(save_loc, 'w') as outfile:
            json.dump(obj, outfile)
        return save_loc
    
    # only for experiment yamls!
    def save_yaml(self, dic):
        assert self.eid != None
        filename = LOC['experiments_dir'] + str(self.eid) + '.yaml'

        with open(filename, 'w') as file:
            yaml_file = yaml.dump(dic, file)
        return yaml_file
    
    def save_pickle(self, obj, save_loc=LOC['viz_data_dir']):
        filename = str(self.eid) + '.pickle'

        with open(save_loc + filename, 'wb') as f:
            pickle.dump(obj, f)
        return save_loc + filename

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
        new.level = self.check(exp_yaml['level'])
        new.date = self.check(exp_yaml['date'])
        new.edited = self.check(exp_yaml['edited'])
        # model
        new.parameters = self.check(exp_yaml['parameters'])
        new.model = self.check(self.load_model(exp_yaml['model'], exp_yaml['model_type']))
        # data
        new.dataset = self.check(CsqaDataset(exp_yaml['dataset']))
        new.testset = self.check(CsqaDataset(exp_yaml['testset']))
        new.rationales = self.check(CoseDataset(exp_yaml['rationales']))
        new.test_rationales = self.check(CoseDataset(exp_yaml['test_rationales']))
        new.preprocessed = self.check(self.load_json(exp_yaml['preprocessed']))
        new.train_predictions = self.check(self.load_json(exp_yaml['train_predictions']))
        new.test_predictions = self.check(self.load_json(exp_yaml['test_predictions']))
        # learnings
        new.evaluation_mode = self.check(exp_yaml['evaluation_mode'])
        new.evaluation_results = self.check(exp_yaml['evaluation_results'])
        new.viz_data = self.check(self.load_pickle(exp_yaml['viz_data']))
        new.viz_mode = self.check(exp_yaml['viz_mode'])
        new.viz = self.check(exp_yaml['viz'])

        return new
    
    def check(self,obj):
        if obj == None: return None
        else: return obj
    
    def load_model(self,path:str, type=str):
        if type == 'RandomClassifier':
            model = RandomClassifier(69)
        else:
            assert False
        model.load_state_dict(torch.load(path))
        return model
    
    def load_json(self,path:str):
        if path == None: return None
        self.data = []
        with open(path, 'r') as json_file:
            json_list = list(json_file)
            for json_str in json_list:
                result = json.loads(json_str)
                self.data.append(result)
        return self.data
    
    def load_pickle(self, path:str):
        if path == None: return None
        with open(path, 'rb') as f:
            result = pickle.load(f)
        return result


    def train(self, output_softmax=False): # TODO shift output_softmax parameter to train.train()!
        t_out = train(self.parameters, self.dataset, self.model)
        self.model = t_out['model']
        self.viz_data['train_loss'] = [float(round(x,4)) for x in t_out['losses']]

        if output_softmax:
            self.train_predictions = t_out['outputs']
        else:
            self.train_predictions = [int(torch.argmax(x).item()) for x in t_out['outputs']]
        return t_out
    
    def evaluate(self):
        result = {'train':{}, 'test':{}}
        for mode in result.keys():
            if mode == 'train':
                gold = [int(x) for x in self.dataset.get_labels(limit=self.parameters['limit'])]
                pred = self.train_predictions
            elif mode == 'test':
                gold = [int(x) for x in self.testset.get_labels(limit=-1)]
                pred = predict(self.parameters, self.model, self.testset)
                self.test_predictions = pred
            
            # do the evaluation
            if 'explainability' in self.evaluation_mode:
                result[mode]['partial_match_score'] = eval.rationale_match(self.rationales, self.rationales)

            if 'efficiency' in self.evaluation_mode:
                pass

            if 'competence' in self.evaluation_mode:
                result[mode]['accuracy'], result[mode]['precision'], result[mode]['recall'] = eval.competence(gold, pred)

        self.evaluation_results = result
    
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
        else:
            pass # TODO save this fig in Experiment?