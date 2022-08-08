from abc import ABC, abstractmethod

from models.bert import BertPipeline
from models.random import RandomClassifier


class Experiment(ABC):

    def __init__(self, params:{}):
        self.complete_set, self.train_set, self.val_set, self.test_set = self.init_data(params)
        self.model = self.model_factory(params['model_type'], params)

    def __call__(self, params:{}):
        print('training...')
        self.train_output = self.train(params)
        print('predicting')
        self.val_pred = self.model(self.val_set)
        print('evaluating')
        self.eval_output = self.evaluate(params)
        print('visualizing')
        self.viz_output = self.viz(params)
        print('experiment done!')
        return self
        
    def evaluate(self, params:{}, split='val'):
        return {
            'competence':self.eval_competence(params), 
            'explainability':self.eval_explainability(params), 
            'efficiency':self.eval_efficiency(params)
        }
    
    def model_factory(self, type:str, params:{}):
        if type == 'Random':
            model = RandomClassifier(params['rnd_seed'])
        elif type == "BERT":
            if 'load_from' in params: print(f"LOADING MODEL FROM {params['load_from']}")
            model = BertPipeline(params=params)
        else:
            raise AttributeError('model_type: "' + type + '" is unknown!')
        return model

    @abstractmethod
    def init_data(self, params:{}):
        ''' returns (complete_dataset, train_data, val_data, test_data) '''
        raise NotImplementedError()

    @abstractmethod
    def train(self, params:{}):
        '''trains self.model on self.train_set'''
        raise NotImplementedError()
    
    @abstractmethod
    def eval_competence(self, params:{}):
        raise NotImplementedError()

    @abstractmethod
    def eval_explainability(self, params:{}):
        raise NotImplementedError()

    @abstractmethod
    def eval_efficiency(self, params:{}):
        raise NotImplementedError()

    @abstractmethod
    def viz(self, params:{}):
        raise NotImplementedError()

