from abc import ABC, abstractmethod

from models.bert import BertPipeline
from models.random import RandomClassifier

from tqdm import tqdm

'''
This class represents a single pass through the pipeline of a single model
and provides means to reproducing such passes/runs by saving important data.

Each experiment consists of the following phases:
    0. Initialization
    1. Training / Loading
    2. Predicting the validation set
    3. Evaluating the modee
    4. Visualizing aspects of the experiment

As mentioned each Experiment subclass correlates to a kind of model:
    - RandomExperiment and RandomClassifier
    - BERTExperiment and BERTClassifier
    - ...
Since this class is abstract most of the actual logic is implemented in the respective sublassess.

'''
class Experiment(ABC):

    def __init__(self, params:{}):
        self.complete_set, self.train_set, self.val_set, self.test_set = self.init_data(params)
        self.model = self.model_factory(params['model_type'], params)

    def __call__(self, params:{}):
        print('training...')
        self.train_output = self.train(params)
        print('predicting...')

        # TODO is there a better way to do this? mb dont use a pipeline???
        # method 3 :puny workaround
        preds = [self.model(sample, **params) for sample in tqdm(self.val_set, desc='predicting:')]
        # TODO find and remove the second tqdm bar during prediction :S
        logits, attentions = zip(*preds)
        self.val_pred = (list(logits), [a[0] for a in attentions])
    
        print('evaluating...')
        self.eval_output = self.evaluate(params)
        print('visualizing...')
        self.viz_output = self.viz(params)
        print('saving...')
        self.save(params)
        print('experiment done!')
        return self
        
    def evaluate(self, params:{}, split='val'):
        if 'skip_evaluation' in params and params['skip_evaluation']: 
            print('SKIPPING EVALUATION (flag was set in param dict!)')
            return None
        return {
            'competence':self.eval_competence(params), 
            'explainability':self.eval_explainability(params), 
            'efficiency':self.eval_efficiency(params)
        }
    
    def model_factory(self, type:str, params:{}):
        ''' This method allows to create model classes from strings'''
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
        '''evaluates the competence of the experiments model; returns {accuracy, precicison, recall}'''
        raise NotImplementedError()

    @abstractmethod
    def eval_explainability(self, params:{}):
        '''evaluates the quantifiable explainability of the model with the aid of the ERASER module; 
        returns a large number of metrics around comprehensiveness and sufficiency'''
        raise NotImplementedError()

    @abstractmethod
    def eval_efficiency(self, params:{}):
        '''evaluates the efficiency of the experiments modele; returns {flops, num_params}'''
        raise NotImplementedError()

    @abstractmethod
    def viz(self, params:{}):
        '''create visualizations of relevant aspects of the experiment'''
        raise NotImplementedError()


    def save(self, params:{}):
        '''save relevant data e.g. evaluations, predictions, etc'''
        raise NotImplementedError()