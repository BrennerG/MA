from abc import ABC, abstractmethod

from models.bert import BertPipeline
from models.random import RandomClassifier
from models.gcn import GCN

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

        # TRAINING
        if not('skip_training' in params and params['skip_training']): # no skip
            print('training...')
            self.train_output = self.train(params)
        else: # skip training
            if not 'load_from' in params or params['load_from'] != None:
                print('WARNING: training will be skipped, but no checkpoint was given (load_from) parameter (=prediction with only pre-trained model)')
            else:
                print(f"MODEL PRELOADED FROM {params['load_from']} - SKIPPING TRAINING!") # this already happened in experiment.model_factory()

        # PREDICTION
        # EVALUATION
        if not('skip_evaluation' in params and params['skip_evaluation']): # no skip
            print('predicting...')
            preds = []
            for sample in tqdm(self.val_set):
                # preds.append(self.model(sample, **params)) # TODO changed this for GCN - still holding for Random and BERT? wich params MUST be passed here? (everything has access to params dict!!!)
                preds.append(self.model(sample))
            logits, attentions = zip(*preds) 
            
            # some models don't have attention
            if any(attentions): 
                self.val_pred = (list(logits), [a[0] for a in attentions])
            else:
                self.val_pred = (list(logits), None)

            print('evaluating...')
            self.eval_output = self.evaluate(params)
        else: # skip evaluation
            print('SKIPPING EVALUATION (flag was set in param dict!)')
            self.val_pred = None
            self.eval_output = None
        
        # VIZ
        print('visualizing...')
        self.viz_output = self.viz(params)

        # SAVE / PERSIST
        print('saving...')
        self.save(params)

        print('experiment done!')
        return self
        
    def evaluate(self, params:{}, split='val'):
        if 'skip_evaluation' in params and params['skip_evaluation']: 
            return None
        return {
            'competence':self.eval_competence(params), 
            'explainability':self.eval_explainability(params), 
            'efficiency':self.eval_efficiency(params)
        }
    
    def model_factory(self, type:str, params:{}):
        ''' This method allows to create model classes from strings'''
        if type == 'Random':
            model = RandomClassifier(params['rnd_seed']) # TODO input whole params dict!
        elif type == "BERT":
            if 'load_from' in params: print(f"LOADING MODEL FROM {params['load_from']}")
            model = BertPipeline(params=params)
        elif type == 'UD_GCN':
            if 'load_from' in params: raise NotImplementedError('Loading for UD_GCN not implemented yet!') # TODO
            model = GCN(params)
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

    @abstractmethod
    def save(self, params:{}):
        '''save relevant data e.g. evaluations, predictions, etc'''
        raise NotImplementedError()