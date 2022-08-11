from abc import ABC, abstractmethod

from models.bert import BertPipeline
from models.random import RandomClassifier

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

        # TODO temporary workaround:
        # huggingface Pipeline parent class (base.py) doesnt recognize self.val_set as Dataset
        # therefore no batching is enabled, which overloads GPU's VRAM
        # SOLUTION: Pipeline accepts: Datasets, lists and generators: so lets give it a generator
        # fails at padding... because first samples shape is considered for padding
        # OPTION 1: give it batches, so the tokenizer properly works
        # fails coz padding method then thinks each batch is a sample... c:
        # OPTION 2: order the val set for this...
        # TODO should the method stay in experiment parent or go into children?
        #def generator_from_dataset(dataset, in_batches=False, order_first=True):
        #    if order_first:
        #        sample_list = list(dataset)
        #    if in_batches:
        #        bs = params['batch_size']
        #        num_batches = int(len(dataset) / bs)
        #        for i in range(num_batches):
        #            yield [dataset[i*bs+x] for x in range(bs)]
        #    else:
        #        for sample in dataset:
        #            yield sample
        #pipeline_iterator = self.model(generator_from_dataset(self.val_set), **params)
        #self.val_pred = list(pipeline_iterator)

        # method 2 - probably not valid
        # this throws an error during postprocessing
        # also the model output from _forward cannot be correct: a batch returns logits.shape (1,5)
        #   -> it should be (batch_size, 5)...
        #from torch.utils.data import DataLoader
        #data_loader = DataLoader(self.val_set, batch_size=1)
        #val_set_iterator = data_loader._get_iterator()
        #self.val_pred = self.model(list(val_set_iterator), **params)

        # vanilla method - overloads RAM
        #self.val_pred = self.model(self.val_set, **params)

        # method 3 :puny workaround
        preds = [self.model(sample, **params) for sample in self.val_set]
        logits, attentions = zip(*preds)
        self.val_pred = (list(logits), list(attentions))
    
        print('evaluating...')
        self.eval_output = self.evaluate(params)
        print('visualizing...')
        self.viz_output = self.viz(params)
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

