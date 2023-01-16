import torch
import os
from tqdm import tqdm
from abc import ABC, abstractmethod
from copy import deepcopy

from argparse import Namespace

from models.bert import BertPipeline
from models.random import RandomClassifier
from models.gcn import GCN
from models.gat import GATForMultipleChoice
from models.bert_gat import BERT_GAT
from models.qagnn.modeling_qagnn import LM_QAGNN
from preproc.ud_preproc import UDParser
from preproc.fourlang_preproc import FourLangParser


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
        self.params = params
        torch.manual_seed(self.params['rnd_seed'])
        self.complete_set, self.train_set, self.val_set, self.test_set = self.init_data()
        self.model = self.model_factory(self.params['model_type'])


    def __call__(self):

        # TRAINING
        if not('skip_training' in self.params and self.params['skip_training']): # no skip
            print('training...')
            self.train_output = self.train()
        else: # skip training
            if not 'load_from' in self.params or self.params['load_from'] == None:
                print('WARNING: training will be skipped, but no checkpoint was given (load_from) parameter (=prediction with only pre-trained model)')
            else:
                print(f"MODEL PRELOADED FROM {self.params['load_from']} - SKIPPING TRAINING!") # this already happened in experiment.model_factory()

        # PREDICTION
        # EVALUATION
        if not('skip_evaluation' in self.params and self.params['skip_evaluation']): # no skip
            print('predicting...')
            # need to softmax logits for evaluation (actually only ERASER)
            prediction_params = deepcopy(self.params)
            prediction_params['softmax_logits'] = True
            preds = []
            with torch.no_grad():
                for sample in tqdm(self.val_set):
                    preds.append(self.model(sample, **prediction_params))  
                self.val_pred = list(zip(*preds))
            # evaluating
            print('evaluating...')
            self.eval_output = self.evaluate()
        else: # skip evaluation
            print('SKIPPING EVALUATION (flag was set in param dict!)')
            self.val_pred = None
            self.eval_output = None
        
        # VIZ
        print('visualizing...')
        self.viz_output = self.viz()

        # SAVE / PERSIST
        print('saving...')
        self.save()

        print('experiment done!')
        return self
        
    def evaluate(self,split='val'):
        if 'skip_evaluation' in self.params and self.params['skip_evaluation']: 
            return None
        return {
            'competence':self.eval_competence(),
            'explainability':self.eval_explainability(),
            'efficiency':self.eval_efficiency()
        }
    
    def model_factory(self, type:str):
        ''' This method allows to create model classes from strings'''
        # print
        if 'load_from' in self.params: print(f"LOADING MODEL FROM {self.params['load_from']}")

        # select model
        if type == 'Random':
            model = RandomClassifier(self.params)

        elif type == "BERT":
            model = BertPipeline(self.params)

        elif type == 'UD_GCN':
            model = GCN(self.params)
            if 'load_from' in self.params:
                if os.path.exists(f"{self.params['load_from']}/model.pt"): 
                    model.load_state_dict(torch.load(f"{self.params['load_from']}/model.pt"))
                else:
                    print(f"load_from location {self.params['load_from']} either not found or empty!")

        elif type == 'UD_GAT':
            model = GATForMultipleChoice(self.params)
            if 'load_from' in self.params:
                if os.path.exists(f"{self.params['load_from']}/model.pt"): 
                    model.load_state_dict(torch.load(f"{self.params['load_from']}/model.pt"))
                else:
                    print(f"load_from location {self.params['load_from']} either not found or empty!")

        elif type == 'BERT_GAT':
            model = BERT_GAT(self.params)
            if 'load_from' in self.params:
                if os.path.exists(f"{self.params['load_from']}/model.pt"): 
                    model.load_state_dict(torch.load(f"{self.params['load_from']}/model.pt"))
                else:
                    print(f"load_from location {self.params['load_from']} either not found or empty!")

        elif type == "qagnn":
            args = Namespace(**self.params)
            num_concepts = self.params['num_concepts'] if 'num_concepts' in self.params else None
            # if 'offset_concepts' in self.params and self.params['offset_concepts']: num_concepts += 2 # TODO why doesn't this work?
            assert num_concepts != None
            model = LM_QAGNN( # TODO have these as defaults in params for args, so that don't have to hardcode params here!
                args=args,
                model_name='bert-large-uncased',
                k=args.k,
                n_ntype=args.num_node_types,
                n_etype=args.num_relation,
                n_concept=num_concepts,
                concept_dim=args.gat_hidden_dim, # TODO this the correct args for the param?
                concept_in_dim=args.concept_dim,
                n_attention_head=args.num_heads,
                fc_dim=args.clf_layer_dim,
                n_fc_layer=args.clf_layer_depth,
                p_emb=args.dropout,
                p_gnn=args.dropout,
                p_fc=args.dropout
            )
            if 'load_from' in self.params:
                model_path = self.params['load_from']
                model_state_dict, old_args = torch.load(model_path, map_location=torch.device('cpu'))
                model = LM_QAGNN(
                    old_args, 
                    old_args.encoder, 
                    k=old_args.k, 
                    n_ntype=4, 
                    n_etype=old_args.num_relation,
                    n_concept=concept_num,
                    concept_dim=old_args.gnn_dim,
                    concept_in_dim=concept_dim,
                    n_attention_head=old_args.att_head_num, 
                    fc_dim=old_args.fc_dim, 
                    n_fc_layer=old_args.fc_layer_num,
                    p_emb=old_args.dropouti, 
                    p_gnn=old_args.dropoutg, 
                    p_fc=old_args.dropoutf,
                    pretrained_concept_emb=cp_emb,
                    freeze_ent_emb=old_args.freeze_ent_emb,
                    init_range=old_args.init_range,
                    encoder_config={}
                )
                model.load_state_dict(model_state_dict)

        else:
            raise AttributeError('model_type: "' + type + '" is unknown!')
        return model
    
    def _graph_parser_factory(self):
        graph_form = self.params['graph_form'] if 'graph_form' in self.params else None
        if graph_form == 'ud':
            return UDParser(self.params)
        elif graph_form == '4lang':
            return FourLangParser(self.params)
        else:
            raise AttributeError(f"No graph formalism '{graph_form}' available! use 'ud' or '4lang'")

    @abstractmethod
    def init_data(self):
        ''' returns (complete_dataset, train_data, val_data, test_data) '''
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        '''trains self.model on self.train_set'''
        raise NotImplementedError()
    
    @abstractmethod
    def eval_competence(self):
        '''evaluates the competence of the experiments model; returns {accuracy, precicison, recall}'''
        raise NotImplementedError()

    @abstractmethod
    def eval_explainability(self):
        '''evaluates the quantifiable explainability of the model with the aid of the ERASER module; 
        returns a large number of metrics around comprehensiveness and sufficiency'''
        raise NotImplementedError()

    @abstractmethod
    def eval_efficiency(self):
        '''evaluates the efficiency of the experiments modele; returns {flops, num_self.params}'''
        raise NotImplementedError()

    @abstractmethod
    def viz(self):
        '''create visualizations of relevant aspects of the experiment'''
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        '''save relevant data e.g. evaluations, predictions, etc'''
        raise NotImplementedError()