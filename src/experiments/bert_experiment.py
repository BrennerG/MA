import math
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import EvalPrediction

from data.huggingface_cose import EraserCosE 
from experiments.experiment import Experiment
import evaluation.eval_util as E
from data.locations import LOC

import yaml
from tqdm import tqdm

import wandb
import os

# TODO test this online (re-run c.dark and verify nothing has changed...!)

class BERTExperiment(Experiment):

    def __init__(self, params):
        super().__init__(params)
        self.avg_rational_lengths = EraserCosE.avg_rational_length(self.complete_set)

    def init_data(self):
        cose = load_dataset(LOC['cose_huggingface'])
        return cose, cose['train'], cose['validation'], cose['test']

    def train(self):
       
        wandb.init(project="BERT_baseline", config=self.params, mode='online' if self.params['wandb_logging']==True else 'disabled')

        result = self.model.train(
            self.complete_set, 
            train_args = TrainingArguments(
                output_dir= self.params['save_loc'],
                evaluation_strategy="epoch",
                learning_rate= self.params['learning_rate'],
                per_device_train_batch_size= self.params['batch_size'],
                per_device_eval_batch_size= self.params['batch_size'],
                num_train_epochs= self.params['epochs'],
                weight_decay=self.params['weight_decay'],
                save_strategy= self.params['save_strategy'],
                overwrite_output_dir= self.params['overwrite_output_dir'],
                no_cuda=(not self.params['use_cuda']),
                report_to='wandb',
                run_name='BERT_insert_name'
            ),
            eval_func=self.huggingface_eval
        )

        wandb.finish()
        return result
    
    def eval_competence(self, probas=None, attn=None):
        if probas==None or attn==None:
            probas, attn = self.val_pred
        results = {}
        results['accuracy'], results['precision'], results['recall'] = E.competence_metrics(self.val_set['label'], probas)
        return results

    def eval_explainability(self, pred=None, attn=None, skip_aopc=False):
        split = 'validation'
        if pred==None or attn==None:
            pred, attn = self.val_pred
        comp_ds = EraserCosE.erase(attn, mode='comprehensiveness', split=split)
        suff_ds = EraserCosE.erase(attn, mode='sufficiency', split=split)
        comp_pred, _ = zip(*self.model(comp_ds, attention=None, softmax_logits=True))
        suff_pred, _ = zip(*self.model(suff_ds, attention=None, softmax_logits=True))
        # calcualte aopc metrics

        aopc_predictions = None
        if not skip_aopc: 
            aopc_intermediate = {}
            for aopc in tqdm(self.params['aopc_thresholds'], desc='explainability_eval: '):
                tokens_to_be_erased = math.ceil(aopc * self.avg_rational_lengths[split])
                # comp
                cds = EraserCosE.erase(attn, mode='comprehensiveness', split=split, k=tokens_to_be_erased)
                cp, _ = zip(*self.model(cds, attention=None, softmax_logits=True))
                cl = E.from_softmax(cp, to='dict') # labels
                # suff
                sds = EraserCosE.erase(attn, mode='sufficiency', split=split, k=tokens_to_be_erased)
                sp, _ = zip(*self.model(sds, attention=None, softmax_logits=True))
                sl = E.from_softmax(sp, to='dict')
                # aggregate
                aopc_intermediate[aopc] = [aopc, cl, sl]
            aopc_predictions = E.reshape_aopc_intermediates(aopc_intermediate, self.params)

        doc_ids = self.val_set['id']
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=aopc_predictions)
        self.er_results = er_results
        return E.classification_scores(results=er_results, mode='val', aopc_thresholds=self.params['aopc_thresholds'], with_ids=doc_ids)
        # E.soft_scores(results=er_results, docids=doc_ids)

    def eval_efficiency(self):
        result = {}
        result['flops'], result['params'] = self.model.efficiency_metrics(self.params)
        return result

    def viz(self):
        return None

    def save(self):
        # check for save location
        if not os.path.exists(self.params['save_loc']):
            os.makedirs(self.params['save_loc'])

        # saving evaluation
        with open(self.params['save_loc']+'evaluation.yaml', 'w') as file:
            documents = yaml.dump(self.eval_output, file)

        # saving attention weights
        if 'save_predictions' in self.params and self.params['save_predictions']:
            prediction_data = {
                xid:{
                    'probas': self.val_pred[0][i].squeeze().tolist(),
                    'attn': self.val_pred[1][i].squeeze().tolist()
                }
                    for i,xid in enumerate(self.val_set['id'])}
            with open(self.params['save_loc']+'predictions_attentions.yaml', 'w') as file:
                documents = yaml.dump(prediction_data, file)
        
        # saving eraser input
        if 'save_eraser_input' in self.params and self.params['save_eraser_input']:
            for x in self.er_results: x.pop('rationales')
            with open(self.params['save_loc']+'eraser_input.yaml', 'w') as file:
                documents = yaml.dump(self.er_results, file)

        return True
    
    def huggingface_eval(self, eval_prediction:EvalPrediction, *args):
        # need to softmax logits for evaluation (actually only ERASER)
        prediction_params = deepcopy(self.params)
        prediction_params['softmax_logits'] = True
        # make predictions
        preds = []
        for sample in tqdm(self.val_set):
            preds.append(self.model(sample, **prediction_params))
        p,a = list(zip(*preds))
        comp = self.eval_competence(p,a)
        expl = self.eval_explainability(p,a, skip_aopc=True)
        return {
            'accuracy': comp['accuracy'],
            'comprehensiveness': expl['comprehensiveness'],
            'sufficiency': expl['sufficiency']
        }