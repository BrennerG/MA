import math
from datasets import load_dataset
from transformers import TrainingArguments

from data.huggingface_cose import EraserCosE 
from experiments.experiment import Experiment
import evaluation.eval_util as E
from data.locations import LOC

import yaml
from tqdm import tqdm

import wandb
import os

# TODO huggingface_cose.py has changed! (added missing '?' to questions!)
# re-run c.dark and verify nothing has changed...!

class BERTExperiment(Experiment):

    def __init__(self, params):
        super().__init__(params)
        self.avg_rational_lengths = EraserCosE.avg_rational_length(self.complete_set)

    def init_data(self, params:{}):
        cose = load_dataset(LOC['cose_huggingface'])
        return cose, cose['train'], cose['validation'], cose['test']

    def train(self, params:{}):
       
        wandb.init(project="BERT_baseline")

        result = self.model.train(
            self.complete_set, 
            train_args = TrainingArguments(
                output_dir= params['save_loc'],
                evaluation_strategy="epoch",
                learning_rate= params['learning_rate'],
                per_device_train_batch_size= params['batch_size'],
                per_device_eval_batch_size= params['batch_size'],
                num_train_epochs= params['epochs'],
                weight_decay=0.01,
                save_strategy= params['save_strategy'],
                overwrite_output_dir= params['overwrite_output_dir'],
                no_cuda=(not params['use_cuda']),
                report_to='wandb',
                run_name='BERT_insert_name'
            )
        )

        wandb.finish()
        return result
    
    def eval_competence(self, params:{}):
        probas, attn = self.val_pred
        results = {}
        results['accuracy'], results['precision'], results['recall'] = E.competence_metrics(self.val_set['label'], probas)
        return results

    def eval_explainability(self, params:{}):
        split = 'validation'
        pred, attn = self.val_pred
        comp_ds = EraserCosE.erase(attn, mode='comprehensiveness', split=split)
        suff_ds = EraserCosE.erase(attn, mode='sufficiency', split=split)
        comp_pred, _ = zip(*self.model(comp_ds, attention=None, softmax_logits=True))
        suff_pred, _ = zip(*self.model(suff_ds, attention=None, softmax_logits=True))
        # calcualte aopc metrics
        aopc_intermediate = {}
        for aopc in tqdm(params['aopc_thresholds'], desc='explainability_eval: '):
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
        aopc_predictions = E.reshape_aopc_intermediates(aopc_intermediate, params)

        doc_ids = self.val_set['id']
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=aopc_predictions)
        self.er_results = er_results
        return E.classification_scores(results=er_results, mode='val', aopc_thresholds=params['aopc_thresholds'], with_ids=doc_ids)
        # E.soft_scores(results=er_results, docids=doc_ids)

    def eval_efficiency(self, params:{}):
        result = {}
        result['flops'], result['params'] = self.model.efficiency_metrics(params)
        return result

    def viz(self, params:{}):
        return None

    def save(self, params:{}):
        # check for save location
        if not os.path.exists(params['save_loc']):
            os.makedirs(params['save_loc'])

        # saving evaluation
        with open(params['save_loc']+'evaluation.yaml', 'w') as file:
            documents = yaml.dump(self.eval_output, file)

        # saving attention weights
        if 'save_predictions' in params and params['save_predictions']:
            prediction_data = {
                xid:{
                    'probas': self.val_pred[0][i].squeeze().tolist(),
                    'attn': self.val_pred[1][i].squeeze().tolist()
                }
                    for i,xid in enumerate(self.val_set['id'])}
            with open(params['save_loc']+'predictions_attentions.yaml', 'w') as file:
                documents = yaml.dump(prediction_data, file)
        
        # saving eraser input
        if 'save_eraser_input' in params and params['save_eraser_input']:
            for x in self.er_results: x.pop('rationales')
            with open(params['save_loc']+'eraser_input.yaml', 'w') as file:
                documents = yaml.dump(self.er_results, file)

        return True