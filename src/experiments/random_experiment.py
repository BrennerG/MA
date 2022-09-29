from experiments.experiment import Experiment
from datasets import load_dataset
from data.huggingface_cose import EraserCosE
from data.locations import LOC
import evaluation.eval_util as E
from data.locations import LOC
import math
import os
import yaml


class RandomClassifierExperiment(Experiment):    
    
    def __init__(self, params):
        super().__init__(params)
        self.avg_rational_lengths = EraserCosE.avg_rational_length(self.complete_set)

    def init_data(self):
        ''' returns (complete_dataset, train_data, val_data, test_data) '''
        cose = load_dataset(LOC['cose_huggingface'])
        if 'debug' in self.params and self.params['debug']:
            return cose, cose['debug_train'], cose['debug_val'], cose['test']
        else:
            return cose, cose['train'], cose['validation'], cose['test']

    def train(self):
        print('NO TRAINING NEEDED FOR RANDOM CLASSIFIER')
        return None
    
    def eval_competence(self):
        probas, attn = self.val_pred
        results = {}
        results['accuracy'], results['precision'], results['recall'] = E.competence_metrics(self.val_set['label'], probas)
        return results


    def eval_explainability(self):
        split = 'validation'
        pred, attn = self.val_pred
        comp_ds = self.reshape_erased_output(EraserCosE.erase(attn, mode='comprehensiveness', split=split))
        suff_ds = self.reshape_erased_output(EraserCosE.erase(attn, mode='sufficiency', split=split))
        comp_pred, _ = self.model(comp_ds, **self.params)
        suff_pred, _ = self.model(suff_ds, **self.params)
        # calcualte aopc metrics
        aopc_intermediate = {}
        for aopc in self.params['aopc_thresholds']:
            tokens_to_be_erased = math.ceil(aopc * self.avg_rational_lengths[split])
            # comp
            cds = self.reshape_erased_output(EraserCosE.erase(attn, mode='comprehensiveness', split=split, k=tokens_to_be_erased))
            cp, _ = self.model(cds, **self.params)
            cl = E.from_softmax(cp, to='dict') # labels
            # suff
            sds = self.reshape_erased_output(EraserCosE.erase(attn, mode='sufficiency', split=split, k=tokens_to_be_erased))
            sp, _ = self.model(sds, **self.params)
            sl = E.from_softmax(sp, to='dict')
            # aggregate
            aopc_intermediate[aopc] = [aopc, cl, sl]
        aopc_predictions = E.reshape_aopc_intermediates(aopc_intermediate, self.params)

        doc_ids = self.val_set['id'] 
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=aopc_predictions)
        return E.classification_scores(results=er_results, mode='val', aopc_thresholds=self.params['aopc_thresholds'], with_ids=doc_ids)
        # E.soft_scores(results=er_results, docids=doc_ids)
    
    def reshape_erased_output(self, inputs):
        questions = [x['question'] for x in inputs]
        answers = [x['answers'] for x in inputs]
        return {
            'question':questions,
            'answers':answers
            }

    def eval_efficiency(self):
        result = {}
        result['flops'], result['params'] = (0,0)
        return result

    def viz(self): pass
    
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
