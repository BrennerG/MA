import math
from datasets import load_dataset

from data.huggingface_cose import EraserCosE
from experiments.experiment import Experiment
import evaluation.eval_util as E
from data.locations import LOC


class BERTExperiment(Experiment):

    def __init__(self, params):
        super().__init__(params)
        self.avg_rational_lengths = EraserCosE.avg_rational_length(self.complete_set)

    def init_data(self, params:{}):
        cose = load_dataset(LOC['cose_huggingface'])
        if 'debug' in params and params['debug']:
            return cose, cose['debug_train'], cose['debug_val'], cose['test']
        else:
            return cose, cose['train'], cose['validation'], cose['test']

    def train(self, params:{}):
        if 'load_from' in params:
            print(f"MODEL PRELOADED FROM {params['load_from']} - SKIPPING TRAINING!")
            return None
        return self.model.train(
            self.complete_set, 
            debug_train_split=('debug' in params and params['debug']),
            save_loc=LOC['bert_checkpoints']
        )
    
    def eval_competence(self, params:{}):
        probas, attn = self.val_pred
        results = {}
        results['accuracy'], results['precision'], results['recall'] = E.competence_metrics(self.val_set['label'], probas)
        return results

    # TODO erase using the dataset objects of this class! (no split='debug_val)
    def eval_explainability(self, params:{}):
        split = 'debug_val' if 'debug' in params and params['debug']==True else 'validation'
        pred, attn = self.val_pred
        comp_ds = EraserCosE.erase(attn, mode='comprehensiveness', split=split)
        suff_ds = EraserCosE.erase(attn, mode='sufficiency', split=split)
        comp_pred, _ = zip(*self.model(comp_ds, attention=None))
        suff_pred, _ = zip(*self.model(suff_ds, attention=None))
        # calcualte aopc metrics
        aopc_intermediate = {}
        for aopc in params['aopc_thresholds']:
            tokens_to_be_erased = math.ceil(aopc * self.avg_rational_lengths[split])
            # comp
            cds = EraserCosE.erase(attn, mode='comprehensiveness', split=split, k=tokens_to_be_erased)
            cp, _ = zip(*self.model(cds, attention=None))
            cl = E.from_softmax(cp, to='dict') # labels
            # suff
            sds = EraserCosE.erase(attn, mode='sufficiency', split=split, k=tokens_to_be_erased)
            sp, _ = zip(*self.model(sds, attention=None))
            sl = E.from_softmax(sp, to='dict')
            # aggregate
            aopc_intermediate[aopc] = [aopc, cl, sl]
        aopc_predictions = E.reshape_aopc_intermediates(aopc_intermediate, params)

        doc_ids = self.val_set['id']
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=aopc_predictions)
        return E.classification_scores(results=er_results, mode='val', aopc_thresholds=params['aopc_thresholds'], with_ids=doc_ids)
        # E.soft_scores(results=er_results, docids=doc_ids)

    def eval_efficiency(self, params:{}):
        result = {}
        result['flops'], result['params'] = self.model.efficiency_metrics()
        return result

    def viz(self, params:{}):
        return None