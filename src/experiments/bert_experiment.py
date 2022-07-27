from experiments.experiment import Experiment
from datasets import load_dataset
from data.huggingface_cose import EraserCosE

import evaluation.eval_util as E
from data.locations import LOC

class BERTExperiment(Experiment):

    def init_data(self, params:{}):
        cose = load_dataset(LOC['cose_huggingface'])
        if 'debug' in params and params['debug']:
            return cose, cose['debug_train'], cose['debug_val'], cose['test']
        else:
            return cose, cose['train'], cose['validation'], cose['test']

    def train(self, params:{}):
        return self.model.train(
            self.complete_set, 
            debug_train_split=('debug' in params and params['debug'])
            # TODO save_loc = '/path/to/dir/'
        )
    
    def load(self):
        return None

    def eval_competence(self, params:{}):
        probas, attn = self.val_pred
        return E.competence_metrics(self.val_set['label'], probas)

    # TODO erase using the dataset objects of this class!
    def eval_explainability(self, params:{}):
        # calculate eraser metrics
        pred, attn = self.val_pred
        comp_ds = EraserCosE.erase(attn, mode='comprehensiveness', split='debug_val')
        suff_ds = EraserCosE.erase(attn, mode='sufficiency', split='debug_val')
        comp_pred, _ = zip(*self.model(comp_ds)) # TODO exclude lime calcs here
        suff_pred, _ = zip(*self.model(suff_ds)) # TODO exclude lime calcs here
        # calcualte aopc metrics
        aopc_intermediate = {}
        for aopc in params['aopc_thresholds']:
            # comp
            cds = EraserCosE.erase(attn, mode='comprehensiveness', split='debug_val')
            cp, _ = zip(*self.model(cds)) # TODO disable lime here
            cl = E.from_softmax(cp, to='dict') # labels
            # suff
            sds = EraserCosE.erase(attn, mode='sufficiency', split='debug_val')
            sp, _ = zip(*self.model(sds)) # TODO disable lime here
            sl = E.from_softmax(sp, to='dict')
            # aggregate
            aopc_intermediate[aopc] = [aopc, cl, sl]
        aopc_predictions = E.reshape_aopc_intermediates(aopc_intermediate, params)

        doc_ids = self.val_set['id'] # TODO only works because we have manually set split to debug_val everywhere...
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=aopc_predictions)
        return E.classification_scores(results=er_results, mode='val', aopc_thresholds=params['aopc_thresholds'], with_ids=doc_ids)

    def eval_efficiency(self, params:{}):
        return self.model.efficiency_metrics()

    def viz(self, params:{}):
        return None