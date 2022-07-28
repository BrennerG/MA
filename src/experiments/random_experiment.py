from experiments.experiment import Experiment
from datasets import load_dataset
from data.huggingface_cose import EraserCosE
from data.locations import LOC
import evaluation.eval_util as E
from data.locations import LOC


class RandomClassifierExperiment(Experiment):    
    
    def init_data(self, params:{}):
        ''' returns (complete_dataset, train_data, val_data, test_data) '''
        cose = load_dataset(LOC['cose_huggingface'])
        if 'debug' in params and params['debug']:
            return cose, cose['debug_train'], cose['debug_val'], cose['test']
        else:
            return cose, cose['train'], cose['validation'], cose['test']

    def train(self, params:{}):
        print('NO TRAINING NEEDED FOR RANDOM CLASSIFIER')
        return None
    
    def eval_competence(self, params:{}):
        probas, attn = self.val_pred
        return E.competence_metrics(self.val_set['label'], probas)

    def eval_explainability(self, params:{}):
        split = 'debug_val' if 'debug' in params and params['debug']==True else 'val'
        pred, attn = self.val_pred
        comp_ds = self.reshape_erased_output(EraserCosE.erase(attn, mode='comprehensiveness', split=split))
        suff_ds = self.reshape_erased_output(EraserCosE.erase(attn, mode='sufficiency', split=split))
        comp_pred, _ = self.model(comp_ds)
        suff_pred, _ = self.model(suff_ds)
        # calcualte aopc metrics
        aopc_intermediate = {}
        for aopc in params['aopc_thresholds']:
            # TODO WHERE THE K AT?
            # comp
            cds = self.reshape_erased_output(EraserCosE.erase(attn, mode='comprehensiveness', split=split))
            cp, _ = self.model(cds)
            cl = E.from_softmax(cp, to='dict') # labels
            # suff
            sds = self.reshape_erased_output(EraserCosE.erase(attn, mode='sufficiency', split=split))
            sp, _ = self.model(sds)
            sl = E.from_softmax(sp, to='dict')
            # aggregate
            aopc_intermediate[aopc] = [aopc, cl, sl]
        aopc_predictions = E.reshape_aopc_intermediates(aopc_intermediate, params)

        doc_ids = self.val_set['id'] # TODO only works because we have manually set split to debug_val everywhere...
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=aopc_predictions)
        return E.classification_scores(results=er_results, mode='val', aopc_thresholds=params['aopc_thresholds'], with_ids=doc_ids)
        # E.soft_scores(results=er_results, docids=doc_ids)
    
    def reshape_erased_output(self, inputs):
        questions = [x['question'] for x in inputs]
        answers = [x['answers'] for x in inputs]
        return {
            'question':questions,
            'answers':answers
            }

    def eval_efficiency(self, params:{}):
        return 0, 0

    def viz(self, params:{}):
        return None