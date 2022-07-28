import numpy as np
# competence metrics
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as pre
from sklearn.metrics import recall_score as rec
# explainability metrics
import evaluation.eraserbenchmark.rationale_benchmark.utils as EU
import evaluation.eraserbenchmark.rationale_benchmark.metrics as EM
from data.locations import LOC

def from_softmax(softmax_predictions=None, to:str='int'): # or STR
    letters = ['A','B','C','D','E']
    intform = [x.squeeze().tolist() for x in softmax_predictions]
    amax = [np.argmax(x) for x in intform]
    if to == 'int':
        return amax
    elif to == 'str':
        return [letters[x] for x in amax]
    elif to == 'dict':
        return [{k:v for k,v in zip(letters, x)} for x in intform]
    else:
        raise ValueError('"to" must be "int","str" or "dict"!')

def competence_metrics(gold_labels:[], predictions:[], rounded=3):
    predictions = from_softmax(predictions, to='int')
    return [float(round(acc(gold_labels, predictions), rounded)), 
            float(round(pre(gold_labels, predictions, average='macro'), rounded)),
            float(round(rec(gold_labels, predictions, average='macro'), rounded))]
        
# transform results
def reshape_aopc_intermediates(intermediate:{}, params:{}):
    result = []
    aopc_thresholds = params['aopc_thresholds']
    size = len(list(intermediate.values())[0][1])
    for i in range(size):
        sample = []
        for aopc in aopc_thresholds:
            sample.append({
                'threshold': aopc,
                "comprehensiveness_classification_scores": intermediate[aopc][1][i],
                "sufficiency_classification_scores": intermediate[aopc][2][i],
            })
        result.append(sample)
    return result

# brings all of the predictions into the correct format for the methods of evaluation above (e.g. soft_scores() and classification_scores())
def create_results(docids, predictions, comp_predicitons, suff_predictions, attentions, aopc_thresholded_scores):
    assert len(docids) == len(predictions) == len(attentions)

    result = []
    labels = from_softmax(predictions, to='str')
    dicprics = from_softmax(predictions, to='dict')
    comp_dicprics = from_softmax(comp_predicitons, to='dict')
    suff_dicprics = from_softmax(suff_predictions, to='dict')

    for i, docid in enumerate(docids):
        dic = {
            'annotation_id': docid, # str
            'rationales': [
                {
                    'docid': docid, # str
                    'soft_rationale_predictions': attentions[i] # List[float]
                }
            ],
            'classification': labels[i], # str

            'classification_scores': dicprics[i], # Dict[str,float],
            'comprehensiveness_classification_scores': comp_dicprics[i], # Dict[str, float] # retrained without top k%%
            'sufficiency_classification_scores': suff_dicprics[i], # Dict[str, float] # retrained with only top K%%

            #'tokens_to_flip': None, # int

            "thresholded_scores": aopc_thresholded_scores[i]
        }
        result.append(dic)

    return result

# classification metrics (accuracy, precision, recall, comprehensiveness, sufficiency)
def classification_scores(results, mode, aopc_thresholds=[0.01, 0.05, 0.1, 0.2, 0.5], with_ids:[]=None):
    if mode == 'train':
        annotations = EU.annotations_from_jsonl(LOC['cose_train'])
    elif mode == 'val':
        annotations = EU.annotations_from_jsonl(LOC['cose_val'])
    elif mode == 'test':
        annotations = EU.annotations_from_jsonl(LOC['cose_test'])
    else:
        raise AttributeError('mode unknown!')
    
    # for debug or custom splits
    if with_ids: 
        annotations = [ann for ann in annotations if ann.annotation_id in with_ids]

    # TODO check for IDs and overlap (are results and annotation in the same order of samples?)
    docs = EU.load_documents(LOC['cose'])
    classifications = EM.score_classifications(results, annotations, docs, aopc_thresholds)

    # parse classification scores from numpy64 to float (for yaml readability!)
    ret = {}
    for k,v, in classifications.items():
        if isinstance(v, np.float64):
            ret[k] = float(v)
        else:
            ret[k] = v
    return ret

# measures the aggreement between the gold rationales and predicted rationales
def soft_scores(results, docids, ds='cose_train'):
    flattened_documents = EM.load_flattened_documents(LOC['cose'], docids=None)
    annotations = EU.annotations_from_jsonl(LOC[ds])
    paired_scoring = EM.PositionScoredDocument.from_results(results, annotations, flattened_documents, use_tokens=True)
    scores = EM.score_soft_tokens(paired_scoring)
    return {k:float(v) for k,v in scores.items()}
