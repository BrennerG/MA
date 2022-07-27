import numpy as np
import torch 
import torch.nn as nn

from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as pre
from sklearn.metrics import recall_score as rec
import evaluation.eraserbenchmark.rationale_benchmark.metrics as EM
import evaluation.eraserbenchmark.rationale_benchmark.utils as EU
from torchstat import stat

from data_access.locations import LOC
import train as T


def explainability_metrics(model, dataset, model_params, evaluation_params, mode='test'):
    # init
    pred, attn = T.predict(model_params, model, dataset)
    if isinstance(attn[0], torch.Tensor): attn_detached = [x.detach().numpy() for x in attn]
    else: attn_detached = attn
    doc_ids = dataset.docids
    # erase for comprehensiveness and sufficiency
    comp_data = dataset.erase(attn, mode='comprehensiveness')
    suff_data = dataset.erase(attn, mode='sufficiency')
    # do predictions
    comp_predictions, _ = T.predict(model_params, model, comp_data, proba=True)
    suff_predictions, _ = T.predict(model_params, model, suff_data, proba=True)
    aopc_predictions = T.predict_aopc_thresholded(model_params, evaluation_params, model, attn, dataset)
    # get results
    er_results = create_results(doc_ids, pred, comp_predictions, suff_predictions, attn_detached, aopc_thresholded_scores=aopc_predictions)
    agreement_auprc = soft_scores(er_results, docids=doc_ids, ds=f'cose_{mode}') # TODO
    clf_scores = classification_scores(results=er_results, mode=mode, aopc_thresholds=evaluation_params['aopc_thresholds'])
    return agreement_auprc, clf_scores

def efficiency_metrics(model:nn.Module, input_size):
    state_dict_keys_before = model.state_dict().keys()
    scores = stat(model, input_size=input_size) # calculates efficiency stats and writes them into model.state_dict()
    # TODO create additional constraints here to get rid of unwanted state_dict_keys (e.g. ends with _shape or sth... - may depend on model too!)
    # TODO state_dict_keys_new = empty after the efficiency calcs for train, bc the state_dict is then filled with the keys already!
    #   this shouldnt matter as the only thing differentiating test/train in this regard is the #samples...
    state_dict_keys_new = [x for x in model.state_dict().keys() if x not in state_dict_keys_before] 
    state_dict = model.state_dict() # updated state_dict
    results = {k:state_dict[k].item() if state_dict[k].shape[0]==1 else state_dict[k].tolist() for k in state_dict_keys_new}
    return results

# returns accuracy, precision, recall
def competence(gold_labels:[], predictions:[], rounded=3):
    predictions = T.from_softmax(predictions, to='int')
    return [float(round(acc(gold_labels, predictions), rounded)), 
            float(round(pre(gold_labels, predictions, average='macro'), rounded)),
            float(round(rec(gold_labels, predictions, average='macro'), rounded))]

''' 
# not used anymore... substituted by soft_scores
# measures the match/overlap of the annotated and predicted rationales
def rationale_match(gold, pred, thresholds=[0.5]):
    transformed_gold = [EM.Rationale.from_annotation(ann) for ann in gold][0]
    transformed_pred = [EM.Rationale.from_annotation(ann) for ann in pred][0]
    return EM.partial_match_score(truth=transformed_gold, pred=transformed_pred, thresholds=thresholds)
'''

# measures the aggreement between the gold rationales and predicted rationales
def soft_scores(results, docids, ds='cose_train'):
    flattened_documents = EM.load_flattened_documents(LOC['cose'], docids=None)
    annotations = EU.annotations_from_jsonl(LOC[ds])
    paired_scoring = EM.PositionScoredDocument.from_results(results, annotations, flattened_documents, use_tokens=True)
    scores = EM.score_soft_tokens(paired_scoring)
    return {k:float(v) for k,v in scores.items()}

# classification metrics (accuracy, precision, recall, comprehensiveness, sufficiency)
def classification_scores(results, mode, aopc_thresholds=[0.01, 0.05, 0.1, 0.2, 0.5]):
    if mode == 'train':
        annotations = EU.annotations_from_jsonl(LOC['cose_train'])
    elif mode == 'val':
        annotations = EU.annotations_from_jsonl(LOC['cose_val'])
    elif mode == 'test':
        annotations = EU.annotations_from_jsonl(LOC['cose_test'])
    else:
        raise AttributeError('mode unknown!')

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

# brings all of the predictions into the correct format for the methods of evaluation above (e.g. soft_scores() and classification_scores())
def create_results(docids, predictions, comp_predicitons, suff_predictions, attentions, aopc_thresholded_scores):
    assert len(docids) == len(predictions) == len(attentions)

    result = []
    labels = T.from_softmax(predictions, to='str')
    dicprics = T.from_softmax(predictions, to='dict')
    comp_dicprics = T.from_softmax(comp_predicitons, to='dict')
    suff_dicprics = T.from_softmax(suff_predictions, to='dict')

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

# This is the input for the ERASER benchmark as described by the original implementation
# it's here for quick reference
'''
THE FOLLOWING TEXT DESCRIBES THE SHAPE OF 'INSTANCES' AS EG FOUND IN METRICS.COMPUTE_AOPC_SCORES(INSTANCES, THRESHOLDS)

    Contents are expected to be jsonl of:
    {
        "annotation_id": str, required
        # these classifications *must not* overlap
        "rationales": List[
            {
                "docid": str, required
                "hard_rationale_predictions": List[{
                    "start_token": int, inclusive, required
                    "end_token": int, exclusive, required
                }], optional,
                # token level classifications, a value must be provided per-token
                # in an ideal world, these correspond to the hard-decoding above.
                "soft_rationale_predictions": List[float], optional.
                # sentence level classifications, a value must be provided for every
                # sentence in each document, or not at all
                "soft_sentence_predictions": List[float], optional.
            }
        ],
        # the classification the model made for the overall classification task
        "classification": str, optional

        # A probability distribution output by the model. We require this to be normalized.
        "classification_scores": Dict[str, float], optional

        # The next two fields are measures for how faithful your model is (the
        # rationales it predicts are in some sense causal of the prediction), and
        # how sufficient they are. We approximate a measure for comprehensiveness by
        # asking that you remove the top k%% of tokens from your documents,
        # running your models again, and reporting the score distribution in the
        # "comprehensiveness_classification_scores" field.
        # We approximate a measure of sufficiency by asking exactly the converse
        # - that you provide model distributions on the removed k%% tokens.
        # 'k' is determined by human rationales, and is documented in our paper.
        # You should determine which of these tokens to remove based on some kind
        # of information about your model: gradient based, attention based, other
        # interpretability measures, etc.
        # scores per class having removed k%% of the data, where k is determined by human comprehensive rationales
        "comprehensiveness_classification_scores": Dict[str, float], optional

        # scores per class having access to only k%% of the data, where k is determined by human comprehensive rationales
        "sufficiency_classification_scores": Dict[str, float], optional

        # the number of tokens required to flip the prediction - see "Is Attention Interpretable" by Serrano and Smith.
        "tokens_to_flip": int, optional

        "thresholded_scores": List[{
            "threshold": float, required,
            "comprehensiveness_classification_scores": like "classification_scores"
            "sufficiency_classification_scores": like "classification_scores"
        }], optional. if present, then "classification" and "classification_scores" must be present
    }
'''