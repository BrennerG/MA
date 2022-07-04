from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as pre
from sklearn.metrics import recall_score as rec

from data_access.locations import LOC
import evaluation.eraserbenchmark.rationale_benchmark.metrics as EM
import evaluation.eraserbenchmark.rationale_benchmark.utils as EU
from train import from_softmax


# returns accuracy, precision, recall
def competence(gold_labels:[], predictions:[], rounded=3):
    predictions = from_softmax(predictions, to='int')
    return [float(round(acc(gold_labels, predictions), rounded)), 
            float(round(pre(gold_labels, predictions, average='macro'), rounded)),
            float(round(rec(gold_labels, predictions, average='macro'), rounded))]

# TODO what does the threshold parameter actually do?
def rationale_match(gold, pred, thresholds=[0.5]):
    transformed_gold = [EM.Rationale.from_annotation(ann) for ann in gold][0]
    transformed_pred = [EM.Rationale.from_annotation(ann) for ann in pred][0]
    return EM.partial_match_score(truth=transformed_gold, pred=transformed_pred, thresholds=thresholds)

def soft_scores(results, docids):
    # TODO store these somewhere else!
    flattened_documents = EM.load_flattened_documents(LOC['cose'], docids=None)
    annotations = EU.annotations_from_jsonl(LOC['cose_train'])
    paired_scoring = EM.PositionScoredDocument.from_results(results, annotations, flattened_documents, use_tokens=True)
    return EM.score_soft_tokens(paired_scoring)

def classification_scores(results, mode, aopc_thresholds=[0.01, 0.05, 0.1, 0.2, 0.5]): # TODO aopc thresholds are now in the parameters
    # TODO store these somewhere else!
    if mode == 'train':
        annotations = EU.annotations_from_jsonl(LOC['cose_train'])
    else:
        annotations = EU.annotations_from_jsonl(LOC['cose_test'])

    docs = EU.load_documents(LOC['cose'])
    # TODO check for IDs and overlap
    return EM.score_classifications(results, annotations, docs, aopc_thresholds)

# TODO this assumes that the order of docids never changed init of the 'experiment' class
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