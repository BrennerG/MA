from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as pre
from sklearn.metrics import recall_score as rec
import evaluation.eraserbenchmark.rationale_benchmark.metrics as EM


# returns accuracy, precision, recall
def competence(gold_labels:[], predictions:[], rounded=3):
    return [float(round(acc(gold_labels, predictions), rounded)), 
            float(round(pre(gold_labels, predictions, average='macro'), rounded)),
            float(round(rec(gold_labels, predictions, average='macro'), rounded))]

# TODO what does the threshold parameter actually do?
def rationale_match(gold, pred, thresholds=[0.5]):
    transformed_gold = [EM.Rationale.from_annotation(ann) for ann in gold][0]
    transformed_pred = [EM.Rationale.from_annotation(ann) for ann in pred][0]
    return EM.partial_match_score(truth=transformed_gold, pred=transformed_pred, thresholds=thresholds)


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