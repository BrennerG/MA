from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as pre
from sklearn.metrics import recall_score as rec


# returns accuracy, precision, recall
def competence(gold_labels:[], predictions:[], rounded=3):
    return [float(round(acc(gold_labels, predictions), rounded)), 
            float(round(pre(gold_labels, predictions, average='macro'), rounded)),
            float(round(rec(gold_labels, predictions, average='macro'), rounded))]