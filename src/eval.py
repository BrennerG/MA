from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as pre
from sklearn.metrics import recall_score as rec


# returns accuracy, precision, recall
def competence(gold_labels:[], predictions:[]):
    return (acc(gold_labels, predictions), 
            pre(gold_labels, predictions, average='macro'),
            rec(gold_labels, predictions, average='macro'))