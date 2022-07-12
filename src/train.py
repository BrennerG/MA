import math
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

def train_custom(P:{}, ds:Dataset, clf):
    clf.train(ds)
    return clf.predict(ds)

def train(P:{}, ds:Dataset, clf:nn.Module):
    epoch_losses = []
    
    if P['batch_size'] > 1:
        ds = DataLoader(ds, batch_size=5, shuffle=True, num_workers=0) # this allows ez batching + shuffling

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(clf.parameters(), lr=P['lr'], momentum=P['momentum'])

    for epoch in range(P['epochs']):
        all_outputs = []
        all_attentions = []
        running_loss = 0

        for i, (question, context, answers, label, evidence) in enumerate(ds):
            optimizer.zero_grad()

            output, attn = clf(question, context, answers)
            output = output.view(P['batch_size'],-1)
            label = torch.tensor([label], requires_grad=True).long()

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            PRINT_EVERY = P['print_every']
            if i % PRINT_EVERY == (PRINT_EVERY-1):
                mean_loss = running_loss / PRINT_EVERY
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {mean_loss:.3f}')
                epoch_losses.append(mean_loss)
                running_loss = 0.0
            
            all_outputs.append(output)
            all_attentions.append(attn)
    
    return {
        'model': clf,
        'outputs': all_outputs,
        'attentions': all_attentions,
        'losses': epoch_losses
    }

def predict(P:{}, clf:nn.Module(), ds:Dataset()):
    predictions = []
    attentions = []

    with torch.no_grad():
        for i, (question, context, answers, label, evidence) in enumerate(ds):
            output, attn = clf(question, context, answers)
            output = output.view(P['batch_size'],-1)
            predictions.append(output)
            attentions.append(attn)

    return predictions, attentions

# this method is for the eraser benchmark
# it allows re-predicting a given dataset by omitting the top x% of tokens,
# where x is a number in the 'aopc_thresholds' parameter.
# this enables evaluation for sufficiency and comprehensiveness as proposed by ERASER.
def predict_aopc_thresholded(model_params:{}, eval_params, clf:nn.Module(), attn:[], ds:Dataset()):
    aopc_thresholds = eval_params['aopc_thresholds'] # TODO put them into P{}
    intermediate = {}
    result = []

    for aopc in aopc_thresholds:
        # TODO calculate the aopc erases on a per_sample basis or take aopc*avg_evidence_len (second approach currently)
        tokens_to_be_erased = math.ceil(aopc * ds.avg_evidence_len)
        # comp
        comp_ds = ds.erase(attn, k=tokens_to_be_erased, mode='comprehensiveness')
        comp_pred, _ = predict(model_params, clf, comp_ds)
        comp_labels = from_softmax(comp_pred, to='dict')
        # suff
        suff_ds = ds.erase(attn, k=tokens_to_be_erased, mode='sufficiency')
        suff_pred, _ = predict(model_params, clf, suff_ds)
        suff_labels = from_softmax(suff_pred, to='dict')
    
        intermediate[aopc] = [aopc, comp_labels, suff_labels]

    # TODO transform results (there is probably a better variant)
    for i,x in enumerate(ds):
        sample = []
        for aopc in aopc_thresholds:

            sample.append({
                'threshold': aopc,
                "comprehensiveness_classification_scores": intermediate[aopc][1][i],
                "sufficiency_classification_scores": intermediate[aopc][2][i],
            })
        result.append(sample)

    return result

# enables transformation from a label_softmax output of the model
# to a dictionary or integer form of labels/predictions
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