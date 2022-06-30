import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from data_access.locations import LOC

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


# TODO CURRENT retrain for aopc_thresholds!
def predict_aopc_thresholded(P:{}, clf:nn.Module(), ds:Dataset()):
    aopc_thresholds = P['aopc_thresholds'] # TODO put them into P{}

    for aopc in aopc_thresholds:
        pass

    return None

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

def retrain(P:{}, clf:nn.Module(), ds:Dataset(), weights:[]):
    # 1 create cut_ds (cutting top-k entries from text) (as Dataset?)
    # 2 return predict cut_ds
    return None