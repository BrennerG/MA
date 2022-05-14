import torch
import torch.nn as nn
import torch.optim as optim
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
        running_loss = 0

        for i, (question, context, answers, label) in enumerate(ds):
            optimizer.zero_grad()

            output = clf(question, context, answers).view(P['batch_size'],-1)
            label = torch.tensor([label], requires_grad=True).long()

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            PRINT_EVERY = P['print_every']
            if i % PRINT_EVERY == (PRINT_EVERY-1):
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_EVERY:.3f}')
                epoch_losses.append(running_loss)
                running_loss = 0.0
            
            all_outputs.append(output)
    
    return {
        'model': clf,
        'outputs': all_outputs,
        'losses': epoch_losses
    }