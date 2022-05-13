import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from data_access.csqa_dataset import CsqaDataset
from data_access.locations import LOC
from models.random_clf import RandomClassifier

LIMIT = -1
EPOCHS = 1
PRINT_EVERY = 1000
BATCH_SIZE = 1
LR = 0.001
MOMENTUM = 0.9

ds = CsqaDataset(LOC['csqa_train'], limit=LIMIT)
clf = RandomClassifier(69)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(clf.parameters(), lr=LR, momentum=MOMENTUM)
# check for batch size here if not --> use dataloader instead
# dataloader = DataLoader(ds, batch_size=5, shuffle=True, num_workers=0) # this allows ez batching + shuffling

for epoch in range(EPOCHS):
    running_loss = 0

    for i, (question, context, answers, label) in enumerate(ds):
        optimizer.zero_grad()

        output = clf(question, context, answers).view(BATCH_SIZE,-1)
        label = torch.tensor([label], requires_grad=True).long()

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % PRINT_EVERY == (PRINT_EVERY-1):
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_EVERY:.3f}')
            running_loss = 0.0