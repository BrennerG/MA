import torch
import io
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

from experiments.experiment import Experiment
from models.ud_preproc import UDParser
from data.locations import LOC

# TODO get it to learn!
# TODO verify saving and loading
# TODO experiment with different q_a graph joining methods!
# TODO allow batching - how?

class UD_GCN_Experiment(Experiment):

    def __init__(self, params:{}):
        assert torch.cuda.is_available()
        self.device = 'cuda:0' if ('use_cuda' in params and params['use_cuda']) else 'cpu'
        self.udparser = UDParser(params=params)
        super().__init__(params)
        self.model.to(self.device)

    def init_data(self, params:{}):
        cose = load_dataset(LOC['cose_huggingface'])
        # add graph edges as new cols to the dataset
        edges = [self.udparser(ds, num_samples=len(ds), split=split, qa_join=params['qa_join']) for (split,ds) in cose.items()]
        for i,split in enumerate(cose):
            cose[split] = cose[split].add_column('qa_graphs', edges[i])
        return cose, cose['train'], cose['validation'], cose['test']

    def train(self, params):
        assert 'learning_rate' in params
        assert 'weight_decay' in params

        loss_fn = CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        acc = Accuracy(num_classes=5)
        self.model.train()
        results = {
            'avg_train_losses':[],
            'avg_test_losses':[],
            'train_accs':[],
            'test_accs':[]
        }

        # LOOP
        for epoch in range(params['epochs']):
            preds = torch.zeros(len(self.train_set))
            losses = []

            for i,sample in enumerate(tqdm(self.train_set, desc=f'epoch={epoch}')):
                optimizer.zero_grad()
                out, _ = self.model(sample)
                preds[i] = torch.argmax(out)
                loss = loss_fn(out,preds[i].long())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            # INTER TRAINING EVAL
            test_preds, test_losses = self.intermediate_evaluation()
            avg_train_loss = np.mean(losses)
            avg_test_loss = np.mean(test_losses)
            train_acc = acc(preds.int(), torch.Tensor(self.train_set['label']).int())
            test_acc = acc(test_preds.int(), torch.Tensor(self.val_set['label']).int())
            print(f"\tavg train loss: {avg_train_loss}")
            print(f"\tavg test  loss: {avg_test_loss}")
            print(f"\ttrain acc: {train_acc}")
            print(f"\ttest  acc: {test_acc}")
            results['avg_train_losses'].append(avg_train_loss)
            results['avg_test_losses'].append(avg_test_loss)
            results['train_accs'].append(train_acc)
            results['test_accs'].append(test_acc)

        return results
    
    def intermediate_evaluation(self):
        self.model.eval()
        preds = torch.zeros(len(self.val_set))
        loss_fn = CrossEntropyLoss()
        losses = []

        for i,sample in enumerate(tqdm(self.val_set, desc='inter-train eval:')):
            out, _ = self.model(sample)
            preds[i] = torch.argmax(out)
            loss = loss_fn(out,preds[i].long())
            losses.append(loss.item())
        
        return preds, losses
    
    def eval_competence(self, params:{}):
        self.model.eval()
        acc = Accuracy(num_classes=5)
        preds = torch.stack([torch.argmax(x) for x in self.val_pred[0]])
        ys = torch.Tensor(self.val_set['label']).int()
        return acc(preds.int(), ys)

    def eval_explainability(self, params:{}):
        return None

    def eval_efficiency(self, params:{}): # TODO
        return None

    def viz(self, params:{}):
        return None

    def save(self, params:{}):
        # no save location, no saving
        if 'save_loc' not in params: return False
        # create save location
        if not os.path.exists(params['save_loc']):
            os.mkdir(params['save_loc'])
        # saving model
        torch.save(self.model.state_dict(), f"{params['save_loc']}/model.pt")
        return True
