import torch
import io
import os
from tqdm import tqdm
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

from experiments.experiment import Experiment
from models.ud_preproc import UDParser
from data.locations import LOC

# TODO verify saving and loading
# TODO experiment with different q_a graph joining methods!
# TODO allow batching - how?

class UD_GCN_Experiment(Experiment):

    def __init__(self, params:{}):
        assert torch.cuda.is_available()
        self.device = 'cuda:0' if ('use_cuda' in params and params['use_cuda']) else 'cpu'
        self.udparser = UDParser()
        super().__init__(params)
        self.model.to(self.device)

    def init_data(self, params:{}):
        num_samples = params['num_samples'] if 'num_samples' in params else -1
        cose = load_dataset(LOC['cose_huggingface'])
        # add graph edges as new cols to the dataset
        edges = [self.udparser(ds, num_samples=num_samples, split=split) for (split,ds) in cose.items()]
        for i,split in enumerate(cose):
            cose[split] = cose[split].add_column('qa_graphs', edges[i])
        return cose, cose['train'], cose['validation'], cose['test']

    def train(self, params):
        assert 'learning_rate' in params
        assert 'weight_decay' in params

        loss_fn = CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        self.model.train()

        # LOOP
        for epoch in range(params['epochs']):
            preds = torch.zeros(len(self.train_set))
            for i,sample in enumerate(tqdm(self.train_set, desc=f'epoch={epoch} training...')):
                optimizer.zero_grad()
                out, _ = self.model(sample)
                preds[i] = torch.argmax(out)
                loss = loss_fn(out,preds[i].long())
                loss.backward()
                optimizer.step()

        return 'NotImplemented: TrainingOutput' # TODO
    
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

    def save(self, params:{}): # TODO save model dict
        # no save location, no saving
        if 'save_loc' not in params: return False
        # create save location
        if not os.path.exists(params['save_loc']):
            os.mkdir(params['save_loc'])
        # saving model
        torch.save(self.model.state_dict(), f"{params['save_loc']}gcn.pt")
        return True