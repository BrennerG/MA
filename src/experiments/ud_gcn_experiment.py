import torch
import io
import os
import json
import math

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
import plotly.express as px
import pandas as pd

from experiments.experiment import Experiment
from models.ud_preproc import UDParser
from data.locations import LOC
from data.huggingface_cose import EraserCosE
import evaluation.eval_util as E

# TODO get it to learn!
# TODO verify saving and loading
# TODO allow batching?
# TODO GAT (rename this)
# TODO experiment with different q_a graph joining methods!

class UD_GCN_Experiment(Experiment):

    def __init__(self, params:{}):
        assert torch.cuda.is_available()
        self.device = 'cuda:0' if ('use_cuda' in params and params['use_cuda']) else 'cpu'
        self.udparser = UDParser(params=params)
        super().__init__(params)
        self.model.to(self.device)
        self.avg_rational_lengths = EraserCosE.avg_rational_length(self.complete_set)

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
        results = {
            'avg_train_losses':[],
            'avg_test_losses':[],
            'train_accs':[],
            'test_accs':[]
        }

        # LOOP
        for epoch in range(params['epochs']):
            preds = torch.zeros(len(self.train_set))
            attentions = []
            losses = []
            self.model.train()

            for i,sample in enumerate(tqdm(self.train_set, desc=f'epoch={epoch}')):
                optimizer.zero_grad()
                target = torch.Tensor([sample['label']]).squeeze().long()
                out, attention = self.model(sample)
                preds[i] = torch.argmax(out)
                loss = loss_fn(out, target)
                losses.append(loss.item())
                attentions.append(attention)
                loss.backward()
                optimizer.step()

            # INTER TRAINING EVAL
            test_preds, test_losses = self.intermediate_evaluation()
            avg_train_loss = np.mean(losses)
            avg_test_loss = np.mean(test_losses)
            train_acc = acc(preds.int(), torch.Tensor(self.train_set['label']).int()).item()
            test_acc = acc(test_preds.int(), torch.Tensor(self.val_set['label']).int()).item()
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

    def eval_explainability(self, params:{}): # TODO only for GAT models
        split = 'validation'
        pred, attn = self.val_pred
        # prepare Comprehensiveness
        comp_ds = EraserCosE.erase(attn, mode='comprehensiveness', split=split)
        comp_edges = self.udparser(comp_ds, num_samples=len(comp_ds), split=split, qa_join=params['qa_join'], use_cache=False)
        for i,sample in enumerate(comp_ds):
            sample['qa_graphs'] = comp_edges[i]
        # prepare Sufficiency
        suff_ds = EraserCosE.erase(attn, mode='sufficiency', split=split)
        suff_edges = self.udparser(suff_ds, num_samples=len(suff_ds), split=split, qa_join=params['qa_join'], use_cache=False)
        for i,sample in enumerate(suff_ds):
            sample['qa_graphs'] = suff_edges[i]
        # predict
        comp_pred, _ = zip(*self.model(comp_ds))
        suff_pred, _ = zip(*self.model(suff_ds))
        # TODO CURRENT! (bug existing from the lines above)

        # calcualte aopc metrics
        aopc_intermediate = {}
        for aopc in tqdm(params['aopc_thresholds'], desc='explainability_eval: '):
            tokens_to_be_erased = math.ceil(aopc * self.avg_rational_lengths[split])
            # comp
            cds = EraserCosE.erase(attn, mode='comprehensiveness', split=split, k=tokens_to_be_erased)
            ce = self.udparser(cds, num_samples=len(cds), split=split, qa_join=params['qa_join'], use_cache=False)
            for i,sample in enumerate(cds):
                sample['qa_graphs'] = ce[i]
            cp, _ = zip(*self.model(cds))
            cl = E.from_softmax(cp, to='dict') # labels
            # suff
            sds = EraserCosE.erase(attn, mode='sufficiency', split=split, k=tokens_to_be_erased)
            se = self.udparser(sds, num_samples=len(sds), split=split, qa_join=params['qa_join'], use_cache=False)
            for i,sample in enumerate(sds):
                sample['qa_graphs'] = se[i]
            sp, _ = zip(*self.model(sds))
            sl = E.from_softmax(sp, to='dict')
            # aggregate
            aopc_intermediate[aopc] = [aopc, cl, sl]
        aopc_predictions = E.reshape_aopc_intermediates(aopc_intermediate, params)

        doc_ids = self.val_set['id']
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=aopc_predictions)
        self.er_results = er_results
        return E.classification_scores(results=er_results, mode='val', aopc_thresholds=params['aopc_thresholds'], with_ids=doc_ids)

    def eval_efficiency(self, params:{}): # TODO
        return None

    def viz(self, params:{}):
        if not 'save_loc' in params: return False
        # VIZ LOSS
        num_epochs = len(self.train_output['avg_train_losses'])
        df = pd.DataFrame(
            list(zip(
                list(range(num_epochs))*2,
                self.train_output['avg_train_losses'] + self.train_output['avg_test_losses'],
                ['train']*num_epochs + ['test']*num_epochs
            )), 
            columns=['epoch', 'loss', 'split']
        )
        fig = px.line(df, x="epoch", y="loss", color='split', title='UD+GCN train loss')
        fig.write_image(f"{params['save_loc']}/loss.png")
        return True

    def save(self, params:{}):
        # no save location, no saving
        if 'save_loc' in params: 
            # create save location
            if not os.path.exists(params['save_loc']):
                os.mkdir(params['save_loc'])
            # saving model
            torch.save(self.model.state_dict(), f"{params['save_loc']}/model.pt")
            # saving cached glove vocabulary # TODO only if using glove...
        # cache used glove embeddings
        with open(LOC['glove_cache'], 'w') as outfile:
            json.dump(self.model.embedding.cached_dict, outfile, sort_keys=True, indent=4)
        return True