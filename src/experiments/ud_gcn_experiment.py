import torch
import io
import os
import json
import math
import yaml

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, Recall, Precision
import plotly.express as px
import pandas as pd
import wandb

from experiments.experiment import Experiment
from preproc.ud_preproc import UDParser
from data.locations import LOC
from data.huggingface_cose import EraserCosE
import evaluation.eval_util as E
from models.gcn import GCN


class UD_GCN_Experiment(Experiment):

    def __init__(self, params:{}):
        assert torch.cuda.is_available()
        self.params = params
        self.device = 'cuda:0' if ('use_cuda' in self.params and self.params['use_cuda']) else 'cpu'
        self.udparser = UDParser(params=self.params)
        super().__init__(self.params)
        self.model.to(self.device)
        self.avg_rational_lengths = EraserCosE.avg_rational_length(self.complete_set)
        wandb.init(config=self.params, mode='online' if self.params['wandb_logging']==True else 'disabled')

    def init_data(self):
        cose = load_dataset(LOC['cose_huggingface'])
        # add graph edges as new cols to the dataset
        edges = [self.udparser(ds, num_samples=len(ds), split=split, qa_join=self.params['qa_join']) for (split,ds) in cose.items()]
        for i,split in enumerate(cose):
            cose[split] = cose[split].add_column('qa_graphs', edges[i])
        return cose, cose['train'], cose['validation'], cose['test']

    def train(self):
        assert 'learning_rate' in self.params
        assert 'weight_decay' in self.params

        do_explainability_eval = 'inter_training_expl_eval' in self.params and self.params['inter_training_expl_eval']==True
        loss_fn = CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])
        acc = Accuracy(num_classes=5)
        results = {
            'avg_train_losses':[],
            'avg_test_losses':[],
            'train_accs':[],
            'test_accs':[]
        }

        # LOOP
        for epoch in range(self.params['epochs']):
            preds = torch.zeros(len(self.train_set),5)
            attentions = []
            losses = []
            self.model.train()

            for i,sample in enumerate(tqdm(self.train_set, desc=f'epoch={epoch}')):
                optimizer.zero_grad()
                target = torch.Tensor([sample['label']]).squeeze().long()
                out, attention = self.model(sample) # TODO call with **args? yes probably!
                preds[i] = out
                loss = loss_fn(out, target)
                losses.append(loss.item())
                attentions.append(attention)
                loss.backward()
                optimizer.step()

            # INTER TRAINING EVAL
            test_preds, test_attn, test_losses = self.intermediate_evaluation()
            avg_train_loss = np.mean(losses)
            avg_test_loss = np.mean(test_losses)
            train_acc = acc(torch.argmax(preds,dim=1), torch.Tensor(self.train_set['label']).int()).item()
            test_acc = acc(torch.argmax(test_preds,dim=1), torch.Tensor(self.val_set['label']).int()).item()
            if do_explainability_eval:
                expl_eval = self.eval_explainability(self.params, pred=test_preds, attn=test_attn, skip_aopc=True)

            # log on wandb
            result_dict = {
                'avg_train_loss': avg_train_loss,
                'avg_test_loss': avg_test_loss,
                'acc_train': train_acc,
                'acc_test': test_acc,
                'comprehensiveness_test': expl_eval['comprehensiveness'] if do_explainability_eval else None,
                'sufficiency_test': expl_eval['sufficiency'] if do_explainability_eval else None
            }
            if 'wandb_logging' in self.params and self.params['wandb_logging']:
                wandb.log(result_dict)
            else:
                print(result_dict)

        return None
    
    def intermediate_evaluation(self):
        self.model.eval()
        preds = torch.zeros(len(self.val_set),5)
        attentions = []
        loss_fn = CrossEntropyLoss()
        losses = []

        for i,sample in enumerate(tqdm(self.val_set, desc='inter-train eval:')):
            out, attn = self.model(sample)
            target = torch.Tensor([sample['label']]).squeeze().long()
            preds[i] = out
            attentions.append(attn)
            loss = loss_fn(out, target)
            losses.append(loss.item())
        
        return preds, attentions, losses
    
    def eval_competence(self):
        self.model.eval()
        acc = Accuracy(num_classes=5)
        prec = Precision(num_classes=5)
        reca = Recall(num_classes=5)
        preds = torch.stack([torch.argmax(x) for x in self.val_pred[0]])
        ys = torch.Tensor(self.val_set['label']).int()
        return {
            'accuracy' : acc(preds.int(), ys).item(), 
            'precision' : prec(preds.int(), ys).item(), 
            'recall' : reca(preds.int(), ys).item()
        }

    def eval_explainability(self, pred=None, attn=None, skip_aopc=False): 
        if isinstance(self.model,GCN): return None
        split = 'validation'
        if pred==None or attn==None:
            pred, attn = self.val_pred
        # prepare Comprehensiveness
        comp_ds = EraserCosE.erase(attn, mode='comprehensiveness', split=split)
        comp_edges = self.udparser(comp_ds, num_samples=len(comp_ds), split=split, qa_join=self.params['qa_join'], use_cache=False)
        for i,sample in enumerate(comp_ds):
            sample['qa_graphs'] = comp_edges[i]
        # prepare Sufficiency
        suff_ds = EraserCosE.erase(attn, mode='sufficiency', split=split)
        suff_edges = self.udparser(suff_ds, num_samples=len(suff_ds), split=split, qa_join=self.params['qa_join'], use_cache=False)
        for i,sample in enumerate(suff_ds):
            sample['qa_graphs'] = suff_edges[i]
        # predict
        comp_pred, _ = zip(*self.model(comp_ds, softmax_logits=True))
        suff_pred, _ = zip(*self.model(suff_ds, softmax_logits=True))

        # give option to do inter-train eval
        aopc_predictions = None
        if not skip_aopc: 
            # calcualte aopc metrics
            aopc_intermediate = {}
            for aopc in tqdm(self.params['aopc_thresholds'], desc='explainability_eval: '):
                tokens_to_be_erased = math.ceil(aopc * self.avg_rational_lengths[split])
                # comp
                cds = EraserCosE.erase(attn, mode='comprehensiveness', split=split, k=tokens_to_be_erased)
                ce = self.udparser(cds, num_samples=len(cds), split=split, qa_join=self.params['qa_join'], use_cache=False)
                for i,sample in enumerate(cds):
                    sample['qa_graphs'] = ce[i]
                cp, _ = zip(*self.model(cds, softmax_logits=True))
                cl = E.from_softmax(cp, to='dict') # labels
                # suff
                sds = EraserCosE.erase(attn, mode='sufficiency', split=split, k=tokens_to_be_erased)
                se = self.udparser(sds, num_samples=len(sds), split=split, qa_join=self.params['qa_join'], use_cache=False)
                for i,sample in enumerate(sds):
                    sample['qa_graphs'] = se[i]
                sp, _ = zip(*self.model(sds, softmax_logits=True))
                sl = E.from_softmax(sp, to='dict')
                # aggregate
                aopc_intermediate[aopc] = [aopc, cl, sl]
            aopc_predictions = E.reshape_aopc_intermediates(aopc_intermediate, self.params)

        doc_ids = self.val_set['id']
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=aopc_predictions)
        self.er_results = er_results
        return E.classification_scores(results=er_results, mode='val', aopc_thresholds=self.params['aopc_thresholds'], with_ids=doc_ids)

    def eval_efficiency(self): # TODO
        return None

    def viz(self):
        return True

    def save(self):
        # no save location, no saving
        if 'save_loc' in self.params: 
            # create save location
            if not os.path.exists(self.params['save_loc']):
                os.mkdir(self.params['save_loc'])
            # saving model
            torch.save(self.model.state_dict(), f"{self.params['save_loc']}/model.pt")

        # cache used glove embeddings
        if self.params['embedding'] == 'glove':
            with open(LOC['glove_cache'], 'w') as outfile:
                json.dump(self.model.embedding.cached_dict, outfile, sort_keys=True, indent=4)
        else:
            raise NotImplementedError('These embeddings have no implementation for caching yet')
        
        # saving evaluation
        with open(self.params['save_loc']+'evaluation.yaml', 'w') as file:
            documents = yaml.dump(self.eval_output, file)

        return True