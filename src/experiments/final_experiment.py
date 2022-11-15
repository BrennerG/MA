import os
import torch
import yaml

from datasets import load_dataset

from experiments.ud_gcn_experiment import UD_GCN_Experiment
from data.locations import LOC
from data.huggingface_cose import EraserCosE
import evaluation.eval_util as E


# 4LANG + BERT + GAT
class FinalExperiment(UD_GCN_Experiment):

    def __init__(self, params:{}):
        # HARD OVERWRITES
        params['graph_form'] = '4lang'
        params['embedding'] = 'albert-base-v2'
        params['model_type'] = 'BERT_GAT'
        super().__init__(params)
        self.model.GP = self.graph_parser

    def init_data(self):
        cose = load_dataset(LOC['cose_huggingface'])
        # add graph edges as new cols to the dataset
        flang_parse = [self.graph_parser(ds, num_samples=len(ds), split=split, qa_join=self.params['qa_join']) for (split,ds) in cose.items()]
        self.graph_parser.save_concepts() # TODO put this in save?
        for i,split in enumerate(cose):
            cose[split] = cose[split].add_column('edges', flang_parse[i][0])
            cose[split] = cose[split].add_column('nodes_to_qa_tokens', flang_parse[i][1])
            cose[split] = cose[split].add_column('concept_ids', flang_parse[i][2])
        return cose, cose['train'], cose['validation'], cose['test']

    def eval_explainability(self, pred=None, attn=None, skip_aopc=False): 
        split = 'validation' # TODO change
        if pred==None or attn==None:
            pred, attn = self.val_pred

        # prepare Comprehensiveness
        comp_ds = EraserCosE.erase(attn, mode='comprehensiveness', split=split)
        comp_edges, comp_nodes_to_qa_tokens, comp_concept_ids = self.graph_parser(comp_ds, num_samples=len(comp_ds), split=split, qa_join=self.params['qa_join'], use_cache=False)
        for i,sample in enumerate(comp_ds):
            sample['edges'] = comp_edges[i]
            sample['nodes_to_qa_tokens'] = comp_nodes_to_qa_tokens[i]
            sample['concept_ids'] = comp_concept_ids[i]

        # prepare Sufficiency
        suff_ds = EraserCosE.erase(attn, mode='sufficiency', split=split)
        suff_edges, suff_nodes_to_qa_tokens, suff_concept_ids = self.graph_parser(suff_ds, num_samples=len(suff_ds), split=split, qa_join=self.params['qa_join'], use_cache=False)
        for i,sample in enumerate(suff_ds):
            sample['edges'] = suff_edges[i]
            sample['nodes_to_qa_tokens'] = suff_nodes_to_qa_tokens[i]
            sample['concept_ids'] = suff_concept_ids[i]

        # erased Predictions
        comp_pred, _ = zip(*self.model(comp_ds, softmax_logits=True))
        suff_pred, _ = zip(*self.model(suff_ds, softmax_logits=True))

        doc_ids = self.val_set['id']
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=None) # TODO aopcs?
        self.er_results = er_results
        # return E.classification_scores(results=er_results, mode='val', aopc_thresholds=self.params['aopc_thresholds'], with_ids=doc_ids)
        return E.classification_scores(results=er_results, mode='custom', aopc_thresholds=self.params['aopc_thresholds'], with_ids=doc_ids)

    def save(self):
        # no save location, no saving
        if 'save_loc' in self.params: 
            # create save location
            if not os.path.exists(self.params['save_loc']):
                os.mkdir(self.params['save_loc'])
            # saving model
            torch.save(self.model.state_dict(), f"{self.params['save_loc']}/model.pt")

        # saving evaluation
        with open(self.params['save_loc']+'evaluation.yaml', 'w') as file:
            documents = yaml.dump(self.eval_output, file)

        return True