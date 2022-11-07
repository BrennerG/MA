import os
import torch
import yaml

from datasets import load_dataset

from experiments.ud_gcn_experiment import UD_GCN_Experiment
from data.locations import LOC


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
        for i,split in enumerate(cose):
            cose[split] = cose[split].add_column('edges', flang_parse[i][0])
            cose[split] = cose[split].add_column('nodes_to_qa_tokens', flang_parse[i][1])
            cose[split] = cose[split].add_column('concept_ids', flang_parse[i][2])
        return cose, cose['train'], cose['validation'], cose['test']
    
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