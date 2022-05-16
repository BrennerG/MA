import json
import pandas as pd
import torch
from torch.utils.data import Dataset

# TODO would transforms be useful here? (e.g. transforming to BERT, masking, ...)
class CsqaDataset(Dataset):

    def __init__(self, path_to_raw:str, limit=-1):
        self.location = path_to_raw
        self.data = []

        with open(path_to_raw, 'r') as json_file:
            json_list = list(json_file)

            #for json_str in json_list:
            if limit > 0:
                for json_str in json_list[:limit]:
                    result = json.loads(json_str)
                    self.data.append(result)
            else:
                for json_str in json_list:
                    result = json.loads(json_str)
                    self.data.append(result)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']['stem']
        answer = [x['text'] for x in self.data[idx]['question']['choices']]
        context = self.data[idx]['question']['question_concept']
        label_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        label = label_map[self.data[idx]['answerKey']]
        #label_tensor = torch.zeros(5)
        #label_tensor[label] = 1
        return question, context, answer, float(label)
    
    def get_labels(self, limit):
        for i in range(len(self)):
            yield self.__getitem__(i)[3]