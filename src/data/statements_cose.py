import json
from data.locations import LOC


class StatementLoader():

    def __init__(self):
        self.data = {}
        self.data.update(self.load('train.statement.jsonl'))
        self.data.update(self.load('dev.statement.jsonl'))
        self.data.update(self.load('test.statement.jsonl'))

    def load(self, filename:str):
        split = {}
        with open(LOC['statements_folder']+filename,'r') as f:
            for line in f:
                x = json.loads(line)
                sample = []
                for s in x['statements']:
                    sample.append(s['statement'])
                split[x['id']] = sample
 
        return split

    def add_statements(self, dataset):
        new_cols = {x:[] for x in dataset}
        for split in dataset:
            for sample in dataset[split]:
                if sample['id'] in self.data:
                    new_cols[split].append(
                        self.data[sample['id']]
                    )
                else:
                    new_cols[split].append(None)

            # add new cols to dataset
            dataset[split] = dataset[split].add_column('statements', new_cols[split])
        return dataset