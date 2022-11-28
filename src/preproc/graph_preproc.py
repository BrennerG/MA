from abc import ABC, abstractmethod

class GraphPreproc(ABC):

    def __init__(self):
        self.root_token = '[ROOT]'

    def __call__(self, dataset, num_samples, split, qa_join, **kwargs):
        return self.parse(dataset, num_samples=num_samples, split=split, qa_join=qa_join, **kwargs)

    @abstractmethod
    def parse(self, dataset, num_samples, split, qa_join, use_cache=True):
        raise NotImplementedError()

    @abstractmethod
    def extract_edges(self, dataset, num_samples=-1, qa_join='none'):
        raise NotImplementedError()

    @abstractmethod
    def tokenize(self, sentence:str):
        raise NotImplementedError()

    @abstractmethod
    def show(self, edge_index, tokens):
        raise NotImplementedError()