import numpy as np

class RandomClassifier():
    
    def __init__(self, rnd_seed:int):
        np.random.seed(rnd_seed)
    
    def __call__(self, inputs, **params):
        if isinstance(inputs['question'], str):
            return self._call_single(inputs, params)
        elif isinstance(inputs['question'], list):
            return self._call_iterable(inputs, params)
    
    def _call_iterable(self, inputs, params):
        size = len(inputs['question'])
        tokens = [x.split() for x in inputs['question']]
        lens = [len(x) for x in tokens]
        num_choices = len(inputs['answers'][0])
        probas = [np.random.rand(num_choices) for x in range(size)]
        attention = [np.random.rand(l) for l in lens]
        return probas, attention
    
    def _call_single(self, inputs, params):
        tokens = inputs['question'].split()
        num_choices = len(inputs['answers'])
        probas = np.random.rand(num_choices)
        attention = np.random.rand(len(tokens))
        return probas, [attention]