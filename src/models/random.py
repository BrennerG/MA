import numpy as np

class RandomClassifier():
    
    def __init__(self, rnd_seed:int):
        np.random.seed(rnd_seed)
    
    def __call__(self, inputs):
        size = len(inputs['question'])
        tokens = [x.split() for x in inputs['question']]
        lens = [len(x) for x in tokens]
        num_choices = len(inputs['answers'][0])
        probas = [np.random.rand(num_choices) for x in range(size)]
        attention = [np.random.rand(l) for l in lens]
        return probas, attention