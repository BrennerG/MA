from models.bert import BertPipeline
from datasets import load_dataset

cose = load_dataset('src/tests/bert/huggingface_cose.py')
pipe = BertPipeline()
# pipe.train(cose, debug_train_split=True)
pipe.load("src/tests/bert/results/checkpoint-1641")
pipe(cose['validation'])

print('done')