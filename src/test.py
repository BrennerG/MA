from models.bert import BertPipeline
from datasets import load_dataset
from tests.bert import bert_cose

#bert_cose.experiment()

cose = load_dataset('src/tests/bert/huggingface_cose.py')
pipe = BertPipeline()
# pipe.train(cose, debug_train_split=True)
pipe.load("src/tests/bert/results/22_7/checkpoint-5470")
probas, attn = pipe(cose['debug_val'])

print('done')