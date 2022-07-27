from models.bert import BertPipeline
from datasets import load_dataset
from tests.bert import bert_cose
import eval as E

#bert_cose.experiment()

cose = load_dataset('src/tests/bert/huggingface_cose.py')
pipe = BertPipeline()
# pipe.train(cose, debug_train_split=True)
pipe.load("src/tests/bert/results/22_7/checkpoint-5470")
probas, attn = pipe(cose['debug_val'])

# eval
comp = E.competence_metrics(cose['debug_val']['label'], probas)
expl = E.explainability_metrics_for_bert_model(pipe, cose['debug_val'])
print('done')