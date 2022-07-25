# dataset loading
from datasets import load_dataset
from transformers import AutoTokenizer
from data_access.cose_dataset import CoseDataset
from .huggingface_cose import EraserCosE

# DataCollatorForMultipleChoice
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

# model
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers import AlbertForMultipleChoice, AlbertTokenizer

# evaluation competence
from evaluate import evaluator
import evaluate
import numpy as np

# evaluation explainability
import lime
from lime.lime_text import LimeTextExplainer
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm


'''
    A SIMPLE BERT PIPELINE FOR COSE
'''


TOKENIZER = AlbertTokenizer.from_pretrained('albert-base-v2') # or TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    first_sentences = [[question]*5 for question in examples['question']]
    second_sentences = examples['answers']
    #flatten
    first_sentences = sum(first_sentences, []) # this flattens the lists :O
    second_sentences = sum(second_sentences, []) # not needed for predict

    tokenized_examples = TOKENIZER(first_sentences, second_sentences, truncation=True, padding=True)
    res = {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()} # rejoin to shape (8752 x 5 x unpadded_len)
    return res

def lime_preprocess_function(examples):
    first_sentences = [examples['question']] * 5
    second_sentences = examples['answers']
    tokenized_examples = TOKENIZER(first_sentences, second_sentences, truncation=True, padding=True)
    res = {k:v for k,v in tokenized_examples.items()}
    return res

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features] # len=16
        batch_size = len(features) # len=16
        num_choices = len(features[0]["input_ids"]) # len=4
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ] 
        flattened_features = sum(flattened_features, []) # len=64

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64) # for training
        return batch


def train():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cose = load_dataset('src/tests/bert/huggingface_cose.py')
    tokenized_cose = cose.map(preprocess_function, batched=True)
    model = AlbertForMultipleChoice.from_pretrained("albert-base-v2") # or model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
    training_args = TrainingArguments(
        output_dir="src/tests/bert/results/22_7/",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy='epoch',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_cose["train"],
        eval_dataset=tokenized_cose["validation"],
        tokenizer=TOKENIZER,
        data_collator=DataCollatorForMultipleChoice(tokenizer=TOKENIZER),
    )
    trainer.train()
    print('done')

# takes ~4 minutes to do the predictions
def eval_competence():
    cose = load_dataset('src/tests/bert/huggingface_cose.py')
    tokenized_cose = cose.map(preprocess_function, batched=True)
    model = AlbertForMultipleChoice.from_pretrained("src/tests/bert/results/checkpoint-1641") 
    # metrics
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load('recall')
    # train
    trainer = Trainer(
        model=model,
        args=None,
        train_dataset=tokenized_cose["train"],
        eval_dataset=tokenized_cose["validation"],
        tokenizer=TOKENIZER,
        data_collator=DataCollatorForMultipleChoice(tokenizer=TOKENIZER),
    )
    # predict
    preds = trainer.predict(tokenized_cose['validation'])
    argmaxd = [np.argmax(x).item() for x in preds.predictions]
    acc = accuracy.compute(references=tokenized_cose['validation']['label'], predictions=argmaxd)
    prec = precision.compute(references=tokenized_cose['validation']['label'], predictions=argmaxd, average='weighted')
    reca = recall.compute(references=tokenized_cose['validation']['label'], predictions=argmaxd, average='weighted')

    # I could also do ... (probably better with multiple metrics!)
    # evaluation = trainer.evaluate(tokenized_cose['validation'])
    print(acc, prec, reca)

def eval_explainability():
    # TODO 
    # 1. ~~get LIME weights~~
    #   - ~~make it run~~
    #   - ~~wrap predict method~~
    #   - ~~get the weights!~~
    # 2. ERASER
    #   - fix weights, can't erase if attention is zeros
    #   - recreate eraser benchmark for huggingface dataset
    #   - reuse experiment.eval() interface(s)!
    # 3. META
    #   - isn't this the same setting as the logistic-regression fail?
    cose = load_dataset('src/tests/bert/huggingface_cose.py')
    tokenized_cose = cose.map(preprocess_function, batched=True)
    model = AlbertForMultipleChoice.from_pretrained("src/tests/bert/results/checkpoint-1641") 
    explainer = LimeTextExplainer(class_names=['A','B','C','D','E'])
    ds_for_lime = EraserCosE.parse_to_lime(ds=tokenized_cose['validation'])

    training_args = TrainingArguments(
        output_dir="src/tests/bert/results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy='epoch',
        log_level='critical'
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_cose["train"],
        eval_dataset=tokenized_cose["validation"],
        tokenizer=TOKENIZER,
        data_collator=DataCollatorForMultipleChoice(tokenizer=TOKENIZER),
    )
    # Multiplexer for (question, answer) -> 'questions+answers'
    def clf_wrapper(input_str:str):
        if isinstance(input_str, list) and len(input_str) == 1: input_str = input_str[0]
        question, answers = input_str.split('[qsep]') # define this somewhere!
        answers = answers.split(' [sep] ')
        mapping = {
            'id': ['none'],
            'question': [question],
            'context': ['none'],
            'answers': [answers],
            'label': [0],
            'rationale': [{
                'docid': "",
                'end_sentence': -1,
                'end_token': -1,
                'start_sentence': -1,
                'start_token': -1,
                'text': 'none',

            }]
        }

        dataset = Dataset.from_dict(mapping=mapping) # dataset = Dataset.from_dict(mapping=mapping, features=EraserCosE._info().features, info=EraserCosE._info())
        tokenized_data = dataset.map(lime_preprocess_function)
        pred = trainer.predict(tokenized_data)
        return pred.predictions

    weights = []
    for i,x in enumerate(tqdm(ds_for_lime)):
        exp = explainer.explain_instance(x, clf_wrapper, num_features=30, num_samples=1)
        weights.append(exp.as_list())

    print('done')


def experiment():
    # prepare data
    cose = load_dataset('src/tests/bert/huggingface_cose.py')
    tokenized_cose = cose.map(preprocess_function, batched=True)
    ds_for_lime = EraserCosE.parse_to_lime(ds=tokenized_cose['validation'])
    # prepare models
    model = AlbertForMultipleChoice.from_pretrained("src/tests/bert/results/checkpoint-1641") 
    explainer = LimeTextExplainer(class_names=['A','B','C','D','E'])

    # Multiplexer for (question, answer) -> 'questions+answers'
    def old_wrapper(input_str:str):
        if isinstance(input_str, list) and len(input_str) == 1: input_str = input_str[0]
        # prep
        question, answers = input_str.split('[qsep]') # define this somewhere!
        answers = answers.split(' [sep] ')
        # encode
        encoding = TOKENIZER([question]*5, answers, return_tensors='pt', padding=True)
        inputs = {k: v.unsqueeze(0) for k, v in encoding.items()}
        pred = model(**inputs, output_attentions=True)
        return pred.logits.detach().numpy()

    def clf_wrapper(input_str:str):
        answers = input_str[0].split('[qsep]')[1].split(' [sep] ')
        rep_questions = [item for item in input_str for i in range(5)]
        rep_answers = answers*len(input_str)
        # encode
        encoding = TOKENIZER(rep_questions, rep_answers, return_tensors='pt', padding=True)
        inputs = {k: v.unsqueeze(0) for k, v in encoding.items()}
        pred = model(**inputs, output_attentions=True)
        logits = pred.logits.detach().numpy().T
        grouped = list(zip(*(iter(logits.flatten()),) * 5)) # group the predictions
        return np.array(grouped)

    weights = []
    for i,x in tqdm(enumerate(ds_for_lime)):
        exp = explainer.explain_instance(x, clf_wrapper, num_samples=10, num_features=30, labels=list(range(5)))
        weights.append(exp.as_list())

    print('done')