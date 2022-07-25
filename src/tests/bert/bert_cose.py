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
import train as T
import eval as E


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

# TODO find a way to bring answers into prediction function without multiplexing ...
def eval_explainability():
    # prepare data
    cose = load_dataset('src/tests/bert/huggingface_cose.py')
    tokenized_cose = cose.map(preprocess_function, batched=True)
    ds_for_lime = EraserCosE.parse_to_lime(ds=tokenized_cose['validation'])

    # prepare models
    model = AlbertForMultipleChoice.from_pretrained("src/tests/bert/results/checkpoint-1641") 
    explainer = LimeTextExplainer(class_names=['A','B','C','D','E'])

    # Multiplexer for (question, answer) -> 'questions+answers' = the prediction function for lime
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

    # get lime explanations
    weights = []
    for i,x in enumerate(tqdm(ds_for_lime)):
        exp = explainer.explain_instance(x, clf_wrapper, num_samples=5, num_features=30, labels=list(range(5)))
        weights.append(exp.as_list())
    
    # bring weights into order of tokens
    ordered_weights = []
    for i,input_str in enumerate(ds_for_lime):
        q = ds_for_lime[i].split('[qsep]')[0].split()
        w = dict(weights[i])
        ordered = [w[p] if (p in q and p in w.keys()) else 0 for p in q ]
        ordered_weights.append(ordered)
    
    # the proper prediction function
    def predict(samples):
        questions = [x['question'] for x in samples]
        answers = [x['answers'] for x in samples]
        rep_questions = sum([[question] * 5 for question in questions], [])
        rep_answers = sum(answers, [])
        encoding = TOKENIZER(rep_questions, rep_answers, return_tensors='pt', padding=True)
        inputs = {k: v.unsqueeze(0) for k, v in encoding.items()}
        pred = model(**inputs, output_attentions=True)
        logits = pred.logits.detach().numpy().T
        grouped = list(zip(*(iter(logits.flatten()),) * 5)) # group the predictions
        return np.array(grouped)
    
    # aopc predictions (redundant with the one in src/train.py)
    def predict_aopc_thresholded(attn:[], aopc_thresholds=[0.01, 0.05, 0.1, 0.2, 0.5]):
        intermediate = {}
        result = []
        # erase for each aopc_thresh
        for aopc in aopc_thresholds:
            # comp
            comp_ds = EraserCosE.erase(ordered_weights, mode='comprehensiveness')
            comp_pred = predict(comp_ds)
            comp_labels = T.from_softmax(comp_pred, to='dict')
            # suff
            suff_ds = EraserCosE.erase(ordered_weights, mode='sufficiency')
            suff_pred = predict(suff_ds)
            suff_labels = T.from_softmax(suff_pred, to='dict')
            # aggregate
            intermediate[aopc] = [aopc, comp_labels, suff_labels]

        # transform results
        assert len(comp_ds) == len(suff_ds)
        for i in range(len(comp_ds)):
            sample = []
            for aopc in aopc_thresholds:
                sample.append({
                    'threshold': aopc,
                    "comprehensiveness_classification_scores": intermediate[aopc][1][i],
                    "sufficiency_classification_scores": intermediate[aopc][2][i],
                })
            result.append(sample)

        return result

    # calculate eraser metrics
    pred = predict(cose['validation']) # regular predictions
    comp_ds = EraserCosE.erase(ordered_weights, mode='comprehensiveness')
    suff_ds = EraserCosE.erase(ordered_weights, mode='sufficiency')
    comp_pred = predict(comp_ds)
    suff_pred = predict(suff_ds)
    aopc_predictions = predict_aopc_thresholded(ordered_weights)

    doc_ids = cose['validation']['id'] # TODO which one?
    er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, ordered_weights, aopc_thresholded_scores=aopc_predictions)
    agreement_auprc = E.soft_scores(er_results, docids=doc_ids, ds='cose_val')
    classification_scores = E.classification_scores(results=er_results, mode='val', aopc_thresholds=[0.01, 0.05, 0.1, 0.2, 0.5]) # TODO test this on full data

    print('done')