# dataset loading
from datasets import load_dataset
from transformers import AutoTokenizer
from .huggingface_cose import EraserCosE

# DataCollatorForMultipleChoice
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

# model
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers import AlbertForMultipleChoice, AlbertTokenizer

'''
    A SIMPLE BERT PIPELINE FOR COSE
'''


# TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER = AlbertTokenizer.from_pretrained('albert-base-v2')

def preprocess_function(examples):
    first_sentences = [[question]*5 for question in examples['question']]
    second_sentences = examples['answers']
    #flatten
    first_sentences = sum(first_sentences, []) # this flattens the lists :O
    second_sentences = sum(second_sentences, [])

    tokenized_examples = TOKENIZER(first_sentences, second_sentences, truncation=True)
    res = {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()} # rejoin to shape (8752 x 5 x unpadded_len)
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
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def run():
    cose = load_dataset('src/bert_cose_sanity/huggingface_cose.py')
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # is now global
    tokenized_cose = cose.map(preprocess_function, batched=True)
    #model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
    model = AlbertForMultipleChoice.from_pretrained("albert-base-v2")

    training_args = TrainingArguments(
        output_dir="src/tests/bert/results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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