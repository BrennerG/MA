import torch

from transformers import Pipeline
from transformers import AlbertForMultipleChoice, AlbertTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from dataclasses import dataclass
from typing import Optional, Union

class BertPipeline(Pipeline):

    def __init__(self):
        super().__init__(
            model = AlbertForMultipleChoice.from_pretrained("albert-base-v2"),
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2'),
            feature_extractor=None,
            modelcard=None,
            framework=None,
            task = '',
            args_parser=None,
            device = -1,
            binary_output = False
        )
        self.trainer = None

    # see documentation in super (essentially it allows for dynamic parameter passing to the pipeline)
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
    
    # TRAINING METHOD
    def train(self, dataset, train_args:TrainingArguments=None, save_loc=None):

        def preprocess_function(examples):
            first_sentences = [[question]*5 for question in examples['question']]
            second_sentences = examples['answers']
            #flatten
            first_sentences = sum(first_sentences, []) # this flattens the lists :O
            second_sentences = sum(second_sentences, []) # not needed for predict

            tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True, padding=True)
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
                batch["labels"] = torch.tensor(labels, dtype=torch.int64) # for training
                return batch

        # training method starts here!
        tokenized = dataset.map(preprocess_function, batched=True)
        overwrite = False

        if train_args == None:
            if save_loc == None:
                save_loc = 'data/experiments/default_bert/'
                overwrite = True
            # arguments for training
            training_args = TrainingArguments(
                output_dir=save_loc,
                evaluation_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=10,
                weight_decay=0.01,
                save_strategy='epoch',
                overwrite_output_dir=overwrite,
                no_cuda=True # TODO yes_cuda!!!
            )
        
        # use trainer api for training
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=self.tokenizer)
        )

        trainer.train()