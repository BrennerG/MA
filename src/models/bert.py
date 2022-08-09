import torch
import numpy as np
import lime

from tqdm import tqdm
from transformers import Pipeline
from transformers import AlbertForMultipleChoice, AlbertTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Union
from lime.lime_text import LimeTextExplainer

from data.locations import LOC
from data.huggingface_cose_old import EraserCosE # TODO change this back after huggingface is fixed and renamed


class BertPipeline(Pipeline):
    
    _DEFAULT_BASE = "albert-base-v2"

    def __init__(self, params:{}):

        
        if 'load_from' in params and params['load_from']: model = AlbertForMultipleChoice.from_pretrained(params['load_from'])
        else: model = AlbertForMultipleChoice.from_pretrained(self._DEFAULT_BASE)

        if 'bert_base' in params and params['bert_base']: self.tokenizer_base = params['bert_base']
        else: self.tokenizer_base = self._DEFAULT_BASE

        super().__init__(
            model = model,
            tokenizer = AlbertTokenizer.from_pretrained(self.tokenizer_base),
            feature_extractor=None,
            modelcard=None,
            framework=None,
            task = '',
            args_parser=None,
            device = -1,
            binary_output = False
        )
        self.trainer = None
        self.cached_inputs = None

    # see documentation in super (essentially it allows for dynamic parameter passing to the pipeline)
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forwards_kwargs = {}
        postprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        if 'attention' in kwargs:
            postprocess_kwargs['attention'] = kwargs['attention']
            # lime parameters
            if postprocess_kwargs['attention'] == 'lime':
                if 'lime_num_features' in kwargs: postprocess_kwargs['lime_num_features'] = kwargs['lime_num_features']
                if 'lime_num_permutations' in kwargs: postprocess_kwargs['lime_num_permutations'] = kwargs['lime_num_permutations']
        return preprocess_kwargs, forwards_kwargs, postprocess_kwargs

    def preprocess(self, inputs, maybe_arg=2):
        if isinstance(inputs, dict): # a single sample is entered
            inputs = [inputs] # ~unsqueeze
        self.cached_inputs = inputs # cache inputs for lime
        questions = [x['question'] for x in inputs]
        answers = [x['answers'] for x in inputs]
        rep_questions = sum([[question] * 5 for question in questions], [])
        rep_answers = sum(answers, [])
        encoding = self.tokenizer(rep_questions, rep_answers, return_tensors='pt', padding=True)
        model_inputs = {k: v.unsqueeze(0) for k, v in encoding.items()} 
        return model_inputs

    def _forward(self, model_inputs):         
        return self.model(**model_inputs, output_attentions=True)
        
    # TODO use transformer attention weights as attn_weights (requires some form of aggregation...)
    # TODO expand parameters
    # attention = {None, 'lime', 'attention', 'random}
    # output = {'proba', 'softmax', 'label', 'dict', 'intform', ...}
    def postprocess(self, model_outputs, attention='lime', output='proba', lime_num_features=30, lime_num_permutations=3):
        logits = model_outputs.logits.detach().numpy().T
        grouped = list(zip(*(iter(logits.flatten()),) * 5)) # group the predictions
        probas = np.array(grouped)

        # attention/weights
        attn_weights = None
        if attention == None:
            attn_weights = None
        elif attention == 'lime':
            attn_weights = self.lime_weights(lime_num_features, lime_num_permutations) # uses cached features
        elif attention == 'zeros':
            attn_weights = [len(x.split(" ")) for x in self.cached_inputs['question']]
            self.cached_inputs = None
        elif attention == 'random':
            raise NotImplementedError()
        elif attention == 'self_attn':
            raise NotImplementedError()
        else:
            raise AttributeError(f"attention mode {attention}")

        # output
        if output == 'proba': return probas, attn_weights
        else: raise AttributeError(f'output mode {output} unknown!')
    
    def load(self, path_to_checkpoint:str):
        loaded_model = AlbertForMultipleChoice.from_pretrained(path_to_checkpoint) 
        self.model = loaded_model
        return loaded_model

    # TODO can lime take a mask to ignore answers in the merge_string instead of merged inputs?
    def lime_weights(self, num_features=30, lime_num_permutations=3):
        # init
        explainer = LimeTextExplainer(class_names=['A','B','C','D','E'])
        ds_for_lime = EraserCosE.parse_to_lime(ds=self.cached_inputs)
        self.cached_inputs = None # reset cached data for safety

        # Multiplexer for (question, answer) -> 'questions+answers' = the prediction function for lime
        def clf_wrapper(input_str:str):
            answers = input_str[0].split('[qsep]')[1].split(' [sep] ')
            rep_questions = [item for item in input_str for i in range(5)]
            rep_answers = answers*len(input_str)
            # encode
            encoding = self.tokenizer(rep_questions, rep_answers, return_tensors='pt', padding=True)
            inputs = {k: v.unsqueeze(0) for k, v in encoding.items()}
            pred = self.model(**inputs, output_attentions=True)
            logits = pred.logits.detach().numpy().T
            grouped = list(zip(*(iter(logits.flatten()),) * 5)) # group the predictions
            return np.array(grouped)

        # get lime explanations
        weights = []
        for i,x in enumerate(tqdm(ds_for_lime)):
            exp = explainer.explain_instance(x, clf_wrapper, num_samples=lime_num_permutations, num_features=num_features, labels=list(range(5)))
            weights.append(exp.as_list())
        
        # bring weights into order of tokens
        ordered_weights = []
        for i,input_str in enumerate(ds_for_lime):
            q = ds_for_lime[i].split('[qsep]')[0].split()
            w = dict(weights[i])
            ordered = [w[p] if (p in q and p in w.keys()) else 0 for p in q ]
            ordered_weights.append(ordered)

        return ordered_weights

    def efficiency_metrics(self):
        cose = load_dataset(LOC['cose_huggingface'])
        tokenized_cose = cose.map(self.preprocess_function, batched=True, batch_size=8752)
        input_dict = {'input_ids':torch.Tensor(tokenized_cose['train']['input_ids'])}
        flops = self.model.floating_point_ops(input_dict, exclude_embeddings=False) # chapter 2.1 relevant for albert: https://arxiv.org/pdf/2001.08361.pdf
        if self.tokenizer_base == 'albert-base-v2':
            nr_params = 11000000 # according to https://huggingface.co/transformers/v3.3.1/pretrained_models.html
        else:
            nr_params = -1
        return flops, nr_params

    # only needed for training, yes I know its ugly
    def preprocess_function(self, examples):
        first_sentences = [[question]*5 for question in examples['question']]
        second_sentences = examples['answers']
        #flatten
        first_sentences = sum(first_sentences, []) # this flattens the lists :O
        second_sentences = sum(second_sentences, []) # not needed for predict

        tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True, padding=True)
        res = {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()} # rejoin to shape (8752 x 5 x unpadded_len)
        return res

    # TRAINING METHOD
    def train(self, dataset, train_args:TrainingArguments=None, debug_train_split=False):

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
        tokenized = dataset.map(self.preprocess_function, batched=True)
        overwrite = False

        # debug splits are tiny
        if debug_train_split: 
            train_data = tokenized['debug_train']
            val_data = tokenized['debug_val']
        else: 
            train_data = tokenized['train']
            val_data = tokenized['validation']

        # use trainer api for training
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=self.tokenizer)
        )

        trainer.train()
