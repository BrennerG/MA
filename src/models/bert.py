import torch
import numpy as np
import lime

from tqdm import tqdm
from transformers import Pipeline
from transformers import AlbertForMultipleChoice, AlbertTokenizer
from transformers import RobertaTokenizer, RobertaForMultipleChoice
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Union
from lime.lime_text import LimeTextExplainer

from data.locations import LOC
from data.huggingface_cose import EraserCosE 

from sklearn.preprocessing import MinMaxScaler


class BertPipeline(Pipeline):
    
    _DEFAULT_BASE = "albert-base-v2"

    def __init__(self, params:{}):

        if params['bert_base'] == 'albert-base-v2':
            _tokenizer = AlbertTokenizer
            _model = AlbertForMultipleChoice
        elif params['bert_base'] == 'roberta-base':
            _tokenizer = RobertaTokenizer
            _model = RobertaForMultipleChoice
        elif params['bert_base'] == 'bert-base-uncased' or params['bert_base'] == 'bert-large-uncased':
            _tokenizer = AutoTokenizer
            _model = AutoModelForMultipleChoice
        else:
            raise AttributeError('Error: BERT Base unknown!')

        # load model from pretrained or checkpoint
        if 'load_from' in params and params['load_from']: 
            model = _model.from_pretrained(params['load_from'])
        else: 
            model = _model.from_pretrained(params['bert_base'])

        super().__init__(
            model = model,
            tokenizer = _tokenizer.from_pretrained(params['bert_base']),
            feature_extractor=None,
            modelcard=None,
            framework=None,
            task = '',
            args_parser=None,
            device = 0 if ('use_cuda' in params and params['use_cuda']) else -1,
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
                if 'lime_scaling' in kwargs: postprocess_kwargs['lime_scaling'] = kwargs['lime_scaling']
        if 'softmax_logits' in kwargs and kwargs['softmax_logits']:
            forwards_kwargs['softmax_logits'] = True
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
        model_inputs = {k: v.unsqueeze(0).to(self.device) for k, v in encoding.items()} 
        return model_inputs

    def _forward(self, model_inputs, softmax_logits=False):
        output = self.model(**model_inputs, output_attentions=True)
        if softmax_logits:
            output.logits = torch.softmax(output.logits,1)
        return output
        
    def postprocess(self, model_outputs, attention='lime', output='proba', lime_num_features=-1, lime_num_permutations=5000, lime_scaling='minmax'):
        logits = model_outputs.logits.detach().numpy().T
        grouped = list(zip(*(iter(logits.flatten()),) * 5)) # group the predictions
        probas = np.array(grouped)

        # attention/weights
        attn_weights = None
        if attention == None:
            attn_weights = None
        elif attention == 'lime':
            attn_weights = self.lime_weights( # uses cached features
                predicted_label=np.argmax(probas), 
                num_features=lime_num_features, 
                num_permutations=lime_num_permutations, 
                scaling=lime_scaling) 
        elif attention == 'zeros':
            attn_weights = [np.zeros(len(x['question'].split())) for x in self.cached_inputs]
        elif attention == 'random':
            attn_weights = [np.random.rand(len(x['question'].split())) for x in self.cached_inputs]
        elif attention == 'self_attention':
            # TODO find another way to aggregate?
            last_layers_attn = model_outputs.attentions[-1]
            attn_for_predicted_label = last_layers_attn[np.argmax(probas)]
            mean_of_heads = torch.mean(attn_for_predicted_label, 0) # take mean of all attention heads
            sum_heads = torch.sum(mean_of_heads, 0)
            l = len(self.cached_inputs[0]['question'].split())
            attn_weights = torch.softmax(sum_heads[:l],0).unsqueeze(0).numpy()
        else:
            raise AttributeError(f"attention mode {attention}")

        # output
        self.cached_inputs = None # reset cached_inputs for cleanliness
        if output == 'proba': return probas, attn_weights
        else: raise AttributeError(f'output mode {output} unknown!')
    
    def load(self, path_to_checkpoint:str):
        loaded_model = AlbertForMultipleChoice.from_pretrained(path_to_checkpoint) 
        self.model = loaded_model
        return loaded_model

    # TODO can lime take a mask to ignore answers in the merge_string instead of merged inputs?
    # attribute 'mask_string'?
    # >>> https://lime-ml.readthedocs.io/en/latest/lime.html#subpackages
    def lime_weights(self, predicted_label, num_features=30, num_permutations=3, scaling='minmax'):
        # init
        if num_features < 1: lime_feature_selection = 'none'
        else: lime_feature_selection = 'auto'
        explainer = LimeTextExplainer(class_names=['A','B','C','D','E'], feature_selection=lime_feature_selection)
        ds_for_lime = EraserCosE.parse_to_lime(ds=self.cached_inputs) 

        # Multiplexer for (question, answer) -> 'questions+answers' = the prediction function for lime
        def clf_wrapper(input_str:str):
            answers = input_str[0].split('[qsep]')[1].split('[sep]')
            rep_questions = [item for item in input_str for i in range(5)]
            rep_answers = answers*len(input_str)
            # TODO IDEA: if labels are a single token - remove it or replace it (with a marker token e.g.:"[SOL]"")
            # encode
            encoding = self.tokenizer(rep_questions, rep_answers, return_tensors='pt', padding=True)
            inputs = {k: v.view(len(input_str), 5, -1) for k,v in encoding.items()}
            preds = np.zeros((len(input_str),5))
            for i in range(len(input_str)):
                out = self.model(**{k:v[i].unsqueeze(0).to(self.device) for k,v in inputs.items()}, output_attentions=True)
                preds[i] = out.logits.squeeze().detach().cpu().numpy()
            return preds

        # get lime explanations
        weights = []
        for i,x in enumerate(ds_for_lime):
            exp = explainer.explain_instance(x, clf_wrapper, num_samples=num_permutations, num_features=num_features, labels=list(range(5)))
            weights.append(exp.as_list(predicted_label))

            # show explanations like this (only during manual debugging)
            # import matplotlib.pylab as plt
            # plt.figure()
            # exp.as_pyplot_figure()
            # plt.show()
        
        # bring weights into order of tokens
        ordered_weights = []
        for i,input_str in enumerate(ds_for_lime):
            q = ds_for_lime[i].split('[qsep]')[0].split()
            w = dict(weights[i])
            ordered = [w[p] if (p in q and p in w.keys()) else 0 for p in q ]
            # normalize
            if scaling == 'softmax':
                ordered_weights.append(torch.softmax(torch.Tensor(ordered),0).tolist())
            elif scaling == 'normalize':
                ordered_weights.append(np.array(ordered) / np.linalg.norm(ordered))
            else:
                ordered_weights.append(ordered)

        # normalize ctd
        if scaling == 'minmax':
            scaler = MinMaxScaler()
            transposed = [[x] for sublist in ordered_weights for x in sublist]
            scaled_weights = scaler.fit_transform(transposed).T
        elif scaling == 'none' or scaling == None:
            scaled_weights = np.array(ordered_weights)
        else:
            raise AttributeError(f"LIME: scaling method '{scaling}' is unknown!")

        return scaled_weights

    def efficiency_metrics(self, params:{}):
        cose = load_dataset(LOC['cose_huggingface'])
        tokenized_cose = cose.map(self.preprocess_function, batched=True, batch_size=8752)
        input_dict = {'input_ids':torch.Tensor(tokenized_cose['train']['input_ids'])}
        flops = self.model.floating_point_ops(input_dict, exclude_embeddings=False) # chapter 2.1 relevant for albert: https://arxiv.org/pdf/2001.08361.pdf
        if params['bert_base'] == 'albert-base-v2':
            nr_params = 11000000 # according to https://huggingface.co/transformers/v3.3.1/pretrained_models.html
        if params['bert_base'] == 'roberta-base':
            nr_params = 125000000
        if params['bert_base'] == 'bert-base-uncased':
            nr_params = 110000000
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
    def train(self, dataset, train_args:TrainingArguments=None):

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

        # use trainer api for training
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=tokenized['train'],
            eval_dataset=tokenized['validation'],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=self.tokenizer)
        )

        trainer.train()
