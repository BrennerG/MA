import json
import datasets

import statistics as stat
import math as math
import numpy as np

import evaluation.eraserbenchmark.rationale_benchmark.utils as EU

from datasets import Dataset

""" ERASER Cos-E Dataset """

_CITATION = 'EMPTY'
_DESCRIPTION = "The Cos-E task of the ERASER Benchmark Suite."
_URL = None # 'http://www.eraserbenchmark.com/zipped/cose.tar.gz'
_FULL = 8752
_LIMIT = 11 # set debug split manually here
COSE_LOC = 'data/eraser/cose/'

class EraserCosEConfig(datasets.BuilderConfig):
    """BuilderConfig for EraserCosE."""

    def __init__(self, **kwargs):
       super(EraserCosEConfig, self).__init__(**kwargs)


class EraserCosE(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        EraserCosEConfig(
            name="cose",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                            datasets.Value("string")
                    ),
                    "label": datasets.Value("int32"),
                    "rationale": datasets.features.Features(
                        {
                            'docid': datasets.Value('string'),
                            'end_sentence': datasets.Value('int32'),
                            'end_token': datasets.Value('int32'),
                            'start_sentence': datasets.Value('int32'),
                            'start_token': datasets.Value('int32'),
                            'text': datasets.Value('string'),
                        }
                    )                
                }
            ),
            supervised_keys=None, # No default supervised_keys (as we have to pass both question and context as input).
            homepage='http://www.eraserbenchmark.com/',
            citation=_CITATION,
            task_templates=None
        )
    
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'mode': 'train'}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={'mode': 'val'}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'mode': 'test'}),
        ]

    def _generate_examples(self, mode):
        parselabel = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        # choosing split
        train, val, test = EU.load_datasets(COSE_LOC)
        if mode=='train': raw = train
        elif mode=='val': raw = val
        elif mode == 'test': raw = test
        else: raise AttributeError(f'CosE split {mode} unknown!')
        # extract
        docids = [x.annotation_id for x in raw]
        docs = EU.load_flattened_documents(COSE_LOC, docids)
        labels = [parselabel[x.classification] for x in raw]

        for i,X in enumerate(raw[:_LIMIT]):
            rationale = list(X.evidences)[0][0]
            question = " ".join(docs[X.annotation_id])
            if question[-1] != '?': question = question + ' ?'
            yield i, {
                'id': X.annotation_id,
                #'question': docs[X.annotation_id],
                'question': question,
                'context': X.query_type,
                'answers': X.query.split(' [sep] '),
                #'answers': X.query.replace(' [sep] ', ' '),
                'label': int(parselabel[X.classification]),
                'rationale': { 
                    'docid': rationale.docid,
                    'end_sentence': rationale.end_sentence,
                    'end_token': rationale.end_token,
                    'start_sentence': rationale.start_sentence,
                    'start_token': rationale.start_token,
                    'text': rationale.text
                }
            }

    @staticmethod
    def erase(weights:[], k=None, split='val', mode='comprehensiveness'): # modes = {'comprehensiveness', 'sufficiency'}
        # get correct split
        train, val, test = EU.load_datasets(COSE_LOC)
        if split=='train': raw = train
        elif split=='val' or split=='validation': raw = val
        elif split=='test': raw = test
        else: raise AttributeError(f'CosE split {mode} unknown!')
        # extract
        docids = [x.annotation_id for x in raw]
        docs = EU.load_flattened_documents(COSE_LOC, docids)
        parselabel = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        labels = [parselabel[x.classification] for x in raw]
        # take average evidence length if no k given!
        if k==None: 
            lens = []
            for evi in [list(X.evidences)[0][0] for X in raw[:_LIMIT]]:
                lens.append(evi.end_token - evi.start_token)
            k = math.ceil(stat.mean(lens))

        ret = []
        for i,X in enumerate(raw[:_LIMIT]):
            rationale = list(X.evidences)[0][0]
            question = docs[X.annotation_id]

            # adjust len of attention weights
            if len(weights[i]) > len(question): 
                attn = weights[i][:len(question)]
            elif len(weights[i]) == len(question): 
                attn = weights[i]
            else: 
                raise AttributeError(f"len(question)={len(question)} can't be longer than len(attention)={len(weights[i])}")

            # get top_k tokens
            if k > len(weights[i]): top_idx = np.array(range(len(weights[i])))
            else: top_idx = np.sort(np.argpartition(weights[i],-k)[-k:])
            # trim top_idx
            if max(top_idx) > len(question):
                top_idx = [x for x in top_idx if x<len(question)]

            # erase the top_k tokens
            if mode=='comprehensiveness':
                erased = [question[x] for x in range(len(question)) if x not in top_idx]
            elif mode=='sufficiency':
                erased = [question[x] for x in top_idx]
            else:
                raise AttributeError('evaluation mode unknown!')

            ret.append({
                'id': X.annotation_id,
                'question': " ".join(erased),
                'context': X.query_type,
                'answers': X.query.split(' [sep] '),
                'label': int(parselabel[X.classification]),
                'rationale': { 
                    'docid': rationale.docid,
                    'end_sentence': rationale.end_sentence,
                    'end_token': rationale.end_token,
                    'start_sentence': rationale.start_sentence,
                    'start_token': rationale.start_token,
                    'text': rationale.text
                }
            })
        return ret

    @staticmethod
    def parse_to_lime(ds:Dataset):
        strings = []
        questions = [d['question'] for d in ds]
        answers = [a['answers'] for a in ds]

        for q,a in zip(questions, answers):
            merged = f"{q}[qsep]{' [sep] '.join(a)}"
            strings.append(merged)

        return strings
    
    @staticmethod
    def avg_rational_length(ds:Dataset):
        result = {}
        for split in list(ds.keys()):
            r_lens = [r['end_token']-r['start_token'] for r in ds[split]['rationale']]
            result[split] = np.mean(r_lens)
        return result
