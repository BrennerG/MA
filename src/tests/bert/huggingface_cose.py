import json
import datasets

from data_access.locations import LOC

import evaluation.eraserbenchmark.rationale_benchmark.utils as EU


""" ERASER Cos-E Dataset """

_CITATION = 'EMPTY'
_DESCRIPTION = "The Cos-E task of the ERASER Benchmark Suite."
_URL = None # 'http://www.eraserbenchmark.com/zipped/cose.tar.gz'
_FULL = 8752


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
        train, val, test = EU.load_datasets(LOC['cose'])
        if mode == 'train': raw = train
        elif mode == 'val': raw = val
        elif mode == 'test': raw = test
        # extract
        docids = [x.annotation_id for x in raw]
        docs = EU.load_flattened_documents(LOC['cose'], docids)
        labels = [parselabel[x.classification] for x in raw]
        # self.avg_evidence_len = round(np.mean([len(x[4].text.split()) for x in self])) # TODO calculate this differently

        for i,X in enumerate(raw):
            rationale = list(X.evidences)[0][0]
            yield i, {
                'id': X.annotation_id,
                #'question': docs[X.annotation_id],
                'question': " ".join(docs[X.annotation_id]),
                'context': X.query_type,
                'answers': X.query.split(' [sep] '),
                #'answers': X.query.replace(' [sep] ', ' '),
                'label': int(parselabel[X.classification]),
                'rationale': { # TODO this might be a bad idea if the original rationale Object is needed later
                    'docid': rationale.docid,
                    'end_sentence': rationale.end_sentence,
                    'end_token': rationale.end_token,
                    'start_sentence': rationale.start_sentence,
                    'start_token': rationale.start_token,
                    'text': rationale.text
                }
            }

    def erase(self, weights:[], k=None, mode='comprehensiveness'): # modes = {'comprehensiveness', 'sufficiency'}
        raise NotImplemented