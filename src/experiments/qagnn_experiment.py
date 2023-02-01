import os
import torch
import wandb
import random
import numpy as np
import torch
import wandb

from argparse import Namespace
import torch.nn as nn
try: from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except: from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForMaskedLM
from tqdm import tqdm
from copy import deepcopy
from torchmetrics import Accuracy, Recall, Precision
from collections import OrderedDict

from experiments.final_experiment import FinalExperiment
from data.locations import LOC
from data.huggingface_cose import EraserCosE
from models.qagnn.utils.utils import *
from models.qagnn.utils.optimization_utils import OPTIMIZER_CLASSES
from models.qagnn.modeling_qagnn import *
from datasets import load_dataset
from data.locations import LOC
from data.statements_cose import StatementLoader
import evaluation.eval_util as E


class QagnnExperiment(FinalExperiment):

    def __init__(self, params:{}):
        # load
        if 'load_from' in params:
            model_path = params['load_from']
            _ , old_args = torch.load(model_path, map_location=torch.device('cpu'))
            self.params = old_args.__dict__
            # so that params can be taken from the args** as overwrites
            self.params['wandb_logging'] = params['wandb_logging']
            self.params['skip_training'] = params['skip_training']
            self.params['node_relevance'] = params['node_relevance']
        else:
            self.params = params
            # class specific overwrites
            self.params['offset_concepts'] = True
            self.params['max_num_nodes'] = params['max_node_num'] # TODO stupid synonym
            self.params['expand'] = None # TODO why do I have to do this?

        torch.manual_seed(self.params['rnd_seed'])
        wandb.init(
            config=self.params, 
            mode='online' if self.params['wandb_logging']==True else 'disabled',
            project='qagnn_4lang'
        )

        # set devices
        self.device0 = torch.device('cuda:0') # encoder
        self.device1 = torch.device('cpu') # decoder

        # prepare graph parser
        self.graph_parser = self._graph_parser_factory()
        # general stuff
        self.complete_set, self.train_set, self.val_set, self.test_set = self.init_data()
        self.params['num_concepts'] = max(self.graph_parser.id2concept)
        self.model = self.model_factory(self.params['model_type']) # .to(self.device) # TODO shift device assignment from train() to here
        # TODO uncomment if we do k-erasing
        #raw_cose = load_dataset(LOC['cose_huggingface'])
        #self.avg_rational_lengths = EraserCosE.avg_rational_length(raw_cose)

    def __call__(self):

        # TRAINING
        if not('skip_training' in self.params and self.params['skip_training']): # no skip
            print('training...')
            self.train_output = self.train()
        else: # skip training
            if not 'load_from' in self.params or self.params['load_from'] == None:
                print('WARNING: training will be skipped, but no checkpoint was given (load_from) parameter (=prediction with only pre-trained model)')
            else:
                print(f"MODEL PRELOADED FROM {self.params['load_from']} - SKIPPING TRAINING!") # this already happened in experiment.model_factory()

        # PREDICTION
        # EVALUATION
        if not('skip_evaluation' in self.params and self.params['skip_evaluation']): # no skip
            print('predicting...')
            # need to softmax logits for evaluation (actually only ERASER)
            prediction_params = deepcopy(self.params)
            prediction_params['softmax_logits'] = True
            #prediction_params['save'] = 'PREDS_DATA.jsonl'
            preds = []
            with torch.no_grad():
                for qids, labels, *input_data in tqdm(self.val_set):
                    preds.append(self.model(*input_data, **prediction_params))

                # DEBATCH
                _val_pred = list(zip(*preds))
                self.val_pred = [None] * 2
                self.val_pred[0] = torch.cat(_val_pred[0], dim=0) # debatch logits

                # DEBATCH AND AVERAGE ATTENTION # TODO mb turn this into a function
                attn = [x.view(-1,5,2,200) for x in _val_pred[1]] # reshape entries # TODO de-hardcode
                attn = torch.cat(attn,dim=0) # concat to sample as first dim
                qa_lens = [[len(a.split()) for a in x['statements']] for x in self.dev_statements]
                self.val_pred[1] = []
                for _attn, _map, _qa_lens in zip(attn, self.flang_dev[1], qa_lens):
                    _attn = _attn.mean(dim=1)
                    grouped_attn = []
                    for a, m, l in zip(_attn, _map, _qa_lens):
                        if any([_ for _ in m if _ != None]):
                            assert max([_ for _ in m if _ != None]) < l
                        attn_vec = [.0] * l
                        for i,p in enumerate(m):
                            if p != None:
                                attn_vec[p] = a[i].item()
                        grouped_attn.append(attn_vec)
                    self.val_pred[1].append(grouped_attn)

            # evaluating
            print('evaluating...')
            self.eval_output = self.evaluate()
        else: # skip evaluation
            print('SKIPPING EVALUATION (flag was set in param dict!)')
            self.val_pred = None
            self.eval_output = None
        
        # VIZ
        print('saving viz data for qualitative analysis...')
        self.viz_output = self.qualitative_analysis()

        # SAVE / PERSIST
        print('saving...')
        self.save()

        print('experiment done!')
        if self.params['wandb_logging']:
            wandb.run.summary['eval_output'] = self.eval_output
        else:
            print(self.eval_output)
        return self
    
    def qualitative_analysis(self):
        dataset = self.complete_set

        # PREDICT
        prediction_params = deepcopy(self.params)
        # prediction_params['softmax_logits'] = True # TODO neecessary?
        prediction_params['detail'] = True
        preds = []
        with torch.no_grad():
            for qids, labels, *input_data in tqdm(dataset.test()):
                preds.append(self.model(*input_data, **prediction_params))

        # DEBATCH
        _val_pred = list(zip(*preds)) # 0:logits, 1:attn, 2:concept_ids, 3:node_type_ids, 4:edge_index_orig, 5:edge_type_orig
        logits, _, concept_ids, node_type_ids, edge_index_orig, edge_type_orig = [[item for sublist in list for item in sublist] for list in _val_pred] # flatten
        attn = torch.cat(_val_pred[1]).view(1238,2,5,200)

        # ADD SOME MORE FLANG DATA
        edges, node2qa_map, concept_names = self.flang_test
        id2concept = self.graph_parser.id2concept

        # PARSE NODE_SCORES TO FLOAT
        if self.node_relevance_test != None:
            for X in self.node_relevance_test:
                for A in X:
                    for key, value in A.items():
                        A[key] = float(value)

        # create a beautiful bulky dictionary
        return {
            'statement_data':self.test_statements,
            'node_scores': self.node_relevance_test,
            'logits': [x.tolist() for x in logits],
            'attn': attn.tolist(),
            'concept_ids': [x.tolist() for x in concept_ids],
            'node_type_ids': [x.tolist() for x in node_type_ids],
            'edge_index_orig': [[x.tolist() for x in edges] for edges in edge_index_orig],
            'edge_type_orig': [[x.tolist() for x in edges] for edges in edge_type_orig],
            '4L_edges': edges,
            '4L_map': node2qa_map,
            '4L_concept_names': concept_names,
            '4L_id2concept': id2concept
        }

    def parse_params_to_args(self, params, **kwargs):
        args = Namespace(**params)

        # overwrite
        for arg in kwargs:
            print('!')

        return args

    def load_qagnn_dataset(self, **kwargs):
        args = self.parse_params_to_args(self.params)
        args.dev_statements = kwargs['dev_statements'] if 'dev_statements' in kwargs else args.dev_statements
        args.train_adj = None
        args.dev_adj = None
        args.test_adj = None

        # LOAD DATA
        dataset = LM_QAGNN_DataLoader(
            args,
            args.train_statements,
            args.train_adj,
            args.dev_statements,
            args.dev_adj,
            args.test_statements,
            args.test_adj,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            device=(self.device0, self.device1),
            model_name=args.encoder,
            max_node_num=args.max_node_num,

            is_inhouse=False,
            #inhouse_train_qids_path="data/qa_gnn/inhouse_split_qids.txt"
            #max_seq_length=args.max_seq_len,)
            #subsample=args.subsample,  # 0.0
            #use_cache=args.use_cache) # True
        )
        return dataset

    def init_data(self):
        args = self.parse_params_to_args(self.params)
        args.train_adj = None
        args.dev_adj = None
        args.test_adj = None
        add_edge_types = args.num_relation == 3
        node_relevance = args.node_relevance
        
        # LOAD DATA
        dataset = LM_QAGNN_DataLoader(
            args,
            args.train_statements,
            args.train_adj,
            args.dev_statements,
            args.dev_adj,
            args.test_statements,
            args.test_adj,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            device=(self.device0, self.device1),
            model_name=args.encoder,
            max_node_num=args.max_node_num,

            is_inhouse=False,
            #inhouse_train_qids_path="data/qa_gnn/inhouse_split_qids.txt"

            #max_seq_length=args.max_seq_len,)
            #subsample=args.subsample,  # 1.0
            #use_cache=args.use_cache) # True
        )

        self.train_statements = self.prepare_qagnn_data(path=LOC['qagnn_statements_train'])
        self.dev_statements = self.prepare_qagnn_data(path=LOC['qagnn_statements_dev'])
        self.test_statements = self.prepare_qagnn_data(path=LOC['qagnn_statements_test'])

        # parse all splits
        self.flang_train, self.flang_dev, self.flang_test = [
            self.graph_parser(
                ds, 
                num_samples=len(ds), 
                split=split, 
                qa_join=self.params['qa_join'], 
                use_cache=args.use_cache,
                max_num_nodes=args.max_node_num,
                expand=args.expand,
                add_edge_types=add_edge_types
            ) for (split, ds) in zip(['train','dev','test'], [self.train_statements, self.dev_statements, self.test_statements])
        ]

        self.graph_parser.save_concepts() # TODO put this in save?

        # node relevance scoring
        if node_relevance:
            self.node_relevance_train = self.node_relevance_scoring(self.flang_train, self.train_statements)
            self.node_relevance_dev = self.node_relevance_scoring(self.flang_dev, self.dev_statements)
            self.node_relevance_test = self.node_relevance_scoring(self.flang_test, self.test_statements)
        else:
            self.node_relevance_train = None
            self.node_relevance_dev = None
            self.node_relevance_test = None

        # add adj data to dataset
        *dataset.train_decoder_data, dataset.train_adj_data = self.add_4lang_adj_data(target_flang=self.flang_train, target_set=self.train_statements, add_edge_types=add_edge_types, relevance_scores=self.node_relevance_train)
        *dataset.dev_decoder_data, dataset.dev_adj_data = self.add_4lang_adj_data(target_flang=self.flang_dev, target_set=self.dev_statements, add_edge_types=add_edge_types, relevance_scores=self.node_relevance_dev)
        *dataset.test_decoder_data, dataset.test_adj_data = self.add_4lang_adj_data(target_flang=self.flang_test, target_set=self.test_statements, add_edge_types=add_edge_types, relevance_scores=self.node_relevance_test)

        return dataset, dataset.train(), dataset.dev(), dataset.test()
    
    def node_relevance_scoring(self, flang, set):
        TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
        LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
        LM_MODEL.cuda()
        LM_MODEL.eval()

        results = []
        for s, (X, concepts) in tqdm(enumerate(zip(set, flang[2])), desc="node relevance scoring"):
            grouped_results = []
            for a,qac in enumerate(concepts):

                # Tokenize
                sents = []
                qac.insert(0, "ab_extra")
                for c, conc in enumerate(qac):
                    if conc == "ab_extra": sent = X['question']
                    else: sent = f"{X['question'].lower()} {conc}."
                    sents.append(TOKENIZER.encode(sent, add_special_tokens=True))
            
                # Rank in Batches
                scores = []
                n_cids = len(qac)
                cur_idx = 0
                batch_size = 50
                while cur_idx < n_cids:

                    # Prepare batch
                    input_ids = sents[cur_idx: cur_idx+batch_size]
                    max_len = max([len(seq) for seq in input_ids])
                    for j, seq in enumerate(input_ids):
                        seq += [TOKENIZER.pad_token_id] * (max_len-len(seq))
                        input_ids[j] = seq
                    input_ids = torch.tensor(input_ids).cuda() #[B, seqlen]
                    mask = (input_ids!=1).long() #[B, seq_len]

                    # Get LM score
                    with torch.no_grad():
                        outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
                        loss = outputs[0] #[B, ]
                        _scores = list(-loss.detach().cpu().numpy()) #list of float
                    scores += _scores
                    cur_idx += batch_size

                assert len(sents) == len(scores) == len(qac)
                cid2score = OrderedDict(sorted(list(zip(qac, scores)), key=lambda x: -x[1])) #score: from high to low
                grouped_results.append(cid2score)

            results.append(grouped_results)

        '''
        # neural scoring
        results = []
        for _sent in tqdm(sents, desc="node relevance scoring - lm scoring"):
            with torch.no_grad():
                input_ids = torch.tensor(_sent).unsqueeze(0).cuda()
                mask = torch.ones(input_ids.shape).long().cuda()
                score, _, _ = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
                results.append(score.item())
        '''
        return results
    
    # load data used by qagnn and parse into compatible format
    def prepare_qagnn_data(self, path):
        parselabel = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        result = []
        sl = StatementLoader()

        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        for i,X in enumerate(data):
            question = X['question']['stem']
            if len(question) > 0 and question[-1] != '?': question = question + ' ?'
            result.append({
                'id': X['id'],
                'question': question,
                'context': X['question']['question_concept'],
                'answers': [x['text'] for x in X['question']['choices']],
                'label': int(parselabel[X['answerKey']]),
                'statements': sl.data[X['id']]
            })
        
        return result
           
    # this overwrites the graph data in the qagnn dataloader with 4lang graph data
    # overwrite: *dataset.{split}_decoder_data, dataset.{split}_adj_data # split = {train, dev, test}
    def add_4lang_adj_data(self, target_flang, target_set, add_edge_types=False, relevance_scores=None):

        import stanza
        nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=True)
        def lemmatize(word, nlp=self.graph_parser.tfl.nlp):
            parse = nlp(word)
            return parse.sentences[0]._words[0].lemma 

        N = len(target_set)
        max_num_nodes = self.params['max_num_nodes'] if 'max_num_nodes' in self.params else 200

        concept_ids = torch.zeros(N, 5, max_num_nodes).long() # [n,nc,n_node]
        node_type_ids = torch.zeros(N, 5, max_num_nodes).long() # [n,nc,n_node]
        node_scores = torch.ones(N, 5, max_num_nodes, 1) # [n,nc,n_nodes,1] = [8459, 5, 200, 1]
        adj_lengths = torch.zeros(N,5) # [n,nc]
        edge_index = [] # [n,nc,[2,e]] list(list(tensor)))
        edge_types = [] # [n,nc,e] list(list(tensor))

        for i,(X_flang_edges, X_flang_mapping, X_flang_concepts, X_og) in enumerate(tqdm(zip(*target_flang, target_set), desc=f"adding 4lang to qagnn data")):
            
            c_edge_index = [None] * 5
            c_edge_types = [None] * 5

            for a in range(5):

                if 'Z_VEC' in X_flang_concepts[a]: # doesn't happen when loaded from memory!
                    X_flang_concepts[a].remove('Z_VEC')

                concept_names = X_flang_concepts[a]

                # NEW NODE TYPE IDS
                nti = torch.Tensor([2] * 200).long() # expanded nodes and padding(?) are (=2)
                nti[0] = 3 # znode (=3)
                idx_from_mapping = [i for i,x in enumerate([None]+X_flang_mapping[a]) if x != None] # idx from 4L map
                # am_idx = [concept_names.index(x) for x in X_og['answers'][a].split() if x in concept_names]
                am_idx = []
                for x in X_og['answers'][a].split():
                    if x in concept_names:
                        am_idx.append(concept_names.index(x))
                    elif lemmatize(x) in concept_names:
                        am_idx.append(concept_names.index(lemmatize(x)))
                    else:
                        #print(f"WARNING: concept {x} not found in test.{i}.{a}") 
                        pass
                qm_idx = [x for x in idx_from_mapping if x not in am_idx]
                for n,x in enumerate(nti):
                    if n in qm_idx: nti[n]=0 # qnodes (=0)
                    elif n in am_idx: nti[n]=1 # anodes (=1) (mostly a single node)
                
                node_type_ids[i,a] = nti

                # CONCEPT IDS
                concept_tensor = torch.Tensor([self.graph_parser.concept2id[c] if c in self.graph_parser.concept2id else self.graph_parser.concept2id['UNK'] for c in concept_names]).long() # TODO might put "UNK" spot here instead of -1!
                concept_ids[i,a] = F.pad(concept_tensor, (0,  max_num_nodes-len(concept_tensor)), 'constant', 1)
                assert concept_tensor.min() >= 0

                # NODE SCORES
                if relevance_scores != None:
                    sample_relevance_scores = relevance_scores[i][a]
                    for c, cname in enumerate(concept_names):
                        node_scores[i,a,c] = np.float64(sample_relevance_scores[cname])

                # ADJ LENGTHS
                assert len(concept_names) == concept_tensor.shape[0]
                adj_lengths[i,a] = len(concept_names)

                # EDGE INDEX
                if add_edge_types:
                    c_edges = torch.Tensor(X_flang_edges[a][0]).view(2,-1).long() +1# because of z-node!
                else:
                    c_edges = torch.Tensor(X_flang_edges[a]).view(2,-1).long() +1# because of z-node!
                c_edge_index[a] = c_edges

                # EDGE TYPES
                if add_edge_types:
                    c_edge_types[a] = torch.Tensor(X_flang_edges[a][1]).long()
                    assert c_edge_types[a].shape[0] == c_edges.shape[1]
                else:
                    c_edge_types[a] = torch.zeros(c_edges.shape[1]).long()

                # INSERT Z-EDGES
                z_edges = [(0,x) for x in idx_from_mapping]
                _c_edges = c_edges.T.tolist()
                _c_edge_types_a = c_edge_types[a].tolist()
                for ze in z_edges:
                    if ze not in _c_edges:
                        _c_edges.append(ze)
                        _c_edge_types_a.append(0)

                c_edge_index[a] = torch.Tensor(_c_edges).long().T
                c_edge_types[a] = torch.Tensor(_c_edge_types_a).long()
            
            edge_index.append(c_edge_index)
            edge_types.append(c_edge_types)

        # *dataset.train_decoder_data, dataset.train_adj_data = concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_types)
        return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_types)

    def train(self):
        
        args = self.parse_params_to_args(self.params)
        dataset = self.complete_set

        # SET RANDOM SEEDS
        random.seed(args.rnd_seed)
        np.random.seed(args.rnd_seed)
        torch.manual_seed(args.rnd_seed)
        if torch.cuda.is_available() and args.use_cuda:
            torch.cuda.manual_seed(args.rnd_seed)

        # GET / CHECK PATHS
        config_path = os.path.join(args.save_loc, 'config.json')
        model_path = os.path.join(args.save_loc, 'model.pt')
        log_path = os.path.join(args.save_loc, 'log.csv')
        export_config(args, config_path)
        check_path(model_path)
        with open(log_path, 'w') as fout:
            fout.write('step,dev_acc,test_acc\n')

        self.model.encoder.to(self.device0)
        self.model.decoder.to(self.device1)

        # OPTIMIZER
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
            {'params': [p for n, p in self.model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
            {'params': [p for n, p in self.model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
            {'params': [p for n, p in self.model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
        ]
        optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

        # SCHEDULE
        if args.lr_schedule == 'fixed':
            try:
                scheduler = ConstantLRSchedule(optimizer)
            except:
                scheduler = get_constant_schedule(optimizer)
        elif args.lr_schedule == 'warmup_constant':
            try:
                scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
            except:
                scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
        elif args.lr_schedule == 'warmup_linear':
            max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
            try:
                scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
            except:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

        print('parameters:')
        for name, param in self.model.decoder.named_parameters():
            if param.requires_grad:
                print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
            else:
                print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
        num_params = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)
        print('\ttotal:', num_params)

        if args.loss == 'margin_rank':
            loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
        elif args.loss == 'cross_entropy':
            loss_func = nn.CrossEntropyLoss(reduction='mean')

        def compute_loss(logits, labels):
            if args.loss == 'margin_rank':
                num_choice = logits.size(1)
                flat_logits = logits.view(-1)
                correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
                correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
                wrong_logits = flat_logits[correct_mask == 0]
                y = wrong_logits.new_ones((wrong_logits.size(0),))
                loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
            elif args.loss == 'cross_entropy':
                loss = loss_func(logits, labels)
            return loss

        ###################################################################################################
        #   Training                                                                                      #
        ###################################################################################################

        print()
        print('-' * 71)
        # has something to do with torch version
        '''
        if args.fp16:
            print ('Using fp16 training')
            scaler = torch.cuda.amp.GradScaler()
        '''

        global_step, best_dev_epoch = 0, 0
        best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
        epoch_loss = 0.0
        start_time = time.time()
        self.model.train()
        freeze_net(self.model.encoder)

        # EPOCHS
        for epoch_id in range(args.epochs):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(self.model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(self.model.encoder)
            self.model.train()

            # SAMPLES
            for qids, labels, *input_data in tqdm(dataset.train()):
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            logits, _ = self.model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                            loss = compute_loss(logits, labels[a:b])
                    else:
                        logits, _ = self.model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                        loss = compute_loss(logits, labels[a:b])
                    loss = loss * (b - a) / bs
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    total_loss += loss.item()
                    epoch_loss += loss.item()

                if args.max_grad_norm > 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                scheduler.step()
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()


                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_last_lr()[0], total_loss, ms_per_batch))
                    total_loss = 0
                    start_time = time.time()
                global_step += 1

            self.model.eval()
            dev_acc = evaluate_accuracy(dataset.dev(), self.model)
            save_test_preds = args.save_model
            if not save_test_preds:
                test_acc = evaluate_accuracy(dataset.test(), self.model) if args.test_statements else 0.0
            else:
                eval_set = dataset.test()
                total_acc = []
                count = 0
                #preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
                #with open(preds_path, 'w') as f_preds:
                with torch.no_grad():
                    for qids, labels, *input_data in tqdm(eval_set):
                        count += 1
                        logits, _, concept_ids, node_type_ids, edge_index, edge_type = self.model(*input_data, detail=True)
                        predictions = logits.argmax(1) #[bsize, ]
                        preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
                        for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
                            acc = int(pred.item()==label.item())
                            #print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                            #f_preds.flush()
                            total_acc.append(acc)
                test_acc = float(sum(total_acc))/len(total_acc)

            # LOG
            if args.wandb_logging:
                result_dict = {
                    'avg_train_loss': epoch_loss / len(dataset.train()),
                    #'avg_test_loss': avg_test_loss, # TODO
                    'acc_train': evaluate_accuracy(dataset.train(), self.model),
                    'acc_dev': dev_acc,
                    'acc_test': test_acc
                    #'comprehensiveness_test': expl_eval['comprehensiveness'] if do_explainability_eval else None, # TODO??? we never used this anyway
                    #'sufficiency_test': expl_eval['sufficiency'] if do_explainability_eval else None # TODO??? we never used this anyway
                }
                wandb.log(result_dict)
            else:
                print('-' * 71)
                print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc, test_acc))
                print('-' * 71)
                with open(log_path, 'a') as fout:
                    fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
            epoch_loss = 0.0

            # SAVE
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                if args.save_model:
                    torch.save([self.model.state_dict(), args], f"{model_path}.{epoch_id}")
                    # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                    #     for p in model.named_parameters():
                    #         print (p, file=f)
                    print(f'model saved to {model_path}.{epoch_id}')
            else:
                if args.save_model:
                    torch.save([self.model.state_dict(), args], f"{model_path}.{epoch_id}")
                    # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                    #     for p in model.named_parameters():
                    #         print (p, file=f)
                    print(f'model saved to {model_path}.{epoch_id}')
            self.model.train()
            start_time = time.time()
            if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break

    def eval_competence(self):
        self.model.eval()
        acc = Accuracy(num_classes=5)
        prec = Precision(num_classes=5)
        reca = Recall(num_classes=5)
        preds = torch.stack([torch.argmax(x) for x in self.val_pred[0]])
        ys = torch.cat(list(x[1] for x in self.val_set))
        return {
            'accuracy' : acc(preds.int(), ys).item(), 
            'precision' : prec(preds.int(), ys).item(), 
            'recall' : reca(preds.int(), ys).item()
        }

    def eval_explainability(self, pred=None, attn=None, skip_aopc=False): 

        # TODO add node_relevance scoring for erased predictions!!

        # PARAMS
        max_num_nodes = self.params['max_num_nodes'] if 'max_num_nodes' in self.params else None
        expand = self.params['expand'] if 'expand' in self.params else None
        pred, attentions = self.val_pred
        # k = round(self.avg_rational_lengths['validation'])

        # writes erased statements to files (coz dataset class needs files)
        def persist_statements(statements, name):
            int_to_label = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
            path = f"{self.params['save_loc']}{name}.dev.statement.jsonl"
            file = open(path, "w")

            for sample in statements: 
                _labels = np.zeros(5)
                _labels[sample['label']] = 0

                res_sample = {
                    "answerKey": int_to_label[sample['label']],
                    "id": sample['id'],
                    "question": {
                        "question_concept": sample['context'],
                        "choices": [
                            {"label": "A", "text": sample['answers'][-1]},
                            {"label": "B", "text": sample['answers'][0]},
                            {"label": "C", "text": sample['answers'][1]},
                            {"label": "D", "text": sample['answers'][2]},
                            {"label": "E", "text": sample['answers'][3]}
                        ],
                        "stem": sample['erased_stem'] if 'erased_stem' in sample else sample['question'],
                    },
                    "statements": [{'label':bool(_labels[i]), 'statement':sample['statements'][i]} for i in range(5)]
                }
                print(json.dumps(res_sample), file=file)

            return None

        # predicts erased datasets and statements
        def predict(dataset, dev_statements, save_file=None):

            add_edge_types = self.params['num_relation'] == 3 if 'num_relation' in self.params else False
            node_relevance = 'node_relevance' in self.params and self.params['node_relevance']
            
            # ADD 4LANG ADJ DATA (=DECODER DATA)
            flang_dev = self.graph_parser(
                dev_statements, 
                num_samples=len(dev_statements),
                split='_dev',
                qa_join=self.params['qa_join'],
                use_cache=False,
                max_num_nodes=max_num_nodes,
                expand=expand,
                add_edge_types=add_edge_types,
                use_existing_concept_ids=True
            ) # 2h (local PC)

            if node_relevance:
                node_relevance_dev = self.node_relevance_scoring(flang_dev, dev_statements) # 30 minutes (local PC)
            else:
                node_relevance_dev = None
            *dataset.dev_decoder_data, dataset.dev_adj_data = self.add_4lang_adj_data(target_flang=flang_dev, target_set=dev_statements, add_edge_types=add_edge_types, relevance_scores=node_relevance_dev)

            # PREDICT
            prediction_params = deepcopy(self.params)
            prediction_params['softmax_logits'] = True
            if save_file != None:
                prediction_params['save'] = save_file
            preds = []
            with torch.no_grad():
                for qids, labels, *input_data in tqdm(dataset.dev()):
                    preds.append(self.model(*input_data, **prediction_params))

            # DEBATCH
            _val_pred = list(zip(*preds))
            logits = torch.cat(_val_pred[0], dim=0) # debatch logits
            return logits

        # STATEMENTS
        dev_statements = self.prepare_qagnn_data(path=LOC['qagnn_statements_dev'])
        comp_statements = deepcopy(dev_statements)
        suff_statements = deepcopy(dev_statements)

        # ERASE
        for idx,(X,ans_attn) in enumerate(zip(dev_statements, attentions)):

            comp_statements[idx]['erased_stem'] = [X['question']] * 5
            suff_statements[idx]['erased_stem'] = [X['question']] * 5
            
            for a,(stmnt,attn) in enumerate(zip(X['statements'],ans_attn)):
                tokens = stmnt.split()
                _tokens = tokens[:-len(X['answers'][a].split())] # only use question stem without answer-ending
                assert len(tokens) == len(attn), "some form of sample mismatch has happened (where?)"
                top_idx = [i for i,x in enumerate(attn) if x>0]
                _top_idx = [i for i,x in enumerate(attn) if x>0 and i<len(_tokens)] # topidx without ids for answers
                if 0 < len(top_idx) < len(tokens): # default case
                    # comp
                    comp_statements[idx]['statements'][a] = " ".join([x for i,x in enumerate(tokens) if i in top_idx])
                    comp_statements[idx]['erased_stem'][a] = " ".join([x for i,x in enumerate(_tokens) if i in _top_idx])
                    # suff
                    suff_statements[idx]['statements'][a] = " ".join([x for i,x in enumerate(tokens) if i not in top_idx])
                    suff_statements[idx]['erased_stem'][a] = " ".join([x for i,x in enumerate(_tokens) if i not in _top_idx])
                elif len(top_idx) == 0: # attn is all 0
                    comp_statements[idx]['statements'][a] = X['answers'][a] # backup method or comp is empty
                    comp_statements[idx]['erased_stem'][a] = ""
                    suff_statements[idx]['statements'][a] = stmnt # everything is selected for suff
                    suff_statements[idx]['erased_stem'][a] = X['question']
                elif len(top_idx) == len(tokens): # attn is non-0 everywhere!
                    comp_statements[idx]['statements'][a] = stmnt # everything is selected for comp
                    comp_statements[idx]['erased_stem'][a] = X['question']
                    suff_statements[idx]['statements'][a] = X['answers'][a] # nothing is in suff, so backup
                    suff_statements[idx]['erased_stem'][a] = "" # nothing is in suff, so backup
                
                #assert comp_statements[idx]['statements'][a] == f"{comp_statements[idx]['erased_stem'][a].replace('?','')} {X['answers'][a]}"
                #assert suff_statements[idx]['statements'][a] == f"{suff_statements[idx]['erased_stem'][a].replace('?','')} {X['answers'][a]}"
        
        # save erased statements bc QAGNN_DataLoader class needs it
        persist_statements(comp_statements, 'comp')
        persist_statements(suff_statements, 'suff')

        # CREATE DATASETS
        comp_dataset = self.load_qagnn_dataset(dev_statements="data/experiments/default/comp.dev.statement.jsonl")
        suff_dataset = self.load_qagnn_dataset(dev_statements="data/experiments/default/suff.dev.statement.jsonl")

        # PREDICT
        comp_logits = predict(comp_dataset, comp_statements, save_file='COMP_DATA.jsonl')
        suff_logits = predict(suff_dataset, suff_statements, save_file='SUFF_DATA.jsonl')
    
        doc_ids = [x['id'] for x in self.dev_statements]
        pred = self.val_pred[0]
        comp_pred = comp_logits
        suff_pred = suff_logits
        attn = self.val_pred[1]
        aopc_thresholded_scores=None
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=None) 
        return E.classification_scores(results=er_results, mode='custom', aopc_thresholds=self.params['aopc_thresholds'], with_ids=doc_ids)

    def save(self):
        # WRITING VIZ DATA
        viz_path = f"{self.params['save_loc']}viz_data.json"
        with open(viz_path, 'w') as file:
            json.dump(self.viz_output, file)
        print(f"wrote viz data to {viz_path}")

        return True

def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            logits, _ = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples

class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    #
    def __init__(self, config):
        super().__init__(config)
    #
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        #
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs