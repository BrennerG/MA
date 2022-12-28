import os
import torch
import wandb
import random
import numpy as np
import torch

from argparse import Namespace
import torch.nn as nn
try: from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except: from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
from tqdm import tqdm
from copy import deepcopy
from torchmetrics import Accuracy, Recall, Precision

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
        # overwrite
        self.params = params
        self.params['offset_concepts'] = True
        torch.manual_seed(self.params['rnd_seed'])
        wandb.init(config=self.params, mode='online' if self.params['wandb_logging']==True else 'disabled')

        # set devices
        '''
        if 'use_cuda' in self.params and self.params['use_cuda']:
            self.device0 = torch.device("cuda:0") 
            self.device1 = torch.device("cuda:0")
        else:
            self.device0 = torch.device("cpu") 
            self.device1 = torch.device("cpu")
        '''
        # 4h30
        '''
        self.device0 = torch.device('cpu') # encoder
        self.device1 = torch.device('cuda:0') # decoder
        '''
        self.device0 = torch.device('cuda:0') # encoder
        self.device1 = torch.device('cpu') # decoder

        # prepare graph parser
        self.graph_parser = self._graph_parser_factory()
        # general stuff
        self.complete_set, self.train_set, self.val_set, self.test_set = self.init_data()
        self.params['num_concepts'] = max(self.graph_parser.id2concept)
        self.params['num_relation'] = 1 # basic case
        self.model = self.model_factory(self.params['model_type']) # .to(self.device) # TODO shift device assignment from train() to here
        raw_cose = load_dataset(LOC['cose_huggingface'])
        self.avg_rational_lengths = EraserCosE.avg_rational_length(raw_cose)

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
        print('visualizing...')
        self.viz_output = self.viz()

        # SAVE / PERSIST
        print('saving...')
        self.save()

        print('experiment done!')
        return self

    def load_qagnn_dataset(self, **kwargs):
        # IMITATE PARAMETERS / ARGS
        args = Namespace(**self.params)

        # qagnn data_loader params
        #args.train_statements = 'data/qa_gnn/train.statement.jsonl'
        args.train_statements = LOC['qagnn_statements_train']
        args.train_adj = None
        #args.dev_statements = 'data/qa_gnn/dev.statement.jsonl'
        args.dev_statements = LOC['qagnn_statements_dev']
        args.dev_adj = None
        #args.test_statements = 'data/qa_gnn/test.statement.jsonl'
        args.test_statements = LOC['qagnn_statements_test']
        args.test_adj = None
        args.batch_size = 32
        args.eval_batch_size = 16
        args.encoder = 'bert-large-uncased'
        args.max_node_num = self.params['max_num_nodes']
        args.drop_partial_batch = False
        args.fill_partial_batch = False

        # overwrite parameters
        args.dev_statements = kwargs['dev_statements'] if 'dev_statements' in kwargs else args.dev_statements

        # fourlang_parser params
        use_cache = self.params['use_cache'] if 'use_cache' in self.params else True
        max_num_nodes = self.params['max_num_nodes'] if 'max_num_nodes' in self.params else None
        expand = self.params['expand'] if 'expand' in self.params else None

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
        return dataset

    def init_data(self):
        # IMITATE PARAMETERS / ARGS
        args = Namespace(**self.params)

        # qagnn data_loader params
        #args.train_statements = 'data/qa_gnn/train.statement.jsonl'
        args.train_statements = LOC['qagnn_statements_train']
        args.train_adj = None
        #args.dev_statements = 'data/qa_gnn/dev.statement.jsonl'
        args.dev_statements = LOC['qagnn_statements_dev']
        args.dev_adj = None
        #args.test_statements = 'data/qa_gnn/test.statement.jsonl'
        args.test_statements = LOC['qagnn_statements_test']
        args.test_adj = None
        args.batch_size = 32
        args.eval_batch_size = 16
        args.encoder = 'bert-large-uncased'
        args.max_node_num = self.params['max_num_nodes']
        args.drop_partial_batch = False
        args.fill_partial_batch = False

        # fourlang_parser params
        use_cache = self.params['use_cache'] if 'use_cache' in self.params else True
        max_num_nodes = self.params['max_num_nodes'] if 'max_num_nodes' in self.params else None
        expand = self.params['expand'] if 'expand' in self.params else None

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
                use_cache=use_cache,
                max_num_nodes=max_num_nodes,
                expand=expand
            ) for (split, ds) in zip(['train','dev','test'], [self.train_statements, self.dev_statements, self.test_statements])
        ]

        self.graph_parser.save_concepts() # TODO put this in save?

        *dataset.train_decoder_data, dataset.train_adj_data = self.add_4lang_adj_data(target_flang=self.flang_train, target_set=self.train_statements)
        *dataset.dev_decoder_data, dataset.dev_adj_data = self.add_4lang_adj_data(target_flang=self.flang_dev, target_set=self.dev_statements)
        *dataset.test_decoder_data, dataset.test_adj_data = self.add_4lang_adj_data(target_flang=self.flang_test, target_set=self.test_statements)

        return dataset, dataset.train(), dataset.dev(), dataset.test()
    
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
    def add_4lang_adj_data(self, target_flang, target_set):
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

                # NODE TYPE IDS
                concept_names = X_flang_concepts[a]
                answer_concepts = X_og['answers'][a].split()
                # set context node
                node_type_ids[i,a,0] = 3
                # set answer type
                am_idx = [concept_names.index(ac) for ac in answer_concepts if ac in concept_names]
                for am_i in am_idx:
                    node_type_ids[i,a,am_i+1] = 1
                # set question type
                qm_idx = [x for x in range(len(concept_names)) if concept_names[x] not in answer_concepts]
                assert len(qm_idx) < 200
                for qm_i in qm_idx:
                    node_type_ids[i,a,qm_i+1] = 0

                # CONCEPT IDS
                concept_tensor = torch.Tensor([self.graph_parser.concept2id[c] for c in concept_names]).long()
                concept_ids[i,a] = F.pad(concept_tensor, (0,  max_num_nodes-len(concept_tensor)), 'constant', 1)
                pass

                # NODE SCORES ?
                pass

                # ADJ LENGTHS
                adj_lengths[i,a] = len(concept_names) + 1

                # EDGE INDEX
                c_edges = torch.Tensor(X_flang_edges[a]).view(2,-1).long()
                c_edge_index[a] = c_edges

                # EDGE TYPES
                c_edge_types[a] = torch.zeros(c_edges.shape[1]).long()
            
            edge_index.append(c_edge_index)
            edge_types.append(c_edge_types)

        # *dataset.train_decoder_data, dataset.train_adj_data = concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_types)
        return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_types)

    def train(self):
        # IMITATE PARAMETERS / ARGS
        args = Namespace(**self.params)
        # FOR MODEL
        args.k = 5
        args.gnn_dim = self.params['gat_hidden_dim']
        args.concept_dim = 1024
        args.att_head_num = 2
        args.fc_dim = 200 # hidden dim for final MLP layer
        args.fc_layer_num = 0 # nr layers for final MLP layer + 1
        args.dropouti = self.params['dropout']
        args.dropoutg = self.params['dropout']
        args.dropoutf = self.params['dropout']
        args.encoder = 'bert-large-uncased'
        args.num_relation = 1
        args.encoder_layer = -1
        # FOR TRAINING
        args.weight_decay = 0.01
        args.encoder_lr = 2e-05
        args.decoder_lr = 0.001
        args.optim = 'radam'
        args.lr_schedule = 'fixed'
        args.loss = 'cross_entropy'
        args.unfreeze_epoch = 4
        args.refreeze_epoch = 10000
        args.mini_batch_size = 1
        args.fp16 = False
        args.max_grad_norm = 1.0
        args.log_interval = 10
        args.save_model = False
        args.save_dir = './saved_models/qagnn/' # in params
        args.max_epochs_before_stop = 10 # in params
        # ACTUALLY ALREADY IN __INIT__()
        # args.dev_statements = LOC['qagnn_statements_dev']
        args.test_statements = LOC['qagnn_statements_test']

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

        
        '''
        # TODO implement model loading!
        if args.load_model_path:
            print (f'loading and initializing model from {args.load_model_path}')
            model_state_dict, old_args = torch.load(args.load_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_state_dict)
        '''

        # TODO move this to init
        self.model.encoder.to(self.device0)
        self.model.decoder.to(self.device1)

        # TODO hardcode this (maybe put this in its own function)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
            {'params': [p for n, p in self.model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
            {'params': [p for n, p in self.model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
            {'params': [p for n, p in self.model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
        ]
        optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

        # TODO is probably always 'fixed' anyway
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
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
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
                preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
                with open(preds_path, 'w') as f_preds:
                    with torch.no_grad():
                        for qids, labels, *input_data in tqdm(eval_set):
                            count += 1
                            logits, _, concept_ids, node_type_ids, edge_index, edge_type = self.model(*input_data, detail=True)
                            predictions = logits.argmax(1) #[bsize, ]
                            preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
                            for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
                                acc = int(pred.item()==label.item())
                                print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                                f_preds.flush()
                                total_acc.append(acc)
                test_acc = float(sum(total_acc))/len(total_acc)

            print('-' * 71)
            print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc, test_acc))
            print('-' * 71)
            with open(log_path, 'a') as fout:
                fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
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

        # PARAMS
        max_num_nodes = self.params['max_num_nodes'] if 'max_num_nodes' in self.params else None
        expand = self.params['expand'] if 'expand' in self.params else None
        pred, attentions = self.val_pred
        k = round(self.avg_rational_lengths['validation'])

        # writes erased statements to files (coz dataset class needs files)
        def persist_statements(statements, name):
            int_to_label = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
            path = f"{self.params['save_loc']}/{name}.dev.statement.jsonl"
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
                        "stem": sample['question'],
                    },
                    "statements": [{'label':bool(_labels[i]), 'statement':sample['statements'][i]} for i in range(4)]
                }
                print(json.dumps(res_sample), file=file)

            return None

        # predicts erased datasets and statements
        def predict(dataset, dev_statements):
            
            # ADD 4LANG ADJ DATA (=DECODER DATA)
            flang_dev = self.graph_parser(
                dev_statements, 
                num_samples=len(dev_statements),
                split='_dev',
                qa_join=self.params['qa_join'],
                use_cache=False,
                max_num_nodes=max_num_nodes,
                expand=expand,
                use_existing_concept_ids=True
            )
            *dataset.dev_decoder_data, dataset.dev_adj_data = self.add_4lang_adj_data(target_flang=flang_dev, target_set=dev_statements)

            # PREDICT
            prediction_params = deepcopy(self.params)
            prediction_params['softmax_logits'] = True
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
            for a,(stmnt,attn) in enumerate(zip(X['statements'],ans_attn)):
                tokens = stmnt.split()
                assert len(tokens) == len(attn), "some form of sample mismatch has happened (where?)"
                top_idx = [i for i,x in enumerate(attn) if x>0]
                if 0 < len(top_idx) < len(tokens): # default case
                    comp_statements[idx][a] = " ".join([x for i,x in enumerate(tokens) if i in top_idx])
                    suff_statements[idx][a] = " ".join([x for i,x in enumerate(tokens) if i not in top_idx])
                elif len(top_idx) == 0: # attn is all 0
                    comp_statements[idx][a] = X['answers'][a] # backup method or comp is empty
                    suff_statements[idx][a] = stmnt # everything is selected for suff
                elif len(top_idx) == len(tokens): # attn is non-0 everywhere!
                    comp_statements[idx][a] = stmnt # everything is selected for comp
                    suff_statements[idx][a] = X['answers'][a] # nothing is in suff, so backup
        # save erased statements bc QAGNN_DataLoader class needs it
        persist_statements(comp_statements, 'comp') 
        persist_statements(suff_statements, 'suff') 

        # CREATE DATASETS
        comp_dataset = self.load_qagnn_dataset(dev_statements="data/experiments/default/comp.dev.statement.jsonl")
        suff_dataset = self.load_qagnn_dataset(dev_statements="data/experiments/default/suff.dev.statement.jsonl")

        # PREDICT
        comp_logits = predict(comp_dataset, comp_statements)
        suff_logits = predict(suff_dataset, suff_statements)
    
        doc_ids = [x['id'] for x in self.dev_statements]
        pred = self.val_pred[0]
        comp_pred = comp_logits
        suff_pred = suff_logits
        attn = self.val_pred[1]
        aopc_thresholded_scores=None
        er_results = E.create_results(doc_ids, pred, comp_pred, suff_pred, attn, aopc_thresholded_scores=None) 
        return E.classification_scores(results=er_results, mode='custom', aopc_thresholds=self.params['aopc_thresholds'], with_ids=doc_ids)

    # def save(self): raise NotImplementedError()

def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            logits, _ = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples