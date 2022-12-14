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

from experiments.final_experiment import FinalExperiment
from data.locations import LOC
from data.huggingface_cose import EraserCosE
from models.qagnn.utils.utils import *
from models.qagnn.utils.optimization_utils import OPTIMIZER_CLASSES
from models.qagnn.modeling_qagnn import *
from datasets import load_dataset
from data.locations import LOC
from data.statements_cose import StatementLoader


class QagnnExperiment(FinalExperiment):

    def __init__(self, params:{}):
        # overwrite
        self.params = params
        self.params['model_type'] == 'qagnn'
        torch.manual_seed(self.params['rnd_seed'])
        wandb.init(config=self.params, mode='online' if self.params['wandb_logging']==True else 'disabled')
        # set devices # TODO don't hardcode
        # self.device = 'cuda:0' if ('use_cuda' in self.params and self.params['use_cuda']) else 'cpu'
        self.device0 = torch.device("cpu") 
        self.device1 = torch.device("cpu")
        # prepare graph parser
        self.graph_parser = self._graph_parser_factory()
        # general stuff
        self.complete_set, self.train_set, self.val_set, self.test_set = self.init_data()
        self.params['num_concepts'] = max(self.graph_parser.id2concept)
        self.params['num_relation'] = 1 # basic case
        self.model = self.model_factory(self.params['model_type']) # .to(self.device) # TODO shift device assignment from train() to here
        raw_cose = load_dataset(LOC['cose_huggingface'])
        self.avg_rational_lengths = EraserCosE.avg_rational_length(raw_cose)

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

            is_inhouse=True, # TODO correct?
            inhouse_train_qids_path="data/qa_gnn/inhouse_split_qids.txt" # TODO above

            #max_seq_length=args.max_seq_len,)
            #subsample=args.subsample,  # 1.0
            #use_cache=args.use_cache) # True
        )

        train_set = self.prepare_qagnn_data(path=LOC['qagnn_statements_train'])
        dev_set = self.prepare_qagnn_data(path=LOC['qagnn_statements_dev'])
        test_set = self.prepare_qagnn_data(path=LOC['qagnn_statements_test'])

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
            ) for (split, ds) in zip(['train','dev','test'], [train_set, dev_set, test_set])
        ]

        self.graph_parser.save_concepts() # TODO put this in save?
        return dataset, train_set, dev_set, test_set
    
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
    def add_4lang_adj_data(self, dataset, split='train'):
        assert split == 'train'
        max_num_nodes = self.params['max_num_nodes'] if 'max_num_nodes' in self.params else 200
        N = len(self.train_set)

        concept_ids = torch.zeros(N, 5, max_num_nodes).long() # [n,nc,n_node]
        node_type_ids = torch.zeros(N, 5, max_num_nodes).long() # [n,nc,n_node]
        node_scores = torch.ones(N, 5, max_num_nodes, 1) # [n,nc,n_nodes,1] = [8459, 5, 200, 1]
        adj_lengths = torch.zeros(N,5) # [n,nc]
        edge_index = [] # [n,nc,[2,e]] list(list(tensor)))
        edge_types = [] # [n,nc,e] list(list(tensor))

        for i,(X_flang_edges, X_mapping, X_flang_concepts, X_og) in enumerate(tqdm(zip(self.flang_train[0], self.flang_train[1], self.flang_train[2], self.train_set), desc='adding 4lang to qagnn data')):
            
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

        *dataset.train_decoder_data, dataset.train_adj_data = concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_types)
        return dataset

    def train(self):
        # IMITATE PARAMETERS / ARGS
        args = Namespace(**self.params)
        # for model
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
        # for training
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
        cp_emb = [np.load(path) for path in args.ent_emb_paths]
        cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

        concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
        print('| num_concepts: {} |'.format(concept_num))
        '''
        
        dataset = self.add_4lang_adj_data(self.complete_set)

        # BUILD MODEL
        print ('args.num_relation', args.num_relation)
        model = LM_QAGNN(
            args, 
            args.encoder, 
            k=args.k, 
            n_ntype=4, 
            n_etype=args.num_relation, 
            n_concept=args.num_concepts,
            concept_dim=args.gnn_dim,
            concept_in_dim=args.concept_dim,
            n_attention_head=args.att_head_num, 
            fc_dim=args.fc_dim, 
            n_fc_layer=args.fc_layer_num,
            p_emb=args.dropouti, 
            p_gnn=args.dropoutg, 
            p_fc=args.dropoutf,
            pretrained_concept_emb=None)
            # freeze_ent_emb=args.freeze_ent_emb,
            # init_range=args.init_range,
            # encoder_config={})

        '''
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
            for qids, labels, *input_data in dataset.train():
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                            loss = compute_loss(logits, labels[a:b])
                    else:
                        logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
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
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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

                model.eval()
                dev_acc = evaluate_accuracy(dataset.dev(), model)
                save_test_preds = args.save_model
                if not save_test_preds:
                    test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
                else:
                    eval_set = dataset.test()
                    total_acc = []
                    count = 0
                    preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
                    with open(preds_path, 'w') as f_preds:
                        with torch.no_grad():
                            for qids, labels, *input_data in tqdm(eval_set):
                                count += 1
                                logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data, detail=True)
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
                        torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                        # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                        #     for p in model.named_parameters():
                        #         print (p, file=f)
                        print(f'model saved to {model_path}.{epoch_id}')
                else:
                    if args.save_model:
                        torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                        # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                        #     for p in model.named_parameters():
                        #         print (p, file=f)
                        print(f'model saved to {model_path}.{epoch_id}')
                model.train()
                start_time = time.time()
                if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                    break

    def eval_explainability(self, pred=None, attn=None, skip_aopc=False): 
        pass

    def save(self):
        pass


class ParsedDataset:

    def __init__(self, complete_set, **kwargs):
        self.train_set = self.init_train(complete_set['train'])
        self.val_set = None
        self.test_set = None
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 1
    
    def init_train(self, train_set):
        set = []
        
        return set

    def train():
        return self.train_set