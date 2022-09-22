from transformers import AlbertTokenizer, AlbertTokenizerFast, AlbertModel

class AlbertEmbedding():
    
    def __init__(self, params:{}):
        self.device = 'cuda:0' if ('use_cuda' in params and params['use_cuda']) else 'cpu'
        self.tokenizer = AlbertTokenizerFast.from_pretrained(params['embedding'])
        self.model = AlbertModel.from_pretrained(params['embedding']).to(self.device)
        self.dim = self.model.config.hidden_size
    
    def __call__(self, text:str, return_bert_map=False): # TODO _call_single and _call_iterable (this is call_single)
        inputs = self.tokenizer(text.split(), return_tensors='pt', return_offsets_mapping=True, is_split_into_words=True)
        # create bert_map
        offset_mapping = inputs.pop('offset_mapping') # not allowed as input for model!
        is_subword = (offset_mapping[0][:,0] != 0).tolist()
        real_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]) 
        skip_these_special_tokens = [x for x in list(self.tokenizer.special_tokens_map.values()) if x != self.tokenizer.unk_token] + ['▁'] # 'x ?' will be tokenized as [x, _, ?] ... # TODO?
        bert_map = []
        c = -1
        for i,is_sub in enumerate(is_subword):
            if real_tokens[i] in skip_these_special_tokens:
                bert_map.append(None)
            elif is_sub: 
                bert_map.append(c)
            else: 
                c+=1
                bert_map.append(c)
        # run through albert
        inputs = {k: v.to(self.device) for k, v in inputs.items()} 
        outputs = self.model(**inputs)
        if return_bert_map: return outputs.last_hidden_state, bert_map
        else: return outputs.last_hidden_state
        # debugging: list(zip(real_tokens, [text.split()[x] if x != None else None for x in bert_map]))