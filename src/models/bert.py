import torch

import torch.nn as nn

from transformers import AlbertTokenizer, AlbertForMultipleChoice


# https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626
class BertClassifier(nn.Module):

    def __init__(self, params:{}, bert:str='albert'):
        super().__init__()
        self.TYPE = "BertClassifier"
        self.rationality = 'soft'
        torch.manual_seed(params['random_seed'])
        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        self.model = AlbertForMultipleChoice.from_pretrained("albert-base-v2")
        self.sm = nn.Softmax(dim=0)
    
    def forward(self, question:str, context:str, answers:[], truth:int):
        label = torch.tensor(int(truth)).unsqueeze(0)
        encoding = self.tokenizer(5*[question], answers, return_tensors="pt", padding=True)

        outputs = self.model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=label, output_attentions=True)  # batch size is 1
        softmaxed = self.sm(outputs.logits.squeeze())

        attentions = outputs.attentions # len=#attention_layers each layer outputs shape (batch_size, num_heads, sequence_length, sequence_length)
        # TODO aggregate the attention layers somehow...
        # TODO cut attentions back from masks
        aggregated_attentions = None 

        return softmaxed, aggregated_attentions