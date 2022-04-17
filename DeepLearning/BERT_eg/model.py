'''
Author: Aman
Date: 2022-04-17 22:51:07
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-17 22:52:09
'''

from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.init as init
from configs import new_layers_config


class MyBERT(nn.Module):
    def __init__(self, new_layers_config=new_layers_config(), bert_model_name="hfl/chinese-bert-wwm-ext", bert_hidden_size=768):
        super(MyBERT, self).__init__()
        self.model_name = bert_model_name
        self.num_labels = new_layers_config.num_labels # for MLP

        # BERT layer
        self.bert = BertModel.from_pretrained(self.model_name)

        # New layers
        self.fc = nn.Linear(bert_hidden_size, new_layers_config.linear_hidden_size)  # BERT -> FC
        self.Tanh = nn.Tanh()
        self.classifier = nn.Linear(new_layers_config.linear_hidden_size, self.num_labels_1)  # FC -> classifier

        self.dropout = nn.Dropout(new_layers_config.linear_dropout)

        # init weights
        self.init_weights()

    def forward(self, batch):
        # Inputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

        # BERT layer
        sequence_output, pooled_output = self.bert(**bert_inputs, return_dict=False) # sequence_output shape: (batch_size, sequence_length, 768)
        
        # linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings

        # New layers
        fc_out = self.dropout(pooled_output)
        fc_out = self.fc(fc_out)
        logits = self.classifier(self.Tanh(fc_out))

        return logits


    def init_weights(self):
        init.kaiming_normal_(self.fc.weight)
        init.kaiming_normal_(self.classifier.weight)
