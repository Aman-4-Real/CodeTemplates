'''
Author: Aman
Date: 2022-04-17 22:42:53
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-17 22:46:37
'''

# import packages here
import linecache
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, data_config):
        super(MyDataset, self).__init__()
        self._filename = file_path
        self._total_len = sum(1 for line in open(self._filename))
        self._tokenizer = tokenizer
        self._max_seq_length = data_config.max_seq_length
        self._max_context_length = data_config.max_context_length
        self._max_response_length = data_config.max_response_length
    
    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split('\t')

        # define what your batch is like:
        batch = {}
        
        # e.g.
        # y_maincatetory, y_subcategory = line[0], line[1]
        # context = line[2:-1]
        # response = line[-1]
        
        # input_ids, attention_mask, segment_ids = self.annotate(context, response)
        # batch = {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'token_type_ids': segment_ids,
        #     'labels_1': int(y_maincatetory),
        #     'labels_2': int(y_subcategory)
        # }
        return batch
        
    # define your other functions
    def annotate(self, ):
        pass

    # def annotate(self, context, response):
    #     if len(context) > self._max_uttr_num: # truncate
    #         context = context[-self._max_uttr_num:]
    #     all_context_sents = []
    #     for sent in context:
    #         sent = "".join(sent.split())
    #         sent = self._tokenizer.tokenize(sent)[:self._max_context_length]
    #         all_context_sents.append(sent)
    #     tokens = [self._tokenizer.cls_token]
    #     segment_ids = [0]
    #     i = 1 # avoid context is none
    #     for i, sent in enumerate(all_context_sents):
    #         tokens.extend(sent + ["[#EOS#]"])
    #         segment_ids.extend([i % 2] * (len(sent) + 1))
    #     tokens += [self._tokenizer.sep_token]
    #     segment_ids += [i % 2]
        
    #     response_tokens = self._tokenizer.tokenize(response)[:self._max_response_length]
    #     tokens += response_tokens
    #     segment_ids += [(i+1) % 2] * len(response_tokens)
    #     tokens += [self._tokenizer.sep_token]
    #     segment_ids += [(i+1) % 2]
        
    #     tokens = tokens[:self._max_seq_length]
    #     segment_ids = segment_ids[:self._max_seq_length]
    #     attention_mask = [1] * len(tokens)
    #     # assert len(tokens) <= self._max_seq_length
    #     while len(tokens) < self._max_seq_length: # pad to max_seq_length
    #         tokens.append(self._tokenizer.pad_token)
    #         segment_ids.append(0)
    #         attention_mask.append(0)
    #     # assert len(tokens) == len(segment_ids) == len(attention_mask) == self._max_seq_length
    #     anno_seq = self._tokenizer.convert_tokens_to_ids(tokens)
    #     input_ids = np.asarray(anno_seq)
    #     attention_mask = np.asarray(attention_mask)
    #     segment_ids = np.asarray(segment_ids)

    #     return input_ids, attention_mask, segment_ids

