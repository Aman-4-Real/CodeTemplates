'''
Author: Aman
Date: 2022-04-17 23:18:12
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-17 23:20:18
'''

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from configs import *
from model import MyBERT
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device_ids = "0,1,2,3"
model_name = "hfl/chinese-bert-wwm-ext"
data_config = data_config()

devices = list(eval(device_ids))
device = torch.device("cuda")

# load tokenizer
EOS = "[#EOS#]"
tokenizer = BertTokenizer.from_pretrained("./vocab/vocab.txt", never_split=[EOS])
# tokenizer.vocab['[#EOS#]'] = tokenizer.vocab.pop('[unused1]')



# load model
model_path = "."
checkpoint = torch.load(model_path)
model = MyBERT(bert_model_name=model_name)
model.to(device)
model = nn.DataParallel(model, device_ids=devices)
model.load_state_dict(checkpoint['model'])


def annotate(context, response):
    # The same function as that in your MyDataset.py
    pass



def predict(posts, response):
    input_ids, attention_mask, segment_ids = annotate(posts, response)
    batch = {
        'input_ids': torch.from_numpy(input_ids).unsqueeze(0),
        'attention_mask': torch.from_numpy(attention_mask).unsqueeze(0),
        'token_type_ids': torch.from_numpy(segment_ids).unsqueeze(0)
    }
    model.eval()
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items()}
        fc_preds = model.forward(batch)

    return fc_preds.tolist()[0]



if __name__ == "__main__":
    res = predict(["中午好","中午好呀","你吃午饭了吗"], "刚刚才吃过了")
    print(res)










