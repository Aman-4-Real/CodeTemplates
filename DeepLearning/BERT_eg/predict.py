'''
Author: Aman
Date: 2022-04-17 23:10:58
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-17 23:16:20
'''

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from configs import *
from model import MyBERT
from MyDataset import MyDataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"

parser = argparse.ArgumentParser()
parser.add_argument("--device_ids", default="0,1,2,3", type=str, help="GPU device ids")
parser.add_argument("--model_name", default="hfl/chinese-bert-wwm-ext", type=str, help="Model name")
parser.add_argument("--test_batch_size", default=128, type=int, help="Test batch size")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
parser.add_argument("--test_data", default=".", type=str, help="Test data") # 4uttrs_140w_sess_log
parser.add_argument("--model_path", default=".", type=str, help="Model path")

global args
args = parser.parse_args()
new_layers_config = new_layers_config()
data_config = data_config()
print(args)


devices = list(eval(args.device_ids))
device = torch.device("cuda")

# load model
checkpoint = torch.load(args.model_path)
# args = checkpoint['args']
model = MyBERT(bert_model_name=args.model_name)
# optimizer.load_state_dict(checkpoint['optimizer'])
model.to(device)
model = nn.DataParallel(model, device_ids=devices)
model.load_state_dict(checkpoint['model'])

# load tokenizer
EOS = "[#EOS#]"
tokenizer = BertTokenizer.from_pretrained(args.model_name, never_split=[EOS])
tokenizer.vocab['[#EOS#]'] = tokenizer.vocab.pop('[unused1]')

print("Loading test data...")
test_data = MyDataset(args.test_data, tokenizer, data_config)
print("Data test loaded.")


def predict(model, test_data):
    model.eval()
    test_dataset = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loss = 0.0
    test_acc = 0.0
    y_preds = []
    y_labels = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataset, ncols=150, leave=False)
        for i, batch in enumerate(epoch_iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.forward(batch)
            ys = batch["labels"]
            y_preds.append(torch.argmax(preds, dim=1))
            y_labels.append(ys)
    test_loss /= len(test_dataset)
    y_preds = torch.cat(y_preds, dim=0)
    y_labels = torch.cat(y_labels, dim=0)
    test_acc = (y_preds == y_labels).sum().item() / len(y_labels) * 100

    return y_preds.tolist(), y_labels.tolist()



if __name__ == "__main__":
    y_preds, y_labels = predict(model, test_data)
    report(y_preds, y_preds)
    conf_mat = confusion_matrix(y_labels, y_preds)
    print("Confusion Matrix:\n", conf_mat)
