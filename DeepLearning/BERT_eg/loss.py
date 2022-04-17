'''
Author: Aman
Date: 2022-04-17 22:53:38
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-17 22:54:42
'''


import torch
import torch.nn as nn

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, ):
        loss = 0.0
        return loss