'''
Author: Aman
Date: 2022-04-17 22:42:05
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-17 22:53:06
'''

class new_layers_config():
    def __init__(self):
        self.num_labels = 3
        self.linear_hidden_size = 256
        self.linear_dropout = 0.0


class data_config():
    def __init__(self):
        self.max_uttr_num = 10
        self.max_seq_length = 512
        self.max_context_length = 50
        self.max_response_length = 50
        