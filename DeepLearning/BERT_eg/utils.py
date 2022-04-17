'''
Author: Aman
Date: 2022-04-17 22:48:26
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-17 22:49:33
'''

import numpy as np
import torch
import datetime
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
# import numpy as np
import warnings
warnings.filterwarnings("ignore")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='model.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        # self.args = args
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     state = {'model': model.state_dict(), 'args': self.args}
    #     torch.save(state, self.path)
    #     self.val_loss_min = val_loss


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def report(labels, preds):
    target_names = ['AAA', 'BBB']
    print("\nReport of Micro-labels:\n")
    
    print(classification_report(labels, preds, target_names=target_names, digits=4))

    return


# def compute_metrics(labels, preds):
#     # calculate accuracy using sklearn's function
#     acc = accuracy_score(labels, preds)
#     p = precision_score(labels, preds, average='macro')
#     r = recall_score(labels, preds, average='macro')
#     f1 = f1_score(labels, preds, average='macro')

#     p_class, r_class, f_class, support_micro = precision_recall_fscore_support(labels, preds, average=None)
#     p_class = [round(x, 4) for x in p_class]
#     r_class = [round(x, 4) for x in r_class]
#     f_class = [round(x, 4) for x in f_class]
#     # support_micro = [round(x, 4) for x in support_micro]    
#     print('Precision: {}'.format(p_class))
#     print('Recall: {}'.format(r_class))
#     print('F1: {}'.format(f_class))
#     print('F1_AVG: {}'.format(np.mean(f_class)))
#     # print('support_micro: {}'.format(support_micro))

#     # auc = roc_auc_score(labels, preds, average='ovo')

#     return {
#       'accuracy': round(acc, 4),
#       'precision_score': round(p, 4),
#       'recall_score': round(r, 4),
#       'f1_score': round(f1, 4),
#     #   'auc_score': auc,
#     }