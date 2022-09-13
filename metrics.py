import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        bceloss = nn.CrossEntropyLoss(reduction="mean")
        assert not target.requires_grad
        
        return bceloss(input, target) * self.train_size + beta * kl


# def lr_linear(epoch_num, decay_start, total_epochs, start_value):
#     if epoch_num < decay_start:
#         return start_value
#     return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)

def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1).astype("float") == targets.data.cpu().numpy())

def precision(outputs, targets):
    if np.sum(outputs[:,1]>0.5)==0:
        return 0
    else:
        return  np.sum((outputs[:,1]>0.5) * (targets==1))/sum(outputs[:,1]>0.5)
#def recall(outputs, targets):
#    if np.sum(outputs[:,1]<=0.5)==0:
##        return 0
#    return  np.sum((outputs[:,1]<=0.5) * (targets==0))/sum(outputs[:,1]<=0.5)

#def F1(outputs, targets):
#    p = precision(outputs, targets)
#    r= recall(outputs, targets)
#    if p!=0 and r!=0:
#        return 2/(1/p+1/r)
#    else:
#        return 0
def sensibility(outputs, targets):
    if np.sum(targets)!=0:
        return np.sum((outputs[:,1]>0.5) * (targets==1))/np.sum(targets)
    else:
        return 0 
def specificity(outputs, targets):
    if np.sum(targets==0)!=0:
        return np.sum((outputs[:,1]<=0.5) * (targets==0))/np.sum(targets==0)
    else:
        return 0

def rocauc(outputs, targets):
    from sklearn.metrics import roc_curve, auc
    y= outputs[:,1]
    fpr, tpr, thresholds = roc_curve(targets, y)
    return(auc(fpr, tpr))
    

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
