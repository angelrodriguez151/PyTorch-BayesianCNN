import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        bceloss = nn.BCELoss()
        assert not target.requires_grad
        
        return bceloss(input, target, reduction='mean') * self.train_size + beta * kl


# def lr_linear(epoch_num, decay_start, total_epochs, start_value):
#     if epoch_num < decay_start:
#         return start_value
#     return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def acc(outputs, targets):
    return np.mean((outputs.cpu().numpy()>0.5).astype("float") == targets.data.cpu().numpy())

def sensibility(outputs, targets):
    return np.sum((outputs.cpu().numpy()>0.5) * (targets.data.cpu().numpy()==1))/np.sum(outputs.cpu().numpy()>0.5)
def specificity(outputs, targets):
    return np.sum((outputs.cpu().numpy()<=0.5) * (targets.data.cpu().numpy()==0))/np.sum(outputs.cpu().numpy()<=0.5)

def rocauc(outputs, targets):
    from sklearn.metrics import roc_curve, auc
    y= outputs.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(targets.data.cpu().numpy(), y)
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
