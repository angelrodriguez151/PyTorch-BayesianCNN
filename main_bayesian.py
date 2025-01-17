from __future__ import print_function

import os
import argparse

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

import data
import utils
import metrics
import config_bayesian as cfg
from models.BayesianModels.Bayesianmymodel import BBBmymodel
from models.BayesianModels.Bayesianmymodel import BBBmymodel1
from models.BayesianModels.Bayesianmymodel import BBBmymodel1Layer
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
from models.BayesianModels.Bayesian1DConv import BBBConv1
from models.BayesianModels.Bayesian1DConv import BBBLinear2

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'mymodel'):
        return BBBmymodel(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'mymodel1'):
        return BBBmymodel1(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'conv1'):
        return BBBConv1(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'features'):
        return BBBLinear2(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '1layer'):
        return BBBmymodel1Layer(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')

def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()
        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []
    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] =  F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)
        beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))


    return valid_loss/len(validloader), np.mean(accs)

def testing(net,  testloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate accuracy and mean roc-auc"""
    
    from torch import nn
    valid_loss = 0.0
    accs = []
    ou = np.empty((0,2))
    la = np.array([])

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = torch.zeros(inputs.shape[0],  net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :,j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)
        beta = metrics.get_beta(i-1, len(testloader), beta_type, epoch, num_epochs)
        accs.append(metrics.acc(log_outputs, labels))

        ou = np.concatenate([ou, np.exp(log_outputs.cpu().numpy())])
        la = np.concatenate([la, labels.cpu().numpy()])
        
    precision=(metrics.precision(ou, la))

    spec = (metrics.specificity(ou, la))
    sens = (metrics.sensibility(ou, la))

    return  np.mean(accs),precision, spec, sens

def tunning_1(dataset, net_type, n_epochs):
    import numpy as np
    sigma_list= [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]
    # values=[]
    # for priorsigma in sigma_list :
    #     cfg.priors["prior_sigma"]= priorsigma
    #     a, b, c =  run(dataset, net_typ,n_epochs)
    #     print(a)
    #     values.append(a[3])
    # cfg.priors["prior_sigma"] = sigma_list[np.argmax(np.array(values))]
    values=[]

    for posteriormu in sigma_list :
        cfg.priors["posterior_mu_initial"]= (0,posteriormu)
        a, b, c =  run(dataset, net_type,n_epochs)
        print(a)
        values.append(a[3])
    cfg.priors["posterior_mu_initial"] = (0,sigma_list[np.argmax(np.array(values))])

    values=[]
    for posteriorsigma1 in sigma_list:
        cfg.priors["posterior_rho_initial"] = (cfg.priors["posterior_rho_initial"][0], posteriorsigma1)
        a, b, c  = run(dataset, net_type, n_epochs)
        print(a)
        values.append(a[3])
    cfg.priors["posterior_rho_initial"] =(cfg.priors["posterior_rho_initial"][0], sigma_list[np.argmax(np.array(values))])

    values=[]
    lista2=[-7,-6,-5,-4,-3,-2,-1]
    for posteriorsigma2 in lista2:
        cfg.priors["posterior_rho_initial"] =(posteriorsigma2, cfg.priors["posterior_rho_initial"][1])
        a, b, c = run(dataset, net_type, n_epochs)
        print(a)
        values.append(a[3])
    cfg.priors["posterior_rho_initial"] = (lista2[np.argmax(np.array(values))], cfg.priors["posterior_rho_initial"][1])

    
    
    
def run(dataset, net_type,n_epochs = cfg.n_epochs):

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors
    print(priors)
    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf
    valid_acc_max = 0
    trainaccuracy = []
    valaccuracy  = []
    valloss=[]
    import time
    t1=time.time()
    print("starting to train")
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)
        trainaccuracy.append(train_acc)
        valaccuracy.append(valid_acc)
        valloss.append(valid_loss)
        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_acc_max<=valid_acc: #valid_loss <= valid_loss_max:
             print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
             torch.save(net.state_dict(), ckpt_name) 
             valid_acc_max = valid_acc
    print("Testing best model yet")
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)
    net.load_state_dict(torch.load(ckpt_name)) 
    net.eval()
    accs,precision,spec, sens = testing(net, test_loader)
    t2= time.time()-t1       
    return (t2, accs,precision, spec, sens), trainaccuracy, valaccuracy, valloss




if __name__ == '__main__':

    run("miset","1layer")
