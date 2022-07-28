############### Configuration file for Bayesian ###############
layer_type = 'lrt' # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.01,
    'posterior_mu_initial': (0, 0.5),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.5),  # (mean, std) normal_
}

n_epochs = 200
lr_start = 0.01
num_workers = 2
valid_size = 0.3
batch_size = 4
train_ens = 1
valid_ens = 1
beta_type = 0.1 # 'Blundell', 'Standard', etc. Use float for const value
