############### Configuration file for Bayesian ###############
layer_type = 'lrt' # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.01,
    'posterior_mu_initial': (0, 0.01),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.01),  # (mean, std) normal_
}

n_epochs = 40
lr_start = 0.0005
num_workers = 2
valid_size = 0.2
batch_size = 32
train_ens = 1
valid_ens = 1
beta_type = 0.01 # 'Blundell', 'Standard', etc. Use float for const value
