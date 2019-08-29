import numpy as np
import matplotlib.pyplot as plt
import torch
#%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datasets import SineData
from math import pi

# Create dataset
dataset = SineData(amplitude_range=(-1., 1.),
                   shift_range=(-.5, .5),
                   num_samples=2000)
from neural_process import NeuralProcess

x_dim = 1
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 50  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder

neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)

from torch.utils.data import DataLoader
from original_training import NeuralProcessTrainer

from utils import context_target_split
batch_size = 2
num_context = 4
num_target = 4


data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Extract a batch from data_loader
for batch in data_loader:
    break

# Use batch to create random set of context points
x, y = batch
x_context, y_context = torch.tensor([[[2.8877],
         [0.2856],
         [1.1107],
         [0.8568]]]), torch.tensor([[[-0.4021],
         [-0.0657],
         [-0.7002],
         [-0.5378]]])


for a in range(5):
    alpha = 2**a/5
    neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)

    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                      num_context_range=(num_context, num_context),
                                      num_extra_target_range=(num_target, num_target), 
                                      print_freq=200, alpha=alpha, MMD=False)



    neuralprocess.training = True
    np_trainer.train(data_loader, 80, x_context, y_context)

for a in range(5):
    alpha = 2**a/5
    neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)

    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                      num_context_range=(num_context, num_context),
                                      num_extra_target_range=(num_target, num_target), 
                                      print_freq=200, alpha=alpha, MMD=True)



    neuralprocess.training = True
    np_trainer.train(data_loader, 80, x_context, y_context)



for a in range(5):
    alpha = 2**a/5
    for v in range(5):
        sigma = 2**v/5
        neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim, True, sigma)

        optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
        np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                          num_context_range=(num_context, num_context),
                                          num_extra_target_range=(num_target, num_target), 
                                          print_freq=200, alpha=alpha, MMD=False, 
                                          fixed_sigma=True, sig = sigma)



        neuralprocess.training = True
        np_trainer.train(data_loader, 30, x_context, y_context)