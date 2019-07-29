import numpy as np
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from progress_bar import printProgressBar

def compute_kernel(x, y):
    return (-(x - y).pow(2)).exp()

def MMD(z_prior, z_posterior, num_samples):
    target_samples_1 = z_prior.sample((num_samples,))
    context_samples_1 = z_posterior.sample((num_samples,))
    target_samples_2 = z_prior.sample((num_samples,))
    context_samples_2 = z_posterior.sample((num_samples,))
    
    t_kernel = compute_kernel(target_samples_1, target_samples_2)
    c_kernel = compute_kernel(context_samples_1, context_samples_2)
    ct_kernel = compute_kernel(target_samples_1, context_samples_1) + compute_kernel(target_samples_2, context_samples_2)
    return (t_kernel + c_kernel - ct_kernel).mean()
    

def lossf(log_p, z_prior, context_z_posterior, test_z_posterior, alpha, MSE):
    
    kld = kl_divergence(context_z_posterior, test_z_posterior).prod()
    return -log_p + alpha*kld


def train(data, cnp, epochs, num_test_maximum, alpha, optimizer):
    cnp.train()
    for epoch in range(epochs):
        total_loss = 0
        iteration = 0
        length = len(data)
        printProgressBar(0, length, prefix = 'Epoch {} Progress:'.format(epoch + 1), suffix = 'Complete', length = 100)
        for i, function in enumerate(data):
            optimizer.zero_grad()
            num_points = function[0].size()[0]
            perm = torch.randperm(num_points)
            num_context = np.random.randint(num_points - num_test_maximum, num_points)
            context_x = function[0][perm][0:num_context]
            context_y = function[1][perm][0:num_context]
            test_x = function[0][perm][num_context:num_points]
            test_y = function[1][perm][num_context:num_points]
            mu, sigma, log_p, en_dist, t_en_dist, MSE = cnp(context_x, context_y, test_x, test_y)
            loss = lossf(log_p, torch.distributions.Normal(torch.zeros(cnp.encoded_size), 1), en_dist, t_en_dist, alpha, MSE)
            
            loss.backward()
            optimizer.step()
    
            total_loss += loss/len(data)
            
            iteration += 1
            
            printProgressBar(i + 1, length, prefix = 'Epoch {} Progress:'.format(epoch + 1), suffix = 'Complete.  Iteration = {}. Average loss = {}.'.format(iteration, total_loss), length = 50)
        print('EPOCH LOSS {}'.format(total_loss))
            
            