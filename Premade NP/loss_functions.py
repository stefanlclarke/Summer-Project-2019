import torch
from random import randint
from neural_process import NeuralProcessImg
from torch import nn
from torch.distributions.kl import kl_divergence
from utils import (context_target_split, batch_context_target_mask,
                   img_mask_to_np_input)

def log_likelihood1(p_y_pred, y_target):
    return p_y_pred.log_prob(y_target).mean(dim=0).sum()

def kl_div(q_target, q_context):
    return kl_divergence(q_target, q_context).mean(dim=0).sum()

def compute_kernel(x, y):
    return (-(x - y).pow(2)).exp()


def MMD(q_target, q_context, num_samples):
    target_samples_1 = q_target.sample((num_samples,))
    context_samples_1 = q_context.sample((num_samples,))
    target_samples_2 = q_target.sample((num_samples,))
    context_samples_2 = q_context.sample((num_samples,))
    
    t_kernel = compute_kernel(target_samples_1, target_samples_2)
    c_kernel = compute_kernel(context_samples_1, context_samples_2)
    ct_kernel = compute_kernel(target_samples_1, context_samples_1) + compute_kernel(target_samples_2, context_samples_2)
    return (t_kernel + c_kernel - ct_kernel).mean()

def ELBO(p_y_pred, y_target, q_target, q_context, alpha):
    result =  - log_likelihood1(p_y_pred, y_target) + alpha*kl_div(q_target, q_context)
    return result

def MMD_ELBO(p_y_pred, y_target, q_target, q_context, alpha):
    result =  - log_likelihood1(p_y_pred, y_target) + alpha*MMD(q_target, q_context, 10)
    return result

def info_vae()