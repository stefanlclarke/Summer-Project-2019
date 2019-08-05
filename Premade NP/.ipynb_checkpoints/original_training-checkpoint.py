import torch
from random import randint
from neural_process import NeuralProcessImg
from torch import nn
from utils import (context_target_split, batch_context_target_mask,
                   img_mask_to_np_input)
from loss_functions import (kl_div, log_likelihood1, compute_kernel, MMD, ELBO)
import loss_functions 
import numpy as np
from math import pi
from matplotlib import pyplot as plt


class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, print_freq=100, alpha = 0.5, MMD=True, fixed_sigma=False, sig=1.):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq
        self.MMD = MMD
        self.fixed_sigma = fixed_sigma
        self.sig = sig

        # Check if neural process is for images
        self.is_img = isinstance(self.neural_process, NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []
        self.alpha = alpha
        
        #for plotting while trianing
        self.x_target_plot = torch.Tensor(np.linspace(-pi, pi, 100)).unsqueeze(1).unsqueeze(0)

    def train(self, data_loader, epochs, x_context_plot, y_context_plot):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    img, _ = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    p_y_pred, q_target, q_context = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    _, y_target = img_mask_to_np_input(img, target_mask)
                else:
                    x, y = data
                    x_context, y_context, x_target, y_target = \
                        context_target_split(x, y, num_context, num_extra_target)
                    p_y_pred, q_target, q_context = \
                        self.neural_process(x_context, y_context, x_target, y_target)
                
                loss = self._loss(p_y_pred, y_target, q_target, q_context, MMD)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))
            
            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))
            
            self.neural_process.training = False

            for i in range(64):
                # Neural process returns distribution over y_target
                p_y_pred = self.neural_process(x_context_plot, y_context_plot, self.x_target_plot)
                # Extract mean of distribution
                mu = p_y_pred.loc.detach()
                std = p_y_pred.stddev.detach()
                plt.plot(self.x_target_plot.numpy()[0], mu.numpy()[0], alpha=0.05,
                         c='b')

            plt.scatter(x_context_plot[0].numpy(), y_context_plot[0].numpy(), c='k')
            
            if self.MMD == True:
                st = 'MMD'
            else:
                st = 'KLD'
                
            if self.fixed_sigma==True:
                ss = 'FIXED_SIGMA={}'.format(self.sig)
            else:
                ss = None
            
            
            plt.savefig('./NewPlots/{}{}alpha{}epoch{}.png'.format(ss, st, self.alpha, epoch))
            plt.clf()
            
            plt.plot(self.x_target_plot.numpy()[0], std.numpy()[0], c='b')
            plt.savefig('./NewPlots/{}{}alpha{}epoch{}_variance.png'.format(ss, st, self.alpha, epoch))
            plt.clf()
            
            self.neural_process.training = True
            

    def _loss(self, p_y_pred, y_target, q_target, q_context, MMD):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        if MMD == True:
            result = loss_functions.MMD_ELBO(p_y_pred, y_target, q_target, q_context, self.alpha)
        else:
            result = loss_functions.ELBO(p_y_pred, y_target, q_target, q_context, self.alpha)
        return result