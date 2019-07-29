import numpy as np
import torch
import torch.nn as nn
from models import encoder, r_to_z, decoder

class NP(nn.Module):
    def __init__(self, encoded_size, x_dim, y_dim):
        super(NP, self).__init__()
        self.encoded_size = encoded_size
        self._encoder = encoder(encoded_size, x_dim, y_dim)
        self._rz = r_to_z(encoded_size)
        self._decoder = decoder(encoded_size, x_dim, y_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, context_x, context_y, target_x, target_y=None):
        en_mu, en_sigma, en_dist = self._rz(self._encoder(context_x, context_y))
        if target_y is not None:
            t_en_mu, t_en_sigma, t_en_dist = self._rz(self._encoder(target_x, target_y))
            representation = t_en_dist.rsample().unsqueeze(0).transpose(0,1)
        else:
            representation = en_dist.rsample().unsqueeze(0).transpose(0, 1)
            t_en_dist = None

        mu, sigma, dist = self._decoder(representation, target_x)
        
        if target_y is not None:
            log_p = dist.log_prob(target_y)
            log_p = log_p.sum()/(torch.tensor(list(target_y.size())).prod())
            
            
            MSE = (mu - target_y).pow(2).sum()/mu.size()[1]
        else:
            log_p = None
            MSE = None
        return mu, sigma, log_p, en_dist, t_en_dist, MSE