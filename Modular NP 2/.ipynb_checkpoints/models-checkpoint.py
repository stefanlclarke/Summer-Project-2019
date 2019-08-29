import numpy as np
import torch
import torch.nn as nn

hidden_layer_size = 100

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)


class encoder(nn.Module):
    def __init__(self, output_sizes, x_dimension, y_dimension):
        super(encoder, self).__init__()
        self.output_sizes = output_sizes
        
        self.fc1 = nn.Linear(x_dimension+y_dimension, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc21 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc22 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, self.output_sizes)
        self.relu = nn.ReLU()
        
    def forward(self, x, y):
        #x and y are 1xn dimensional torch tensors
        
        xy = torch.cat([x, y], dim=1)
        l1 = self.relu(self.fc1(xy))
        l21 = self.relu(self.fc21(l1))
        l22 = self.relu(self.fc22(l21))
        l2 = self.relu(self.fc2(l22))
        R = self.fc3(l2)
    
        r_agg = torch.mean(R, dim=0).reshape(1, self.output_sizes)
        return r_agg
    
class r_to_z(nn.Module):
    def __init__(self, encoded_size):
        super(r_to_z, self).__init__()
        self.encoded_size = encoded_size
        
        self.fc1 = nn.Linear(self.encoded_size, hidden_layer_size)
        self.fc2_mu = nn.Linear(hidden_layer_size, self.encoded_size)
        self.fc2_logvar = nn.Linear(hidden_layer_size, encoded_size)
        self.relu = nn.ReLU()
        
    def forward(self, r):
        mu = self.fc2_mu(self.relu(self.fc1(r))).reshape(self.encoded_size)
        logvar = self.fc2_logvar(self.relu(self.fc1(r)))
        std = torch.exp(logvar.mul(1/2)).reshape(self.encoded_size)
        dist = torch.distributions.Normal(mu, std)
        
        mu_out = mu.reshape(self.encoded_size, 1)
        sigma_out = std.reshape(self.encoded_size, 1)
        return mu_out, sigma_out, dist
    
class decoder(nn.Module):
    def __init__(self, encoded_size, x_dimension, y_dimension):
        super(decoder, self).__init__()
        self.encoded_size = encoded_size
        
        self.fc1 = nn.Linear(self.encoded_size + x_dimension, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc21 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc22 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3_mu = nn.Linear(hidden_layer_size, y_dimension)
        self.fc3_sigma = nn.Linear(hidden_layer_size, y_dimension)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        
    def forward(self, z, x):
        #x and z are 1xn dimensional torch tensors
        
        zmulti = torch.cat([z for i in range(x.size()[0])], dim=1).transpose(0,1)
        xz = torch.cat([x, zmulti], dim=1)
        
        nicolas = self.relu(self.fc1(xz))
        l2 = self.relu(self.fc2(nicolas))
        l21 = self.relu(self.fc21(l2))
        l22 = self.relu(self.fc22(l21))
        out_mu = self.fc3_mu(l22)
        out_logvar = self.fc3_sigma(l22)
        out_sigma = self.softplus(out_logvar)
        
        dist = torch.distributions.Normal(out_mu, out_sigma)
        
        return out_mu, out_sigma, dist
    
    
class convlayers(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(convlayers, self).__init__()
        self.convdim = outputSize(outputSize(input_dimension, 2, 2, 0), 2, 2, 0)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(self.convdim**2, self.output_dimension)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #x shape([1, 1, input_dimension, input_dimension])
        l1 = self.pool1(self.relu(self.conv1(x)))
        l2 = self.pool2(self.relu(self.conv2(l2)))
        l2 = l2.view(1, self.convdim**2)
        l3 = self.fc1(l2)
        return l3
    
class deconvlayers(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(deconvlayers, self).__init__()
        self.insize = outputSize(outputSize(output_dimension, 2, 2, 0), 2, 2, 0)
        self.fc1 = nn.Linear(input_dimension, self.insize**2)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        l1 = self.relu(self.fc1(x))
        l1 = l1.view(1, 1, self.insize, self.insize)
        l2 = self.relu(self.deconv1(self.unpool1(l1)))
        l3 = self.relu(self.deconv2(self.unpool2(l2)))
        return l3