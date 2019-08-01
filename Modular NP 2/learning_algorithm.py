import numpy as np
import torch
import torch.nn as nn

from training import train
from NP import NP
import matplotlib.pyplot as plt
import random
from PIL import Image
from training import train
from NP import NP
from mpl_toolkits.mplot3d import Axes3D

import torchvision
import torchvision.datasets as datasets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

def img_to_vector(imgs):
    data = [torch.from_numpy(np.asarray(img, dtype="int32").reshape(28*28, 1)).unsqueeze(0) for img in imgs]
    return torch.cat(data, dim=0).float()

def vector_to_image(vec):
    array = vec.reshape(28, 28).detach().numpy()
    img = Image.fromarray(array)
    return img.convert('RGB')
                             
mnist_trainset_images = [mnist_trainset[i][0] for i in range(len(mnist_trainset)) if mnist_trainset[i][1] == 2]
mnist_trainset_digits = [mnist_trainset[i][1] for i in range(len(mnist_trainset)) if mnist_trainset[i][1] == 2]
                             
mnist_trainset_arrays = img_to_vector(mnist_trainset_images)
mnist_trainset_digits = torch.tensor(mnist_trainset_digits).unsqueeze(0).transpose(0,1).float()

a = torch.linspace(1, 28, 28)
b = torch.linspace(1, 28, 28)
x_t = a.repeat(28).unsqueeze(0).transpose(0,1)
y_t = b.repeat(28,1).t().contiguous().view(-1).unsqueeze(0).transpose(0,1)

mesh = torch.cat([x_t, y_t], dim=1)

train_data = []
for function in mnist_trainset_arrays:
    train_data.append([mesh, function*1/256])
    
model = NP(128, 2, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

points_test = 500
num_functions_train = 1000

test_x = mesh
perm = torch.randperm(784)
context_y = train_data[0][1][perm][0:points_test]
context_x = train_data[0][0][perm][0:points_test]

context_y1 = [[i,train_data[0][1][i]] for i in perm[0:points_test]]
draw_y1 = []
for i in range(784):
    if i in perm[0:points_test]:
        draw_y1.append(train_data[0][1][i])
    else:
        draw_y1.append(torch.tensor([0.5]))
draw_y1 = torch.cat(draw_y1).reshape(28*28, 1)
img3 = vector_to_image(draw_y1*256)

for i in range(50):
    train(train_data, model, 10, 700, 0.5, optimizer)
    mu, sigma, log_p, en_dist, t_en_dist, MSE = model(context_x, context_y, test_x)
    img1 = vector_to_image(mu.squeeze()*256)
    img2 = vector_to_image(train_data[0][1]*256)
    display(img2)
    display(img3)
    display(img1)
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(train_data[0][0].numpy()[:,0], train_data[0][0].numpy()[:,1], train_data[0][1].detach().numpy(), c='blue', alpha=0.5)
    ax.scatter(train_data[0][0].numpy()[:,0], train_data[0][0].numpy()[:,1], mu.detach().numpy(), c='red', alpha=0.5)
    plt.savefig('./Models/epoch{}img'.format(10*i))
    
    
    if i % 10 == 0:
        torch.save(model.state_dict(), './Models/epoch{}'.format(10*i))
    
    '''
    plt.plot(train_data[1][0].numpy(), train_data[1][1].numpy())
    plt.plot(train_data[1][0].detach().numpy(), mu.detach().numpy())
    plt.show()
    '''