import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


######### Neural Network Architecture ############

class MaxPool(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        return self.pool(x)

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

class Log_Softmax(nn.Module):
  def forward(self, x):
    return F.log_softmax(x, dim = 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(1, 32, 3, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, 3, 1),
                                     nn.ReLU(),
                                     MaxPool(kernel_size = 2),
                                     nn.Dropout(0.25),
                                     Flatten(),
                                     nn.ReLU(),
                                     nn.Linear(9216, 128),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(128, 10),
                                     Log_Softmax()
                                     ])

    ################ Precision Switching Function ####################

    def set_precision(self, layer_num, layer_precision):
          self.layers[layer_num].weight = nn.Parameter(self.layers[layer_num].weight.to(dtype=layer_precision))
          self.layers[layer_num].bias = nn.Parameter(self.layers[layer_num].bias.to(dtype=layer_precision))
          self.layers[layer_num].to(dtype=layer_precision)

    ##################################################################

    def forward(self, x):
      for i in range(len(self.layers)-1):
        weights_exist = any(param.requires_grad for param in self.layers[i].parameters())
        if weights_exist:
          x = x.to(dtype = self.layers[i].weight.dtype)
          x = self.layers[i](x)
        else:
          x = self.layers[i](x)
      output = self.layers[-1](x)
      return output

  ######## Get the indices of layers that contain weights ########

    def get_trainable_layer_indices(self):
      trainable_layer_indices = []
      for i in range(len(self.layers)):
         weights_exist = any(param.requires_grad for param in self.layers[i].parameters())
         if weights_exist:
          trainable_layer_indices.append(i)
      return trainable_layer_indices