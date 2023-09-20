from __future__ import print_function
from app import change_precision, update_precision_settings_file
from helper_funcs import *
from conv_net import Net
from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import random
import requests
import json
from requests.exceptions import ConnectionError
import time
import csv
import os

torch.manual_seed(seed_value)
# torch.set_default_dtype(torch.float64)    # Uncomment to perform full training in Float64 precision

################# Sanity Check Function for Weight Precision ################

def sanity_check_precision(model, precision_dict):
  for key in precision_dict.keys():
    if model.layers[key].weight.dtype == precision_dict[key] and model.layers[key].bias.dtype == precision_dict[key]:
      continue
    else:
      print("Sanity Check status: FAILED")
      return False
  print("Sanity Check status: OKAY!")
  return True

############## Train and Test functions ##############

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    layer_weights = list(model.parameters())
    trainable_layer_indices = model.get_trainable_layer_indices()
    precision_dict = get_layer_precision_in_dict(layer_weights, trainable_layer_indices)
    for batch_idx, (data, target) in enumerate(train_loader):
        flag = 0
        start_time = time.time()
        if batch_idx % precision_change_freq == 0:
          print("Current Precision Values:", precision_dict)
          precision_settings = get_precision_settings()
          if precision_settings == None:
            pass
          else:
            precision_dict_new = convert_dict(precision_settings, trainable_layer_indices)
            for key in precision_dict_new.keys():
              if precision_dict_new[key] != precision_dict[key]:
                flag = 1
                model.set_precision(key, precision_dict_new[key])
                precision_dict[key] = precision_dict_new[key]
              else:
                pass
          if flag:
            print("Updated Precision Values:", precision_dict)
          else:
            print("Continuing with same precision values")
        correct = 0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_acc = 100. * correct / len(data)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        batch_time = end_time - start_time
        send_metrics(round(loss.item(), 5), round(train_acc, 5), round(batch_time, 5))
        log_training_data(epoch, batch_idx, loss.item(), train_acc, batch_time, precision_dict, 'training_log_f32.csv')
        if batch_idx % log_interval == 0:
            sanity_check_status = sanity_check_precision(model, precision_dict)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss.item(), correct, len(data),
            train_acc))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

############### GPU_device_check ################

use_cuda = torch.cuda.is_available()
if use_cuda:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

############ Defining Training Arguments ############\

train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}
if use_cuda:
  cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
  train_kwargs.update(cuda_kwargs)
  test_kwargs.update(cuda_kwargs)

########### Dataset and Pre-processing #############

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

############ Creating Model and Optimizer instances ###########

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=alpha)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

############### Training Loop ############

training_start_time = time.time()
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

training_end_time = time.time()

####### Total training time and maximum memory used for the training #######

total_training_time = training_end_time - training_start_time
print('Total training time: {:.4f} min'.format(total_training_time/60))
print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

if save_model:
   torch.save(model.state_dict(), "mnist_cnn.pt")


###### Model Size ######

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.8f}MB'.format(size_all_mb))


clean_json()



