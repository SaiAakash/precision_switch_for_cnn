import requests 
import json
import torch
from requests.exceptions import ConnectionError
import os
import time
import csv

########## Helper Functions ###########

####### Function to get precision of all trainable layers(layers with weights) of the CNN ######

def get_layer_precision_in_dict(layer_data, trainable_layer_indices):
  layer_precision = {}
  for layer_index in trainable_layer_indices:
    for i in range(0, len(layer_data), 2):
      layer_precision[layer_index] = layer_data[i].dtype
  return layer_precision

####### Function to read precision settings from a json file #######

def get_precision_settings():
  try:
    with open('precision_settings.json', 'r') as file:
      return json.load(file)
  except FileNotFoundError:
    print('Precision Settings File Does Not Exist \n')
    print('Continuing training with default precision \n')
    return None
  
####### Function to convert the data read from the json file to a suitable dictionary format ######## 

def convert_dict(precision_dict, trainable_indices):
  converted_dict = {}
  for key in precision_dict.keys():
    converted_dict[trainable_indices[int(key) - 1]] = eval(precision_dict[key])
  return converted_dict

####### Function to send the metrics to the flask app for display on the webpage #######

def send_metrics(loss, accuracy, batch_time):
    data = {
        'loss': loss,
        'accuracy': accuracy,
        'batch_time': batch_time
    }
    try:
        response = requests.post('http://localhost:5000/update_metrics', json=data)
        return response
    except ConnectionError as ce:
        return None

####### Function to log the training data to a csv file ########

def log_training_data(epoch, batch_idx, loss, accuracy, batch_time, precision_dict, csv_filename='training_log_mpt.csv'):
    if not os.path.isfile(csv_filename):
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Batch', 'Loss', 'Accuracy', 'Batch_Training_Time', 'Precison_Settings'])

    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, batch_idx, loss, accuracy, batch_time, precision_dict])

####### Function to delete the json file once the training is complete #######

def clean_json():
  if os.path.exists('precision_settings.json'):
    os.remove('precision_settings.json')
