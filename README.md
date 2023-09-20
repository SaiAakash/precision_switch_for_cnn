# Precision Switching for CNN

A simple interactive dashboard to switch precision of weights and activations for different layers of a convolutional neural network during training.

**Installation Instructions** 

```bash
$ git clone https://github.com/SaiAakash/precision_switch_for_cnn.git
$ cd precision_switch_for_cnn
```bash

Creating a Conda environment

```bash
$ conda create --name precision_switch_env python=3.8
$ conda activate precision_switch_env

**Installing the required dependencies**

```bash
$ python -m pip install flask
$ python -m pip install torchvision==0.15.1
$ python -m pip install matplotlib pandas

**Running the training script**

```bash
$ python mnist_training.py

**Running the precision switching framework along with the training script**

In one terminal, run the following command

```bash
$ python app.py

Select precision for different layers on the dashboard.

In the second terminal, run the training script

```bash
$ python mnist_training.py

Keep changing the precision settings in the dashboard to train the CNN with different precision combinations for different layers during the training.

To test the model, run the following command to predict on individual images from the MNIST dataset.

```bash
$ python make_predictions.py
