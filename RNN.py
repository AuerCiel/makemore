import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures

# read in all the words
words = open('Neural Network_Zero to hero/makemore/names.txt', 'r').read().splitlines()