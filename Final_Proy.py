# importing libraries 
import os, math, random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# fixed randomness for reproducible runs
seed = 42  # random
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device

# project configuration
data_dir = "./Data" # folder with NASA files
fd_id = "FD001"   # dataset split to use
w_size = 120     # window size (cycles) 50, 
b_size = 64    # batch size 128,
lr = 1e-3       # learning rate
wd = 1e-4       # weight decay
ep = 30         # number of epochs 50
loss_type = "huber"  # loss_type  # 'huber' or 'mse'
noise_sigma = 0.05  # gaussian augment scale

print(os.getcwd())
print ("")
print(os.listdir("./Data"))

# 3 operating settings + 21 sensors (we drop constant sensors later)
cols_keep = list(range(3, 26))  # keep columns in this range 

