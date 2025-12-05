# %% [markdown]
# # Remaining Useful Life Prediction
#
# Remaining useful life (RUL) prediction estimates how many operational cycles remain  
# before a machine fails. This is central to predictive maintenance—scheduling repairs  
# before failure occurs rather than reacting to breakdowns or replacing components on  
# fixed schedules. The challenge lies in learning degradation patterns from multivariate  
# sensor data where failure modes are complex and equipment operates under varying  
# conditions.
#
# The NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS) contains run-to-  
# failure data from turbofan engines. The dataset includes four subsets (FD001, FD002,  
# FD003, FD004) with increasing complexity based on operating conditions and failure  
# modes. Each engine runs until failure in the training set, providing complete  
# degradation trajectories. The test set provides partial trajectories and you must  
# predict RUL at the final observed timestep.

# %% [markdown]
# # Suggested Approach 
from IPython.display import Image, display
display(Image(filename="img1.png"))


# %% 
# importing libraries 
import os, math, random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
print ("libraries imported succesfully")

# %% 
# fixed randomness for reproducible runs
seed = 42  # random
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device

# %% 
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

# %% 
# loads datset 
def load_fd_split(data_dir: str, fd_id: str):
    """
    loads train/test and RUL files for a given fd subset
   # Input:  data_dir and fd_id 'FD001' ...
   # Output: train_raw [Ntrain, 26], test_raw [Ntest, 26], rul_true [Neng_test]
""" 
    tr_path = os.path.join(data_dir, f"train_{fd_id}.txt")
    te_path = os.path.join(data_dir, f"test_{fd_id}.txt")
    ru_path = os.path.join(data_dir, f"RUL_{fd_id}.txt")

    # space-separated
    train_raw = np.loadtxt(tr_path)
    test_raw  = np.loadtxt(te_path)
    rul_true  = np.loadtxt(ru_path).reshape(-1)  # only one RUL per test engine
    
    return train_raw, test_raw, rul_true

train_raw, test_raw, rul_true = load_fd_split(data_dir, fd_id)


# feature  extraction and normalization
# raw columns: [0:id, 1:cycle, 2:setting1, 3:setting2, ..., sensors...]

idx_id, idx_t = 0, 1
feat_idx = np.array(cols_keep) - 1  # convert to 0 based

# extracts features only
X_tr_full = train_raw[:, feat_idx]   # features for train
X_te_full = test_raw[:,  feat_idx]  # features for test
eng_tr = train_raw[:, idx_id].astype(int)
eng_te = test_raw[:,  idx_id].astype(int)
t_tr = train_raw[:, idx_t].astype(int)
t_te = test_raw[:,  idx_t].astype(int)

# drops constant sensors (zero variance)
var_tr = X_tr_full.var(axis=0)
keep_mask = var_tr > 1e-6
X_tr_full = X_tr_full[:, keep_mask]
X_te_full = X_te_full[:, keep_mask]
feat_names_kept = np.arange(feat_idx.size)[keep_mask]
n_feat = X_tr_full.shape[1]  # number of kept features

# normalizes using traning stats only
mu = X_tr_full.mean(axis=0)    # mean per feature
sd = X_tr_full.std(axis=0) + 1e-8   # std per feature
X_tr_full = (X_tr_full - mu) / sd
X_te_full = (X_te_full - mu) / sd

# re-attach engine id + cycle columns 
train_rec = np.concatenate([train_raw[:, [idx_id, idx_t]], X_tr_full], axis=1)  # [id, t, feats...]
test_rec  = np.concatenate([test_raw[:,  [idx_id, idx_t]], X_te_full], axis=1)

# %% 
# compute RUL for training rows

def compute_train_rul(rec: np.ndarray):
    """
    Computes Remaining Useful Life (RUL) for each row of the TRAIN set
    for every engine: RUL = (last cycle) - (current cycle)
   """
   
    eng = rec[:, 0].astype(int)
    t   = rec[:, 1].astype(int)
    rul = np.zeros_like(t)
    
    for e in np.unique(eng):
        m = (eng == e)
        t_e = t[m]
        t_last = t_e.max()
        rul[m] = t_last - t_e
        
    return rul

# extracts last rows for TEST set
def compute_test_last_windows(rec: np.ndarray, rul_true: np.ndarray):
    """
    For test set:
    1 extract one final row per engine
    2 return those rows + their true RUL
    3 this is the window to make predictions on
    """ 
    
    eng = rec[:, 0].astype(int)
    t   = rec[:, 1].astype(int)
    engines = np.unique(eng)
    rows_last = []
    
    for i, e in enumerate(engines):
        m = eng == e
        idx = np.argmax(t[m]) #  last cycle idx
        rows_last.append(rec[m][idx])
    y_true = rul_true.copy()
    
    return rows_last, y_true

y_tr_full = compute_train_rul(train_rec)  # per row RUL in train

# %%
# %%