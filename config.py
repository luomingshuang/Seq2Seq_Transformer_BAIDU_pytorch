import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
input_dim = 80  # dimension of feature
window_size = 25  # window size for FFT (ms)
stride = 10  # window stride for FFT (ms)
hidden_size = 512
embedding_dim = 512
cmvn = True  # apply CMVN on feature
num_layers = 4
LFR_m = 4
LFR_n = 3
sample_rate = 16000  # aishell

# Training parameters
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 2  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
IGNORE_ID = -1
sos_id = 0
eos_id = 1
num_train = 120098
num_dev = 14326
num_test = 7176

vocab_size = 94
word_length = 6
imgs_padding = 75

imgs_path = '.../BAIDU/lip_imgs'
annotation_root = '.../BAIDU/sentences_data'
trn_txt = '.../BAIDU/sentences_data/trn_seqs_ys.txt'
val_txt = '.../BAIDU/sentences_data/val_seqs_ys.txt'
tst_txt = '.../BAIDU/sentences_data/tst_seqs_ys.txt'

pickle_file = 'BAIDU.pickle'
