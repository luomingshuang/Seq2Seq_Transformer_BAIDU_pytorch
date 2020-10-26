import pickle
import random
import cv2
import torch
import os,math

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import pickle_file, IGNORE_ID, imgs_path, word_length, imgs_padding

from cvtransforms import *
#from utils import extract_feature

#words_dict = {0: '<sos>', 1: '<eos>', 2: 'BIN', 3: 'GREEN', 4: 'AT', 5: 'A', 6: 'FIVE', 7: 'SOON', 8: 'BLUE', 9: 'F', 10: 'ONE', 11: 'SIX', 12: 'PLEASE', 13: 'TWO', 14: 'SEVEN', 15: 'AGAIN', 16: 'THREE', 17: 'G', 18: 'EIGHT', 19: 'NOW', 20: 'ZERO', 21: 'H', 22: 'L', 23: 'FOUR', 24: 'NINE', 25: 'N', 26: 'BY', 27: 'B', 28: 'R', 29: 'S', 30: 'Z', 31: 'U', 32: 'IN', 33: 'M', 34: 'E', 35: 'T', 36: 'WITH', 37: 'Y', 38: 'O', 39: 'RED', 40: 'WHITE', 41: 'LAY', 42: 'D', 43: 'J', 44: 'K', 45: 'Q', 46: 'X', 47: 'P', 48: 'PLACE', 49: 'C', 50: 'I', 51: 'V', 52: 'SET'}

def HorizontalFlip(batch_img):
    if random.random() > 0.5:
        batch_img = batch_img[:,:,::-1,...]
    return batch_img

def FrameRemoval(batch_img):
    for i in range(batch_img.shape[0]):
        if(random.random() < 0.05 and 0 < i):
            batch_img[i] = batch_img[i - 1]
    return batch_img
    
    
def ColorNormalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img

def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        feature, trn = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        if len(feature.shape) == 2:
            feature, trn = elem
            input_length = feature.shape[0]
            input_dim = feature.shape[1]
            padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
            padded_input[:input_length, :] = feature
            padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=IGNORE_ID)
            batch[i] = (padded_input, padded_target, input_length)
        else:
            feature, trn = elem
            input_length = feature.shape[0]
            #print(feature[0].shape)
            input_dim1 = feature[0].shape[0]
            input_dim2 = feature[0].shape[1]
            input_dim3 = feature[0].shape[2]
            padded_input = np.zeros((max_input_len, input_dim1, input_dim2, input_dim3), dtype=np.float32)
            padded_input[:input_length, :, :, :] = feature
            padded_target = np.pad(trn, (0, word_length - len(trn)), 'constant', constant_values=IGNORE_ID)
            batch[i] = (padded_input, padded_target, input_length)
    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
        else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i * n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


class AiShellDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)
        #print(data)
        self.samples = data[split]
        print(len(self.samples))
        
        self.samples = list(filter(lambda sample:len(sample['trn'])<=word_length, self.samples))
        #print(self.samples['trn'])
        print('loading {} {} samples...'.format(len(self.samples), split))

    def __getitem__(self, i):
        sample = self.samples[i]
        img_folder = sample['name']
        img_files = sample['imgs']
        labels = sample['trn']
        #print(len(labels))
        images = [cv2.imread(os.path.join(img_folder, file)) for file in img_files]
        images = [cv2.resize(img, (96, 96)) for img in images]
        
        for i in range(imgs_padding-len(images)):
            images.append(np.zeros((96, 96, 3), dtype=np.uint8))
        
        #print('every image shape is:', images[0].shape)
        ###gray
        images = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                      for img in images], axis=0) / 255.0
        ###color
        #images = np.stack([img for img in images]) / 255.0
        
        #print('images is:')
        #print(images.shape)
        labels = np.pad(labels, (0, word_length - len(labels)), 'constant', constant_values=IGNORE_ID)
        if(self.split == 'train'):
            images = RandomCrop(images, (88, 88))
            images = ColorNormalize(images)
            images = HorizontalFlip(images)
            images = RandomDrop(images)
        elif self.split == 'val' or self.split == 'tst':
            images = CenterCrop(images, (88, 88))
            images = ColorNormalize(images)
        #print(images.shape)
        return torch.FloatTensor(np.ascontiguousarray(images)), torch.LongTensor(labels)

        #return feature, trn
        #print(vid.shape)
        #print(len(trn), vid.transpose(0,1,2,3).shape)
        #return torch.FloatTensor(vid.transpose(3, 0, 1, 2)), torch.LongTensor(trn)
        #return torch.FloatTensor(vid), torch.LongTensor(trn)


    def __len__(self):
        return len(self.samples)

    def _load_vid(self, p): 
        #files = sorted(os.listdir(p))
        files = sorted(os.listdir(p), key=lambda x:int(x.split('.')[0]))        
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        # array = [cv2.resize(im, (50, 100)).reshape(50, 100, 3) for im in array]
        array = [cv2.resize(im, (160, 160)) for im in array]
        #print(p, len(array))
        array = np.stack(array, axis=0)
        return array
    
    def _load_boundary(self, arr, time):
        st = math.floor(float(time[0]) * 25)
        ed = math.ceil(float(time[1]) * 25)
        return arr[st:ed]

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    def _padding_ones(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.ones(size))
        return np.stack(array, axis=0)


if __name__ == "__main__":
    import torch
    from utils import parse_args
    from tqdm import tqdm

    args = parse_args()
    #train_dataset = AiShellDataset(args, 'train')

    train_dataset = AiShellDataset(args, 'train')
    #print(train_dataset[0][0].size())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,  pin_memory=True, shuffle=True, num_workers=args.num_workers)

    valid_dataset = AiShellDataset(args, 'val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers)
    ##collate_fn=pad_collate, 
    tst_dataset = AiShellDataset(args, 'tst')
    tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers)

    print('train_dataset: ', len(train_dataset), len(train_loader))
    print('valid_dataset: ', len(valid_dataset), len(valid_loader))
    print('tst_dataset: ', len(tst_dataset), len(tst_loader))
    # for i, batch in enumerate(train_loader):
    #     padded_input, padded_target = batch
    #     print(padded_input.size())
    #
    # print(len(train_dataset))
    # print(len(train_loader))
    #
    # feature = train_dataset[10][0]
    # print(feature.shape)
    #
    # trn = train_dataset[10][1]
    # print(trn)
    #
    # with open(pickle_file, 'rb') as file:
    #     data = pickle.load(file)
    # IVOCAB = data['IVOCAB']
    #
    # print([IVOCAB[idx] for idx in trn])
    #
    # for data in train_loader:
    #     print(data)
    #     break

    max_len = 0

    for data in tqdm(train_loader):
        feature = data[0]
        # print(feature.shape)
        if feature.shape[1] > max_len:
            max_len = feature.shape[1]

    print('max_len: ' + str(max_len))