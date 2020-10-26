import argparse
import numpy as np
import torch
import os
import torch.nn as nn

# from torch import nn
from tqdm import tqdm
import editdistance

from config import device, print_freq, vocab_size, sos_id, eos_id, IGNORE_ID, word_length
from data_gen import AiShellDataset, pad_collate
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

import pickle
pklfile = 'char_list.pkl'
with open(pklfile, 'rb') as file:
    data = pickle.load(file)
char_list = data
print(char_list)
#torch.cuda.set_device(0)
#os.environ['CUDA_VISIBLE_DEVICES']='2,3,0'

def wer_compute(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        #print(word_pairs)
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        encoder = Encoder(512, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder)
        # print(model)
        #model = nn.DataParallel(model, device_ids=[2,3]).cuda()

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    else:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        
        # optimizer = checkpoint['optimizer']
        # optimizer._update_lr()
        #lr = 0.0002
        optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    logger = get_logger()

    # Move to GPU, if available
    #model = model.to(device)
    #print(model)
    #model = model.module
    #model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    #print(model)
    model = model.to(device)

    # Custom dataloaders
    train_dataset = AiShellDataset(args, 'train')
    #print(train_dataset[0][0].size())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    valid_dataset = AiShellDataset(args, 'val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    # Epochs
    k = 0
    for epoch in range(0, 1):
        # One epoch's validation
        wer = valid(valid_loader=valid_loader,model=model,logger=logger)

        # Check if there was an improvement
        is_best = wer < best_loss
        #is_best = train_loss < best_loss
        best_loss = min(wer, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        #save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def parse_args_decode():
    parser = argparse.ArgumentParser(
        "End-to-End Automatic Speech Recognition Decoding.")
    # decode
    parser.add_argument('--beam_size', default=5, type=int,
                        help='Beam size')
    parser.add_argument('--nbest', default=2, type=int,
                        help='Nbest size')
    parser.add_argument('--decode_max_len', default=100, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    args = parser.parse_args()
    return args

args_decode = parse_args_decode()

def valid(valid_loader, model, logger):
    #print(model)
    model = model.module.module
    model.eval()
    #print(model)
    losses = AverageMeter()
    pred_all_txt = []
    gold_all_txt = []
    # Batches
    wer = float(0)
    i = 0
    for data in tqdm(valid_loader):
        #i += 1
        #if i == 10:
         #  break
        # Move to GPU, if available
        padded_input, padded_target = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        #input_lengths = input_lengths.to(device)
        if padded_target.size(1) <= word_length:
            with torch.no_grad():
                # Forward prop.
                preds_nbest = model.recognize(padded_input, char_list, args_decode)
                #print('the preds_nbest result: ', preds_nbest[0]['yseq'])
                #print(padded_target.cpu().numpy()[0])
                
                gold_txt = []
                
                for arr in padded_target.cpu().numpy():
                    #print(arr)
                    golds = [char_list[one] for one in arr if one not in (sos_id, eos_id, -1)]
                    gold_txt.append(' '.join(golds))
                
                #print(gold_txt)
                
                gold_all_txt.extend(gold_txt)
                
                
                pred_argmax = preds_nbest[0]['yseq']
                preds = pred_argmax
                #print(preds, preds.cpu().numpy())
                pred_txt = []
                #for one in preds:
                    #preds = []
                    #for one in arr:
                    #if one != 0:
                     #       preds.append(char_list[one])
                    #elif one != 1:
                     #       preds.append(char_list[one])
                    #else:
                     #       break
                preds = [char_list[one] for one in preds if one not in (sos_id, eos_id, -1)]
                pred_txt.append(' '.join(preds))
                    #print(pred_txt)
                #print(pred_txt)
                pred_all_txt.extend(pred_txt)
            
                #print(' '.join(golds))
                #print(pred_argmax, gold)
        else:
            break
        #print(pred_all_txt)
    #print(gold_all_txt)
    wer = wer_compute(pred_all_txt, gold_all_txt)
    print('wer: ', wer)

    return wer

def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()