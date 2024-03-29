import numpy as np
import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
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
    
    writer = SummaryWriter()
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
        #start_epoch = checkpoint['epoch'] + 1
        #epochs_since_improvement = checkpoint['epochs_since_improvement']
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    valid_dataset = AiShellDataset(args, 'val')
    #valid_dataset = AishellDataset(args, 'tst')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    # Epochs
    k = 0
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss, n = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger, k=k)
        k = n
        print('train_loss: ', train_loss)
        writer.add_scalar('model_{}/train_loss'.format(word_length), train_loss, epoch)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model_{}/learning_rate'.format(word_length), lr, epoch)
        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        valid_loss, wer = valid(valid_loader=valid_loader,model=model,logger=logger)
        writer.add_scalar('model_{}/valid_loss'.format(word_length), valid_loss, epoch)
        writer.add_scalar('model_{}/valid_wer'.format(word_length), wer, epoch)

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
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, optimizer, epoch, logger, k):
    writer = SummaryWriter()
    
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    # Batches
    n = k
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        pred, gold = model(padded_input, padded_target)
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()
        
        writer.add_scalar('model_{}/train_iteration_loss'.format(word_length), loss.item(), n)
        n += 1
        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), loss=losses))

        #if n%100 == 0:
         #   break

    return losses.avg, n


def valid(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()
    pred_all_txt = []
    gold_all_txt = []
    # Batches
    wer = float(0)
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, padded_target = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        #input_lengths = input_lengths.to(device)
        if padded_target.size(1) <= word_length:
            with torch.no_grad():
                # Forward prop.
                pred, gold = model(padded_input, padded_target)
                loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
                pred_argmax = pred.argmax(-1)
                preds = pred_argmax
                #print(preds, preds.cpu().numpy())
                pred_txt = []
                for arr in preds.cpu().numpy():
                    preds = []
                    for one in arr:
                        if one != eos_id:
                            preds.append(char_list[one])
                        else:
                            break
                    pred_txt.append(' '.join(preds))
                    #print(pred_txt)
                pred_all_txt.extend(pred_txt)

                gold_txt = []
                for arr in gold.cpu().numpy():
                    golds = [char_list[one] for one in arr if one not in (sos_id, eos_id, -1)]
                    gold_txt.append(' '.join(golds))
                gold_all_txt.extend(gold_txt)

            # Keep track of metrics
            losses.update(loss.item())
        else:
            break

    wer = wer_compute(pred_all_txt, gold_all_txt)
    print('wer: ', wer)
    
    # Print status
    logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return losses.avg, wer


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
