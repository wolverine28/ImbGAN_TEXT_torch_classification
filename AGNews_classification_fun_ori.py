# -*- coding:utf-8 -*-

import os
import random
import math
import dill 

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import torchtext
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset

from classifier import Classifier
from utils import get_oversampled_loader,  save_checkpoint, get_imbalanced_loader, MyDataset_origin, get_overlapped_datset, get_datset
from word_correction import tokenizer


def run(IR,rep,GPU_NUM,OR):

    output_file = '{:s}/rep_{:02d}_IR_{:.4f}_ROS_AGNews.csv'.format('./output',rep,IR)
    if os.path.exists(output_file):
        return 0

    #########################################################################################
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    # Additional Infos
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(GPU_NUM))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')
    #########################################################################################

    # ================== Parameter Definition =================
    USE_CUDA = True
    BATCH_SIZE = 128
    SEQ_LEN = 100

    emb_dim = 200
    hidden_dim = 100
    u_sample_ratio = IR
    # ================== Dataset Definition =================
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    train_iter = datasets.AG_NEWS(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>","<pad>","<sos>"], min_freq=7)
    vocab.set_default_index(vocab["<unk>"])

    UNK, PAD, SOS = 0, 1, 2

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    def collate_batch(batch, SEQ_LEN):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            processed_text = torch.cat((torch.tensor([SOS]),processed_text))
            if len(processed_text)>SEQ_LEN:
                processed_text = processed_text[:SEQ_LEN]
            else:
                pad = torch.tensor(PAD).repeat(SEQ_LEN-len(processed_text))
                processed_text = torch.cat((processed_text,pad))
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.stack(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)


    train_iter, test_iter = datasets.AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    custom_collate_batch = lambda batch: collate_batch(batch,SEQ_LEN+1)

    VOCAB_SIZE = len(vocab)
    print('VOCAB_SIZE : {}'.format(VOCAB_SIZE))
    vocab.itos = vocab.get_itos()


    # Undersampling
    negative_subset = [i for i in train_dataset if np.in1d(i[0],[1,2])]
    positive_subset = [i for i in train_dataset if np.in1d(i[0],[3,4])]
    subset = positive_subset
    if u_sample_ratio != 1:
        count = int(len(subset)/u_sample_ratio)
        subset_idx = np.random.choice(range(len(subset)),count)
        subset = np.array(subset)[subset_idx]
    train_dataset = np.concatenate((np.array(negative_subset),subset))

    ori_label = train_dataset[:,0].astype(np.int)
    bin_label = (ori_label>=3)*1
    train_dataset = MyDataset_origin(train_dataset[:,1],bin_label,ori_label)

    negative_subset = [i for i in test_dataset if np.in1d(i[0],[1,2])]
    positive_subset = [i for i in test_dataset if np.in1d(i[0],[3,4])]
    subset = positive_subset
    if u_sample_ratio != 1:
        count = int(len(subset)/u_sample_ratio)
        subset_idx = np.random.choice(range(len(subset)),count)
        subset = np.array(subset)[subset_idx]
    test_dataset = np.concatenate((np.array(negative_subset),subset))

    ori_label = test_dataset[:,0].astype(np.int)
    bin_label = (ori_label>=3)*1
    test_dataset = MyDataset_origin(test_dataset[:,1],bin_label,ori_label)
    # ================== Dataloader Definition =================
    if OR:
        train_dataset = get_overlapped_datset(train_dataset,3)
        test_dataset = get_datset(test_dataset)
    else:
        train_dataset = get_datset(train_dataset)
        test_dataset = get_datset(test_dataset)


    train_loader, test_loader = get_imbalanced_loader(train_dataset,test_dataset,BATCH_SIZE,SEQ_LEN,vocab)
    # train_loader, test_loader = get_oversampled_loader(train_dataset,test_dataset,BATCH_SIZE,SEQ_LEN,vocab)

    # batch = next(iter(train_loader))
    # print([TEXT.vocab.itos[i] for i in batch[0][0]])

    # ================== Model Definition =================
    classifier = Classifier(num_voca=VOCAB_SIZE,emb_dim=emb_dim,hidden_dim=hidden_dim,use_cuda=USE_CUDA)
    optimizer = optim.Adam(classifier.parameters(),lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    if USE_CUDA:
        classifier = classifier.cuda()
        criterion = criterion.cuda()


    def accuracy(preds, y):
        preds = (torch.sigmoid(preds.data)>0.5).view(-1)
        acc = torch.sum(preds == y) / len(y)
        return acc

    def compute_BA(preds, labels):

        TP = torch.logical_and(preds==1,labels==1).sum()
        FP = torch.logical_and(preds==1,labels==0).sum()
        TN = torch.logical_and(preds==0,labels==0).sum()
        FN = torch.logical_and(preds==0,labels==1).sum()

        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)

        # if target.sum()!=0:

        BA = (TPR+TNR)/2
        return BA

    # ================== Training Loop =================
    N_EPOCH = 50
    for i in range(N_EPOCH):
        classifier.train()
        train_len, train_acc, train_loss  = 0, [], []
        for batch_no, batch in enumerate(train_loader):
            optimizer.zero_grad()

            label = batch[0]
            text = batch[1]
            if USE_CUDA:
                text, label = text.cuda(), label.float().cuda()
            # label.data.sub_(1)
            
            pred = classifier(text)
            loss = criterion(pred.view(-1),label)

            predicted = torch.round(torch.sigmoid(pred.data)).view(-1)
            acc = compute_BA(predicted,label)

            train_loss.append(loss.item())
            train_acc.append(acc.item())

            loss.backward()
            optimizer.step()
        train_epoch_loss = np.mean( train_loss )
        train_epoch_acc = np.mean( train_acc )
        classifier.eval()

        with torch.no_grad():
            test_pred, test_label = [], []
            test_loss = []
            for batch in test_loader:
                text = batch[0]
                label = batch[1]
                if USE_CUDA:
                    text, label = text.cuda(), label.float().cuda()
                
                pred = classifier(text)
                loss = criterion(pred.view(-1),label)
                predicted = torch.round(torch.sigmoid(pred.data)).view(-1)

                test_pred.append(predicted)
                test_label.append(label)
                test_loss.append(loss.item())
            test_pred = torch.cat(test_pred)
            test_label = torch.cat(test_label)

            acc = compute_BA(test_pred,test_label)
            print('epoch:{}/{} epoch_train_loss:{:.4f},epoch_train_acc:{:.4f}'
                ' epoch_val_loss:{:.4f},epoch_val_acc:{:.4f}'.format(i+1, N_EPOCH,
                    train_epoch_loss.item(), train_epoch_acc.item(),
                    np.mean(test_loss), acc))

    pred_res = torch.stack((test_pred,test_label)).T
    pred_res = pred_res.cpu().numpy().astype(np.int8)
    np.savetxt(output_file, pred_res,delimiter=',')
    # if (i+1)%10==0:
    #     save_checkpoint(acc, classifier, optimizer, i+1, 0, './log', index=True)

if __name__=="__main__":
    run(10,0,0)