import os

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

from torch.utils.data.sampler import WeightedRandomSampler
class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)

def collate_batch(batch,TEXT):
    label_list, text_list, offsets = [], [], [0]
    labelset = np.array(['neg','pos'])
    for _ in batch:
        data = TEXT.numericalize([_.text[:TEXT.fix_length]])[0]
        if data.size(0)<TEXT.fix_length:
            pad_size = TEXT.fix_length-data.size(0)
            pad = torch.tensor(2).repeat((pad_size)).type(torch.LongTensor)
            data = torch.cat([data,pad])
        label = np.where(labelset == _.label)[0][0]
        text_list.append(data)
        label_list.append(label)
        offsets.append(data.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.stack(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return text_list, label_list

def get_imbalanced(tr_dataset,ts_dataset,batch_size,TEXT):

    cb = lambda batch: collate_batch(batch, TEXT)

    sampler = StratifiedBatchSampler(np.array([i.label for i in tr_dataset]), batch_size=batch_size)
    train_loader = DataLoader(tr_dataset, batch_sampler=sampler,
                                            shuffle=False, num_workers=int(8),collate_fn=cb)

    # train_loader = DataLoader(trainset,batch_size=BATCH_SIZE, num_workers=int(8),collate_fn=collate_batch)
    # batch = next(iter(dataloader))
    # batch[0]

    sampler = StratifiedBatchSampler(np.array([i.label for i in ts_dataset]), batch_size=batch_size)
    test_loader = DataLoader(ts_dataset, batch_sampler=sampler,
                                            shuffle=False, num_workers=int(8),collate_fn=cb)

    print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_loader)))
    print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_loader)))

    return train_loader, test_loader

def get_oversampled(tr_dataset,ts_dataset,batch_size, TEXT):

    cb = lambda batch: collate_batch(batch, TEXT)

    length = tr_dataset.__len__()

    labels = [i.label for i in tr_dataset]
    num_sample_per_class = Counter(labels)

    selected_list = []
    for i in range(0, length):
        _ = tr_dataset.__getitem__(i)
        label = _.label
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_sample_per_class[label])
            # selected_list.append(num_sample_per_class[label]/np.sum(list(num_sample_per_class.values())))
            # num_sample_per_class[label] -= 1

    sampler = WeightedRandomSampler(selected_list, len(selected_list))
    train_loader = DataLoader(tr_dataset, batch_size=batch_size,
                                 sampler=sampler, num_workers=0, drop_last=True,collate_fn=cb)

    # train_loader = DataLoader(trainset,batch_size=BATCH_SIZE, num_workers=int(8),collate_fn=collate_batch)
    batch = next(iter(train_loader))
    print(Counter(batch[1].numpy()))

    sampler = StratifiedBatchSampler(np.array([i.label for i in ts_dataset]), batch_size=batch_size)
    test_loader = DataLoader(ts_dataset, batch_sampler=sampler,
                                            shuffle=False, num_workers=int(0),collate_fn=cb)

    print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_loader)))
    print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_loader)))

    return train_loader, test_loader

def save_checkpoint(acc, model, optim, epoch, SEED, LOGDIR, index=False):
    # Save checkpoint.
    print('Saving..')

    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if index:
        ckpt_name = 'ckpt_epoch' + str(epoch) + '_' + str(SEED) + '.t7'
    else:
        ckpt_name = 'ckpt_' + str(SEED) + '.t7'

    ckpt_path = os.path.join(LOGDIR, ckpt_name)
    torch.save(state, ckpt_path)