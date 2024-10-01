#!/usr/bin/env python
# coding:utf-8

from data_modules.dataset import ClassificationDataset
from data_modules.collator import Collator
from torch.utils.data import DataLoader

import torch
import numpy as np
import random

def data_loaders(config, vocab, data={'train': None, 'val': None, 'test': None}, tokenizer=None, tokenized=False, drop_last=True):
    """
    get data loaders for training and evaluation
    :param config: helper.configure, Configure Object
    :param vocab: data_modules.vocab, Vocab Object
    :param data: on-memory data, Dict{'train': List[str] or None, ...}
    :param tokenizer: bert tokenizer for tokenizing input document
    :return: -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    on_memory = data['train'] is not None
    collate_fn = Collator(config, vocab)
    train_dataset = ClassificationDataset(config, vocab, stage='TRAIN', on_memory=on_memory, corpus_lines=data['train'], tokenizer=tokenizer, tokenized=tokenized)
    # (train_dataset[0])
    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,  # using args
                              shuffle=True,
                            #   num_workers=config.train.device_setting.num_workers,
                              collate_fn=collate_fn,
                              pin_memory=False,
                              drop_last=drop_last)

    val_dataset = ClassificationDataset(config, vocab, stage='VAL', on_memory=on_memory, corpus_lines=data['val'], tokenizer=tokenizer, tokenized=tokenized)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.eval.batch_size,  # using args
                            shuffle=True,
                            # num_workers=config.train.device_setting.num_workers,
                            collate_fn=collate_fn,
                            pin_memory=False,
                            drop_last=drop_last)

    test_dataset = ClassificationDataset(config, vocab, stage='TEST', on_memory=on_memory, corpus_lines=data['test'], tokenizer=tokenizer, tokenized=tokenized)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.test.batch_size,  # using args
                             shuffle=False,
                            #  num_workers=config.train.device_setting.num_workers,
                             collate_fn=collate_fn,
                             pin_memory=False,
                             drop_last=drop_last)

    return train_loader, val_loader, test_loader
