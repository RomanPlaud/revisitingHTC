#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, config, vocab, device):
        """
        origin class of the classification model
        :param config: helper.configure, Configure object
        :param vocab: data_modules.vocab, Vocab object
        :param device: torch.device, config.train.device_setting.device
        """
        super(Classifier, self).__init__()

        self.config = config
        self.device = device
        self.segregate = False
        self.vocab = vocab
        if config.text_encoder.type == "bert":
            self.list = nn.ModuleList()
            if config.model.classifier.num_layer == 1:
                if config.train.losstype in ["leaf_softmax", "leaf_softmax_with_margin"]:
                    self.list.append(nn.Linear(768, len(vocab.idxleaf2idxnodes)))
                else :
                    if hasattr(config.model.classifier, 'segregate') and config.model.classifier.segregate:
                        self.segregate = True
                        last_layer = nn.ModuleList([nn.Linear(768, len(vocab.levels[i])) for i in vocab.levels.keys()])
                        self.list.append(last_layer)
                    else:   
                        self.list.append(nn.Linear(768, len(vocab.v2i['label'].keys())))
            else:
                self.list.append(nn.Linear(768, config.model.classifier.hidden_dimension))
                for _ in range(config.model.classifier.num_layer-2):
                    self.list.append(nn.Linear(config.model.classifier.hidden_dimension, config.model.classifier.hidden_dimension))
                if config.train.losstype in ["leaf_softmax", "leaf_softmax_with_margin"]:
                    self.list.append(nn.Linear(config.model.classifier.hidden_dimension, len(vocab.idxleaf2idxnodes)))
                else :
                    if hasattr(config.model.classifier, 'segregate') and config.model.classifier.segregate:
                        self.segregate = True
                        last_layer = nn.ModuleList([nn.Linear(config.model.classifier.hidden_dimension, len(vocab.levels[i])) for i in vocab.levels.keys()])
                        self.list.append(last_layer)
                    else:
                        self.list.append(nn.Linear(config.model.classifier.hidden_dimension, len(vocab.v2i['label'].keys())))

        self.dropout = nn.Dropout(p=config.model.classifier.dropout)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        """
        forward pass
        :param inputs: torch.FloatTensor, (batch, len(CNNs) * top_k, num_kernels)
        :return: logits, torch.FloatTensor (batch, N)
        """
        if self.config.text_encoder.type == "bert":
            token_output = inputs
            for i, layer in enumerate(self.list):
                if i == len(self.list) - 1:
                    if self.segregate:
                        logits_tmp = [layer[i](token_output) for i in range(len(layer))]
                        logits = torch.cat(logits_tmp, dim=1)
                        for level, indices in self.vocab.levels.items():
                            logits[:, indices] = logits_tmp[level]
                    else:
                        logits = layer(token_output)
                else:
                    token_output = layer(token_output)
                    token_output = self.activation(token_output)
        else:
            token_output = torch.cat(inputs, 1)
            token_output = token_output.view(token_output.shape[0], -1)
            logits = self.dropout(self.linear(token_output))
        return logits, token_output

