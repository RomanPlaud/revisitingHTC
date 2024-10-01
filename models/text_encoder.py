#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, BertPreTrainedModel


class BertTextEncoder(BertPreTrainedModel):
    def __init__(self, config):
        """
        Bert Encoder for text representation
        :param config: [warning] this config is not the customized config but the BertPretrainedConfig
        """
        super(BertTextEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, batch):
        """
        :param inputs: torch.FloatTensor, embedding, (batch, max_len, embedding_dim)
        :param seq_lens: torch.LongTensor, (batch, max_len)
        :return:
        """
        outputs = self.bert(batch['input_ids'], attention_mask=batch['attention_mask'])
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return pooled_output

    @staticmethod
    def __children(module):
        return module if isinstance(module, (list, tuple)) else list(module.children())

    def __apply_leaf(self, module, func):
        c = self.__children(module)
        if isinstance(module, nn.Module):
            func(module)
        if len(c) > 0:
            for leaf in c:
                self.__apply_leaf(leaf, func)

    def __set_trainable(self, module, flag):
        self.__apply_leaf(module, lambda m: self.__set_trainable_attr(m, flag))

    @staticmethod
    def __set_trainable_attr(module, flag):
        module.trainable = flag
        for p in module.parameters():
            p.requires_grad = flag

    def freeze(self):
        self.__set_trainable(self.bert, False)

    def unfreeze(self, start_layer, end_layer):

        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        self.__set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            self.__set_trainable(self.bert.encoder.layer[i], True)