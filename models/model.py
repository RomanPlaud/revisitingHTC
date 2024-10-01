#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
from models.origin import Classifier

from models.text_encoder import BertTextEncoder
from transformers import BertForSequenceClassification




DATAFLOW_TYPE = {
    'Origin': 'origin'
}


class MODEL(nn.Module):
    def __init__(self, config, vocab, model_type, model_mode='TRAIN'):
        """
        Hierarchy-Aware Global Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(MODEL, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device

        self.label_map =  vocab.v2i['label'] #vocab.v2i['token'],

        # self.token_embedding = EmbeddingLayer(
        #     vocab_map=self.token_map,
        #     embedding_dim=config.embedding.token.dimension,
        #     vocab_name='token',
        #     config=config,
        #     padding_index=vocab.padding_index,
        #     pretrained_dir=config.embedding.token.pretrained_file,
        #     model_mode=model_mode,
        #     initial_type=config.embedding.token.init_type
        # )

        self.dataflow_type = DATAFLOW_TYPE[model_type]

        # self.text_encoder = TextEncoder(config)

        if self.config.text_encoder.type == "bert":
            self.text_encoder = BertTextEncoder.from_pretrained(self.config.text_encoder.bert_model_dir)
            ## add a condition to freeze bert parameters
            if hasattr(self.config.text_encoder, 'freeze_bert') and self.config.text_encoder.freeze_bert:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        # else:
            # self.text_encoder = TextEncoder(config)

        if self.dataflow_type == 'origin':
             self.model = Classifier(config=config,
                                        vocab=vocab,
                                        device=self.device)
        else:

            pass
                
            
    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.model.parameters()})
        return params

    def forward(self, batch):
        """
        forward pass of the overall architecture
        :param batch: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
        :return: 
        """

        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        # embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))

        # get the length of sequences for dynamic rnn, (batch_size, 1)
        seq_len = batch['token_len']
        if self.config.text_encoder.type == 'bert':
            token_output = self.text_encoder(batch)
        # else:
        #     token_output = self.text_encoder(embedding, seq_len)

        if self.dataflow_type == 'origin':
            logits, token_output = self.model(token_output)
        else : 
            logits = self.model(token_output)
        return logits, token_output
