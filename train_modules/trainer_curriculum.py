#!/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from train_modules.evaluation_metrics_curriculum import evaluate, evaluate_fast
import torch
import tqdm
import numpy as np
method = {"bce" : [.5, 'threshold'], 
          "conditional_softmax": [None, 'top_down_max'], 
          "conditional_sigmoid": [.5, 'top_down_threshold'], 
          "leaf_softmax": [None, "leaf_softmax"]}



class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, vocab, config):
        """
        :param model: Computational Graph
        :param criterion: train_modules.ClassificationLoss object
        :param optimizer: optimization function for backward pass
        :param vocab: vocab.v2i -> Dict{'token': Dict{vocabulary to id map}, 'label': Dict{vocabulary
        to id map}}, vocab.i2v -> Dict{'token': Dict{id to vocabulary map}, 'label': Dict{id to vocabulary map}}
        :param config: helper.Configure object
        """
        super(Trainer, self).__init__()
        self.model = model
        self.vocab = vocab
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer

        self.scheduler = scheduler

    def update_lr(self):
        """
        (callback function) update learning rate according to the decay weight
        """
        logger.warning('Learning rate update {}--->{}'
                       .format(self.optimizer.param_groups[0]['lr'],
                               self.optimizer.param_groups[0]['lr'] * self.config.train.optimizer.lr_decay))
        for param in self.optimizer.param_groups:
            param['lr'] = self.config.train.optimizer.learning_rate * self.config.train.optimizer.lr_decay

    def run(self, data_loader, epoch, stage, mode='TRAIN', mask_level=None):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        predict_probs = []
        target_labels_fast = []
        total_loss = 0.0
        num_batch = data_loader.__len__()
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            if i>5:
                
                break
        # for batch in tqdm.tqdm(data_loader):
            logits = self.model(batch)
            if self.config.train.loss.recursive_regularization.flag:
                if self.config.structure_encoder.type == "TIN":
                    recursive_constrained_params = [m.weight for m in self.model.hiagm.graph_model.model.model[0].linears_prediction]
                else:
                    recursive_constrained_params = self.model.hiagm.list[-1].weight
            else:
                recursive_constrained_params = None
            if self.config.train.losstype in ['conditional_softmax', 'conditional_sigmoid', 'leaf_softmax']:
                loss, pred = self.criterion(logits,
                                  batch['label'].to(self.config.train.device_setting.device),
                                  recursive_constrained_params)
            else : 
                loss = self.criterion(logits,
                                batch['label'].to(self.config.train.device_setting.device),
                                recursive_constrained_params)
            total_loss += loss.item()

            if mode == 'TRAIN':
                # if self.config.text_encoder.type == "bert":
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.text_encoder.type == "bert":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                    # print(self.optimizer.param_groups[0]['lr'])


            predict_results = torch.sigmoid(logits).detach().cpu().numpy()
            if self.config.train.losstype in ['conditional_softmax', 'conditional_sigmoid', 'leaf_softmax'] :
                predict_results = pred.detach().cpu().numpy()

            predict_probs.extend(predict_results)
            # target_labels.extend(batch['label_list'])
            target_labels_fast.extend(batch['label'].detach().cpu().numpy())

        total_loss = total_loss / num_batch
        if mode == 'EVAL':
            # metrics = evaluate(predict_probs,
            #                    target_labels,
            #                    self.vocab,
            #                    self.config.eval.threshold)
            if self.config.train.losstype in method.keys():
                args_evaluate = method[self.config.train.losstype]
            else : 
                args_evaluate = method["bce"]
            metrics = evaluate_fast(predict_probs, np.array(target_labels_fast), self.vocab, *args_evaluate, self.config, per_level_acc=True, mask_level=mask_level[1])      
            
            logger.info("%s performance at epoch %d for levels %s---" % (stage, epoch, mask_level[0]))
            logger.info("Loss: %.4f" % total_loss)
            for k, v in metrics.items():
                logger.info("%s: %.4f" % (k, v))
            logger.info("\n")
                    
            return metrics

    def train(self, data_loader, epoch):
        """
        training module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.train()
        return self.run(data_loader, epoch, 'Train', mode='TRAIN')

    def eval(self, data_loader, epoch, stage, mask_level=None):
        """
        evaluation module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, TRAIN/DEV/TEST, log the result of the according corpus
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.eval()
        return self.run(data_loader, epoch, stage, mode='EVAL', mask_level=mask_level)
