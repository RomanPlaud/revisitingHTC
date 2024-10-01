#!/usr/bin/env python
# coding:utf-8
# import copy


import helper.logger as logger
from train_modules.evaluation_metrics import evaluate, evaluate_fast, evaluate_top_down_threshold
import torch
import tqdm
import numpy as np
method = {"bce" : [.5, 'threshold'], 
          "conditional_softmax": [None, 'top_down_max'], 
          "conditional_sigmoid": [None, 'top_down_max'], 
          "leaf_softmax": [None, "top_down_max"], 
          "parameter_sharing_softmax": [None, "top_down_max"]}

# def calculate_param_norm(model1, model2):
#     param_norms = {}
#     for (name, param1), (_, param2) in zip(model1.named_parameters(), model2.named_parameters()):
#         if "weight" in name:
#             param_norm = torch.norm(param1.data - param2.data, p=2)
#             param_norms[name] = param_norm.item()
#     return param_norms

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

    def run(self, data_loader, epoch, stage, mode='TRAIN'):
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
        logits_all = []
        embeddings = []
        total_loss = 0.0
        num_batch = data_loader.__len__()
        # for i, batch in enumerate(tqdm.tqdm(data_loader)):
        #     if i>3:
        #         break
        for batch in tqdm.tqdm(data_loader):
            logits, embedding = self.model(batch)
            embedding = embedding.detach().cpu()
            logits_clone = logits.clone()
            # check if config.train has attribute 'loss'
            if hasattr(self.config.train, 'loss') and self.config.train.loss.recursive_regularization.flag:
                if self.config.structure_encoder.type == "TIN":
                    recursive_constrained_params = [m.weight for m in self.model.hiagm.graph_model.model.model[0].linears_prediction]
                else:
                    recursive_constrained_params = self.model.hiagm.list[-1].weight
            else:
                recursive_constrained_params = None
            if self.config.train.losstype in ['conditional_softmax', 'conditional_sigmoid', 'leaf_softmax', 'parameter_sharing_softmax', 'leaf_softmax_with_margin', 'parameter_sharing_softmax_with_margin', 'conditional_softmax_with_logit_adjustment']:
                loss, pred = self.criterion(logits,
                                  batch['label'].to(self.config.train.device_setting.device),
                                  mode)
                
            else : 
                loss = self.criterion(logits,
                                batch['label'].to(self.config.train.device_setting.device),
                                recursive_constrained_params)
            total_loss += loss.item()

            if mode == 'TRAIN':
                # if self.config.text_encoder.type == "bert":
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)

                self.optimizer.zero_grad()

                # before_model = copy.deepcopy(self.model.hiagm)

                loss.backward()


                if self.config.text_encoder.type == "bert":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)

                # print(torch.norm(self.model.hiagm.list[-1][0].weight.grad, p=2, dim=1).mean())
                # grad_1 = torch.norm(self.model.hiagm.list[-1][1].weight.grad, p=2, dim=1)
                # print(grad_1[grad_1!=0].mean())
                # grad_2 = torch.norm(self.model.hiagm.list[-1][2].weight.grad, p=2, dim=1)
                # print(grad_2[grad_2!=0].mean())

                self.optimizer.step()

                # norms = calculate_param_norm(before_model, self.model.hiagm)

                # for layer_name in norms:
                #     print(f"Layer {layer_name}: {norms[layer_name]}")

                if self.scheduler is not None:
                    self.scheduler.step()

            if mode == 'EVAL' or mode == 'INFERENCE_ONLY':

                
                if self.config.train.losstype in ['conditional_softmax', 'conditional_sigmoid', 'leaf_softmax', "parameter_sharing_softmax", "leaf_softmax_with_margin", "parameter_sharing_softmax_with_margin", "conditional_softmax_with_logit_adjustment"] :
                    predict_results = pred.detach().cpu().numpy()
                else : 
                    predict_results = torch.sigmoid(logits).detach().cpu().numpy()

                predict_probs.extend(predict_results)
                # target_labels.extend(batch['label_list'])
                target_labels_fast.extend(batch['label'].detach().cpu().numpy())
                logits_all.extend(logits_clone.detach().cpu().numpy())
                embeddings.append(embedding)


            # predict_results = torch.sigmoid(logits).detach().cpu().numpy()
            # if self.config.train.losstype in ['conditional_softmax', 'conditional_sigmoid', 'leaf_softmax', "parameter_sharing_softmax", "leaf_softmax_with_margin", "parameter_sharing_softmax_with_margin"] :
            #     predict_results = pred.detach().cpu().numpy()

            # predict_probs.extend(predict_results)
            # # target_labels.extend(batch['label_list'])
            # target_labels_fast.extend(batch['label'].detach().cpu().numpy())

        total_loss = total_loss / num_batch
        if mode == 'EVAL':
            if stage == 'TEST':
                predict_probs = np.array(predict_probs)
                logits_all = np.array(logits_all)
                target_labels_fast = np.array(target_labels_fast)
                ## save the predict results
                path = self.config.log.filename.replace('.log', '_')
                pickle.dump(predict_probs, open(path+"pred.pickle", "wb"))
                pickle.dump(logits, open(path+"logits.pickle", "wb"))
                pickle.dump(labels, open(path+ "labels.pickle", "wb"))

            
            max_depth = max(list(self.vocab.levels.keys()))
            # metrics = evaluate(predict_probs,
            #                    target_labels,
            #                    self.vocab,
            #                    self.config.eval.threshold)
            if hasattr(self.config.eval, "type"):
                if stage == "DEV":
                    metrics = evaluate_top_down_threshold(predict_probs, np.array(target_labels_fast), self.vocab, self.config.eval.threshold, self.config, self.config.eval.type, metrics=["micro", "macro"], modes=["standard"], depths=[max_depth])
                elif stage == 'TEST':
                    metrics = evaluate_top_down_threshold(predict_probs, np.array(target_labels_fast), self.vocab, self.config.eval.threshold, self.config, self.config.eval.type, metrics=["micro", "macro"], modes=['standard'], depths=[max_depth])
            else : 
                if self.config.train.losstype in method.keys():
                    args_evaluate = method[self.config.train.losstype]
                else : 
                    args_evaluate = method["bce"]
                metrics = evaluate_fast(predict_probs, np.array(target_labels_fast), self.vocab, *args_evaluate, self.config, per_level_acc=True) 

            logger.info("%s performance at epoch %d ---" % (stage, epoch))
            logger.info("Loss: %.4f" % total_loss)
            if hasattr(self.config.eval, "type"):
                if stage == "DEV":
                    logger.info("Micro precision: %.4f" % metrics["micro"]["standard"]["precision"])
                    logger.info("Micro recall: %.4f" % metrics["micro"]["standard"]["recall"])
                    logger.info("Micro F1: %.4f" % metrics["micro"]["standard"]["f1_score"])
                    logger.info('\n')
                    logger.info("Macro precision: %.4f" % metrics["macro"]["standard"][max_depth]["precision"])
                    logger.info("Macro recall: %.4f" % metrics["macro"]["standard"][max_depth]["recall"])
                    logger.info("Macro F1 max depth: %.4f" % metrics["macro"]["standard"][max_depth]["f1_score"])
                    logger.info('\n')

                elif stage == 'TEST':
                    for metric in ["micro", "macro"]:
                        # for mode in ["lca", "standard"]:
                        for mode in ['standard']:
                            if metric == "micro":
                                logger.info(f"{metric}-{mode} precision: {metrics[metric][mode]['precision']}")
                                logger.info(f"{metric}-{mode} recall: {metrics[metric][mode]['recall']}")
                                logger.info(f"{metric}-{mode} F1_score: {metrics[metric][mode]['f1_score']}")
                                logger.info("\n")
                            elif metric == "macro":
                                for depth in [max_depth]:
                                    logger.info(f"{metric}-{mode}-{depth} precision: {metrics[metric][mode][depth]['precision']}")
                                    logger.info(f"{metric}-{mode}-{depth} recall: {metrics[metric][mode][depth]['recall']}")
                                    logger.info(f"{metric}-{mode}-{depth} F1_score: {metrics[metric][mode][depth]['f1_score']}")
                                    logger.info("\n")
                                
            else: 
                for k, v in metrics.items():
                    logger.info("%s: %.4f" % (k, v))
                logger.info("\n")
                    
            return metrics, total_loss
        if mode=="INFERENCE_ONLY":
            # print(predict_probs)
            return predict_probs, np.array(target_labels_fast), logits_all, embeddings

    def train(self, data_loader, epoch):
        """
        training module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.train()
        return self.run(data_loader, epoch, 'Train', mode='TRAIN')

    def eval(self, data_loader, epoch, stage):
        """
        evaluation module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, TRAIN/DEV/TEST, log the result of the according corpus
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.eval()
        return self.run(data_loader, epoch, stage, mode='EVAL')

    def inference(self, data_loader, epoch, stage):
        self.model.eval()
        return self.run(data_loader, epoch, stage, mode='INFERENCE_ONLY')