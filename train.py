#!/usr/bin/env python
# coding:utf-8
import helper.logger as logger
from models.model import MODEL
import torch
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.criterions import ClassificationLoss, MATCHLoss, CHAMPLoss, ConditionalSofmax, ConditionalSoftmaxV2, ConditionalSigmoid, PSSoftmaxWithMargin, LeafSoftmaxWithMargin, ConditionalSoftmaxWithLogitAdjustment
from train_modules.trainer import Trainer
from helper.utils import load_checkpoint, save_checkpoint, compute_learning_rates
from helper.arg_parser import get_args

import time
import random
import numpy as np
import pprint
import warnings

from transformers import BertTokenizer
from helper.lr_schedulers import get_linear_schedule_with_warmup
from helper.adamw import AdamW

import pickle

warnings.filterwarnings("ignore")


def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.learning_rate,  # using args
            # lr=config.train.optimizer.learning_rate,
                                params=params,
                                weight_decay=args.l2rate)
    else:
        raise TypeError("Recommend the Adam optimizer")


def train(config, args):
    logger.info(args.config_file)
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config)

    ## test 
    # path = "predictions_runs/" + str(config.data.dataset) + '/' + str(config.data.name)

    # predict_probs, logits_all, labels = [], [], []
    # pickle.dump(predict_probs, open(path + '/prob.pickle', "wb"))
    # pickle.dump(logits_all, open(path + '/logits.pickle', "wb"))
    # pickle.dump(labels, open(path + '/labels.pickle', "wb"))
    # print('passed test')

    tokenizer = BertTokenizer.from_pretrained(config.text_encoder.bert_model_dir)

    tokenized = hasattr(config.data, 'tokenized') and config.data.tokenized
    # get data
    train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab, tokenizer=tokenizer, tokenized=tokenized)

    # build up model
    model = MODEL(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')
    
    model.to(config.train.device_setting.device)

    # Code for counting parameters
    # from thop import clever_format
    # print(model)
    # def count_parameters(model):
    #     total = sum(p.numel() for p in model.parameters())
    #     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return total, trainable
    #
    # total_params, trainable_params = count_parameters(model)
    # total_params, trainable_params = clever_format([total_params, trainable_params], "%.4f")
    # print("Total num of parameters: {}. Trainable parameters: {}".format(total_params, trainable_params))
    # sys.exit()

    # define training objective & optimizer

    if config.train.losstype == 'champ':
        criterion = CHAMPLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                              corpus_vocab.v2i['label'],
                              config.train.loss.champ_regularization.beta)
    elif config.train.losstype == 'match':
        params = None
        if config.train.loss.classification == 'focal' and hasattr(config.train.loss, 'focal_parameters'):
            params = {alpha : config.train.loss.focal_parameters.alpha, 
                     gamma : config.train.loss.focal_parameters.gamma}
        criterion = MATCHLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                              corpus_vocab.v2i['label'],
                            #   recursive_penalty=config.train.loss.recursive_regularization.penalty,
                            #   recursive_constraint=config.train.loss.recursive_regularization.flag, 
                            #   proba_penalty=config.train.loss.probability_regularization.penalty,
                            #   proba_constraint=config.train.loss.probability_regularization.flag,
                              loss = config.train.loss.classification, 
                              params = params)
    elif config.train.losstype == 'standard':
        criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                                        corpus_vocab.v2i['label'],
                                        recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                        recursive_constraint=config.train.loss.recursive_regularization.flag)
    elif config.train.losstype == 'conditional_softmax':
        criterion = ConditionalSoftmaxV2(os.path.join(config.data.data_dir, config.data.hierarchy),
                                        corpus_vocab.v2i['label'], corpus_vocab.levels)
    elif config.train.losstype == 'conditional_softmax_with_logit_adjustment':
        criterion = ConditionalSoftmaxWithLogitAdjustment(os.path.join(config.data.data_dir, config.data.hierarchy),
                                corpus_vocab.v2i['label'], corpus_vocab.levels,
                                corpus_vocab.proba_vector, config.train.logit_adjustment_tau, 
                                device=config.train.device_setting.device)
    elif config.train.losstype == 'conditional_sigmoid':
        criterion = ConditionalSigmoid(os.path.join(config.data.data_dir, config.data.hierarchy),
                                        corpus_vocab.v2i['label'], corpus_vocab.levels)
        
    elif config.train.losstype == 'leaf_softmax':
        criterion = LeafSoftmaxWithMargin(os.path.join(config.data.data_dir, config.data.hierarchy), 
                                            corpus_vocab.v2i['label'],
                                            corpus_vocab.idxleaf2idxnodes, 
                                            corpus_vocab.graph_hierarchy,
                                            alpha=0.0,
                                            device=config.train.device_setting.device)
    elif config.train.losstype == "parameter_sharing_softmax":
        criterion = PSSoftmaxWithMargin(os.path.join(config.data.data_dir, config.data.hierarchy), 
                              corpus_vocab.v2i['label'],
                              corpus_vocab.idxleaf2idxnodes, 
                              corpus_vocab.graph_hierarchy,
                              alpha=0.0,
                              device=config.train.device_setting.device)
        
    elif config.train.losstype == "leaf_softmax_with_margin":
        criterion = LeafSoftmaxWithMargin(os.path.join(config.data.data_dir, config.data.hierarchy), 
                                            corpus_vocab.v2i['label'],
                                            corpus_vocab.idxleaf2idxnodes, 
                                            corpus_vocab.graph_hierarchy,
                                            alpha=config.train.loss.margin_coefficient,
                                            device=config.train.device_setting.device)        
    elif config.train.losstype == "parameter_sharing_softmax_with_margin":
        criterion = PSSoftmaxWithMargin(os.path.join(config.data.data_dir, config.data.hierarchy), 
                              corpus_vocab.v2i['label'],
                              corpus_vocab.idxleaf2idxnodes, 
                              corpus_vocab.graph_hierarchy,
                              alpha=config.train.loss.margin_coefficient,
                              device=config.train.device_setting.device)

    torch.autograd.set_detect_anomaly(True)
    if config.text_encoder.type == "bert":
        t_total = int(len(train_loader) * (config.train.end_epoch-config.train.start_epoch))

        param = list(model.named_parameters())
        if hasattr(config.train.optimizer, "set_to_zero_decay_bert") and not config.train.optimizer.set_to_zero_decay_bert:
            no_decay = []
        else : 
            no_decay = ['bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in param if 'bert' in n and not any(nd in n for nd in no_decay)],
             'weight_decay': args.l2rate, 'lr': config.train.optimizer.learning_rate},
            {'params': [p for n, p in param if 'bert' in n and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.train.optimizer.learning_rate}]

        if hasattr(config.train.optimizer, "set_to_zero_decay_classif") and not config.train.optimizer.set_to_zero_decay_classif:
            print('not zeroing the weight decay')
            no_decay = []
        else : 
            no_decay = ['bias', 'LayerNorm.weight']

        if hasattr(config.model.classifier, "segregate") and config.model.classifier.segregate :
            ## raise an error if len(config.train.optimzer.learning_rates_classifier) != len(corpus_vocab.levels)
            if hasattr(config.train.optimizer, 'learning_rates_classifier') and len(config.train.optimizer.learning_rates_classifier) != len(corpus_vocab.levels):
                raise ValueError("The number of learning rates for classifiers is not equal to the number of levels")
            elif not hasattr(config.train.optimizer, 'learning_rates_classifier'):
                config.train.optimizer.learning_rates_classifier = compute_learning_rates(args.learning_rate, corpus_vocab.levels)
                print(config.train.optimizer.learning_rates_classifier)
            print(config.train.optimizer.learning_rates_classifier)
            for i in range(config.model.classifier.num_layer):
                if i != (config.model.classifier.num_layer-1):
                    print([n for n, p in param if 'model.list.'+str(i) in n and not any(nd in n for nd in no_decay)])
                    grouped_parameters.append({'params': [p for n, p in param if 'model.list.'+str(i) in n and not any(nd in n for nd in no_decay)],
                                            'weight_decay': args.l2rate, 'lr': args.learning_rate})
                    if no_decay != []:
                        grouped_parameters.append({'params': [p for n, p in param if 'model.list.'+str(i) in n and any(nd in n for nd in no_decay)],
                                            'weight_decay': 0.0, 'lr': args.learning_rate})
                else:
                    for j, lr in enumerate(config.train.optimizer.learning_rates_classifier):
                        print([n for n, p in param if 'model.list.'+str(i)+'.'+str(j) in n and not any(nd in n for nd in no_decay)])
                        grouped_parameters.append({'params': [p for n, p in param if 'model.list.'+str(i)+'.'+str(j) in n and not any(nd in n for nd in no_decay)],
                                                'weight_decay': args.l2rate, 'lr': lr})
                        if no_decay != []:
                            grouped_parameters.append({'params': [p for n, p in param if 'model.list.'+str(i)+'.'+str(j) in n and any(nd in n for nd in no_decay)],
                                                    'weight_decay': 0.0, 'lr': lr})
        else:        
            grouped_parameters += [
                {'params': [p for n, p in param if 'bert' not in n and not any(nd in n for nd in no_decay)],
                'weight_decay': args.l2rate, 'lr': args.learning_rate}]
            if no_decay != []:
                grouped_parameters.append({'params': [p for n, p in param if 'bert' not in n and any(nd in n for nd in no_decay)],
                                           'weight_decay': 0.0, 'lr': args.learning_rate})

        # print(grouped_parameters)
        warmup_steps = int(t_total * 0.1)
        optimizer = AdamW(grouped_parameters, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
    else:
        optimizer = set_optimizer(config, model)
        scheduler = None
    
    # get epoch trainer
    trainer = Trainer(model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      vocab=corpus_vocab,
                      config=config)

    # set origin log
    best_epoch = [-1, -1]
    best_performance = [0.0, 0.0]
    best_loss = float('inf')
    '''
        ckpt_dir
            begin-time_dataset_model
                best_micro/macro-model_type-training_params_(tin_params)
                                            
    '''
    # model_checkpoint = config.train.checkpoint.dir
    
    model_checkpoint = os.path.join(args.ckpt_dir, args.begin_time + config.train.checkpoint.dir)  # using args
    model_name = config.model.type
    if hasattr(config, 'structure_encoder') and config.structure_encoder.type == "TIN":
        model_name += '_' + str(args.tree_depth) + '_' + str(args.hidden_dim) + '_' + args.tree_pooling_type + '_' + str(args.final_dropout) + '_' + str(args.hierar_penalty)
    wait = 0
    if not os.path.isdir(model_checkpoint):
        # os.mkdir(model_checkpoint)
        os.makedirs(model_checkpoint)
    elif args.load_pretrained:
        # loading previous checkpoint
        dir_list = os.listdir(model_checkpoint)
        dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
        latest_model_file = ''
        for model_file in dir_list[::-1]:  # best or latest ckpt
            if model_file.startswith('best'):
                continue
            else:
                latest_model_file = model_file
                break
        if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
            logger.info('Loading Previous Checkpoint...')
            logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
            best_performance, config = load_checkpoint(model_file=os.path.join(model_checkpoint, latest_model_file),
                                                       model=model,
                                                       config=config,
                                                       optimizer=optimizer)
            logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
                best_performance[0], best_performance[1]))
    
    for epoch in range(config.train.start_epoch, config.train.end_epoch):
        max_depth = max(list(corpus_vocab.levels.keys()))
        # epochs += 1
        start_time = time.time()
        trainer.train(train_loader, epoch)
        # trainer.eval(train_loader, epoch, 'TRAIN')
        performance, total_loss = trainer.eval(dev_loader, epoch, 'DEV')

        # record results for each epoch
        print("[Val] epoch: %d micro_f1: %.4f\t macro_f1_depth_max: %.4f" \
                    % (epoch, performance['micro']['standard']['f1_score'], performance['macro']['standard'][max_depth]['f1_score']))
        # saving best model and check model
        # if not (performance['micro']['standard']['f1_score'] >= best_performance[0] or performance['macro']['standard'][max_depth]['f1_score'] >= best_performance[1]):
        #     wait += 1
        #     # reduce LR on plateau
        #     if wait % config.train.optimizer.lr_patience == 0:
        #         logger.warning("Performance has not been improved for {} epochs, updating learning rate".format(wait))
        #         trainer.update_lr()
        #     # early stopping
        #     if wait == config.train.optimizer.early_stopping:
        #         logger.warning("Performance has not been improved for {} epochs, stopping train with early stopping"
        #                        .format(wait))
        #         break

        # if performance['micro']['standard']['f1_score'] > best_performance[0]:
        #     wait = 0
        #     logger.info('Improve Micro-F1 {}% --> {}%'.format(best_performance[0], performance['micro']['standard']['f1_score']))
        #     best_performance[0] = performance['micro']['standard']['f1_score']
        #     best_epoch[0] = epoch
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'model_type': config.model.type,
        #         'state_dict': model.state_dict(),
        #         'best_performance': best_performance,
        #         'optimizer': optimizer.state_dict()
        #     }, os.path.join(model_checkpoint, 'best_micro_' + model_name))
        # if performance['macro']['standard'][max_depth]['f1_score'] > best_performance[1]:
        #     wait = 0
        #     logger.info('Improve Macro-F1-max-depth {}% --> {}%'.format(best_performance[1], performance['macro']['standard'][max_depth]['f1_score']))
        #     best_performance[1] = performance['macro']['standard'][max_depth]['f1_score']
        #     best_epoch[1] = epoch
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'model_type': config.model.type,
        #         'state_dict': model.state_dict(),
        #         'best_performance': best_performance,
        #         'optimizer': optimizer.state_dict()
        #     }, os.path.join(model_checkpoint, 'best_macro_max_depth' + model_name))
        # Save the model based on the lowest total_loss
        if total_loss < best_loss:
            wait = 0
            logger.info('Total loss improved from {:.4f} to {:.4f}'.format(best_loss, total_loss))
            best_loss = total_loss
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_checkpoint, 'best_loss_' + model_name))
        else:
            wait += 1
            # Reduce LR on plateau
            if wait % config.train.optimizer.lr_patience == 0:
                logger.warning("Loss has not improved for {} epochs, updating learning rate".format(wait))
                trainer.update_lr()
            # Early stopping
            if wait == config.train.optimizer.early_stopping:
                logger.warning("Loss has not improved for {} epochs, stopping training with early stopping".format(wait))
                break

        # if epoch % 10 == 1:
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'model_type': config.model.type,
        #         'state_dict': model.state_dict(),
        #         'best_performance': best_performance,
        #         'optimizer': optimize.state_dict()
        #     }, os.path.join(model_checkpoint, model_name + '_epoch_' + str(epoch)))

        logger.info('Epoch {} Time Cost {} secs.'.format(epoch, time.time() - start_time))
        # total_time += time.time() - start_time

    # print("Average training time per epoch: {} secs.".format(float(total_time) / epochs))

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_loss_' + model_name)

    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=model,
                        config=config,
                        optimizer=optimizer)
        # start_time = time.time()
        # performance = trainer.eval(test_loader, best_epoch[0], 'TEST')
        predict_probs, labels, logits_all, _ = trainer.inference(test_loader, best_epoch, 'TEST')

        path = "predictions_runs/" + str(config.data.dataset) + '/' + str(config.data.name)

        pickle.dump(predict_probs, open(path + '/prob.pickle', "wb"))
        pickle.dump(logits_all, open(path + '/logits.pickle', "wb"))
        pickle.dump(labels, open(path + '/labels.pickle', "wb"))
        print('passed test')

        # print("Inference time A : {} secs.".format(time.time() - start_time))
        # record best micro test performance
        print("Best Micro-f1 on epoch: %d, [Test] performance↓\nMicro-f1: %.4f\nMacro-f1-max-depth: %.4f" \
                    % (best_epoch, performance['micro']['standard']['f1_score'], performance['macro']['standard'][max_depth]['f1_score']))
    # remove file of best micro model
    # os.remove(best_epoch_model_file)

    # best_epoch_model_file = os.path.join(model_checkpoint, 'best_macro_max_depth' + model_name)
    # if os.path.isfile(best_epoch_model_file):
    #     load_checkpoint(best_epoch_model_file, model=model,
    #                     config=config,
    #                     optimizer=optimizer)
    #     performance = trainer.eval(test_loader, best_epoch[1], 'TEST')
    #     # record best macro test performance
    #     print("Best Macro-f1-max-depth on epoch: %d, [Test] performance↓\nMicro-f1: %.4f\nMacro-f1-max-depth: %.4f" \
    #                 % (best_epoch[1], performance['micro']['standard']['f1_score'], performance['macro']['standard'][max_depth]['f1_score']))
    # # os.remove(best_epoch_model_file)
    return 


if __name__ == "__main__":
    args = get_args()

    pprint.pprint(vars(args))
    configs = Configure(config_json_file=args.config_file)
    configs.update(vars(args))

    # if configs.train.device_setting.device == 'cuda':
    #     os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))

    # else:
    #     os.system("CUDA_VISIBLE_DEVICES=''")



    random_seed = random.randint(1, 1000)
    print('random seed: ', random_seed)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')

    logger.Logger(configs)

    # if not os.path.isdir(configs.train.checkpoint.dir):
    #     os.mkdir(configs.train.checkpoint.dir)

    # train(config)
    train(configs, args)