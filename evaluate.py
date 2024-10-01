from data_modules.vocab import Vocab
from helper.configure import Configure
import numpy as np
import torch
import os
from data_modules.data_loader import data_loaders
from transformers import BertTokenizer
from models.model import MODEL
from train_modules.criterions import ClassificationLoss, MATCHLoss, CHAMPLoss, ConditionalSofmax, ConditionalSoftmaxV2, ConditionalSigmoid, PSSoftmaxWithMargin, LeafSoftmaxWithMargin, ConditionalSoftmaxWithLogitAdjustment
from train_modules.trainer import Trainer
import pickle
from sklearn.metrics import f1_score, hamming_loss
from train_modules.evaluation_metrics import compute_curve_hf1
from helper.utils import preprocess_predictions

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--output_file', type=str)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    configs = args.config_file
    config = Configure(config_json_file=configs)
    corpus_vocab = Vocab(config)
    config.add("hidden_dim", 512)
    config.add("final_dropout", 0.5)
    config.add("tree_depth", 2)
    config.add("tree_pooling_type", "sum")


    tokenizer = BertTokenizer.from_pretrained(config.text_encoder.bert_model_dir)

    tokenized = hasattr(config.data, 'tokenized') and config.data.tokenized

    # get data
    train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab, tokenizer=tokenizer, tokenized=tokenized, drop_last=True)

    # build up model
    model = MODEL(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')

    model_file = args.model_file
    print(model_file)
    checkpoint_model = torch.load(model_file) #, map_location=torch.device('cpu'))
    # del checkpoint_model['state_dict']['text_encoder.bert.embeddings.position_ids']
    model.load_state_dict(checkpoint_model['state_dict'])
    model.to(config.train.device_setting.device)

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

    trainer = Trainer(model=model,
                    criterion=criterion,
                    optimizer=None,
                    scheduler=None,
                    vocab=corpus_vocab,
                    config=config)


    predict_probs, labels, _, _ = trainer.inference(test_loader, -1, 'TEST')

    # preprocess the prediction 
    # predict_probs = np.array(predict_probs)
    # relations = corpus_vocab.hierarchy
    # predict_probs = np.array(preprocess_predictions(predict_probs, relations))

    hf1_auc = compute_curve_hf1(predict_probs, labels)
    hamming_loss = hamming_loss(labels, predict_probs > 0.5)
    f1_score_micro = f1_score(labels, predict_probs > 0.5, average='micro')
    f1_score_macro = f1_score(labels, predict_probs > 0.5, average='macro')
    ## evaluate the model

    # print('HF1-AUC: ', hf1_auc)
    print('Hamming Loss: ', hamming_loss)
    print('F1 Score Micro: ', f1_score_micro)
    print('F1 Score Macro: ', f1_score_macro)

    # save the results

    with open(args.output_file, 'wb') as f:
        pickle.dump({'hf1_auc': hf1_auc, 'hamming_loss': hamming_loss, 'f1_score_micro': f1_score_micro, 'f1_score_macro': f1_score_macro}, f)



