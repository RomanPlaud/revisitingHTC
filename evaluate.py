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

configs = "configs/aaaa_final_wos/vanilla_bert_wos_leaf_softmax.json"
config = Configure(config_json_file=configs)
corpus_vocab = Vocab(config)
config.add("hidden_dim", 512)
config.add("final_dropout", 0.5)
config.add("tree_depth", 2)
config.add("tree_pooling_type", "sum")


file = open("wos_hierarchy.pickle", "wb")
h = corpus_vocab.hierarchy
pickle.dump(h, file)


tokenizer = BertTokenizer.from_pretrained(config.text_encoder.bert_model_dir)

tokenized = hasattr(config.data, 'tokenized') and config.data.tokenized

# get data
train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab, tokenizer=tokenizer, tokenized=tokenized, drop_last=True)

# build up model
model = MODEL(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')

model_file = "ckpt/0827_1307_vanilla_bert_wos_leaf_softmax/best_loss_Origin"
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
                            recursive_penalty=config.train.loss.recursive_regularization.penalty,
                            recursive_constraint=config.train.loss.recursive_regularization.flag, 
                            proba_penalty=config.train.loss.probability_regularization.penalty,
                            proba_constraint=config.train.loss.probability_regularization.flag,
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
elif config.train.losstype == 'conditional_softmax_with_logit_adjustment':
        criterion = ConditionalSoftmaxWithLogitAdjustment(os.path.join(config.data.data_dir, config.data.hierarchy),
                                corpus_vocab.v2i['label'], corpus_vocab.levels,
                                corpus_vocab.proba_vector, config.train.loss.logit_adjustment.tau, 
                                device=config.train.device_setting.device)

trainer = Trainer(model=model,
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                vocab=corpus_vocab,
                config=config)


# predict_probs_val, labels_val, logits_val = trainer.inference(dev_loader, -1, 'VAL')
# pickle.dump(predict_probs_val, open("predictions_runs/wv4/cond_softmax/prob_val4.pickle", "wb"))
# pickle.dump(logits_val, open("predictions_runs/wv4/cond_softmax/logits_val4.pickle", "wb"))
# pickle.dump(labels_val, open("predictions_runs/wv4/cond_softmax/labels_val4.pickle", "wb"))

predict_probs, labels, logits, embeddings = trainer.inference(test_loader, -1, 'TEST')

# pickle.dump(embeddings, open("predictions_runs/hwv_depth2/hitin/prob_1.pickle", "wb"))
pickle.dump(predict_probs, open("predictions_runs/wos_nll/leaf_softmax/probs_1.pickle", "wb"))
pickle.dump(logits, open("predictions_runs/wos_nll/leaf_softmax/logits_1.pickle", "wb"))
pickle.dump(labels, open("predictions_runs/wos_nll/leaf_softmax/labels_1.pickle", "wb"))