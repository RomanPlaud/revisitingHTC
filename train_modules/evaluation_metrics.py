#!/usr/bin/env python
# coding:utf-8

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from helper.utils import get_hierarchy_relations, preprocess_predictions
import os
import torch
import networkx as nx



def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f


def evaluate(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        sample_predict_label_list = [id2label[i] for i in sample_predict_id_list]

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1}


def evaluate_fast(predictions, labels, vocab, threshold=.5, Type='threshold', configs=None, per_level_acc=False):
    if Type == 'threshold':
        predictions = np.array([prediction>threshold for prediction in predictions])
    elif Type == 'top_down_max' or Type == 'top_down_threshold':
        relations = get_hierarchy_relations(os.path.join(configs.data.data_dir, configs.data.hierarchy), vocab.v2i['label'], add_root_relation=True)
        predictions_new = []
        for prediction in predictions:
            idx = -1
            prediction_new = np.zeros_like(prediction)
            while idx in relations.keys():
                maxi = np.max(prediction[relations[idx]], axis=0)
                idx_level = np.argmax(prediction[relations[idx]], axis=0)
                idx = relations[idx][idx_level]
                if Type == 'top_down_threshold':
                    if maxi < threshold:
                        break
                prediction_new[idx] = 1
            predictions_new.append(prediction_new)
        predictions = np.array(predictions_new)

    elif Type == 'leaf_softmax':
        relations = get_hierarchy_relations(os.path.join(configs.data.data_dir, configs.data.hierarchy), vocab.v2i['label'], add_root_relation=True)
        predictions_new = []
        for prediction in predictions:
            prediction_new = np.zeros_like(prediction)
            prediction_leaf = prediction[list(vocab.idxleaf2idxnodes.values())]
            leaf_idx = np.argmax(prediction_leaf, axis=0)
            leaf_idx = vocab.idxleaf2idxnodes[leaf_idx]
            prediction_new[leaf_idx] = 1
            ## find the key in relations which value contain leaf_idx
            for level in reversed(sorted(list(vocab.levels.keys()))[:vocab.idx2levels[leaf_idx]]):
                for node in vocab.levels[level]:
                    if (node in relations.keys()) and (leaf_idx in relations[node]):
                        leaf_idx = node
                        prediction_new[leaf_idx] = 1
                        break
            predictions_new.append(prediction_new) 
        predictions = np.array(predictions_new)
       

    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    metrics = {'micro_f1': micro_f1, 'macro_f1': macro_f1}

    if per_level_acc:
        for level in vocab.levels.keys():
            labels_level = labels[:, vocab.levels[level]]
            predictions_level = predictions[:, vocab.levels[level]]
            metrics["acc level " + str(level)] = accuracy_score(labels_level, predictions_level)

    return metrics

def evaluate_top_down_threshold(predictions, labels, vocab, threshold=.5, configs=None, type = "set", metrics = ["micro", "macro"], modes = ["lca", "standard"], depths = [0, 1, 2]):
    relations = get_hierarchy_relations(os.path.join(configs.data.data_dir, configs.data.hierarchy), vocab.v2i['label'], add_root_relation=True)

    def _infer_recursive(father, prediction, threshold, current_path, paths):
        if father==-1 or prediction[father] >= threshold:
            if father not in relations.keys():
                paths.append(current_path)
            else:
                if np.max(prediction[relations[father]]) >= threshold:
                    for child in relations[father]:
                        _infer_recursive(child, prediction, threshold, current_path + [child], paths)
                else:
                    if current_path == []:
                        paths.append([-1])
                    else : 
                        paths.append(current_path)

    
    def compute_set(path_label, paths_predictions, lca_bool=False, mode='micro', depth=None):
        if mode == 'micro':
            most_specific_predictions = paths_predictions
            most_specific_label = path_label
            if lca_bool : 
                most_specific_predictions = [path[-1] for path in paths_predictions]
                most_specific_label = path_label[-1]
        if mode == 'macro':
            most_specific_predictions = [path[:depth+1] for path in paths_predictions]
            most_specific_label = path_label[:depth+1]
            if lca_bool :
                most_specific_predictions = [path[:depth+1][-1] for path in paths_predictions]
                most_specific_label = path_label[:depth+1][-1]

        r_i = []
        if lca_bool:
            lcas = [nx.lowest_common_ancestor(vocab.graph_hierarchy, most_specific_label, node) for node in most_specific_predictions]
            most_specific_lca = lcas[np.argmax([vocab.idx2levels[lca] if lca!=-1 else -1 for lca in lcas])]
            r_i = [lca for lca in lcas if lca!=most_specific_lca]

        augmented_predictions = set()
        for i, pred in enumerate(most_specific_predictions):
            if lca_bool:
                path = list(nx.shortest_path(vocab.graph_hierarchy, lcas[i], pred))
            else:
                path = pred
            augmented_predictions = augmented_predictions.union(set(path))
        augmented_predictions = (augmented_predictions.difference(set(r_i))).difference(set([-1]))

        if lca_bool:
            augmented_label = set(list(nx.shortest_path(vocab.graph_hierarchy, most_specific_lca, most_specific_label)))
        else : 
            augmented_label = set(most_specific_label)
            most_specific_label = most_specific_label[-1]
        augmented_label = augmented_label.difference(set([-1]))

        intersection = augmented_predictions.intersection(augmented_label)

        # print(intersection, augmented_predictions, augmented_label)

        if mode == 'micro':
            return len(intersection), len(augmented_predictions), len(augmented_label)
        if mode == 'macro':
            return len(intersection), len(augmented_predictions), len(augmented_label), most_specific_label


    results = {metric : {mode : {} for mode in modes}  for metric in metrics }
    for metric in metrics:
        for mode in modes:
            if metric == 'micro':
                results[metric][mode] = {'pred': [], 'label': [], 'intersection': []}
            if metric == 'macro':
                for depth in depths:
                    results[metric][mode][depth] = {'pred': [], 'label': [], 'intersection': [], 'most_specific_label': []}

    for prediction, label in zip(predictions, labels):

        label_paths = []
        _infer_recursive(-1, label, threshold, [], label_paths)
        label_path = label_paths[0]

        predicted_paths = []
        _infer_recursive(-1, prediction, threshold, [], predicted_paths)

        
        for metric in metrics:
            for mode in modes : 
                if metric == 'micro':
                    intersection, augmented_predictions, augmented_label = compute_set(label_path, predicted_paths, mode=='lca', 'micro')
                    results[metric][mode]['pred'].append(augmented_predictions)
                    results[metric][mode]['label'].append(augmented_label)
                    results[metric][mode]['intersection'].append(intersection)
                if metric == 'macro':
                    for depth in depths :
                        if len(label_path)>=depth+1:
                            intersection, augmented_predictions, augmented_label, most_specific_label = compute_set(label_path, predicted_paths, mode=='lca', 'macro', depth)
                            results[metric][mode][depth]['pred'].append(augmented_predictions)
                            results[metric][mode][depth]['label'].append(augmented_label)
                            results[metric][mode][depth]['intersection'].append(intersection)
                            results[metric][mode][depth]['most_specific_label'].append(most_specific_label)
    
    for metric in metrics:
        for mode in modes :
            if metric == 'micro':
                precision = np.sum(results[metric][mode]['intersection']) / np.sum(results[metric][mode]['pred'])
                recall = np.sum(results[metric][mode]['intersection']) / np.sum(results[metric][mode]['label'])
                results[metric][mode]['precision'] = precision
                results[metric][mode]['recall'] = recall

                del results[metric][mode]['pred']
                del results[metric][mode]['intersection']
                del results[metric][mode]['label']

                if precision + recall == 0:
                    results[metric][mode]['f1_score'] = 0
                else :
                    results[metric][mode]['f1_score'] = 2 * precision * recall / (precision + recall)

            if metric == 'macro':
                for depth in depths:
                    precisions = []
                    recalls = []
                    for most_specific_label in results[metric][mode][depth]['most_specific_label']:
                        idx = np.where(np.array(results[metric][mode][depth]['most_specific_label']) == most_specific_label)[0]
                        if np.sum(np.array(results[metric][mode][depth]['pred'])[idx]) != 0 : 
                            precision = np.sum(np.array(results[metric][mode][depth]['intersection'])[idx]) / np.sum(np.array(results[metric][mode][depth]['pred'])[idx])
                        else : 
                            precision = 0
                        recall = np.sum(np.array(results[metric][mode][depth]['intersection'])[idx]) / np.sum(np.array(results[metric][mode][depth]['label'])[idx])
                        precisions.append(precision)
                        recalls.append(recall)

                    del results[metric][mode][depth]['pred']
                    del results[metric][mode][depth]['intersection']
                    del results[metric][mode][depth]['label']
                    del results[metric][mode][depth]['most_specific_label']

                    precision = np.mean(precisions)
                    recall = np.mean(recalls)
                    results[metric][mode][depth]['precision'] = precision
                    results[metric][mode][depth]['recall'] = recall

                    if precision + recall == 0:
                        results[metric][mode][depth]['f1_score'] = 0.0
                    else:
                        results[metric][mode][depth]['f1_score'] = 2 * precision * recall / (precision + recall)

    return results


def evaluate_very_fast_aux(probs, labels, vocab, type_of_averaging, depth):
    probs = np.array(preprocess_predictions(probs, vocab.hierarchy))
    if type_of_averaging == "micro":
        precision = precision_score(labels.ravel(), probs.ravel() > 0.5)
        recall = recall_score(labels.ravel(), probs.ravel() > 0.5)
        f1 = f1_score(labels.ravel(), probs.ravel() > 0.5)
        return precision, recall, f1
    elif type_of_averaging == 'macro':
        idx = [j for k in range(depth+1) for j in vocab.levels[k]]
        ps, rs = [], []
        for j in vocab.levels[depth]:
            idx_label = np.where(labels[:, j] == 1)[0]

            if len(idx_label) != 0:
                precision = precision_score(labels[idx_label][:, idx].ravel(), probs[idx_label][:, idx].ravel() > 0.5)
                recall = recall_score(labels[idx_label][:, idx].ravel(), probs[idx_label][:, idx].ravel() > 0.5)
                ps.append(precision)
                rs.append(recall)
        p, r = np.mean(ps), np.mean(rs)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        return p, r, f1

def evaluate_very_fast(probs, labels, vocab, depth_max):
    metrics = {'micro': {}, 'macro': {}}
    precision, recall, f1 = evaluate_very_fast_aux(probs, labels, vocab, 'micro', depth_max)
    metrics['micro']['precision'] = precision
    metrics['micro']['recall'] = recall
    metrics['micro']['f1_score'] = f1

    precision, recall, f1 = evaluate_very_fast_aux(probs, labels, vocab, 'macro', depth_max)
    metrics['macro']['precision'] = precision
    metrics['macro']['recall'] = recall
    metrics['macro']['f1_score'] = f1

    return metrics
