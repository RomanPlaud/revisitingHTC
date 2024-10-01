#!/usr/bin/env python

import codecs
import torch
# from models.structure_model.tree import Tree
import networkx as nx


def load_checkpoint(model_file, model, config, optimizer=None):
    """
    load models
    :param model_file: Str, file path
    :param model: Computational Graph
    :param config: helper.configure, Configure object
    :param optimizer: optimizer, torch.Adam
    :return: best_performance -> [Float, Float], config -> Configure
    """
    checkpoint_model = torch.load(model_file)
    config.train.start_epoch = checkpoint_model['epoch'] + 1
    best_performance = checkpoint_model['best_loss']
    model.load_state_dict(checkpoint_model['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_model['optimizer'])
    return best_performance, config


def save_checkpoint(state, model_file):
    """
    :param state: Dict, e.g. {'state_dict': state,
                              'optimizer': optimizer,
                              'best_performance': [Float, Float],
                              'epoch': int}
    :param model_file: Str, file path
    :return:
    """
    torch.save(state, model_file)


def get_hierarchy_relations(hierar_taxonomy, label_map, root=None, fortree=False, add_root_relation=False):
    """
    get parent-children relationships from given hierar_taxonomy
    parent_label \t child_label_0 \t child_label_1 \n
    :param hierar_taxonomy: Str, file path of hierarchy taxonomy
    :param label_map: Dict, label to id
    :param root: Str, root tag
    :param fortree: Boolean, True : return label_tree -> List
    :return: label_tree -> List[Tree], hierar_relation -> Dict{parent_id: List[child_id]}
    """
    label_tree = dict()
    label_tree[0] = root
    hierar_relations = {}
    with codecs.open(hierar_taxonomy, "r", "utf8") as f:
        for line in f:
            line_split = line.rstrip().split('\t')
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label not in label_map:
                if fortree and parent_label == 'Root':
                    parent_label_id = -1
                elif add_root_relation and parent_label == 'Root':
                    parent_label_id = -1
                else:
                    continue
            else:
                parent_label_id = label_map[parent_label]
            children_label_ids = [label_map[child_label] \
                                  for child_label in children_label if child_label in label_map]
            hierar_relations[parent_label_id] = children_label_ids
            if fortree:
                assert (parent_label_id + 1) in label_tree
                parent_tree = label_tree[parent_label_id + 1]

                for child in children_label_ids:
                    assert (child + 1) not in label_tree
                    child_tree = Tree(child)
                    parent_tree.add_child(child_tree)
                    label_tree[child + 1] = child_tree
    if fortree:
        return hierar_relations, label_tree
    else:
        return hierar_relations

def label_distance(g: nx.Graph, label1: str, label2: str) -> int:
    return nx.shortest_path_length(g, source=label1, target=label2)


def construct_graph(hierar_taxonomy, label_map):
    taxonomy = nx.Graph()
    with codecs.open(hierar_taxonomy, "r", "utf8") as f:
        for line in f:
            line_split = line.rstrip().split('\t')
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label == 'Root':
                parent_label_id = -1
            else:
                parent_label_id = label_map[parent_label]
            for child_label in children_label:
                if child_label in label_map.keys():
                    taxonomy.add_edge(parent_label_id, label_map[child_label])
    return taxonomy

def compute_learning_rates(lr, levels):
    lrs = [lr]
    for i in sorted(list(levels.keys()))[:-1]:
        lrs.append(lrs[-1] * (len(levels[i+1])/len(levels[i])))
    return list(lrs)
        
        
def preprocess_predictions(predictions, relations):
    new_predictions = []
    for p in predictions:
        pred = p.copy()
        for key, value in relations.items():
            for v in value:
                if pred[v] > pred[key]:
                    pred[v] = pred[key]
        new_predictions.append(pred)
    return new_predictions