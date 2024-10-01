#!/usr/bin/env python
# coding: utf-8

import torch
from helper.utils import get_hierarchy_relations, construct_graph, label_distance
from torch.nn import LogSoftmax, Softmax, LogSigmoid
import networkx as nx
import torch
from helper.utils import get_hierarchy_relations
import torch.nn.functional as F
import torchvision.ops as ops

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = ops.sigmoid_focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        return loss


class ClassificationLoss(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 recursive_penalty,
                 recursive_constraint=True):
        """
        Criterion class, classfication loss & recursive regularization
        :param taxonomic_hierarchy:  Str, file path of hierarchy taxonomy
        :param label_map: Dict, label to id
        :param recursive_penalty: Float, lambda value <- config.train.loss.recursive_regularization.penalty
        :param recursive_constraint: Boolean <- config.train.loss.recursive_regularization.flag
        """
        super(ClassificationLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map)
        self.recursive_penalty = recursive_penalty
        self.recursive_constraint = recursive_constraint

    def _recursive_regularization(self, params, device):
        def cal_reg(param, device):
            """
            recursive regularization: constraint on the parameters of classifier among parent and children
            :param params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
            :param device: torch.device -> config.train.device_setting.device
            :return: loss -> torch.FloatTensor, ()
            """
            rec_reg = 0.0
            for i in range(len(param)):
                if i not in self.recursive_relation.keys():
                    continue
                child_list = self.recursive_relation[i]
                if not child_list:
                    continue
                child_list = torch.tensor(child_list).to(device)
                child_params = torch.index_select(param, 0, child_list)
                parent_params = torch.index_select(param, 0, torch.tensor(i).to(device))
                parent_params = parent_params.repeat(child_params.shape[0], 1)
                _diff = parent_params - child_params
                diff = _diff.view(_diff.shape[0], -1)
                rec_reg += 1.0 / 2 * torch.norm(diff, p=2) ** 2
            return rec_reg

        reg = 0.0
        if type(params) == list:
            for p in params:
                reg += cal_reg(p, device)
        # elif type(params) == torch.Tensor:
        else:
            reg = cal_reg(params, device)


        return reg


    def forward(self, logits, targets, recursive_params):
        """
        :param logits: torch.FloatTensor, (batch, N)
        :param targets: torch.FloatTensor, (batch, N)
        :param recursive_params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        """
        device = logits.device
        if self.recursive_constraint:
            loss_class = self.loss_fn(logits, targets)
            loss_reg = self.recursive_penalty * self._recursive_regularization(recursive_params, device)
            loss = loss_class + loss_reg
        else:
            loss = self.loss_fn(logits, targets)
        return loss

#!/usr/bin/env python
# coding: utf-8


class MATCHLoss(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 recursive_penalty,
                 proba_penalty=0.0,
                 recursive_constraint=True, 
                 proba_constraint=False, 
                 loss = 'bce', 
                 params=None):
        """
        Criterion class, classfication loss & recursive regularization
        :param taxonomic_hierarchy:  Str, file path of hierarchy taxonomy
        :param label_map: Dict, label to id
        :param recursive_penalty: Float, lambda value <- config.train.loss.recursive_regularization.penalty
        :param recursive_constraint: Boolean <- config.train.loss.recursive_regularization.flag
        """
        super(MATCHLoss, self).__init__()
        if loss == 'BCEWithLogitsLoss':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif loss == 'focal':
            if params is not None :    
                self.loss_fn = FocalLoss(**params)
            else : 
                self.loss_fn = FocalLoss()
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map)
        self.recursive_penalty = recursive_penalty
        self.recursive_constraint = recursive_constraint

        self.proba_penalty = proba_penalty
        self.proba_constraint = proba_constraint

    def _recursive_regularization(self, params, device):
        def cal_reg(param, device):
            """
            recursive regularization: constraint on the parameters of classifier among parent and children
            :param params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
            :param device: torch.device -> config.train.device_setting.device
            :return: loss -> torch.FloatTensor, ()
            """
            rec_reg = 0.0
            for i in range(len(param)):
                if i not in self.recursive_relation.keys():
                    continue
                child_list = self.recursive_relation[i]
                if not child_list:
                    continue
                child_list = torch.tensor(child_list).to(device)
                child_params = torch.index_select(param, 0, child_list)
                parent_params = torch.index_select(param, 0, torch.tensor(i).to(device))
                parent_params = parent_params.repeat(child_params.shape[0], 1)
                _diff = parent_params - child_params
                diff = _diff.view(_diff.shape[0], -1)
                rec_reg += 1.0 / 2 * torch.norm(diff, p=2) ** 2
            return rec_reg

        reg = 0.0
        if type(params) == list:
            for p in params:
                reg += cal_reg(p, device)
        # elif type(params) == torch.Tensor:
        else:
            reg = cal_reg(params, device)


        return reg
    
    def _proba_regularization(self, preds, device):
        def cal_reg_prob(preds, device):
            """
            recursive regularization: constraint on the parameters of classifier among parent and children
            :param params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
            :param device: torch.device -> config.train.device_setting.device
            :return: loss -> torch.FloatTensor, ()
            """
            prob_reg = 0.0
            for i in range(preds.shape[1]):
                if i not in self.recursive_relation.keys():
                    continue
                child_list = self.recursive_relation[i]
                if not child_list:
                    continue
                child_list = torch.tensor(child_list).to(device)
                child_proba = torch.index_select(preds, 1, child_list)
                parent_proba = preds[:, [i]]
                parent_proba = parent_proba.repeat(1, child_proba.shape[1])

                _diff = F.relu(child_proba - parent_proba)
                prob_reg += _diff.sum()
            return prob_reg


        reg = cal_reg_prob(preds.sigmoid(), device)
        return reg


    def forward(self, logits, targets, recursive_params):
        """
        :param logits: torch.FloatTensor, (batch, N)
        :param targets: torch.FloatTensor, (batch, N)
        :param recursive_params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        """
        device = logits.device
        loss = self.loss_fn(logits, targets)
        if self.recursive_constraint:
            loss_reg = self.recursive_penalty * self._recursive_regularization(recursive_params, device)
            loss += loss_reg
        if self.proba_constraint:
            loss_prob = self.proba_penalty * self._proba_regularization(logits, device)
            loss += loss_prob

        return loss
    

class CHAMPLoss(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 beta = .2):
        """
        Criterion class, classfication loss & recursive regularization
        :param taxonomic_hierarchy:  Str, file path of hierarchy taxonomy
        :param label_map: Dict, label to id
        :param recursive_penalty: Float, lambda value <- config.train.loss.recursive_regularization.penalty
        :param recursive_constraint: Boolean <- config.train.loss.recursive_regularization.flag
        """
        super(CHAMPLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.taxonomic_hierarchy = construct_graph(taxonomic_hierarchy, label_map)
        self.n = len(label_map.keys())
        self._eps = 10e-15
        self.beta = beta
        self._distance_matrix_()
    
    def _distance_matrix_(self):
        self.distance_matrix = torch.zeros(self.n, self.n, dtype=torch.float)
        for nodei in self.taxonomic_hierarchy.nodes():
            for nodej in self.taxonomic_hierarchy.nodes():
                d = label_distance(self.taxonomic_hierarchy, nodei, nodej)
                self.distance_matrix[nodei][nodej], self.distance_matrix[nodej][nodei] = d, d

        self.max_dist = torch.max(self.distance_matrix)
        self.normalised_distance = (((self.distance_matrix / (self.max_dist + self._eps)) + 1).pow(2) - 1)

    def forward(self, logits, labels, _):

        device = logits.device
        labels = labels.to(device)
        prediction = logits.to(device).sigmoid()
        idx = torch.where(labels==1)
        ones_tensor = torch.ones_like(labels, dtype=torch.float).to(device)  # (bs, lab)
        distance: torch.Tensor = self.distance_matrix.unsqueeze(0).to(device)  # (1, lab, lab)
        zero_f = torch.zeros(1, dtype=torch.float).to(device)

        # Add by 1 in order to avoid edge cases of minm distance since distance[i][j] = 0
        distance = torch.where(distance > -1, distance + 1, zero_f)

        # Mask distance matrix by ground truth values
        distance = labels.unsqueeze(1) * distance

        # Masked values in above step will be set to 0. In order to compute minm later,
        # we reset those values to a high number greater than max distance
        distance = torch.where(distance < 1., self.max_dist + 2, distance).float()

        # Setting indices with minm values in a column to 1 and others to 0,
        # such that for row i and column j, if distance[i][j] = 1, then pred label i is mapped to ground truth value j
        distance = torch.where(distance == distance.min(dim=2, keepdim=True)[0], 1, 0)  # (bs, lab, lab)

        # Refill our concerned binarized values (when distance is 1) with their respective normalised distances
        normalised = self.normalised_distance.to(device).unsqueeze(0)
        distance = torch.where(distance > 0, normalised, zero_f)

        # Modify distance according to how much impact we want from distance penalty in loss calculation
        distance = torch.where(distance != 0., self.beta * distance + 1., zero_f)

        # Computing (1 - p) [part mis-prediction term]
        term1 = (ones_tensor - prediction).unsqueeze(1)
        # Computing log (1-p) [part of mis-prediction term]
        term1 = torch.where(term1 != 0, -torch.log(term1 + self._eps), zero_f)

        # Computing log (p) [part of correct prediction term]
        term2 = torch.where(prediction != 0, -torch.log(prediction + self._eps), zero_f)  # *(alpha)

        # Computing binarized matrix with indices of correct predictions as 1
        correct_ids = labels.unsqueeze(1) * torch.eye(self.n).unsqueeze(0).to(device)

        # Computing loss
        loss1 = torch.matmul(term1, distance).squeeze()
        loss2 = torch.matmul(term2.unsqueeze(1), correct_ids).squeeze()
        loss =  loss1 + loss2
        # print(loss1.mean(), loss2.mean()) 

        return loss.sum(1).mean()

class ConditionalSofmax(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map):
        super(ConditionalSofmax, self).__init__()
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map, add_root_relation=True)
        
    def forward(self, pred, target, _):
        for key in self.recursive_relation.keys():
            pred[:,self.recursive_relation[key]] = LogSoftmax(dim=1)(pred[:,self.recursive_relation[key]])
        for key in self.recursive_relation.keys():
            if key != -1 : 
                pred[:,self.recursive_relation[key]] = pred[:,self.recursive_relation[key]]  +  pred[:, [key]]

        # print(torch.argmax(pred[:, self.recursive_relation[-1]], dim=1))
        # print(torch.argmax(target[:, self.recursive_relation[-1]], dim=1))
        # print('- - - - - - - - - - - -')
        # print(torch.max(torch.exp(pred[:, self.recursive_relation[-1]]), dim=1)[0])
        # print('------------------------')
        loss = -(pred * target)
        loss = loss.sum(dim=1).mean()
        pred = torch.exp(pred)
        return loss, pred

class ConditionalSoftmaxV2(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 levels):
        super(ConditionalSoftmaxV2, self).__init__()
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map, add_root_relation=True)
        self.levels = levels
        
    def forward(self, pred, target, mode='train'):
        for key in self.recursive_relation.keys():
            pred[:,self.recursive_relation[key]] = LogSoftmax(dim=1)(pred[:,self.recursive_relation[key]])
        
        if not mode=='TRAIN':
            pred_clone = pred.clone()
            for level in self.levels:
                for node in self.levels[level] : 
                    if node in self.recursive_relation.keys():
                        pred_clone[:,self.recursive_relation[node]] = pred[:,self.recursive_relation[node]]  +  pred_clone[:, [node]]
            
            pred_clone = torch.exp(pred_clone)
        else : 
            pred_clone = None
        # print(pred_clone[:, self.recursive_relation[-1]].max(dim=1)[0])

        # print(torch.argmax(pred[:, self.recursive_relation[-1]], dim=1))
        # print(torch.argmax(target[:, self.recursive_relation[-1]], dim=1))
        # print('- - - - - - - - - - - -')
        # print(torch.max(torch.exp(pred[:, self.recursive_relation[-1]]), dim=1)[0])
        # print('------------------------')
        loss = -(pred * target)
        loss = loss.sum(dim=1).mean()
        return loss, pred_clone
    
class ConditionalSoftmaxWithLogitAdjustment(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 levels, 
                 probs,
                 tau = 1.0, 
                 device = 'cuda'):
        super(ConditionalSoftmaxWithLogitAdjustment, self).__init__()
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map, add_root_relation=True)
        self.levels = levels
        self.logit_adjustment = tau * torch.log(probs)
        self.logit_adjustment = self.logit_adjustment.to(device)

    def forward(self, pred, target, mode='train'):
        pred_old = pred.clone()
        pred = pred + self.logit_adjustment
        for key in self.recursive_relation.keys():
            pred[:,self.recursive_relation[key]] = LogSoftmax(dim=1)(pred[:,self.recursive_relation[key]])
            pred_old[:,self.recursive_relation[key]] = LogSoftmax(dim=1)(pred_old[:,self.recursive_relation[key]])
        
        if not mode=='TRAIN':
            pred_clone = pred_old.clone()
            for level in self.levels:
                for node in self.levels[level] : 
                    if node in self.recursive_relation.keys():
                        pred_clone[:,self.recursive_relation[node]] = pred_old[:,self.recursive_relation[node]]  +  pred_clone[:, [node]]
            
            pred_clone = torch.exp(pred_clone)
        else : 
            pred_clone = None
        # print(pred_clone[:, self.recursive_relation[-1]].max(dim=1)[0])

        # print(torch.argmax(pred[:, self.recursive_relation[-1]], dim=1))
        # print(torch.argmax(target[:, self.recursive_relation[-1]], dim=1))
        # print('- - - - - - - - - - - -')
        # print(torch.max(torch.exp(pred[:, self.recursive_relation[-1]]), dim=1)[0])
        # print('------------------------')
        loss = -(pred * target)
        loss = loss.sum(dim=1).mean()
        return loss, pred_clone
    

class ConditionalSigmoid(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map, 
                 levels):
        super(ConditionalSigmoid, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map, add_root_relation=True)
        self.epsilon = 1e-7
        self.levels = levels

    def forward(self, pred, target, mode):
        
        pred = pred.sigmoid()

        if not mode=='TRAIN':

            pred_clone = pred.clone()
            for level in self.levels:
                for node in self.levels[level] : 
                    if node in self.recursive_relation.keys():
                        pred_clone[:,self.recursive_relation[node]] = pred[:,self.recursive_relation[node]] * pred_clone[:, [node]]
        else :
            pred_clone = None
        # print(pred_clone[:,self.levels[0]].max(dim=1))

        mask = torch.zeros_like(pred)
        mask[:, self.recursive_relation[-1]] = 1

        # print("level_1 : ", pred_clone[:, self.recursive_relation[-1]].max(dim=1)[0].round(decimals=2))
        
        # pred_level_1 = []
        for b, idx in target.argwhere():
            if int(idx) in self.recursive_relation.keys():
                indices = self.recursive_relation[int(idx)]
                mask[b, indices] = 1
        
        #         pred_level_1.append(float(pred_clone[b, indices].max().detach().cpu().round(decimals=2)))
        # # print("level_2 : ", pred_level_1)

        pred = torch.clamp(pred, self.epsilon, 1.0 - self.epsilon)
        # print(pred[:, self.recursive_relation[-1]].max(dim=1)[0])

        # print(pred * target)
        # print((1 - target) * mask * (1 - pred))

        loss = -(torch.log(pred) * target + (1- target) * mask * torch.log(1 - pred))
        loss = loss.sum(dim=1).mean() 

        return loss, pred_clone
    

class LeafSoftmax(torch.nn.Module):
    def __init__(self, 
                 hierarchy,
                 label_map,
                 leaves_to_nodes, 
                 levels):
        super(LeafSoftmax, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.recursive_relation = get_hierarchy_relations(hierarchy, label_map, )
        self.leaves_to_nodes = leaves_to_nodes
        self.levels = levels

    def forward(self, pred, target, mode):
        label = target[:, list(self.leaves_to_nodes.values())]
        loss = self.loss_fn(pred, label)

        if not mode=='TRAIN':

            ## construct pred 
            pred = pred.softmax(dim=1)
            pred_nodes = torch.zeros_like(target.detach().cpu())
            pred_nodes[:, list(self.leaves_to_nodes.values())] = pred.detach().cpu()
            for level in sorted(self.levels.keys(), reverse=True)[1:]:
                for node in self.levels[level] : 
                    if node in self.recursive_relation.keys():
                        pred_nodes[:,node] = pred_nodes[:,self.recursive_relation[node]].sum(dim=1)
        else : 
            pred_nodes = None

        # print(pred.max(dim=1)[0])
        # print(pred_nodes.max(dim=1)[0])

        return loss, pred_nodes

class PSSoftmaxWithMargin(torch.nn.Module):
    def __init__(self, 
                 hierarchy,
                 label_map, 
                 leaves_to_nodes, 
                 graph_hierarchy,
                 alpha,
                 device):
        super(PSSoftmaxWithMargin, self).__init__()
        self.recursive_relation = get_hierarchy_relations(hierarchy, label_map, add_root_relation=True)
        self.graph_hierarchy = graph_hierarchy.to_undirected()
        self.alpha = alpha
        self.matrix_al = torch.zeros(len(leaves_to_nodes.keys()), len(label_map.keys()))
        self.leaves_to_nodes = leaves_to_nodes
        self.node_to_leaves = dict(zip(leaves_to_nodes.values(), leaves_to_nodes.keys()))
        self.get_matrix_al(self.recursive_relation[-1])
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.matrix_al = self.matrix_al.to(device)
        self.compute_dist_matrix()
        self.dist_matrix = self.dist_matrix.to(device)

    def forward(self, pred, target, mode):
        ## multiply matrix_al with pred 
        pred_leaves = torch.matmul(pred, self.matrix_al.T)
        target_leaves = target[:, list(self.node_to_leaves.keys())]
        margin = (self.alpha * self.dist_matrix[target_leaves.argmax(dim=1), :])
        pred_leaves_with_margin = pred_leaves + margin
        loss = self.loss_fn(pred_leaves_with_margin, target_leaves)

        if not mode=='TRAIN':

            ## construct pred
            pred_leaves = pred_leaves.softmax(dim=1)
            pred_nodes = torch.matmul(pred_leaves, self.matrix_al)
            # print(pred_nodes[:, self.recursive_relation[-1]].max(dim=1)[0])
        else : 
            pred_nodes = None

        return loss, pred_nodes

    def _get_matrix_al_aux(self, node, list_fathers):
        if node in self.recursive_relation.keys():
            for child in self.recursive_relation[node]:
                self._get_matrix_al_aux(child, list_fathers + [node])
        else:
            self.matrix_al[self.node_to_leaves[node], list_fathers + [node]] = 1

    def get_matrix_al(self, nodes):
        for node in nodes:
            self._get_matrix_al_aux(node, [])

    def compute_dist_matrix(self):
        leaves = list(self.leaves_to_nodes.values())
        self.dist_matrix = torch.zeros((len(leaves), len(leaves)))
        for i in range(len(leaves)):
            for j in range(i+1, len(leaves)):
                path = nx.shortest_path(self.graph_hierarchy, leaves[i], leaves[j])
                self.dist_matrix[i, j] = len(path)
                self.dist_matrix[j, i] = len(path)  
        self.dist_matrix = self.dist_matrix / self.dist_matrix.max()


class LeafSoftmaxWithMargin(torch.nn.Module):
    def __init__(self, 
                 hierarchy,
                 label_map, 
                 leaves_to_nodes,
                 graph_hierarchy, 
                 alpha,
                 device):
        super(LeafSoftmaxWithMargin, self).__init__()
        self.recursive_relation = get_hierarchy_relations(hierarchy, label_map, add_root_relation=True)
        self.graph_hierarchy = graph_hierarchy.to_undirected()
        self.alpha = alpha
        self.matrix_al = torch.zeros(len(leaves_to_nodes.keys()), len(label_map.keys()))
        self.leaves_to_nodes = leaves_to_nodes
        self.node_to_leaves = dict(zip(leaves_to_nodes.values(), leaves_to_nodes.keys()))
        self.get_matrix_al(self.recursive_relation[-1])
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.matrix_al = self.matrix_al.to(device)
        self.compute_dist_matrix()
        self.dist_matrix = self.dist_matrix.to(device)

    def forward(self, pred, target, mode):
        ## multiply matrix_al with pred 
        target_leaves = target[:, list(self.node_to_leaves.keys())]
        margin = (self.alpha * self.dist_matrix[target_leaves.argmax(dim=1), :])
        pred_leaves = pred + margin
        loss = self.loss_fn(pred_leaves, target_leaves)

        if not mode=='TRAIN':

            ## construct pred
            pred_leaves = pred.softmax(dim=1)
            pred_nodes = torch.matmul(pred_leaves, self.matrix_al)
            # print(pred_nodes[:, self.recursive_relation[-1]].max(dim=1)[0])
        
        else:
            pred_nodes = None

        return loss, pred_nodes

    def _get_matrix_al_aux(self, node, list_fathers):
        if node in self.recursive_relation.keys():
            for child in self.recursive_relation[node]:
                self._get_matrix_al_aux(child, list_fathers + [node])
        else:
            self.matrix_al[self.node_to_leaves[node], list_fathers + [node]] = 1

    def get_matrix_al(self, nodes):
        for node in nodes:
            self._get_matrix_al_aux(node, [])

    def compute_dist_matrix(self):
        leaves = list(self.leaves_to_nodes.values())
        self.dist_matrix = torch.zeros((len(leaves), len(leaves)))
        for i in range(len(leaves)):
            for j in range(i+1, len(leaves)):
                path = nx.shortest_path(self.graph_hierarchy, leaves[i], leaves[j])
                self.dist_matrix[i, j] = len(path)
                self.dist_matrix[j, i] = len(path)  
        self.dist_matrix = self.dist_matrix / self.dist_matrix.max()

