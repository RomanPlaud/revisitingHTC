#!/usr/bin/env python
# coding:utf-8

import pickle
from collections import Counter
import helper.logger as logger
import tqdm
import os
import json
from helper.utils import get_hierarchy_relations
import torch
import networkx as nx

class Vocab(object):
    def __init__(self, config):
        """
        vocabulary class for text classification, initialized from pretrained embedding file
        and update based on minimum frequency and maximum size
        :param config: helper.configure, Configure Object
        :param min_freq: int, the minimum frequency of tokens
        :param special_token: List[Str], e.g. padding and out-of-vocabulary
        :param max_size: int, maximum size of the overall vocabulary
        """
        logger.info('Processing....')
        self.corpus_files = {"TRAIN": os.path.join(config.data.data_dir, config.data.train_file),
                             "VAL": os.path.join(config.data.data_dir, config.data.val_file),
                             "TEST": os.path.join(config.data.data_dir, config.data.test_file)}
        counter = Counter()
        self.config = config
        # counter for tokens
        self.freqs = {'label': counter.copy()}
        self.freqs2 = {'label': counter.copy()}
        # vocab to index
        self.v2i = {'label': dict()}
        # index to vocab
        self.i2v = {'label': dict()}

        if False:
            pass

        # if not os.path.isdir(self.config.vocabulary.dir):
        #     os.system('mkdir ' + str(self.config.vocabulary.dir))
        # label_dir = os.path.join(self.config.vocabulary.dir, self.config.data.dataset + '_' + self.config.vocabulary.label_dict)
        # if os.path.isfile(label_dir) :
        #     with open(label_dir, 'r') as f_in:
        #         for i, line in enumerate(f_in):
        #             data = line.rstrip().split('\t')
        #             assert len(data) == 2
        #             self.v2i['label'][data[0]] = i
        #             self.i2v['label'][i] = data[0]
        else:
            logger.info('Generating Vocabulary from Corpus...')
            self._count_vocab_from_corpus()
            # print(len(self.freqs['label'].keys()))
            for field in self.freqs.keys():
                temp_vocab_list = list(self.freqs[field].keys())
                for i, k in enumerate(temp_vocab_list):
                    self.v2i[field][k] = i
                    self.i2v[field][i] = k

        self._get_leaf_label()
        self._get_per_level_index()
        self._get_mask_for_progressive_learning()
        self._create_graph()
        self._creat_probs_vector()


    # def _load_pretrained_embedding_vocab(self):
    #     """
    #     initialize counter for word in pre-trained word embedding
    #     """
    #     pretrained_file_dir = self.config.embedding.token.pretrained_file
    #     with open(pretrained_file_dir, 'r', encoding='utf8') as f_in:
    #         logger.info('Loading vocabulary from pretrained embedding...')
    #         for line in tqdm.tqdm(f_in):
    #             data = line.rstrip('\n').split(' ')
    #             if len(data) == 2:
    #                 # first line in pretrained embedding
    #                 continue
    #             v = data[0]
    #             self.freqs['token'][v] += self.min_freq + 1

    def _count_vocab_from_corpus(self):
        """
        count the frequency of tokens in the specified corpus
        """
        for corpus in self.corpus_files.keys():
            mode = 'ALL'
            with open(self.corpus_files[corpus], 'r') as f_in:
                logger.info('Loading ' + corpus + ' subset...')
                for line in tqdm.tqdm(f_in):
                    data = json.loads(line.rstrip())
                    self._count_vocab_from_sample(data, mode)
                    if corpus in ['TRAIN', 'VAL']:
                        self._count_vocab_from_sample_for_la(data, mode)

    def _count_vocab_from_sample(self, line_dict, mode='ALL'):
        """
        update the frequency from the current sample
        :param line_dict: Dict{'token': List[Str], 'label': List[Str]}
        """
        for k in self.freqs.keys():
            if mode == 'ALL':
                for t in line_dict[k]:
                    self.freqs[k][t] += 1

    def _count_vocab_from_sample_for_la(self, line_dict, mode='ALL'):
        """
        update the frequency from the current sample
        :param line_dict: Dict{'token': List[Str], 'label': List[Str]}
        """
        for k in self.freqs.keys():
            if mode == 'ALL':
                for t in line_dict[k]:
                    self.freqs2[k][t] += 1

    # def _shrink_vocab(self, k, max_size=None):
    #     """
    #     shrink the vocabulary
    #     :param k: Str, field <- 'token', 'label'
    #     :param max_size: int, the maximum number of vocabulary
    #     """
    #     logger.info('Shrinking Vocabulary...')
    #     tmp_dict = Counter()
    #     for v in self.freqs[k].keys():
    #         if self.freqs[k][v] >= self.min_freq:
    #             tmp_dict[v] = self.freqs[k][v]
    #     if max_size is not None:
    #         tmp_list_dict = tmp_dict.most_common(max_size)
    #         self.freqs[k] = Counter()
    #         for (t, v) in tmp_list_dict:
    #             self.freqs[k][t] = v
    #     logger.info('Shrinking Vocabulary of tokens: ' + str(len(self.freqs[k])))

    def _get_leaf_label(self):
        self.hierarchy = get_hierarchy_relations(os.path.join(os.path.join(self.config.data.data_dir, self.config.data.hierarchy)), self.v2i['label'], add_root_relation=True)
        n_nodes = len(self.v2i['label'].keys())
        leaves = [i for i in range(n_nodes) if i not in self.hierarchy.keys()]
        self.idxleaf2idxnodes = dict(zip(range(len(leaves)), leaves))

    def _get_per_level_index(self):
        children = self.hierarchy[-1]
        self.levels = {}
        self.idx2levels = {}
        i = 0
        while children!=[]:
            self.levels[i] = children
            for ch in children:
                self.idx2levels[ch] = i
            children = [c for ch in children for c in self.hierarchy.get(ch, [])]            
            i+=1
    
    def _get_mask_for_progressive_learning(self):
        nb_levels = len(self.levels.keys())
        self.mask_per_level = {str(nb_levels-1)+'/'+str(nb_levels-1) : torch.ones(len(self.v2i['label'].keys()))}
        for i in range(nb_levels-2, -1, -1):
            mask_i = self.mask_per_level[str(i+1)+'/'+str(nb_levels-1)].clone()
            mask_i[self.levels[i+1]] = 0
            self.mask_per_level[str(i)+'/'+str(nb_levels-1)] = mask_i
        self.mask_per_level = dict(reversed(self.mask_per_level.items()))

            
    def _create_graph(self):
        self.graph_hierarchy = nx.DiGraph()
        for k, v in self.hierarchy.items():
            for c in v:
                self.graph_hierarchy.add_edge(k, c)            

    def _creat_probs_vector(self):
        for k in self.freqs['label'].keys():
            if k not in self.freqs2['label'].keys():
                self.freqs2['label'][k] = 1
        count = dict(self.freqs2['label'])
        self.proba_vector = torch.zeros(len(self.v2i['label']))
        for _, value in self.hierarchy.items():
            names_children = [self.i2v['label'][v] for v in value]
            sum_brothers = sum([count[v] for v in names_children])
            for v in names_children:
                i = self.v2i['label'][v]
                self.proba_vector[i] = count[v]/sum_brothers