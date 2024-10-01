import argparse
import os
from tqdm import tqdm
import json
from data_modules.vocab import Vocab
from helper.configure import Configure
import numpy as np
from transformers import BertTokenizer

import sys
sys.path.append('../')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_train_path', type=str, default='',
                        help='location of the data corpus')
    parser.add_argument('--data_test_path', type=str, default='',
                        help='location of the data corpus')
    parser.add_argument('--data_valid_path', type=str, default='',
                        help='location of the data corpus')
    parser.add_argument('--max_length', type=int, default=512,
                        help='max sequence length')
    parser.add_argument('--config_file', type=str)

    return parser.parse_args()


def tokenize_dataset(args):

    config = Configure(config_json_file=args.config_file)
    corpus_vocab = Vocab(config)
    # open a .json files
    lines_train = []
    with open(args.data_train_path, 'r') as f:
        for line in f:
            lines_train.append(json.loads(line))
    lines_test = []
    with open(args.data_test_path, 'r') as f:
        for line in f:
            lines_test.append(json.loads(line))
    lines_valid = []
    with open(args.data_valid_path, 'r') as f:
        for line in f:
            lines_valid.append(json.loads(line))

    new_lines_train = []
    for line in tqdm(lines_train):
        new_line = {}
        r = tokenizer(line['token'], padding='max_length', truncation=True, max_length=args.max_length)
        for k, v in r.items():
            new_line[k] = v
        new_line['label_id'] = [corpus_vocab.v2i['label'][l] for l in line['label']]
        new_line['label'] = line['label']
        new_lines_train.append(new_line)
    
    new_lines_test = []
    for line in tqdm(lines_test):
        new_line = {}
        r = tokenizer(line['token'], padding='max_length', truncation=True, max_length=args.max_length)
        for k, v in r.items():
            new_line[k] = v
        new_line['label_id'] = [corpus_vocab.v2i['label'][l] for l in line['label']]
        new_line['label'] = line['label']
        new_lines_test.append(new_line)
    
    new_lines_valid = []
    for line in tqdm(lines_valid):
        new_line = {}
        r = tokenizer(line['token'], padding='max_length', truncation=True, max_length=args.max_length)
        for k, v in r.items():
            new_line[k] = v
        new_line['label_id'] = [corpus_vocab.v2i['label'][l] for l in line['label']]
        new_line['label'] = line['label']
        new_lines_valid.append(new_line)
    
    # write the tokenized file
    with open(args.data_train_path.replace('.json', '_tokenized.json'), 'w') as f:
        for line in new_lines_train:
            line = json.dumps(line)
            f.write(line + '\n')
    f.close()
    with open(args.data_test_path.replace('.json', '_tokenized.json'), 'w') as f:
        for line in (new_lines_test):
            line = json.dumps(line)
            f.write(line + '\n')
    f.close()
    with open(args.data_valid_path.replace('.json', '_tokenized.json'), 'w') as f:
        for line in (new_lines_valid):
            line = json.dumps(line)
            f.write(line + '\n')
    f.close()

    
    
if __name__ == '__main__':
    args = parse_args()
    tokenize_dataset(args)


