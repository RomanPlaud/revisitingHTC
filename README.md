# Revisiting Hierarchical Text Classification: Inference and Metrics

**This is the official implementation of [Revisiting Hierarchical Text Classification : Inference and Metrics](), CoNLL 2024.**

This repository is based on [HITIN repo](https://github.com/Rooooyy/HiTIN)

## Abstract 

Hierarchical text classification (HTC) is the task of assigning labels to a text within a structured space organized as a hierarchy. Recent works
treat HTC as a conventional multilabel classification problem, therefore evaluating it as such. We instead propose to evaluate models based on specifically designed hierarchical metrics and we demonstrate the intricacy of metric choice and prediction inference method.  We introduce a new challenging dataset and we 
evaluate fairly, recent sophisticated models, comparing them with a range of simple but strong baselines, including a new theoretically motivated loss. 
Finally, we show that those baselines are very often competitive with the latest models. This highlights the importance of carefully considering the evaluation methodology when proposing new methods for HTC.

## Requirements

Create a conda environnement as follows : 
```shell
conda create -n revisiting_htc_env --file revisiting-htc.txt
```

## Data preparation


### Datasets

Please manage to acquire the original datasets refer to recent implementations to obtain WOS, RCV1 and BGC. 
- For RCV1 and WOS, follow this [repo](https://github.com/Rooooyy/HiTIN/tree/master)  
- For BGC follows this [repo](https://gitlab.com/distration/dsi-nlp-publib/-/blob/main/htc-survey-22/src/dataset_tools/blurb/)

In any case the must exactlty match the format of data/hwv_train.json namely each line must contain a dictionnary whose keys are :
- 'token' containing a list in which it is the raw input text
- 'label' containing the list of labels

Example: 

```json
{
  "token": [
    "Paris (French pronunciation: \u200b[pa\u0281i] (listen)) is the capital and most populous city of France [...]\nThe Tour de France bicycle race finishes on the Avenue des Champs-\u00c9lys\u00e9es in Paris."
  ],
  "label": [
    "Geography",
    "Cities",
    "Europe (Cities)",
    "Western Europe (Europe (Cities))",
    "France (Western Europe) (Western Europe (Europe (Cities)))"
  ]
}

```

In addition a taxonomy file must me added which must match the hwv.taxonomy namely a txt file which have the following properties :
- each line represent a parent followed by all its children 
- each line is separted by '\n'
- in each line each name is separated by '\t'
- each element must exactly cover all labels represented in the dataset used

Example : 

```txt
Root	Technology	Society and social sciences	Arts	Philosophy and religion	Biological and health sciences	Physical sciences	Everyday life	Mathematics (Root)	Geography	History	People
```

### Tokenization 

For a faster running training you can tokenize your dataset. Here is how you should do with hwv dataset

```shell
python3 tokenize_dataset.py --data_train_path data/HWV/hwv_train.json --data_test_path data/HWV/hwv_test.json --data_valid_path data/HWV/hwv_val.json --config_file data/HWV/config_hwv.json
```

We provide guidelines for HWV dataset (but you can easily apply it to other datasets with the same code)

```shell
bash preprocess_hwv.sh
```


## Train

To reproduce the results of our article : 

```shell
bash bash_files/train_hwv_cond_softmax_la.sh
```

(or any other bash file contained in the folder bash_files)

NB: if you dataset is not tokenized please set "tokenized" to false in the config file and change the names of the paths to dataset

