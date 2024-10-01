# Revisiting Hierarchical Text Classification: Inference and Metrics

This repository is based on Hitin[https://github.com/Rooooyy/HiTIN]


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
-'token' containing a list in which it is the raw input text
-'label' containing the list of labels 

In addition a taxonomy file must me added which must match the hwv.taxonomy namely a txt file which have the following properties :
-each line represent a parent followed by all its children 
-each line is separted by '\n'
-in each line each name is separated by '\t'

### Tokenization 

For a faster running training you can tokenize your dataset. 

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

