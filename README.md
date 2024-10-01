# Revisiting Hierarchical Text Classification: Inference and Metrics

**Official implementation of [Revisiting Hierarchical Text Classification: Inference and Metrics], CoNLL 2024.**

Based on [HITIN repo](https://github.com/Rooooyy/HiTIN).

## Abstract

Hierarchical Text Classification (HTC) assigns labels to text in a structured hierarchical space. Recent approaches treat it as multilabel classification without considering hierarchical relationships. We propose specialized hierarchical metrics and novel inference methods. A new challenging dataset is introduced to fairly compare sophisticated models and competitive baselines, with a focus on evaluation methodology.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/RomanPlaud/revisitingHTC.git
    cd revisitingHTC
    ```
2. Create and activate the conda environment:
    ```bash
    conda create -n revisiting_htc_env --file revisiting-htc.txt
    conda activate revisiting_htc_env
    ```

## Dataset Preparation

### Datasets
Obtain the RCV1, WOS, and BGC datasets by referring to:
- [HiTIN repo for RCV1 and WOS](https://github.com/Rooooyy/HiTIN/tree/master)
- [BGC dataset from this repo](https://gitlab.com/distration/dsi-nlp-publib/-/blob/main/htc-survey-22/src/dataset_tools/blurb/)

Ensure the datasets match the format:
```json
{
  "token": ["Sample input text"],
  "label": ["Category", "Subcategory", "Further Subcategory"]
}

```

In addition a taxonomy file (such as [hwv.taxonomy](data/HWV/hwv.taxonomy)) is required where each line represents a parent category followed by its children, separated by tabs. Ensure all labels used in the dataset are covered.

Example : 

```txt
Root	Science	Technology	Arts
Science	Physics	Chemistry	Biology
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

