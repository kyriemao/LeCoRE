# LeCoRE (WWW 2023)

This is the temporary repository of our WWW 2023 submission: 
Learning Denoised and Interpretable Session Representation for Conversational Search




## Running Environment
Main packages:
- python 3.8.13
- pytorch 1.10.1
- transformers 4.21.2
- numpy: 1.22.4

**Our implementation is based on the excellent open-source [SPLADE repository](https://github.com/naver/splade). Thanks to it!**


## Running Steps

### 1. Download and preprocess data.

The four used public datasets can be downloaded from [QReCC](https://github.com/apple/ml-qrecc), [TopiOCQA](https://github.com/McGill-NLP/topiocqa), [CAsT-19 and CAsT-20](https://www.treccast.ai/). Refer to the [preprocess folder] for data preprocessing and finally move all preprocessed data into a ''datasets'' folder.

### 2. Index passages
We use the pre-trained ad-hoc SPLADE model "naver/efficient-splade-V-large-doc", which can be downloaded in huggingface, to generate passage embeddings:

```python 
# Replacing $Dataset_name with "QReCC", "TopiOCQA" or "CAsT"
python index.py --dataset=$Dataset_name \
--collection_path=$Collection_path \
--pretrained_doc_encoder_path="naver/efficient-splade-V-large-doc" \
--output_index_dir_path=$output_index_dir_path \
--per_gpu_index_batch_size=256 \
--max_doc_length=256 \
--force_emptying_dir \
```

### 3. Train LeCoRE

We provide an example script for training LeCoRE on QReCC. Please run:
```python
bash scripts/train.sh
```


### 4. Evaluate LeCoRE

We provide an example script for evaluating LeCoRE on QReCC. Please run:
```python
bash scripts/test.sh 4
```

