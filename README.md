# Project Title — NewsCLIPpings Fine-Tuning

**Goal:** Fine-tune CLIP on image-text dataset on binary classification task to evaluate mismatched image-caption pairs. 

### Overview
- Built pipeline for preprocessing ~550K image–text pairs.
- Implemented fine-tuning with PyTorch + custom classification head.
- Evaluated model accuracy and explainability via gradient-based attributions.

### Tech Stack
Python, PyTorch, Transformers, Parquet, Docker, AWS EC2

### Results
Achieved ~66.7% binary accuracy, comparable with EMNLP 2021 paper results. 

### Notes
Repo includes base training and evaluation scripts used for experiments.
