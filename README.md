Do you have any questions, suggestions, feedback or need help?
Please open an [issue](https://github.com/mxschmdt/mrqa-gen-al/issues/new/choose)!

# Introduction
[![arXiv](https://img.shields.io/badge/arXiv-2211.14880-b31b1b.svg?style=flat)](https://arxiv.org/abs/2211.14880)
![Generic badge](https://img.shields.io/badge/ICANN_2024-Link_soon-GREEN.svg?style=flat)

This repository contains code to run data generation for Machine Reading Question Answering (MRQA) with Active Learning (AL) as described in our paper.

# Prerequisites

The scripts in this repository are tested with python 3.12.\
Make sure to install all required packages first.\
For example, create a virtual environment with

```bash
virtualenv <path-to-env> --python=python3.12
```

and activate it:

```bash
source <path-to-env>/bin/activate
```

Afterwards you can install the packages from requirements.txt into your virtualenv: 
```bash
pip install -r requirements.txt
```

# Run scripts
There are two scripts in this repository, `run_training.py` and `run_gen.py`.\
Use the `--help` flag for available commands (they are mostly borrowed from [transformers' Trainer](https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/trainer#transformers.TrainingArguments)).\
`run_training.py` is for training an MRQA or question-answer generation model (including AL), while data generation is performed using `run_gen.py`.

For example, to run training for 3 epochs with a batch size of 32 on the SQuAD 1.1 train dataset and evaluate on the SQuAD 1.1 dev dataset using RoBERTa as the underlying LM, run the following command:
```bash
python run_training.py train rc --output_dir models/rc/squad --num_train_epochs 3 --datasets squad:train --transformer roberta-base --eval-datasets squad:validation --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --learning_rate 3e-5 --disable_no_answer
```
Similarly, replace the task (`rc` in this case) with `qa2s` to perform question-answer pair generation using a two-step decoding process, e.g.,
```bash
python run_training.py train qa2s --output_dir models/gen/squad --num_train_epochs 3 --datasets squad:train --transformer facebook/bart-large --eval-datasets squad:validation --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --learning_rate 3e-5 --disable_no_answer
```

Results will be logged to tensorboard (in the `runs` folder of the output directory or specified using `--logging_dir`), and other libraries (CometML, Wandb, etc.) if installed and enabled for transformers.

To make use of AL, e.g., run
```bash
python run_training.py al --output_dir models/al/bioasq --num_train_epochs 3 --datasets st-bioasq:train --gen_transformer facebook/bart-base --rc_transformer roberta-base --eval-datasets st-bioasq:validation --rc_pretrained <path-to-rc-model-checkpoint> --gen_pretrained <path-to-gen-model-checkpoint> --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --learning_rate 3e-5 --mode rt --samples 50 --rounds 4 --rc_freeze_encoder --logging_steps 1 --eval_steps 2 --max_input_length 1024
```

Finally, here is an example for generating data for the BioASQ domain using abstracts from PubMed as input:
```bash
python run_gen.py qa2s --qg_model <path-to-gen-model-checkpoint> --dataset pubmed-20 --batch_size 8 --seq2seq 1 --token_type_ids 0 --max_gen_length 300 --max_input_length 1024 --num_worker 10 --output_dir data_gen/bioasq
```

The generated data can then be used to train an RC model.

# Data

There are scripts included to be used with the `datasets` library for downloading and processing the datasets.
Custom datasets can further be defined in the config file `data/datasets.ini`.
You may also set a cache directory in `config.ini`.

The MRQA Shared Task 2019 datasets can also be found [here](https://github.com/mrqa/MRQA-Shared-Task-2019#training-data).

You can load any dataset that is compatible with HF's datasets library.
See [https://huggingface.co/datasets](https://huggingface.co/datasets) for a list of available datasets.


# Troubleshooting

If you have issues running the scripts above on mps devices (e.g., on Mac), then use the `--use_cpu` flag.