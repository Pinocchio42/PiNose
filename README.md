# PiNose
This repo is the official implementation of the paper "Transferable and Efficient Non-Factual Content Detection via Probe Training with Offline Consistency Checking".

This repository contains two main directories: `offline_consistency` and `llm_probe`. Each directory contains scripts for different stages of the data processing and model training pipeline.  
The relevant data and checkpoint can be downloaded from [here](https://drive.google.com/drive/folders/14BvCjyOgR02ytLAuUWfuiydbgmW6iYtL?usp=sharing).

## offline_consistency

The `offline_consistency` directory contains scripts for constructing the training data offline. The scripts should be run in the following order:

1. `question_bootstrapping.py`: This script is used to generate the initial set of questions.
2. `diverse_response_generation.py`: This script generates a diverse set of responses for each question.
3. `peer_review_gathering.py`: This script collects peer reviews for each response.
4. `peer_review_filtering.py`: This script filters the peer reviews to obtain the final training data.

Each sample in the training data is in the format of [Question, Response, Label].

## llm_probe

The `llm_probe` directory contains scripts for probing the language model (LLM). The scripts should be run in the following order:

1. `extract_hidden_state.py`: This script extracts the hidden states of the LLM when it reads the training samples. These hidden states are used as features.
2. `train_classifier.py`: This script trains a classifier using the extracted features.
3. `test_classifier.py`: This script tests the trained classifier on a test dataset.

Please ensure that you have the necessary permissions and environment set up before running these scripts.
