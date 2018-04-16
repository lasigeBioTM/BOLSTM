#!/usr/bin/env bash
set -e
# $1: model name (in models/)
# $2: corpus path
python src/train_rnn.py preprocessing_predict ddi $1 $2 words wordnet common_ancestors concat_ancestors
