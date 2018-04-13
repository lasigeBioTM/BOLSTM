#!/usr/bin/env bash
set -e
# $1: corpus path
python src/train_rnn.py preprocessing ddi temp/predict_corpus $1
python src/train_rnn.py predict temp/predict_corpus full_model words wordnet common_ancestors concat_ancestors
