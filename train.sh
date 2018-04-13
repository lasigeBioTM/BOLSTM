#!/usr/bin/env bash
set -e
# $1: corpus path
python src/train_rnn.py preprocessing ddi temp/train_corpus $1
python src/train_rnn.py train temp/dditrain full_model words wordnet common_ancestors concat_ancestors