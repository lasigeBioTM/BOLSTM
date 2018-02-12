#!/usr/bin/env bash
set -x
set -e
python3 src/train_rnn.py preprocessing ddi temp/dditrain data/DDICorpus/Train/MedLine/ data/DDICorpus/Train/DrugBank/
python3 src/train_rnn.py preprocessing ddi temp/dditrainmedline data/DDICorpus/Train/MedLine/
python3 src/train_rnn.py preprocessing ddi temp/dditraindrugbank data/DDICorpus/Train/DrugBank/
python3 src/train_rnn.py preprocessing ddi temp/dditest data/DDICorpus/Test/DDIExtraction/All/
python3 src/train_rnn.py preprocessing ddi temp/dditestdrugbank data/DDICorpus/Test/DDIExtraction/DrugBank/
python3 src/train_rnn.py preprocessing ddi temp/dditestmedline data/DDICorpus/Test/DDIExtraction/MedLine/
