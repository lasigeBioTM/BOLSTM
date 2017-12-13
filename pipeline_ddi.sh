#!/usr/bin/env bash
set -x
set -e
python src/train_rnn.py preprocessing ddi temp/dditrain data/DDICorpus/Train/DrugBank/ data/DDICorpus/Train/MedLine/
#python src/train_rnn.py preprocessing ddi temp/dditest data/DDICorpus/Test/DDIExtraction/All/
python src/train_rnn.py train temp/dditrain
python src/train_rnn.py predict temp/dditest
python src/ddi_evaluation.py dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ dditest_results.txt.tsv
