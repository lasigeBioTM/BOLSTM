#!/usr/bin/env bash
set -x
set -e
#python src/train_rnn.py preprocessing ddi temp/dditrain data/DDICorpus/Train/DrugBank/ data/DDICorpus/Train/MedLine/
#python src/train_rnn.py preprocessing ddi temp/dditrainmedline data/DDICorpus/Train/MedLine/
#python src/train_rnn.py preprocessing ddi temp/dditraindrugbank data/DDICorpus/Train/DrugBank/
#python src/train_rnn.py preprocessing ddi temp/dditest data/DDICorpus/Test/DDIExtraction/All/
python src/train_rnn.py preprocessing ddi temp/dditestdrugbank data/DDICorpus/Test/DDIExtraction/DrugBank
python src/train_rnn.py preprocessing ddi temp/dditestmedline data/DDICorpus/Test/DDIExtraction/MedLine

python src/train_rnn.py train temp/dditrain
#python src/train_rnn.py train temp/dditrainmedline
python src/train_rnn.py predict temp/dditest
python src/ddi_evaluation.py dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ dditest_results.txt.tsv

python src/train_rnn.py predict temp/dditestdrugbank
python src/ddi_evaluation.py dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/DrugBank dditest_results.txt.tsv

python src/train_rnn.py predict temp/dditestmedline
python src/ddi_evaluation.py dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/MedLine dditest_results.txt.tsv

#python src/train_rnn.py train temp/dditraindrugbank
#python src/train_rnn.py predict temp/dditest
#python src/ddi_evaluation.py dditest_results.txt
#java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ dditest_results.txt.tsv
