#!/usr/bin/env bash
set -x
set -e
python3 src/train_rnn.py train temp/dditrain temp/dditest
#python3 src/train_rnn.py train temp/dditrainmedline
python3 src/train_rnn.py predict temp/dditest
python3 src/ddi_evaluation.py dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ dditest_results.txt.tsv

#python3 src/train_rnn.py predict temp/dditestdrugbank
#python3 src/ddi_evaluation.py dditestdrugbank_results.txt
#java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/DrugBank dditestdrugbank_results.txt.tsv

#python3 src/train_rnn.py predict temp/dditestmedline
#python3 src/ddi_evaluation.py dditestmedline_results.txt
#java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/MedLine dditestmedline_results.txt.tsv

#python3 src/train_rnn.py train temp/dditraindrugbank
#python3 src/train_rnn.py predict temp/dditest
#python3 src/ddi_evaluation.py dditest_results.txt
#java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ dditest_results.txt.tsv
