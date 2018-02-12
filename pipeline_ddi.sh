#!/usr/bin/env bash
set -x
set -e

python3 src/train_rnn.py train temp/dditrain words words
python3 src/train_rnn.py predict temp/dditest words words
python3 src/ddi_evaluation.py convert words_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ words_dditest_results.txt.tsv

python3 src/train_rnn.py train temp/dditrain wordnet words wordnet
python3 src/train_rnn.py predict temp/dditest wordnet words wordnet
python3 src/ddi_evaluation.py convert wordnet_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ wordnet_dditest_results.txt.tsv

python3 src/train_rnn.py train temp/dditrain chebi_common words common_ancestors
python3 src/train_rnn.py predict temp/dditest chebi_common words common_ancestors
python3 src/ddi_evaluation.py convert chebi_common_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ chebi_common_dditest_results.txt.tsv

python3 src/train_rnn.py train temp/dditrain chebi_concat words concat_ancestors
python3 src/train_rnn.py predict temp/dditest chebi_concat words concat_ancestors
python3 src/ddi_evaluation.py convert chebi_concat_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ chebi_concat_dditest_results.txt.tsv

python3 src/train_rnn.py train temp/dditrain full_model words wordnet common_ancestors concat_ancestors
python3 src/train_rnn.py predict temp/dditest full_model words wordnet common_ancestors concat_ancestors
python3 src/ddi_evaluation.py convert full_model_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ full_model_dditest_results.txt.tsv

python3 src/ddi_evaluation.py analyze words_dditest_results wordnet_dditest_results full_model_dditest_results
python3 src/ddi_evaluation.py analyze words_dditestdrugbank_results wordnet_dditestdrugbank_results full_model_dditestdrugbank_results
python3 src/ddi_evaluation.py analyze words_dditestmedline_results wordnet_dditestmedline_results full_model_dditestmedline_results