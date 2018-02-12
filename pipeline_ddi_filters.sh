#!/usr/bin/env bash
set -x
set -e

#python3 src/train_rnn.py train temp/dditrain model_common words common_ancestors
#python3 src/train_rnn.py train temp/dditrainmedline
#python3 src/train_rnn.py predict temp/dditest model_unidirectional words
#python3 src/ddi_evaluation.py convert model_unidirectional_dditest_results.txt
#java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ model_unidirectional_dditest_results.txt.tsv
#python3 src/ddi_evaluation.py analyze model_unidirectional_dditest_results.txt model_wordnet_dditest_results.txt

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

python3 src/train_rnn.py train_test temp/dditrain words words
python3 src/train_rnn.py predict temp/dditest words words
python3 src/ddi_evaluation.py convert words_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ words_dditest_results.txt.tsv

python3 src/train_rnn.py train_test temp/dditrain wordnet words wordnet
python3 src/train_rnn.py predict temp/dditest wordnet words wordnet
python3 src/ddi_evaluation.py convert wordnet_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ wordnet_dditest_results.txt.tsv

python3 src/train_rnn.py train_test temp/dditrain chebi_common words common_ancestors
python3 src/train_rnn.py predict temp/dditest chebi_common words common_ancestors
python3 src/ddi_evaluation.py convert chebi_common_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ chebi_common_dditest_results.txt.tsv

python3 src/train_rnn.py train_test temp/dditrain chebi_concat words concat_ancestors
python3 src/train_rnn.py predict temp/dditest chebi_concat words concat_ancestors
python3 src/ddi_evaluation.py convert chebi_concat_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ chebi_concat_dditest_results.txt.tsv

python3 src/train_rnn.py train_test temp/dditrain full_model words wordnet common_ancestors concat_ancestors
python3 src/train_rnn.py predict temp/dditest full_model words wordnet common_ancestors concat_ancestors
python3 src/ddi_evaluation.py convert full_model_dditest_results.txt
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ full_model_dditest_results.txt.tsv

#python3 src/ddi_evaluation.py analyze model_unidirectional_dditest_results.txt model_wordnet_dditest_results.txt
