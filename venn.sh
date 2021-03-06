#!/usr/bin/env bash
set -x
set -e
java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/All/ words_dditest_results.txt.tsv
#python3 src/ddi_evaluation.py analyze words_dditest_results.txt wordnet_dditest_results.txt chebi_common_dditest_results.txt chebi_concat_dditest_results.txt
python3 src/ddi_evaluation.py analyze words_dditest_results wordnet_dditest_results full_model_dditest_results

java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/DrugBank/ words_dditestdrugbank_results.txt.tsv
#python3 src/ddi_evaluation.py analyze words_dditestdrugbank_results.txt wordnet_dditestdrugbank_results.txt chebi_common_dditestdrugbank_results.txt chebi_concat_dditestdrugbank_results.txt
python3 src/ddi_evaluation.py analyze words_dditestdrugbank_results wordnet_dditestdrugbank_results full_model_dditestdrugbank_results

java -jar evaluateDDI.jar data/DDICorpus/Test/DDIExtraction/MedLine/ words_dditestmedline_results.txt.tsv
#python3 src/ddi_evaluation.py analyze words_dditestmedline_results.txt wordnet_dditestmedline_results.txt chebi_common_dditestmedline_results.txt chebi_concat_dditestmedline_results.txt
python3 src/ddi_evaluation.py analyze words_dditestmedline_results wordnet_dditestmedline_results full_model_dditestmedline_results
