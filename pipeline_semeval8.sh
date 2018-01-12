#!/usr/bin/env bash
set -x
set -e
#python src/train_rnn.py preprocessing semeval8 temp/semeval8train data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT
#python src/train_rnn.py preprocessing semeval8 temp/semeval8test data/SemEval2010_task8_all_data/SemEval2010_task8_testing/TEST_FILE.txt
python src/train_rnn.py train temp/semeval8train
python src/train_rnn.py predict temp/semeval8test
python src/semeval8_evaluation.py semeval8test_results.txt
perl data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl semeval8test_results.txt.tsv semeval8_goldstandard.txt
