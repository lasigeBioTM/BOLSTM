# BO-LSTM: Classifying Relations via Long Short-Term Memory Networks along Biomedical Ontologies

## Getting Started
Code necessary to run the BO-LSTM model. Use the Dockerfile or the [docker hub image](https://hub.docker.com/r/andrelamurias/bolstm/) setup the experimental environment.

Use *prepare_ddi.sh* to generate training instances and *pipeline_ddi.sh* to run the experiments.

Use *train.sh* to train a BO-LSTM classification model and *predict.sh* to run the model on new data.

## Preparing data
*python3 src/train_rnn.py preprocessing ddi temp/dditrain data/DDICorpus/Train/MedLine/ data/DDICorpus/Train/DrugBank/*

This command pre-processes the documents in "data/DDICorpus/Train/MedLine/" and "data/DDICorpus/Train/DrugBank/", which should be in the "ddi" format and saves the data generated to temp/dditrain.
Before training or making predictions on new documents, it is necessary to pre-process the input data.
The objective of this phase is to convert text data into numeric vectors, which we refer to as training data.
The script produces three types of inputs for each pair of entities (candidate relation):
* Word indices of the shortest dependency path
* Wordnet classes of the words in the shortest dependency path
* Ancestors of the each entity according to the reference ontology

Furthermore, the word and wordnet indices are separated into right and left side.


## Train model
![BOLSTM architecture](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6323831/bin/12859_2018_2584_Fig2_HTML.jpg)

*python3 src/train_rnn.py train temp/dditrain full_model words wordnet common_ancestors concat_ancestors*

This command trains a BOLSTM model named "full_model" based on the data from "temp/dditrain", using the following input channels: "words wordnet common_ancestors concat_ancestors".
The architecture and hyperparameters of the model are defined in models.py.
The model is saved to "models/full_model" and can be used to classify new documents afterwards.

## Predict new data
*python3 src/train_rnn.py predict temp/dditest full_model words wordnet common_ancestors concat_ancestors*

This command uses model "full_model" to predict annotate the documents of "temp/dditest".
The input channels should be specified again. 
The output is a file with one line per entity pair and its predicted label.
