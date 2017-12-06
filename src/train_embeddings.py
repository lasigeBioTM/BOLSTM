# https://rare-technologies.com/word2vec-tutorial/

import logging
import os
import json
import xml.etree.ElementTree as ET
import spacy
import numpy as np
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nlp = spacy.load('en_core_web_lg', disable=['ner'])

def load_vocab(vocab_path='map.json'):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


# load sentences using spacy
class DDICorpusSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        tree = ET.parse(self.dirname)
        root = tree.getroot()
        for drug in root:
            # use just the description for now
            #for line in open(os.path.join(self.dirname, fname)):
            #for x in drug:
            #    print(x)
            #print(drug.find("{http://www.drugbank.ca}description"))
            # print(drug.find("{http://www.drugbank.ca}name").text)
            sentence_text = drug.find("{http://www.drugbank.ca}description").text
            #print(sentence_text)
            if sentence_text:
                sentence_text = sentence_text.replace(";", ",")
                sentence_text = sentence_text.replace("*", " ")
                sentence_text = sentence_text.replace(":", ",")
                parsed_sentence = nlp(sentence_text)
                yield [t.text for t in parsed_sentence]


def generate_embeddings(files_dir, embeddings_path, vocab_path):
    sentences = DDICorpusSentences(files_dir)  # a memory-friendly iterator

    model = gensim.models.Word2Vec(iter=1, min_count=5, size=200, workers=4) # an empty model, no training yet
    model.build_vocab(sentences)  # can be a non-repeatable, 1-pass generator
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)   # can be a non-repeatable, 1-pass generator
    weights = model.wv.syn0
    np.save(open(embeddings_path, 'wb'), weights)

    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))

def main():
    generate_embeddings("data/drugbank/full_database.xml", "embeddings/words.npz", "embeddings/vocab.map.json")

if __name__ == "__main__":
    main()