#open ontology

import logging
from itertools import combinations
import sys
import os
import pickle
import atexit

import numpy as np
import obonet
import networkx
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import spacy
import en_core_web_md
nlp = en_core_web_md.load(disable=['ner'])



#from DiShIn import ssm
#ssm.semantic_base("/src/DiShIn/chebi.db")

global chebi_cache
global paths_cache
global chemical_entity
global role
global subatomic_particle
global application
global multiple_match_count
global no_match_count

chebi_cache_file = "../temp/chebi_cache.pickle"

# store string-> chebi ID
if os.path.isfile(chebi_cache_file):
    logging.info("loading chebi...")
    chebi_cache = pickle.load(open(chebi_cache_file, "rb"))
    loadedchebi = True
    logging.info("loaded chebi dictionary with %s entries", str(len(chebi_cache)))
else:
    chebi_cache = {}
    loadedchebi = False
    logging.info("new chebi dictionary")


def exit_handler():
    print('Saving chebi dictionary...!')
    pickle.dump(chebi_cache, open(chebi_cache_file, "wb"))

atexit.register(exit_handler)

# parse chebi ontology
paths_cache = {} # store chebi ID->paths
chemical_entity = "CHEBI:24431"
role = "CHEBI:50906"
subatomic_particle = "CHEBI:36342"
application = "CHEBI:33232"
root_concept = "CHEBI:00000"
# TODO: create ROOT concept


#dis_vec_table: 601:10
#pos_vec_tanle: 45:10
#dic_vec_table 4k:200
pos_vecs= []
with open("../data/pos_indexes.pkl", "rb") as f:
    pos_index = pickle.load(f)
for w in pos_index:
    #print(w, nlp.vocab.get_vector(w))
    pos_vecs.append(np.zeros(10))

onto_vecs= []
with open("../data/onto_indexes.pkl", "rb") as f:
    onto_index = pickle.load(f)
for w in onto_index:
    #print(w, nlp.vocab.get_vector(w))
    onto_vecs.append(np.zeros(300))

dis_vecs= []
for w in range(601):
    #print(w, nlp.vocab.get_vector(w))
    dis_vecs.append(np.zeros(10))

with open("../data/word_indexes.pkl", "rb") as f:
    word_index = pickle.load(f)
word_vecs = []
for w in word_index:
    #print(w, nlp.vocab.get_vector(w))
    word_vecs.append(nlp.vocab.get_vector(w))

#order concepts
# create embadding matrix with size n
# associate ancestors to entity
f = open("vec.pkl", 'wb')
pickle.dump(np.array(word_vecs), f)
pickle.dump(np.array(pos_vecs), f)
pickle.dump(np.array(dis_vecs), f)
pickle.dump(np.array(onto_vecs), f)
f.close()
#f = open("vec2.pkl", 'wb')
#pickle.dump(np.array(onto_vecs), f)
#f.close()
