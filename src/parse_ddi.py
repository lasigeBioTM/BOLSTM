import xml.etree.ElementTree as ET
import sys
import os
import logging
from subprocess import PIPE, Popen
from parse_text import process_sentence, parse_sentence
from chebi_path import get_lowest_common_ascestor_path, load_chebi, get_all_shortest_paths_to_root, map_to_chebi, get_common_ancestors



NORELATION = 0
MECHANISM = 1
EFFECT = 2
ADVICE = 3
INTTYPE = 4

pairtypes = (MECHANISM, EFFECT, ADVICE, INTTYPE)
label_to_pairtype = {"effect": EFFECT, "mechanism": MECHANISM, "advise": ADVICE, "int": INTTYPE}


def get_ancestors(sentence_labels, sentence_entities, name_to_id, synonym_to_id, id_to_name):
    """
    obtain the path to lowest common ancestor of each entity of each pair
    :param sentence_labels: list of (e1, e2)
    :param sentence_entities: dictionary mapping entity ID to ((e_start, e_end), text, paths_to_root)
    :return: left and right paths to LCA
    """
    right_paths = []
    left_paths = []
    common_ancestors = []
    for p in sentence_labels:
        #paths1 = sentence_entities[p[0]][2]
        #paths2 = sentence_entities[p[1]][2]
        #lca, left_path, right_path = get_lowest_common_ascestor_path(paths1, paths2, id_to_name)
        #left_paths.append(left_path + [lca])
        #right_paths.append(right_path + [lca])
        #print(p[0], sentence_entities[p[0]])
        instance_ancestors = get_common_ancestors(sentence_entities[p[0]][2], sentence_entities[p[1]][2])
        # print("common ancestors:", sentence_entities[p[0]][1:], sentence_entities[p[1]][1:], instance_ancestors)
        common_ancestors.append(instance_ancestors)
    #return (left_paths, right_paths)
    return (common_ancestors, common_ancestors)

def get_sentence_entities(base_dir, name_to_id, synonym_to_id):
    entities = {} # sentence_id -> entities
    for f in os.listdir(base_dir):
        logging.info("processing entities: {}".format(f))
        #if f != "L-Glutamine_ddi.xml":
        #    continue
        tree = ET.parse(base_dir + f)
        root = tree.getroot()
        for sentence in root:
            sentence_id = sentence.get("id")
            sentence_entities = {}
            if len(sentence.findall('pair')) > 0: # skip sentences without pairs
                for e in sentence.findall('entity'):
                    sep_offsets = e.get("charOffset").split(";")
                    offsets = []
                    for o in sep_offsets:
                        start, end = o.split("-")
                        offsets.append(int(start))
                        offsets.append(int(end)+1)
                    e_text = e.get("text")
                    chebi_name = map_to_chebi(e_text, name_to_id, synonym_to_id)
                    if chebi_name in name_to_id:
                        chebi_id = name_to_id[chebi_name]
                    else:
                        chebi_id = synonym_to_id[chebi_name][0]
                    #chebi_name = ""
                    #e_path = get_all_shortest_paths_to_root(chebi_name, is_a_graph, name_to_id, synonym_to_id, id_to_name)
                    #e_path = []
                    #sentence_entities[e.get("id")] = (offsets, e_text, e_path)
                    sentence_entities[e.get("id")] = (offsets, e_text, chebi_id)
                entities[sentence_id] = sentence_entities
    return entities

def parse_ddi_sentences(base_dir, entities):

    parsed_sentences = {}
    # first iterate all documents, and preprocess all sentences
    token_seq = {}
    for f in os.listdir(base_dir):
        logging.info("parsing {}".format(f))
        tree = ET.parse(base_dir + f)
        root = tree.getroot()
        for sentence in root:
            sentence_id = sentence.get("id")
            if len(sentence.findall('pair')) > 0:  # skip sentences without pairs
                parsed_sentence = parse_sentence(sentence.get("text"), entities[sentence_id])
                parsed_sentences[sentence_id] = parsed_sentence
                tokens = []
                for t in parsed_sentence:
                    tokens.append(t.text.replace(" ", "_").replace('\t', '_').replace('\n', '_'))
                #sentence_file.write("{}\t{}\t.\n".format(sentence_id, "\t".join(tokens)))
                token_seq[sentence_id] = tokens

    chunk_size = 500
    wordnet_tags = {}
    sent_ids = list(token_seq.keys())
    chunks = [sent_ids[i:i + chunk_size] for i in range(0, len(sent_ids), chunk_size)]
    for i, chunk in enumerate(chunks):
        sentence_file = open("temp/sentences_{}.txt".format(i), 'w')
        for sent in chunk:
            sentence_file.write("{}\t{}\t.\n".format(sent, "\t".join(token_seq[sent])))
        sentence_file.close()
        os.chdir("sst-light-0.4/")
        sst_args = ["./sst", "bitag",
                    "./MODELS/WSJPOSc_base_20", "./DATA/WSJPOSc.TAGSET",
                    "./MODELS/SEM07_base_12", "./DATA/WNSS_07.TAGSET",
                    "../temp/sentences_{}.txt".format(i), "0", "0"]
        p = Popen(sst_args, stdout=PIPE)
        p.communicate()
        os.chdir("..")
        with open("temp/sentences_{}.txt.tags".format(i)) as f:
            output = f.read()
        sstoutput = parse_sst_results(output)
        wordnet_tags.update(sstoutput)

    return parsed_sentences, wordnet_tags

def parse_sst_results(results):
    sentences = {}
    lines = results.strip().split("\n")
    for l in lines:
        values = l.split("\t")
        wntags = [x.split(" ")[-1].split("-")[-1] for x in values[1:]]
        sentences[values[0]] = wntags
        if values[0].startswith("DDI-MedLine.d185"):
            print(values[0], wntags)
    return sentences

def get_ddi_sdp_instances(base_dir):
    """
    Parse DDI corpus, return vectors of SDP of each relation instance
    :param base_dir: directory containing semeval XML documents and annotations
    :return: instances (vectors), classes (0/1) and labels (eid1, eid2)
    """
    is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()
    entities = get_sentence_entities(base_dir, name_to_id, synonym_to_id)

    parsed_sentences, wordnet_sentences = parse_ddi_sentences(base_dir, entities)
    #print(wordnet_sentences.keys())
    # print(sstoutput)
    left_instances = []
    right_instances = []
    left_ancestors = []
    right_ancestors = []
    left_wordnet = []
    right_wordnet = []
    classes = []
    labels = []
    for f in os.listdir(base_dir):
        logging.info("generating instances: {}".format(f))
        #if f != "L-Glutamine_ddi.xml":
        #    continue
        tree = ET.parse(base_dir + f)
        root = tree.getroot()
        for sentence in root:
            sentence_id = sentence.get("id")
            sentence_pairs = {(p.get("e1"), p.get("e2")): label_to_pairtype[p.get("type")] for p in
                              sentence.findall('pair') if p.get("ddi") == "true"}
            if len(sentence.findall('pair')) > 0: # skip sentences without pairs
                sentence_entities = entities[sentence_id]
                parsed_sentence = parsed_sentences[sentence_id]
                wordnet_sentence = wordnet_sentences[sentence_id]
                # sentence_pairs: {(e1id, e2id): pairtype_label}

                sentence_labels, sentence_we_instances,\
                sentence_wn_instances, sentence_classes = process_sentence(parsed_sentence,
                                                                           sentence_entities,
                                                                           sentence_pairs,
                                                                           wordnet_sentence)


                sentence_ancestors = get_ancestors(sentence_labels, sentence_entities,
                                                   name_to_id, synonym_to_id, id_to_name)




                labels += sentence_labels
                left_instances += sentence_we_instances[0]
                right_instances += sentence_we_instances[1]
                classes += sentence_classes
                left_ancestors += sentence_ancestors[0]
                right_ancestors += sentence_ancestors[1]

                left_wordnet += sentence_wn_instances[0]
                right_wordnet += sentence_wn_instances[1]
    return labels, (left_instances, right_instances), classes,\
           (left_ancestors, right_ancestors), (left_wordnet, right_wordnet)



# print("Drugbank")
# labels, instances, classes = get_ddi_sdp_instances("data/ddi2013Train/DrugBank/")
# print(len(labels), len(instances), len(classes))
# # print(labels[:10], instances[:10], classes[:10])
# print("average path size", sum([len(a) for a in instances])/len(instances))
# print("# of true relations", classes.count(1))
# print("# of total relations", len(classes))
#
# print("Medline")
# labels, instances, classes = get_ddi_sdp_instances("data/ddi2013Train/MedLine/")
# print(len(labels), len(instances), len(classes))
# # print(labels[:10], instances[:10], classes[:10])
# print("average path size", sum([len(a) for a in instances])/len(instances))
# print("# of true relations", classes.count(1))
# print("# of total relations", len(classes))
