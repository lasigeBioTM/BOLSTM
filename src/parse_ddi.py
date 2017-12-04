import xml.etree.ElementTree as ET
import sys
import os
import logging

from parse_text import process_sentence

NORELATION = 0
MECHANISM = 1
EFFECT = 2
ADVICE = 3
INTTYPE = 4

pairtypes = (MECHANISM, EFFECT, ADVICE, INTTYPE)
label_to_pairtype = {"effect": EFFECT, "mechanism": MECHANISM, "advise": ADVICE, "int": INTTYPE}

def get_ddi_sdp_instances(base_dir):
    """
    Parse DDI corpus, return vectors of SDP of each relation instance
    :param base_dir: directory containing semeval XML documents and annotations
    :return: instances (vectors), classes (0/1) and labels (eid1, eid2)
    """
    instances = []
    classes = []
    labels = []

    for f in os.listdir(base_dir):
        logging.info(f)
        #if f != "L-Glutamine_ddi.xml":
        #    continue
        tree = ET.parse(base_dir + f)
        root = tree.getroot()
        for sentence in root:
            sentence_entities = {e.get("id"): (e.get("charOffset"), e.get("text")) for e in sentence.findall('entity')}
            sentence_pairs = {(p.get("e1"), p.get("e2")): label_to_pairtype[p.get("type")] for p in sentence.findall('pair') if p.get("ddi") == "true"}
            # sentence_pairs: {(e1id, e2id): pairtype_label}

            sentence_labels, sentence_instances,sentence_classes = process_sentence(sentence.get("text"),
                                                                                  sentence_entities,
                                                                                  sentence_pairs)
            labels += sentence_labels
            instances += sentence_instances
            classes += sentence_classes
    return labels, instances, classes

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
