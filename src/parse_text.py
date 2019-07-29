from itertools import combinations
import numpy as np
import spacy
import sys
from spacy.tokenizer import Tokenizer
import re
from subprocess import PIPE, Popen
import os
import logging
import networkx as nx
import en_core_web_sm
import string
from neg_gv import neg_gv_list


SSTDIR = "sst-light-0.4/"
TEMP_DIR = "temp/"


def prevent_sentence_segmentation(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc


nlp = en_core_web_sm.load(disable=["ner"])
nlp.add_pipe(prevent_sentence_segmentation, name="prevent-sbd", before="parser")


# https://stackoverflow.com/a/41817795/3605086
def get_network_graph_spacy(document):
    """
    Convert the dependencies of the spacy document object to a networkX graph
    :param document: spacy parsed document object
    :return: networkX graph object and nodes list
    """

    edges = []
    nodes = []
    # ensure that every token is connected
    # edges.append(("ROOT", '{0}-{1}'.format(list(document)[0].lower_, list(document)[0].i)))
    for s in document.sents:
        edges.append(("ROOT", "{0}-{1}".format(s.root.lower_, s.root.i)))
    for token in document:
        nodes.append("{0}-{1}".format(token.lower_, token.i))
        # edges.append(("ROOT", '{0}-{1}'.format(token.lower_, token.i)))
        # print('{0}-{1}'.format(token.lower_, token.i))
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            # print("----", '{0}-{1}'.format(child.lower_, child.i))
            edges.append(
                (
                    "{0}-{1}".format(token.lower_, token.i),
                    "{0}-{1}".format(child.lower_, child.i),
                )
            )
    return nx.Graph(edges), nodes


def get_head_tokens(entities, sentence):
    """
    :param entities: dictionary mapping entity IDs to (offset, text)
    :param sentence: sentence parsed by spacy
    :return: dictionary mapping head tokens word-idx to entity IDs
    """
    sentence_head_tokens = {}
    for eid in entities:
        offset = (entities[eid][0][0], entities[eid][0][-1])
        # starts = {tok.i: tok.idx for tok in doc}
        # entity_tokens = sentence.char_span(offset[0], offset[1])
        entity_tokens = [
            (t, i) for i, t in enumerate(sentence.token) if t.beginChar == offset[0]
        ]
        # if not entity_tokens:
        # try to include the next char
        # entity_tokens = sentence.char_span(offset[0], offset[1] + 1)
        #    entity_tokens = [t for t in sentence.token if t.beginChar == offset[0]]

        if not entity_tokens:
            logging.warning(
                (
                    "no tokens found:",
                    entities[eid],
                    sentence.text,
                    "|".join(
                        [
                            "{}({}-{})".format(t.word, t.beginChar, t.endChar)
                            for t in sentence.token
                        ]
                    ),
                )
            )
            # sys.exit()
        else:
            head_token = "{0}-{1}".format(
                entity_tokens[0][0].word.lower(), entity_tokens[0][1]
            )
            if head_token in sentence_head_tokens:
                logging.warning(
                    (
                        "head token conflict:",
                        sentence_head_tokens[head_token],
                        entities[eid],
                    )
                )
            sentence_head_tokens[head_token] = eid
    return sentence_head_tokens


def get_head_tokens_spacy(entities, sentence, positive_entities):
    """
    :param entities: dictionary mapping entity IDs to (offset, text)
    :param sentence: sentence parsed by spacy
    :return: dictionary mapping head tokens word-idx to entity IDs
    """
    sentence_head_tokens = {}
    pos_gv = set()
    neg_gv = set()
    for eid in entities:
        offset = (entities[eid][0][0], entities[eid][0][-1])
        # starts = {tok.i: tok.idx for tok in doc}
        entity_tokens = sentence.char_span(offset[0], offset[1])
        # if not entity_tokens:
        # try to include the next char
        #    entity_tokens = sentence.char_span(offset[0], offset[1] + 1)

        i = 1
        while not entity_tokens and i + offset[1] < len(sentence.text) + 1:
            entity_tokens = sentence.char_span(offset[0], offset[1] + i)
            i += 1

        i = 0
        while not entity_tokens and offset[0] - i > 0:
            entity_tokens = sentence.char_span(offset[0] - i, offset[1])
            i += 1

        if not entity_tokens:
            logging.warning(
                (
                    "no tokens found:",
                    entities[eid],
                    sentence.text,
                    "|".join([t.text for t in sentence]),
                )
            )
        else:
            head_token = "{0}-{1}".format(
                entity_tokens.root.lower_, entity_tokens.root.i
            )
            if eid in positive_entities:
                pos_gv.add(entity_tokens.root.head.lower_)
            else:
                neg_gv.add(entity_tokens.root.head.lower_)
            if head_token in sentence_head_tokens:
                logging.warning(
                    (
                        "head token conflict:",
                        sentence_head_tokens[head_token],
                        entities[eid],
                    )
                )
            sentence_head_tokens[head_token] = eid
    return sentence_head_tokens, pos_gv, neg_gv


def run_sst(token_seq):
    chunk_size = 500
    wordnet_tags = {}
    sent_ids = list(token_seq.keys())
    chunks = [sent_ids[i : i + chunk_size] for i in range(0, len(sent_ids), chunk_size)]
    for i, chunk in enumerate(chunks):
        sentence_file = open("{}/sentences_{}.txt".format(TEMP_DIR, i), "w")
        for sent in chunk:
            sentence_file.write("{}\t{}\t.\n".format(sent, "\t".join(token_seq[sent])))
        sentence_file.close()
        sst_args = [
            "sst",
            "bitag",
            "{}/MODELS/WSJPOSc_base_20".format(SSTDIR),
            "{}/DATA/WSJPOSc.TAGSET".format(SSTDIR),
            "{}/MODELS/SEM07_base_12".format(SSTDIR),
            "{}/DATA/WNSS_07.TAGSET".format(SSTDIR),
            "{}/sentences_{}.txt".format(TEMP_DIR, i),
            "0",
            "0",
        ]
        p = Popen(sst_args, stdout=PIPE)
        p.communicate()
        with open("{}/sentences_{}.txt.tags".format(TEMP_DIR, i)) as f:
            output = f.read()
        sstoutput = parse_sst_results(output)
        wordnet_tags.update(sstoutput)

    return wordnet_tags


def parse_sst_results(results):
    sentences = {}
    lines = results.strip().split("\n")
    for l in lines:
        values = l.split("\t")
        wntags = [x.split(" ")[-1].split("-")[-1] for x in values[1:]]
        sentences[values[0]] = wntags
    return sentences


def parse_sentence_spacy(sentence_text, sentence_entities):
    # use spacy to parse a sentence
    for e in sentence_entities:
        idx = sentence_entities[e][0]
        sentence_text = (
            sentence_text[: idx[0]]
            + sentence_text[idx[0] : idx[1]].replace(" ", "_")
            + sentence_text[idx[1] :]
        )

    # clean text to make tokenization easier
    sentence_text = sentence_text.replace(";", ",")
    sentence_text = sentence_text.replace("*", " ")
    sentence_text = sentence_text.replace(":", ",")
    sentence_text = sentence_text.replace(" - ", " ; ")
    parsed = nlp(sentence_text)

    return parsed


def process_sentence_spacy(
    sentence,
    sentence_entities,
    sentence_pairs,
    positive_entities,
    wordnet_tags=None,
    mask_entities=True,
    min_sdp_len=0,
    max_sdp_len=15,
):
    """
    Process sentence to obtain labels, instances and classes for a ML classifier
    :param sentence: sentence processed by spacy
    :param sentence_entities: dictionary mapping entity ID to ((e_start, e_end), text, paths_to_root)
    :param sentence_pairs: dictionary mapping pairs of known entities in this sentence to pair types
    :return: labels of each pair (according to sentence_entities,
            word vectors and classes (pair types according to sentence_pairs)
    """

    left_word_vectors = []
    right_word_vectors = []
    left_wordnets = []
    right_wordnets = []
    classes = []
    labels = []

    graph, nodes_list = get_network_graph_spacy(sentence)
    sentence_head_tokens, pos_gv, neg_gv = get_head_tokens_spacy(
        sentence_entities, sentence, positive_entities
    )
    # print(neg_gv - pos_gv)
    entity_offsets = [sentence_entities[x][0][0] for x in sentence_entities]
    # print(sentence_head_tokens)
    for (e1, e2) in combinations(sentence_head_tokens, 2):
        # print()
        # print(sentence_head_tokens[e1], e1, sentence_head_tokens[e2], e2)
        # reorder according to entity ID
        if int(sentence_head_tokens[e1].split("e")[-1]) > int(
            sentence_head_tokens[e2].split("e")[-1]
        ):
            e1, e2 = e2, e1

        e1_text = sentence_entities[sentence_head_tokens[e1]]
        e2_text = sentence_entities[sentence_head_tokens[e2]]

        if e1_text[1].lower() == e2_text[1].lower():
            # logging.debug("skipped same text: {} {}".format(e1_text, e2_text))
            continue

        middle_text = sentence.text[e1_text[0][-1] : e2_text[0][0]]

        # if middle_text.strip() == "or" or middle_text.strip() == "and":
        # logging.debug("skipped entity list: {} {} {}".format(e1_text, middle_text, e2_text))
        #    continue

        if middle_text.strip() in string.punctuation:
            #    logging.debug("skipped punctuation: {} {} {}".format(e1_text, middle_text, e2_text))
            continue

        # if len(middle_text) < 3:
        #    logging.debug("skipped entity list: {} {} {}".format(e1_text, middle_text, e2_text))
        #    continue

        head_token1_idx = int(e1.split("-")[-1])
        head_token2_idx = int(e2.split("-")[-1])
        try:
            sdp = nx.shortest_path(graph, source=e1, target=e2)

            if len(sdp) < min_sdp_len or len(sdp) > max_sdp_len:
                # logging.debug("skipped short sdp: {} {} {}".format(e1_text, str(sdp), e2_text))
                continue

            neg = False
            is_neg_gv = False
            for i, element in enumerate(sdp):
                token_idx = int(element.split("-")[-1])  # get the index of the token
                token_text = element.split("-")[0]
                if (i == 1 or i == len(sdp) - 2) and token_text in neg_gv_list:
                    logging.info("skipped gv {} {}:".format(token_text, str(sdp)))
                    # is_neg_gv = True
                sdp_token = sentence[token_idx]  # get the token obj
                # if any(c.dep_ == 'neg' for c in sdp_token.children):
                #    neg = True

            if neg or is_neg_gv:
                continue

            # if len(sdp) < 3: # len=2, just entities
            #    sdp = [sdp[0]] + nodes_list[head_token1_idx-2:head_token1_idx]
            #    sdp += nodes_list[head_token2_idx+1:head_token2_idx+3] + [sdp[-1]]
            # print(e1_text[1:], e2_text[1:], sdp)
            # if len(sdp) == 2:
            # add context words
            vector = []
            wordnet_vector = []
            negations = 0
            head_token_position = None
            for i, element in enumerate(sdp):
                if element != "ROOT":
                    token_idx = int(
                        element.split("-")[-1]
                    )  # get the index of the token
                    sdp_token = sentence[token_idx]  # get the token obj

                    # if any(c.dep_ == 'neg' for c in sdp_token.children):
                    # token is negated!
                    #    vector.append("not")
                    #    negations += 1
                    # logging.info("negated!: {}<->{} {}: {}".format(e1_text,  e2_text, sdp_token.text, sentence.text))

                    if mask_entities and sdp_token.idx in entity_offsets:
                        vector.append("drug")
                    else:
                        vector.append(sdp_token.text)
                    if wordnet_tags:
                        wordnet_vector.append(wordnet_tags[token_idx])
                    # print(element, sdp_token.text, head_token, sdp)
                    head_token = "{}-{}".format(
                        sdp_token.head.lower_, sdp_token.head.i
                    )  # get the key of head token
                    # head token must not have its head in the path, otherwise that would be the head token
                    # in some cases the token is its own head
                    if head_token not in sdp or head_token == element:
                        # print("found head token of:", e1_text, e2_text, sdp_token.text, sdp)
                        head_token_position = i + negations
                    # vector.append(parsed[token_idx].text)
            # print(vector)
            if head_token_position is None:
                print("head token not found:", e1_text, e2_text, sdp)
                sys.exit()
            else:
                left_vector = vector[: head_token_position + 1]
                right_vector = vector[head_token_position:]
                left_wordnet = wordnet_vector[: head_token_position + 1]
                right_wordnet = wordnet_vector[head_token_position:]
            # word_vectors.append(vector)
            left_word_vectors.append(left_vector)
            right_word_vectors.append(right_vector)
            left_wordnets.append(left_wordnet)
            right_wordnets.append(right_wordnet)
            # if (sentence_head_tokens[e1], sentence_head_tokens[e2]) in sentence_pairs:
            #    print(sdp, e1, e2, sentence_text)
            # print(e1_text, e2_text, sdp, sentence_text)
            # instances.append(sdp)
        except nx.exception.NetworkXNoPath:
            # pass
            logging.warning("no path:", e1_text, e2_text, graph.nodes())
            left_word_vectors.append([])
            right_word_vectors.append([])
            left_wordnets.append([])
            right_wordnets.append([])
            # print("no path:", e1_text, e2_text, sentence_text, parsed.print_tree(light=True))
            # sys.exit()
        except nx.NodeNotFound:
            logging.warning(
                (
                    "node not found:",
                    e1_text,
                    e2_text,
                    e1,
                    e2,
                    list(sentence),
                    graph.nodes(),
                )
            )
            left_word_vectors.append([])
            right_word_vectors.append([])
            left_wordnets.append([])
            right_wordnets.append([])

        labels.append((sentence_head_tokens[e1], sentence_head_tokens[e2]))
        # print(sentence_head_tokens[e1], sentence_head_tokens[e2])
        if (sentence_head_tokens[e1], sentence_head_tokens[e2]) in sentence_pairs:
            classes.append(
                sentence_pairs[(sentence_head_tokens[e1], sentence_head_tokens[e2])]
            )
        else:
            classes.append(0)
    return (
        labels,
        (left_word_vectors, right_word_vectors),
        (left_wordnets, right_wordnets),
        classes,
        pos_gv,
        neg_gv,
    )
