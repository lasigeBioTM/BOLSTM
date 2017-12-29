from itertools import combinations
import numpy as np
import spacy
import sys
from spacy.tokenizer import Tokenizer
import re
import logging
import networkx as nx


def prevent_sentence_segmentation(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc


#nlp = spacy.load('en_core_web_lg', disable=['ner'])
#tokenizer = Tokenizer(nlp.vocab)

#def create_tokenizer(nlp):
#    prefix_re = spacy.util.compile_suffix_regex(nlp.Defaults.prefixes)
#    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
    # infix_re = spacy.util.compile_infix_regex((r'''(?<=[\w*])(;)(?=[\w*])''',))
#    infix_re = spacy.util.compile_suffix_regex(nlp.Defaults.infixes + (r'''(?<=[A-Za-z])[;](?=[A-Za-z])''', ';'))

#    return Tokenizer(nlp.vocab,
#            rules={},
#            prefix_search=prefix_re.search,
#            suffix_search=suffix_re.search,
#            infix_finditer=infix_re.finditer
#            )
#nlp = spacy.load('en_core_web_lg', disable=['ner'], create_make_doc=create_tokenizer)


# https://stackoverflow.com/a/41817795/3605086
def get_network_graph(document):
    """
    Convert the dependencies of the spacy document object to a networkX graph
    :param document: spacy parsed document object
    :return: networkX graph object
    """

    edges = []
    # ensure that every token is connected
    #edges.append(("ROOT", '{0}-{1}'.format(list(document)[0].lower_, list(document)[0].i)))
    for s in document.sents:
        edges.append(("ROOT", '{0}-{1}'.format(s.root.lower_, s.root.i)))
    for token in document:
        #edges.append(("ROOT", '{0}-{1}'.format(token.lower_, token.i)))
        # print('{0}-{1}'.format(token.lower_, token.i))
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            #print("----", '{0}-{1}'.format(child.lower_, child.i))
            edges.append(('{0}-{1}'.format(token.lower_, token.i),
                          '{0}-{1}'.format(child.lower_, child.i)))
    return nx.Graph(edges)


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
        entity_tokens = sentence.char_span(offset[0], offset[1])
        if not entity_tokens:
            # try to include the next char
            entity_tokens = sentence.char_span(offset[0], offset[1] + 1)

        if not entity_tokens:
            logging.warning(("no tokens found:", entities[eid], sentence.text, '|'.join([t.text for t in sentence])))
        else:
            head_token = '{0}-{1}'.format(entity_tokens.root.lower_,
                                          entity_tokens.root.i)
            if head_token in sentence_head_tokens:
                logging.warning(("head token conflict:", sentence_head_tokens[head_token], entities[eid]))
            sentence_head_tokens[head_token] = eid
    return sentence_head_tokens


def process_sentence(sentence_text, sentence_entities, sentence_pairs):
    """
    Process sentence to obtain labels, instances and classes for a ML classifier
    :param sentence_text: sentence text string
    :param sentence_entities: dictionary mapping entity ID to ((e_start, e_end), text, paths_to_root)
    :param sentence_pairs: dictionary mapping pairs of known entities in this sentence to pair types
    :return: labels of each pair (according to sentence_entities,
            word vectors and classes (pair types according to sentence_pairs)
    """
    nlp = spacy.load('en_core_web_lg', disable=['ner'])
    nlp.add_pipe(prevent_sentence_segmentation, name='prevent-sbd', before='parser')
    left_word_vectors = []
    right_word_vectors = []
    classes = []
    labels = []

    # clean text to make tokenization easier
    sentence_text = sentence_text.replace(";", ",")
    sentence_text = sentence_text.replace("*", " ")
    sentence_text = sentence_text.replace(":", ",")


    #print(sentence_entities)
    parsed = nlp(sentence_text)
    graph = get_network_graph(parsed)
    sentence_head_tokens = get_head_tokens(sentence_entities, parsed)

    for (e1, e2) in combinations(sentence_head_tokens, 2):
        labels.append((sentence_head_tokens[e1], sentence_head_tokens[e2]))
        if (sentence_head_tokens[e1], sentence_head_tokens[e2]) in sentence_pairs:
            classes.append(sentence_pairs[(sentence_head_tokens[e1], sentence_head_tokens[e2])])
        else:
            classes.append(0)
        e1_text = sentence_entities[sentence_head_tokens[e1]]
        e2_text = sentence_entities[sentence_head_tokens[e2]]
        try:
            sdp = nx.shortest_path(graph, source=e1, target=e2)
            vector = []
            left_vector = []
            right_vector = []
            head_token_position = None
            for i, element in enumerate(sdp):
                if element != "ROOT":
                    token_idx = int(element.split("-")[-1]) # get the index of the token
                    sdp_token = parsed[token_idx] #get the token obj
                    head_token = "{}-{}".format(sdp_token.head.lower_, sdp_token.head.i) # get the key of head token
                    #vector.append(sdp_token.vector)
                    vector.append(sdp_token.text)
                    #print(element, sdp_token.text, head_token, sdp)
                    # head token must not have its head in the path, otherwise that would be the head token
                    # in some cases the token is its own head
                    if head_token not in sdp or head_token == element:
                        #print("found head token of:", e1_text, e2_text, sdp_token.text, sdp)
                        head_token_position = i
                    #vector.append(parsed[token_idx].text)
            # print(vector)
            if head_token_position is None:
                print("head token not found:", e1_text, e2_text, sdp)
                sys.exit()
            else:
                left_vector = vector[:head_token_position+1]
                right_vector = vector[head_token_position:]
            #word_vectors.append(vector)
            left_word_vectors.append(left_vector)
            right_word_vectors.append(right_vector)
            # if (sentence_head_tokens[e1], sentence_head_tokens[e2]) in sentence_pairs:
            #    print(sdp, e1, e2, sentence_text)
            # print(e1_text, e2_text, sdp, sentence_text)
            # instances.append(sdp)
        except nx.exception.NetworkXNoPath:
            # pass
            logging.warning("no path:", e1_text, e2_text, graph.nodes())
            left_word_vectors.append([])
            right_word_vectors.append([])
            # print("no path:", e1_text, e2_text, sentence_text, parsed.print_tree(light=True))
            # sys.exit()
        except nx.NodeNotFound:
            logging.warning(("node not found:", e1_text, e2_text, e1, e2, list(parsed), graph.nodes()))
            left_word_vectors.append([])
            right_word_vectors.append([])
    return labels, (left_word_vectors, right_word_vectors), classes



