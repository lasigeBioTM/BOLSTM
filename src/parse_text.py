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
import en_core_web_lg
import corenlp
import string

neg_gv_list = {'cerubidine', 'trial', '5-fu', 'total', 'multivitamins', 'elemental', 'nitroprusside', 'chlortetracycline', 'transdermal', 'altered', 'promethazine', 'ml', 'fluoroquinolones', 'cephalothin_sodium', 'amiloride', 'tambocor', 'blocking_agents', 'immunosuppressives', 'weight', 'than', 'nabumetone', 'entacapone', 'fexofenadine', 'cytosine_arabinoside', 'drug', 'metaclopramide', 'divalproex_sodium', 'desloratadine', 'database', 'hydantoins', 'benazepril', 'amoxicillin', 'restricted', 'tendency', 'iron_supplements', 'azathioprine', 'exist', 'imidazole', 'half', 'anxiolytics', 'regimen', 'angiotensin-_converting_enzyme_(ace)_inhibitors', 'uroxatral', 'cefoperazone', 'other', 'wshow', 'andusing', '12', 'dobutamine', 'addiction', '500', 'potential', 'lead', 'eliminated', 'transferase', 'leflunomide', 'digitalis_preparations', 'stadol_ns', 'desbutyl_levobupivacaine', 'glibenclamide', 'vinblastine', 'aripiprazole', 'appear', 'oxidase', 'blunt', 'seriraline', 'bedtime', 'arimidex', 'dextromethorphan', 'lanoxin', 'cabergoline', 'oxacillin', 'naprosyn', 'users', 'iloprost', 'local', 'trifluoperazine', 'cefmenoxime', 'plaquenil', 'excess', 'chlorpromazine', 'misused', 'antibiotic', 'involving', 'stanozolol', 'antimycobacterial', 'zdv', 'antidiabetic_products', 'chlorothiazide', 'orlistat', 'bleomycin', 'latamoxef', 'somatostatin_analog', 'slows', 'alternatives', 'make', 'atenolol', 'corresponding', 'seen', 'l50', 'ribavirin', 'dynacirc', 'coumarin_derivatives', 'glyceryl_trinitrate', 'propofol', 'tacrine', 'mepron', 'excreted', 'examining', 'triflupromazine', 'iron_supplement', 'deltasone', 'amlodipine', 'nandrolone', 'antidiabetic_agents', 'antipsychotic_drugs', 'pa', 'containing_compounds', 'er', 'trimethoprim', 'glycoprotein', 'calcitriol', 'multiple', 'angiotensin_ii_receptor_antagonists', 'coa_reductase_inhibitor', 'nonsteroidal_antiinflammatories', 'infused', 'fluvastatin', 'reversible', 'mycophenolate', 'fell', 'vitamin_b3', 'maoi_antidepressants', '-treated', 'aeds', 'induction', 'hypoglycemic_agents', 'antifungal', 'salicylic_acid', 'gabapentin', 'fibrates', 'carvedilol', 'neuromuscular_blocking_agents', 'mesoridazine', 'require', 'fibrinogen', 'predispose', 'anakinra', 'somatostatin_analogs', 'magnesium_hydroxide_antacids', 'pregnancy', ';', 'therefore', 'antiarrhythmic_agents', 'surgery', 'conversion', 'monoamine_oxidase_inhibitor', 'serum', 'cardiac_glycosides', 'fosphenytoin', 'adrenergic_receptor_blockers', 'detected', 'grepafloxacin', 'systemic_corticosteroids', 'nucleoside_reverse_transcriptase_inhibitors', 'divalproex', 'thiocyanate', 'metrizamide', 'included', 'immunosuppressants', 'terbutaline', 'mycophenolate_mofetil', 'modify', 'blocker', 'valsartan', 'sulfoxone', 'distribution', 'famciclovir', 'minutes', 'chelating', 'immunosuppressive_drugs', 'accelerate', 'thrombolytic_agents', 'twice', 'promazine', 'bactrim', 'psychotropic_drugs', 'borne', 'novoseven', 'hivid', 'cromolyn_sodium', 'converting_enzyme_(ace)_inhibitors', 'cleared', 'transport', 'oruvail', 'experience', 'depletion', 'synkayvite', 'chlorthalidone', 'cyp1a2', 'produces', 'hypoglycemia', 'pegasys', 'diagnostic', 'mixing', 'oxc', 'hydroxyurea', 'and/or', 'requiring', 'mtx', 'lithium_carbonate', 'fibric_acid_derivatives', 'rifapentine', 'furafylline', 'dihydropyridine_calcium_antagonists', 'intensified', 'withdrawal', 'ameliorate', 'levonorgestrol', 'rofecoxib', 'ganglionic', 'anaprox', 'hiv_protease_inhibitors', 'studied', 'phenobarbitol', 'threohydrobupropion', 'antithyroid_drugs', 'alg', 'intoxication', 'anagrelide', 'assessed', 'nothiazines', 'terminated', 'coa_reductase_inhibitors', 'ticlopidine', 'cefazolin', 'cyp3a4', 'oxcarbazepine', 'hypokalemia', 'yielded', 'descarboethoxyloratadine', 'oxandrolone', 'leads', 'tranexamic_acid', 'dexmedetomidine', 'pancuronium', 'antacid', 'resorcinol', 'going', 'lenalidomide', 'influence', 'modified', 'pyrantel', 'droperidol', 'replacement', 'benzylpenicillin', 'acting_beta2-agonists', 'n=29', 'sequence', 'utilize', 'gram', 'interferences', 'nicotinic_acid', 'influenced', 'examples', 'min', 'salicylate', 'sulfur', 'keppra', 'iodoquinol', 'hours', 'trimeprazine', 'vitamin_d2', 'tolerated', 'procarbazine', 'volunteers', 'anions', 'increasing', 'etretinate', 'p450', 'nafcillin', 'cyp2c9', 'considered', 'prednisone', 'zofran', 'drawn', 'isradipine', 'lodosyn', 'substrates', 'orencia', 'debrisoquin', 'indicate', 'peginterferon', 'fortified', 'sulfisoxazole', 'tranylcypromine', 'antacid_products', 'antipsychotic_agents', 'antidiabetic_drugs', 'sucralfate', 'hemostasis', 'medrol', 'aminoglutethimide', 'clotrimazole', 'propanolol', 'monotherapy', 'irinotecan', 'identified', '/', 'somatrem', 'acetophenazine', 'gold', 'dirithromycin', 'sympathomimetics', 'erbitux', 'catalyzed', 'indanavir', 'ergonovine', 'lowered', 'infusion', 'combination', 'linezolid', 'substrate', 'differences', 'lowers', 'concomitant', 'nondepolarizing', 'meq', 'sparfloxacin', 'parameters', 'r', 'adjustments', 'prednisolone_sodium_succinate', 'nimodipine', 'tolerance', 'motrin', 'pill', 'sulfadoxine', 'mayuse', 'occurred', 'ci', 'flucoxacillin', 'metoclopramide', 'rifamipicin', 'responsive', 'cycles', 'trials', 'loop_diuretics', 'exhibits', 'folic_acid', 'ceftazidime', 'h2_antagonists', 'lansoprazole', 'escitalopram', 'methylprednisolone', 'antidepressant', 'accounts', 'vitamin_d3', 'gestodene', 'blocking_drugs', 'contribution', 'substances', 'tranylcypromine_sulfate', 'ritanovir', 'nizatidine', 'ingesting', 'buride', 'wthionamide', 'pravastatin', 'gleevec', 'index', 'tikosyn', 'cefotetan', 'antipsychotic_medications', 'aralen', 'performed', 'phenelzine', 'plicamycin', 'possibility', 'betablockers', 'isoenzymes', 'diphenylhydantoin', 'propatenone', 'eproxindine', 'alone', 'determined', 'evaluated', 'profiles', 'bioavailabillty', 'protamine', 'hyperreflexia', 'vitamin_a', 'vitamin_k_antagonists', 'medicine', 'cytokines', 'hydrocodone', 'vs.', 'methylthiotetrazole', 'tested', 'insert', 'antiacid', 'an', 'differ', 'invalidate', 'antiemetics', 'mellaril', 'dosed', 'range', 'bepridil', 'activated_prothrombin_complex_concentrates', 'inactivate', 'exercised', 'etomidate', 'vecuronium', 'coronary_vasodilators', 'dependent', 'anticholinesterases', 'prochlorperazine', 'r-', 'oxymetholone', 'aprepitant', 'ics', 'iressa', 'mephenytoin', 'ramipril', 'novum', 'medication', 'contains', 'diminished', 'activate', 'lam', 'sterilize', 'methandrostenolone', 'antipyrine', 'hydralazine', 'celecoxib', 'hydramine', 'exists', 'antipyretics', 'adenocard', 'besides', 'alpha-', 'cinacalcet', 'demonstrate', 'lomefloxacin', 'cephalothin', 'prolixin', 'concentrates', 'tests', 'analyses', 'proton_pump_inhibitors', 'mean', 'maintained', 'interferon', 'anticholinergic_agents', 'phenformin', 'failed', 'utilization', 'codeine', 'pediapred', 'isosorbide_dinitrate', 'oxaprozin', 'calcium_channel_antagonists', 'magnesium_sulfate', 'nonsteroidal_antiinflammatory_drugs', 'albuterol', 'prazosin', 'replacing', 'expanders', 'showed', 'hypercalcemia', 'benzothiadiazines', 'aza', 'humira', 'aminopyrine', 'cefamandole_naftate', '1/35', 'tolazoline', 'channel_blockers', 'thyroid_hormones', 'orudis', 'selegiline', 'analgesics', 'antagonists', 'ganglionic_blocking_agents', 'antagonism', 'pseudoephedrine', 'calcium_channel_blocking_drugs', 'oxide', 'chemotherapeutic_agents', 'cations', 'tend', 'undergo', 'includes', 'butazone', 'peak', 'sulfonamide', 'enzymes', '%', 'gabitril', 'acarbose', 'simvastatin', 'mixed', 'ethionamide', 'a', 'cyp2d6', 'ergot', 'metabolites', 'interrupted', 'carmustine', 'antianxiety_drugs', 'about', 'decarboxylase_inhibitor', 'hctz', 'advil', 'isosorbide_mononitrate', 'naltrexone', 'experienced', 'niacin', 'potassium_chloride', 'andtolbutamide', 'established', 'streptomycin', 'circulating', 'components', 'induces', 'dihydropyridine_derivative', 'caution', 'clonidine', 'piroxicam', 'phenylpropanolamine', 'label', 'indicated', 'pharmacokinetics', 'im', 'potassium_sparing_diuretics', 'adrenocorticoids', 'ocs', 'penicillin', 'conducted', 'desethylzaleplon', 'felbatol', 'nitrates', 'reviewed', 'smx', 'disease', 'cream', 'control', 'adefovir_dipivoxil', 'ethotoin', 'corticosteroid', 'voltaren', 'antivirals', 'protease_inhibitor', 'furazolidone', 'estrogen', 'investigated', 'mix', 'dapsone_hydroxylamine', 'cefamandole', 'mitotane', 'poisoning', 'metoprolol', 'dopa_decarboxylase_inhibitor', 'incombination', 'nisoldipine', 'diltiazem_hydrochloride', 'adjustment', 'tnf_blocking_agents', 'etodolac', 'phenelzine_sulfate', 'minus', 'formed', 'lower', 'show', 'cardiovasculars', 'sympathomimetic_bronchodilators', 'nitrofurantoin', 'calcium_channel_blocking_agents', 'oxymetazoline', 'neuroleptic', 'tetracyclic_antidepressants', 'steroid_medicine', 'arb', 'phenytoin_sodium', '5-dfur', 'bronchodilators', 'confirmed', 'among', 'sulphenazole', 'antiretroviral_nucleoside_analogues', 'binding', 'imatinib', 'cylates', 'plasmaconcentrations', 'acetohydroxamic_acid', 'inducing'}



os.environ["CORENLP_HOME"] = "stanford-corenlp-full-2017-06-09/"

SSTDIR = "sst-light-0.4/"

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
nlp = en_core_web_lg.load(disable=['ner'])
nlp.add_pipe(prevent_sentence_segmentation, name='prevent-sbd', before='parser')


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
    #edges.append(("ROOT", '{0}-{1}'.format(list(document)[0].lower_, list(document)[0].i)))
    for s in document.sents:
        edges.append(("ROOT", '{0}-{1}'.format(s.root.lower_, s.root.i)))
    for token in document:
        nodes.append('{0}-{1}'.format(token.lower_, token.i))
        #edges.append(("ROOT", '{0}-{1}'.format(token.lower_, token.i)))
        # print('{0}-{1}'.format(token.lower_, token.i))
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            #print("----", '{0}-{1}'.format(child.lower_, child.i))
            edges.append(('{0}-{1}'.format(token.lower_, token.i),
                          '{0}-{1}'.format(child.lower_, child.i)))
    return nx.Graph(edges), nodes


def get_network_graph_corenlp(sent):

    edges = []
    nodes = []
    # ensure that every token is connected
    #edges.append(("ROOT", '{0}-{1}'.format(list(document)[0].lower_, list(document)[0].i)))
    #edges.append(("ROOT", '{0}-{1}'.format(s.root.lower_, s.root.i)))
    if sent.basicDependencies.root:
        nodes.append("{}-{}".format(sent.token[sent.basicDependencies.root[0]-1].word.lower(),
                                    sent.basicDependencies.root[0]-1))
    else:
        print("no root!", corenlp.to_text(sent))
    for e in sent.basicDependencies.edge:
        head_idx = e.source - 1
        head_text = sent.token[head_idx].word

        child_idx = e.target - 1
        child_text = sent.token[child_idx].word
        child_label = '{0}-{1}'.format(child_text.lower(), child_idx)
        if child_label not in nodes:
            nodes.append(child_label)
        #edges.append(("ROOT", '{0}-{1}'.format(token.lower_, token.i)))
        # print('{0}-{1}'.format(token.lower_, token.i))
        # FYI https://spacy.io/docs/api/token
        #for child in token.children:
            #print("----", '{0}-{1}'.format(child.lower_, child.i))
        edges.append(('{0}-{1}'.format(head_text.lower(), head_idx),
                    child_label))
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
        #entity_tokens = sentence.char_span(offset[0], offset[1])
        entity_tokens = [(t,i) for i, t in enumerate(sentence.token) if t.beginChar == offset[0]]
        #if not entity_tokens:
            # try to include the next char
            #entity_tokens = sentence.char_span(offset[0], offset[1] + 1)
        #    entity_tokens = [t for t in sentence.token if t.beginChar == offset[0]]

        if not entity_tokens:
            logging.warning(("no tokens found:", entities[eid], sentence.text,
                             '|'.join(["{}({}-{})".format(t.word, t.beginChar, t.endChar) for t in sentence.token])))
            #sys.exit()
        else:
            head_token = '{0}-{1}'.format(entity_tokens[0][0].word.lower(),
                                          entity_tokens[0][1])
            if head_token in sentence_head_tokens:
                logging.warning(("head token conflict:", sentence_head_tokens[head_token], entities[eid]))
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
        #if not entity_tokens:
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
                ("no tokens found:", entities[eid], sentence.text, '|'.join([t.text for t in sentence])))
        else:
            head_token = '{0}-{1}'.format(entity_tokens.root.lower_,
                                          entity_tokens.root.i)
            if eid in positive_entities:
                pos_gv.add(entity_tokens.root.head.lower_)
            else:
                neg_gv.add(entity_tokens.root.head.lower_)
            if head_token in sentence_head_tokens:
                logging.warning(("head token conflict:", sentence_head_tokens[head_token], entities[eid]))
            sentence_head_tokens[head_token] = eid
    return sentence_head_tokens, pos_gv, neg_gv


def run_sst(token_seq):
    chunk_size = 500
    wordnet_tags = {}
    sent_ids = list(token_seq.keys())
    chunks = [sent_ids[i:i + chunk_size] for i in range(0, len(sent_ids), chunk_size)]
    for i, chunk in enumerate(chunks):
        sentence_file = open("temp/sentences_{}.txt".format(i), 'w')
        for sent in chunk:
            sentence_file.write("{}\t{}\t.\n".format(sent, "\t".join(token_seq[sent])))
        sentence_file.close()
        sst_args = ["sst", "bitag",
                    "{}/MODELS/WSJPOSc_base_20".format(SSTDIR), "{}/DATA/WSJPOSc.TAGSET".format(SSTDIR),
                    "{}/MODELS/SEM07_base_12".format(SSTDIR), "{}/DATA/WNSS_07.TAGSET".format(SSTDIR),
                    "temp/sentences_{}.txt".format(i), "0", "0"]
        p = Popen(sst_args, stdout=PIPE)
        p.communicate()
        with open("temp/sentences_{}.txt.tags".format(i)) as f:
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
        sentence_text = sentence_text[:idx[0]] + sentence_text[idx[0]:idx[1]].replace(" ", "_") + sentence_text[idx[1]:]

    # clean text to make tokenization easier
    sentence_text = sentence_text.replace(";", ",")
    sentence_text = sentence_text.replace("*", " ")
    sentence_text = sentence_text.replace(":", ",")
    sentence_text = sentence_text.replace(" - ", " ; ")
    parsed = nlp(sentence_text)

    return parsed

corenlp_parser = corenlp.CoreNLPClient(annotators="tokenize pos depparse".split())

def parse_sentence_corenlp(sentence_text, sentence_entities):
    for e in sentence_entities:
        idx = sentence_entities[e][0]
        sentence_text = sentence_text[:idx[0]] + sentence_text[idx[0]:idx[1]].replace(" ", "_") + sentence_text[idx[1]:]

    # clean text to make tokenization easier
    sentence_text = sentence_text.replace(";", ",")
    sentence_text = sentence_text.replace("*", " ")
    sentence_text = sentence_text.replace(":", ",")
    sentence_text = sentence_text.replace(".", ",")
    sentence_text = sentence_text.replace("/", " ")
    ann = corenlp_parser.annotate(sentence_text)
    if len(ann.sentence) > 1:
        print(len(ann.sentence))
        print(sentence_text)
        print(corenlp.to_text(ann.sentence[1]))
        sys.exit()

    return ann.sentence[0]


def process_sentence_corenlp(sentence, sentence_entities, sentence_pairs, wordnet_tags=None):
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

    graph, nodes_list = get_network_graph_corenlp(sentence)
    sentence_head_tokens = get_head_tokens(sentence_entities, sentence)
    for (e1, e2) in combinations(sentence_head_tokens, 2):
        if int(sentence_head_tokens[e1].split("e")[-1]) > int(sentence_head_tokens[e2].split("e")[-1]):
            e1, e2 = e2, e1

        e1_text = sentence_entities[sentence_head_tokens[e1]]
        e2_text = sentence_entities[sentence_head_tokens[e2]]

        if e1_text[1].lower() == e2_text[1].lower():
            logging.debug("skipped same text: {} {}".format(e1_text, e2_text))
            continue

        if e2_text[0][0] - e1_text[0][-1] < 3:
            logging.debug("skipped sentence list: {} {}".format(e1_text, e2_text))
            continue

        labels.append((sentence_head_tokens[e1], sentence_head_tokens[e2]))
        if (sentence_head_tokens[e1], sentence_head_tokens[e2]) in sentence_pairs:
            classes.append(sentence_pairs[(sentence_head_tokens[e1], sentence_head_tokens[e2])])
        else:
            classes.append(0)
        head_token1_idx = int(e1.split("-")[-1])
        head_token2_idx = int(e2.split("-")[-1])
        try:
            sdp = nx.shortest_path(graph, source=e1, target=e2)
            #if len(sdp) < 3: # len=2, just entities
            #    sdp = [sdp[0]] + nodes_list[head_token1_idx-2:head_token1_idx]
            #    sdp += nodes_list[head_token2_idx+1:head_token2_idx+3] + [sdp[-1]]
            # print(e1_text[1:], e2_text[1:], sdp)
            #if len(sdp) == 2:
                # add context words
            vector = []
            wordnet_vector = []

            head_token_position = None
            for i, element in enumerate(sdp):
                token_idx = int(element.split("-")[-1]) # get the index of the token
                #sdp_token = sentence[token_idx] #get the token obj
                sdp_token = sentence.token[token_idx]  # get the token obj
                #if sdp_token.idx in entity_offsets:
                #    vector.append("drug")
                #    print("masked entity {}".format(sdp_token.text))
                #else:
                    #vector.append(sdp_token.vector)
                vector.append(sdp_token.word)
                if wordnet_tags:
                    wordnet_vector.append(wordnet_tags[token_idx])
                #print(element, sdp_token.text, head_token, sdp)
                #head_token = "{}-{}".format(sdp_token.head.lower_, sdp_token.head.i)  # get the key of head token
                #head_token = [t for t in graph.edges() if t in ]
                #print(graph.edges())
                head_token = [e[0] for e in graph.edges() if e[1] == element]
                #  head token must not have its head in the path, otherwise that would be the head token
                # in some cases the token is its own head
                #print(sdp_token.word, head_token)
                if len(head_token) == 0 or head_token[0] not in sdp or head_token[0] == element:
                    #print("found head token of:", e1_text, e2_text, sdp_token.text, sdp)
                    head_token_position = i
                #vector.append(parsed[token_idx].text)
            # print(vector)
            if head_token_position is None:
                print("head token not found:", e1_text, e2_text, sdp)
                sys.exit()
            else:
                if head_token_position == len(vector) - 1:
                    head_token_position -= 1
                left_vector = vector[:head_token_position+1]
                right_vector = vector[head_token_position:]
                left_wordnet = wordnet_vector[:head_token_position+1]
                right_wordnet = wordnet_vector[head_token_position:]
            #word_vectors.append(vector)
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
            logging.warning(("node not found:", e1_text, e2_text, e1, e2, corenlp.to_text(sentence), graph.nodes()))
            left_word_vectors.append([])
            right_word_vectors.append([])
            left_wordnets.append([])
            right_wordnets.append([])
    return labels, (left_word_vectors, right_word_vectors), (left_wordnets, right_wordnets), classes




def process_sentence_spacy(sentence, sentence_entities, sentence_pairs, positive_entities,
                           wordnet_tags=None, mask_entities=True,
                           min_sdp_len=0, max_sdp_len=15):
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
    sentence_head_tokens, pos_gv, neg_gv = get_head_tokens_spacy(sentence_entities, sentence, positive_entities)
    #print(neg_gv - pos_gv)
    entity_offsets = [sentence_entities[x][0][0] for x in sentence_entities]
    #print(sentence_head_tokens)
    for (e1, e2) in combinations(sentence_head_tokens, 2):
        #print()
        #print(sentence_head_tokens[e1], e1, sentence_head_tokens[e2], e2)
        # reorder according to entity ID
        if int(sentence_head_tokens[e1].split("e")[-1]) > int(sentence_head_tokens[e2].split("e")[-1]):
            e1, e2 = e2, e1

        e1_text = sentence_entities[sentence_head_tokens[e1]]
        e2_text = sentence_entities[sentence_head_tokens[e2]]

        if e1_text[1].lower() == e2_text[1].lower():
            #logging.debug("skipped same text: {} {}".format(e1_text, e2_text))
            continue


        middle_text = sentence.text[e1_text[0][-1]:e2_text[0][0]]

        #if middle_text.strip() == "or" or middle_text.strip() == "and":
            #logging.debug("skipped entity list: {} {} {}".format(e1_text, middle_text, e2_text))
        #    continue

        if middle_text.strip() in string.punctuation:
        #    logging.debug("skipped punctuation: {} {} {}".format(e1_text, middle_text, e2_text))
            continue

        #if len(middle_text) < 3:
        #    logging.debug("skipped entity list: {} {} {}".format(e1_text, middle_text, e2_text))
        #    continue




        head_token1_idx = int(e1.split("-")[-1])
        head_token2_idx = int(e2.split("-")[-1])
        try:
            sdp = nx.shortest_path(graph, source=e1, target=e2)

            if len(sdp) < min_sdp_len or len(sdp) > max_sdp_len:
                #logging.debug("skipped short sdp: {} {} {}".format(e1_text, str(sdp), e2_text))
                continue

            neg = False
            is_neg_gv = False
            for i, element in enumerate(sdp):
                token_idx = int(element.split("-")[-1])  # get the index of the token
                token_text = element.split("-")[0]
                if (i == 1 or i == len(sdp) - 2) and token_text in neg_gv_list:
                    logging.info("skipped gv {} {}:".format(token_text, str(sdp)))
                    #is_neg_gv = True
                sdp_token = sentence[token_idx]  # get the token obj
                #if any(c.dep_ == 'neg' for c in sdp_token.children):
                #    neg = True

            if neg or is_neg_gv:
                continue

            #if len(sdp) < 3: # len=2, just entities
            #    sdp = [sdp[0]] + nodes_list[head_token1_idx-2:head_token1_idx]
            #    sdp += nodes_list[head_token2_idx+1:head_token2_idx+3] + [sdp[-1]]
            # print(e1_text[1:], e2_text[1:], sdp)
            #if len(sdp) == 2:
                # add context words
            vector = []
            wordnet_vector = []
            negations = 0
            head_token_position = None
            for i, element in enumerate(sdp):
                if element != "ROOT":
                    token_idx = int(element.split("-")[-1]) # get the index of the token
                    sdp_token = sentence[token_idx]  # get the token obj


                    #if any(c.dep_ == 'neg' for c in sdp_token.children):
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
                    #print(element, sdp_token.text, head_token, sdp)
                    head_token = "{}-{}".format(sdp_token.head.lower_, sdp_token.head.i)  # get the key of head token
                    # head token must not have its head in the path, otherwise that would be the head token
                    # in some cases the token is its own head
                    if head_token not in sdp or head_token == element:
                        #print("found head token of:", e1_text, e2_text, sdp_token.text, sdp)
                        head_token_position = i + negations
                    #vector.append(parsed[token_idx].text)
            # print(vector)
            if head_token_position is None:
                print("head token not found:", e1_text, e2_text, sdp)
                sys.exit()
            else:
                left_vector = vector[:head_token_position+1]
                right_vector = vector[head_token_position:]
                left_wordnet = wordnet_vector[:head_token_position+1]
                right_wordnet = wordnet_vector[head_token_position:]
            #word_vectors.append(vector)
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
            logging.warning(("node not found:", e1_text, e2_text, e1, e2, list(sentence), graph.nodes()))
            left_word_vectors.append([])
            right_word_vectors.append([])
            left_wordnets.append([])
            right_wordnets.append([])

        labels.append((sentence_head_tokens[e1], sentence_head_tokens[e2]))
        #print(sentence_head_tokens[e1], sentence_head_tokens[e2])
        if (sentence_head_tokens[e1], sentence_head_tokens[e2]) in sentence_pairs:
            classes.append(sentence_pairs[(sentence_head_tokens[e1], sentence_head_tokens[e2])])
        else:
            classes.append(0)
    return labels, (left_word_vectors, right_word_vectors), (left_wordnets, right_wordnets), classes, pos_gv, neg_gv



