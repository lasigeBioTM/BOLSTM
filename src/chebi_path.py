import obonet
import networkx
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

from itertools import combinations
import sys

global chebi_cache
chebi_cache = {} # store string-> chebi ID

def load_chebi(path="ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo"):
    print("loading chebi...")
    #graph = obonet.read_obo("data/chebi.obo")
    graph = obonet.read_obo(path)
    #print([dir(d) for u,v,d in graph.edges(data=True)])
    #sys.exit()
    graph = graph.to_directed()
    is_a_graph=networkx.MultiDiGraph([(u,v,d) for u,v,d in graph.edges(data=True) if d['edgetype'] == "is_a"] )
    print(networkx.is_directed_acyclic_graph(is_a_graph))
    id_to_name = {id_: data['name'] for id_, data in graph.nodes(data=True)}
    name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True)}
    synonym_to_id = {}
    for n in graph.nodes(data=True):
        # print(n[1].get("synonym"))
        for syn in n[1].get("synonym", []):
            syn_name = syn.split('"')
            if len(syn_name) > 2:
                syn_name = syn.split('"')[1]
                synonym_to_id.setdefault(syn_name, []).append(n[0])
            else:
                print(syn.split('"'))

    #print(synonym_to_id)
    print("done.")
    return is_a_graph, name_to_id, synonym_to_id, id_to_name




def get_path_to_root(drugname, is_a_graph, name_to_id, synonym_to_id, id_to_name):
    source_id = name_to_id.get(drugname)
    if source_id is None:
        source_id = synonym_to_id[drugname][0]
        print(drugname, "could be one of these:", synonym_to_id[drugname])
    try:
        path = networkx.shortest_path(
            is_a_graph,
            source=source_id,
            target=name_to_id['chemical entity']
        )
    except networkx.exception.NetworkXNoPath:
        path = networkx.shortest_path(
            is_a_graph,
            source=source_id,
            target=name_to_id['role']
        )

    #for path in paths:
    print('•', ' ⟶ '.join(id_to_name[node] for node in path))
    #print(path)
    return path # can use IDs or names


def get_all_shortest_paths_to_root(drugname, is_a_graph, name_to_id, synonym_to_id, id_to_name):
    source_id = name_to_id.get(drugname)
    paths = []
    if source_id is None:
        source_id = synonym_to_id[drugname][0]
        print(drugname, "could be one of these:", synonym_to_id[drugname])
    #print("successors")
    #print([id_to_name[x] for x in is_a_graph.successors(source_id)])
    #print("pred")
    #print([id_to_name[x] for x in is_a_graph.predecessors(source_id)])
    try:
        print("trying chemical entity subontology..", drugname)
        paths = networkx.all_shortest_paths(
            is_a_graph,
            source=source_id,
            target=name_to_id['chemical entity']
        )
        paths = list(paths)
    except networkx.exception.NetworkXNoPath:
        print("trying role subontology..", drugname)
        paths = networkx.all_shortest_paths(
            is_a_graph,
            source=source_id,
            target=name_to_id['role']
        )
        paths = list(paths)

    # for path in paths:
    print(drugname)
    for path in paths:
        print('•', ' ⟶ '.join(id_to_name[node] for node in path))
    print()
    # print(path)
    return paths  # can use IDs or names


def get_lowest_common_ascestor_path(paths1, paths2, id_to_name):
    # find the LCA and return the paths to it
    lowest = None
    lowest_dist = 100
    i_lowest1 = paths1[0] # return first full path if none found
    i_lowest2 = paths2[0]
    for p1 in paths1:
        for p2 in paths2: # use intersection?
            for i1, e1 in enumerate(p1):

                if e1 in p2 and p2.index(e1) + i1 < lowest_dist:
                    lowest_dist = p2.index(e1) + i1
                    print("new best dist:", lowest_dist, id_to_name[e1])
                    lowest = e1
                    i_lowest2 = p2[:p2.index(e1)]
                    i_lowest1 = p1[:i1]
                    print([id_to_name[j] for j in i_lowest1])
                    print([id_to_name[j] for j in i_lowest2])
                    print()
    return lowest, i_lowest1, i_lowest2


def map_to_chebi(text, name_to_id, synonym_to_id):
    """
    Get best ChEBI name for text
    :param text: input text
    :param name_to_id:
    :param synonym_to_id:
    :return:
    """
    if text in name_to_id or text in synonym_to_id:
        return text
    elif text in chebi_cache:
        return chebi_cache[text]
    drug = process.extractOne(text, name_to_id.keys(), scorer=fuzz.token_sort_ratio)
    print("best names of ", text, ":", drug)
    if drug[1] < 70:
        drug_syn = process.extract(text, synonym_to_id.keys(), limit=10, scorer=fuzz.token_sort_ratio)
        print("best synonyms of ", text, ":", drug_syn)
        print("synonyms", text, drug, drug_syn)
        if drug_syn[0][1] > drug[1]:
            drug = drug_syn[0]
    chebi_cache[text] = drug[0]
    return drug[0]

def main():
    is_a_graph, name_to_id, synonym_to_id, id_to_name = load_chebi()
    choices = name_to_id.keys()
    #drug_names = ["Catecholamine-depleting drugs", "reserpine", "beta-blocking agents"]
    #drug_names = ["Catecholamine-depleting", "reserpine", "beta-blocking", "acebutolol", "catecholamine depletors"]
    drug_names = ["tertiary alcohol", "citric acid-d4"]
    drug_ids = []
    for d in drug_names:
        chebi_name = map_to_chebi(d, name_to_id, synonym_to_id)
        drug_ids.append(chebi_name)

    print(drug_ids)
    get_all_shortest_paths_to_root(drug_ids[0], is_a_graph, name_to_id, synonym_to_id, id_to_name)

    for dpair in combinations(drug_ids, 2):
        paths1 = get_all_shortest_paths_to_root(dpair[0], is_a_graph, name_to_id, synonym_to_id, id_to_name)
        paths2 = get_all_shortest_paths_to_root(dpair[1], is_a_graph, name_to_id, synonym_to_id, id_to_name)
        print()
        #print(paths1)
        #print(paths2)
        if paths1 and paths2:
            get_lowest_common_ascestor_path(paths1, paths2, id_to_name)
    #get_lowest_common_ascestor_path(path1, path3)
    #get_lowest_common_ascestor_path(path2, path3)

if __name__ == "__main__":
    main()