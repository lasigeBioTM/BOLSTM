import obonet
import networkx
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

from itertools import combinations
import sys

#graph = obonet.read_obo("data/chebi.obo")
graph = obonet.read_obo("ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo")
#print([dir(d) for u,v,d in graph.edges(data=True)])
#sys.exit()
is_a_graph=networkx.Graph([(u,v,d) for u,v,d in graph.edges(data=True) if d['edgetype'] == "is_a"] )
networkx.is_directed_acyclic_graph(is_a_graph)
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
choices = name_to_id.keys()


drug_names = ["Catecholamine-depleting drugs", "reserpine", "beta-blocking agents"]
drug_ids = []
for d in drug_names:
    drug = process.extractOne(d, choices, scorer=fuzz.token_sort_ratio)
    print("best synonyms", process.extract(d, synonym_to_id.keys(), limit=10, scorer=fuzz.token_sort_ratio))
    if drug[1] < 70:
        drug_syn = process.extract(d, synonym_to_id.keys(), limit=10, scorer=fuzz.token_sort_ratio)
        print("synonyms", d, drug, drug_syn)
        if drug_syn[0][1] > drug[1]:
            drug = drug_syn[0]
    drug_ids.append(drug[0])
#print(drug1)
#print(drug2)
#print(drug3)
print(drug_ids)
#drug1 = drug1[0][0]
#drug2 = drug2[0][0]
#drug3 = drug3[0][0]

def get_path_to_root(drugname):
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

def get_lowest_common_ascestor_path(path1, path2):
    # find the LCA and return the paths to it
    for i1, e1 in enumerate(path1):
        if e1 in path2:
            print(e1)
            print(path1)
            print(path2)
            print()
    return None, path1, path2


for dpair in combinations(drug_ids, 2):
    path1 = get_path_to_root(dpair[0])
    path2 = get_path_to_root(dpair[1])
    print()
    print()
    get_lowest_common_ascestor_path(path1, path2)
#get_lowest_common_ascestor_path(path1, path3)
#get_lowest_common_ascestor_path(path2, path3)