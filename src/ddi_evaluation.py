import sys
import itertools
# open file from train_rnn predict and convert to expected semeval format

from parse_ddi import pairtypes, label_to_pairtype
pairtype_to_label = {v: k for k, v in label_to_pairtype.items()}
print(pairtype_to_label)


if sys.argv[1] == "convert":
    with open(sys.argv[2], 'r') as f, open(sys.argv[2] + ".tsv", 'w') as ff:
        for line in f:
            values = line.strip().split(" ")
            if len(values) == 3:
                sentence_id = ".".join(values[0].split(".")[:3])
                if values[2] != "0":
                    inttype = pairtype_to_label[int(values[2])]
                    ff.write("|".join((sentence_id, values[0], values[1], "1", inttype)))
                else:
                    ff.write("|".join((sentence_id, values[0], values[1], "0", "null")))
                ff.write("\n")

elif sys.argv[1] == "analyze":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import venn

    gold_pairs = {"effect": [], "mechanism": [], "advise": [], "int": []}
    with open("goldDDI.txt", 'r') as f:
        for line in f:
            values = line.strip().split("|")
            pair = (values[1], values[2])
            if values[-1] != "null":
                gold_pairs[values[-1]].append(pair)
    all_pairs = [set(itertools.chain.from_iterable(gold_pairs.values()))]
    all_labels = ["Gold standard"]
    results = {}
    for results_file in sys.argv[2:]:
        results[results_file] = {"effect": [], "mechanism": [], "advise": [], "int": []}
        with open(results_file, 'r') as f:
            for line in f:
                values = line.strip().split("|")
                if len(values) > 3:
                    pair = (values[1], values[2])
                    if values[-1] != "null":
                        results[results_file][values[-1]].append(pair)

        all_pairs.append(set(itertools.chain.from_iterable(results[results_file].values())))
        all_labels.append(results_file)
    print(all_labels)
    #print(results)
    labels = venn.get_labels(all_pairs , fill=['number', 'logic'])
    if len(all_pairs) == 2:
        fig, ax = venn.venn2(labels, names=all_labels)
    elif len(all_pairs) == 3:
        fig, ax = venn.venn3(labels, names=all_labels)
    elif len(all_pairs) == 4:
        fig, ax = venn.venn4(labels, names=all_labels)
    elif len(all_pairs) == 5:
        fig, ax = venn.venn5(labels, names=all_labels)
    elif len(all_pairs) == 6:
        fig, ax = venn.venn6(labels, names=all_labels)
    fig.savefig('{}.png'.format("_".join(all_labels)), bbox_inches='tight')
    plt.close()