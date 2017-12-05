import sys
# open file from train_rnn predict and convert to expected semeval format

from parse_ddi import pairtypes, label_to_pairtype
pairtype_to_label = {v: k for k, v in label_to_pairtype.items()}
print(pairtype_to_label)

with open(sys.argv[1], 'r') as f, open(sys.argv[1] + ".tsv", 'w') as ff:
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

