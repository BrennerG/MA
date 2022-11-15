import json

# LOC = 'data/4lang_parses/cose_train_8752_to-root_.json'
# LOC = 'data/4lang_parses/cose_validation_1086_to-root_.json'
LOC = 'data/4lang_parses/cose_test_1079_to-root_.json'

data = []
with open(LOC, 'r') as fp:
    for line in fp:
        data.append(json.loads(line))

bad_apples = []
edges = data[0][0]
for i,sample in enumerate(edges):
    lens = [len(x) for x in sample]
    if 0 in lens: bad_apples.append(i)

print('DONE')