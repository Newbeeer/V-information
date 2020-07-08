import numpy as np




f = open('./net3_expression_data.tsv')
lines = f.readlines()

a = np.zeros((805,4511))
for idx,line in enumerate(lines):
    if idx == 0:
        continue
    line = line.split('\t')
    for i in range(len(line)):
        line[i] = float(line[i])
    a[idx-1] = line

np.save('gene_data_3.npy',a)

