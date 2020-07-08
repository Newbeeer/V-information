
import numpy as np

#195 * 1643
'''
f = open('./ground_truth_1.tsv')
lines = f.readlines()
trans = 195
all_gene = 1643
test_pair = []
test_mat = np.zeros((trans,all_gene))
for idx,line in enumerate(lines):

    line = line.split('\t')
    if line[2][0] == '1':
        gene_idx_1 = int(line[0][1:]) - 1
        gene_idx_2 = int(line[1][1:]) - 1

        test_pair += [[gene_idx_1,gene_idx_2]]
        test_mat[gene_idx_1][gene_idx_2] = 1


label = []
for i in range(trans):
    for j in range(all_gene):
        if i == j:
            continue
        label += [test_mat[i,j]]

label = np.array(label).astype(np.int32)

'''
test_mat = np.load('20gene_true.npy')
label = []
for i in range(20):
    for j in range(20):
        if i == j:
            continue
        label += [test_mat[i,j]]

label = np.array(label).astype(np.int32)