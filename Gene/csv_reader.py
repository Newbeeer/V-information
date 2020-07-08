import numpy as np


'''
f = open('./insilico.csv')
lines = f.readlines()
a = np.zeros((660,20))

for idx,line in enumerate(lines):
    if idx == 0:
        continue
    line = line.split(',')
    line = line[5:]
    line[19] = line[19][:-1]
    a[idx-1] = line

print(a)
np.save("20gene.npy",a)

'''

f = open('./trueGraph.csv')
lines = f.readlines()
a = np.zeros((20,20))

for idx,line in enumerate(lines):

    line = line.split(',')
    line[19] = line[19][:-1]

    for j in range(20):

        if line[j] == '1' and idx!=j:
            a[idx,j] = 1
        else:
            a[idx,j] = 0


np.save("20gene_true.npy",a)
print(a)
