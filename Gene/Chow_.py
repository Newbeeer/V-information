from util import *
import mixed
from edmonds import *
from kde import mutual_kde,test_MutualInfo
import numpy as np
from test import label
from sklearn import metrics
import argparse
import math

parser = argparse.ArgumentParser(description='Chow')
parser.add_argument('--d', type=float, default=1.)

args = parser.parse_args()
trans = 20
all_gene = 20

def test(method,node_num,mat):

    mat = mat.reshape(-1)
    mat_pair = []
    mat_sort = np.argsort(mat)[::-1][:100]

    correct = 0.0

    for i in range(mat_sort.shape[0]):
        row_idx = int(mat_sort[i]/node_num)
        col_idx = mat_sort[i] % node_num

        if [row_idx,col_idx] in test_pair:
            correct += 1

    print(method," accuracy:",correct/100)

def cal_auc(method,value):


    f,t,threshold = metrics.roc_curve(label,value,pos_label=1)

    print(method," auc score:",metrics.auc(f,t))

    return metrics.auc(f,t)


def main(sample_num,node_num,d,method = 'KSG'):



    mat = np.zeros((trans,all_gene))
    value = []
    for i in range(trans):
        for j in range(all_gene):
            if i == j:
                continue
            if method == 'KSG':
                mat[i,j] = mixed.KSG(X[:,i],X[:,j])
            elif method == 'Noisy_KSG':
                mat[i, j] = mixed.Noisy_KSG(X[:, i], X[:, j])
            elif method == 'Partitioning':
                mat[i, j] = mixed.Partitioning(X[:, i], X[:, j])
            elif method == 'Mixed_KSG':
                mat[i, j] = mixed.Mixed_KSG(X[:, i], X[:, j])
            elif method == 'mutual_kde':
                mat[i, j] = mutual_kde(X[:, i], X[:, j])
            value += [mat[i,j]]

    #test(method,node_num,mat)
    return cal_auc(method,np.array(value))
    #np.save(method + '_' + str(d)+'.npy', mat)


def F_infomation_tree(sample_num,node_num,d):

    mat = np.zeros((trans, all_gene))
    value = []
    for i in range(trans):
        for j in range(all_gene):
            if i == j:
                continue
            mat[i, j] = mixed.F_mlp_gaussian(X[:, i], X[:, j])
            value += [mat[i,j]]


    #test("F-information",node_num,mat)

    return cal_auc("F-information",np.array(value))
    #np.save("F-information_"+str(d)+'.npy',mat)

if __name__ == '__main__':


    d = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    collect = {'F':[],'P':[],'K':[],'NK':[],'MK':[],'KDE':[]}
    for drop_rate in d:
        print("D:",drop_rate)

        num = np.zeros(6)
        seed = 1234
        for i in range(10):
            X = np.load('20gene.npy')
            np.random.seed(seed)

            idx = np.array([i for i in range(X.shape[0])])
            idx = np.random.choice(idx, int(X.shape[0] * drop_rate), replace=False)

            X = X[idx]

            sample_num = X.shape[0]
            node_num = X.shape[1]

            #num[0] += F_infomation_tree(sample_num, node_num,drop_rate)
            #num[1] += main(sample_num, node_num,drop_rate, method='KSG')
            #num[2] += main(sample_num, node_num, drop_rate, method='Partitioning')
            #num[3] += main(sample_num, node_num, drop_rate, method='Noisy_KSG')
            #num[4] += main(sample_num, node_num,drop_rate, method='Mixed_KSG')
            num[5] += main(sample_num, node_num,drop_rate, method='mutual_kde')

            seed += 1
        #print("D:{}, average auc:{}",num[0]/10)


        collect['F'] += [num[0]/10]
        collect['K'] += [num[1]/10]
        collect['P'] += [num[2]/10]
        collect['NK'] += [num[3]/10]
        collect['MK'] += [num[4]/10]
        collect['KDE'] += [num[5]/10]
    num = 10
    # np.save('F_'+str(num)+'.npy', collect['F'])
    # np.save('P_'+str(num)+'.npy', collect['P'])
    # np.save('K_'+str(num)+'.npy', collect['K'])
    # np.save('NK_'+str(num)+'.npy', collect['NK'])
    # np.save('MK_'+str(num)+'.npy', collect['MK'])
    np.save('KDE_'+str(num)+'.npy', collect['KDE'])


    #TODO design ideal experiment
    #TODO try different F-families