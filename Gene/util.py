import numpy as np

class Graph(object):

    def __init__(self,mat):

        self.weighted_mat = mat  # symmetric
        self.node_num = mat.shape[0]

    def Prim(self):

        T = np.zeros((self.node_num,self.node_num))
        node = [0]

        while len(node) != self.node_num:
            max_weight = -100000
            max_index = -1
            max_index_ad = -1
            for i in range(self.node_num):
                if i in node:
                    continue
                for j in node:

                    if self.weighted_mat[i,j] > max_weight:
                        max_weight = self.weighted_mat[i,j]
                        max_index = i
                        max_index_ad = j
            node += [max_index]
            T[max_index,max_index_ad] = 1
            T[max_index_ad,max_index] = 1

        return T



def test_Graph():

    mat = np.array([[0,5,8],[5,0,3],[8,3,0]])
    g = Graph(mat)
    t = g.Prim()
    print(t)

#test_Graph()

def star_graph_simulate(sample_num,node_num):

    '''

    :param graph_num: Number of simulated graphs
    :param node_num: x_0 is the centric node
    :return: Simulated Graph
    '''

    X = np.zeros((sample_num,node_num))
    X[:,0] = np.random.beta(0.5,0.5,sample_num)

    for j in range(1,node_num):
        X[:,j] = np.random.normal(X[:,0],j*0.1,size=sample_num)

        #X[i, 1:int(node_num / 2)] = np.random.normal(X[i, 0], 1, size=int(node_num / 2 - 1))
        #X[i, int(node_num / 2):] = np.random.normal(2*X[i, 0] + 1, 0.1, size=int(node_num / 2))

    return X

def star_graph_simulate_discrete(sample_num,node_num,N):

    '''

    :param graph_num: Number of simulated graphs
    :param node_num: x_0 is the centric node
    : N : size of alphabet
    :return: Simulated Graph
    '''
    P_X_1 = np.random.beta(2,2,N)
    T = np.random.beta(2,2,(node_num,N,N))
    # normalization
    P_X_1 = P_X_1 / P_X_1.sum()
    T = T / T.sum(2).reshape(node_num,N,1)
    X = np.zeros((sample_num,node_num))
    X[:,0] = np.random.choice(N,sample_num,p=P_X_1)

    for i in range(sample_num):
        for j in range(1,node_num):
            X[i,j] = np.random.choice(N,1,p=T[ j, int(X[i,0])])
            #if X[i,0] == 5:
                #print("Node_number:{}, value: {}".format(X[i,0],X[i,j]))


    return X

def Sigmoid(x):

    return 1/(1+np.exp(-x))


def star_graph_simulate_discrete_logistic_distribution(sample_num,node_num,N):

    '''
    :param graph_num: Number of simulated graphs
    :param node_num: x_0 is the centric node
    : N : size of alphabet
    :return: Simulated Graph
    '''
    P_X_1 = np.random.beta(2,2,N)

    #how about we random generate a mean for every x ?

    random_mean = np.random.randint(low=0,high=N, size=(node_num,N)) # node_number. x , associated mean

    # normalization
    P_X_1 = P_X_1 / P_X_1.sum()

    X = np.zeros((sample_num,node_num))
    X[:,0] = np.random.choice(N,sample_num,p=P_X_1)

    # Tabuler
    # T[i]: discretized logistic distribution with mean i
    T = np.zeros((N,N))
    sigma = 1
    for i in range(N):
        for k in range(N):
            if k == 0:
                T[i][0] = Sigmoid((0+0.5 - i)/sigma) - Sigmoid((-1000-0.5-i)/sigma)
            elif k== N-1:
                T[i][N-1] = Sigmoid((1000+0.5 - i)/sigma) - Sigmoid((N-1-0.5-i)/sigma)
            else:
                T[i][k] = Sigmoid((k + 0.5 - i)/sigma) - Sigmoid((k-0.5-i)/sigma)

    for i in range(sample_num):
        for j in range(1,node_num):
            # associated mean
            #mu = random_mean[j,int(X[i, 0])] # determined by the node index j and the value of the root
            mu = (int(X[i, 0]) + j) % N
            X[i,j] = np.random.choice(N,1,p=T[mu])


    return X

def ratio(T,Groud):

    edge_num = Groud.sum()/2
    edge_diff = ((T * (1-Groud)) + (Groud * (1-T))).sum()/4

    return edge_diff / edge_num



