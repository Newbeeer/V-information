import numpy as np
from est_entro import est_entro_JVHW, est_entro_MLE, formalize_sample,est_entro_logistic,est_entro_MLE_F

def est_MI_JVHW(X, Y):
    """This function returns our scalar estimate of mutual information I(X;Y)
    when both X and Y are vectors, and returns a row vector consisting
    of the estimate of mutual information between each corresponding column
    of X and Y when they are matrices.
    Input:
    ----- X, Y: two vectors or matrices with the same size, which can only
                contain integers.
    Output:
    ----- est: the estimate of the mutual information between input vectors
               or that between each corresponding column of the input
               matrices. The output data type is double.
    """
    [X, Y, XY] = formalize(X, Y)
    # I(X,Y) = H(X) + H(Y) - H(X,Y)
    return np.maximum(0, est_entro_JVHW(X) + est_entro_JVHW(Y) - est_entro_JVHW(XY))

def est_MI_JVHW_F(X, Y, N):
    """This function returns our scalar estimate of mutual information I(X;Y)
    when both X and Y are vectors, and returns a row vector consisting
    of the estimate of mutual information between each corresponding column
    of X and Y when they are matrices.
    Input:
    ----- X, Y: two vectors or matrices with the same size, which can only
                contain integers.
    Output:
    ----- est: the estimate of the mutual information between input vectors
               or that between each corresponding column of the input
               matrices. The output data type is double.
    """
    [X, Y, XY] = formalize(X, Y)

    xbins = np.zeros((N))
    for x_i in range(N):
        xbins[x_i] = (X == x_i).sum()

    xbins = xbins / X.shape[0]
    H_y_x = 0
    X = X[:,0] - 1

    for x_i in range(N):

        if (X == x_i).sum() == 0:
            continue
        H_y_x += xbins[x_i] * est_entro_JVHW(Y[np.argwhere(X == x_i).reshape(-1)].reshape(-1,1))


    return np.maximum(0, est_entro_JVHW(Y) - H_y_x)

def est_MI_MLE_F(X, Y, N):
    """This function returns our scalar estimate of mutual information I(X;Y)
    when both X and Y are vectors, and returns a row vector consisting
    of the estimate of mutual information between each corresponding column
    of X and Y when they are matrices.
    Input:
    ----- X, Y: two vectors or matrices with the same size, which can only
                contain integers.
    Output:
    ----- est: the estimate of the mutual information between input vectors
               or that between each corresponding column of the input
               matrices. The output data type is double.
    """
    [X, Y, XY] = formalize(X, Y)

    xbins = np.zeros((N))
    for x_i in range(N):
        xbins[x_i] = (X == x_i).sum()

    xbins = xbins / X.shape[0]
    H_y_x = 0
    X = X[:,0] - 1

    for x_i in range(N):

        if (X == x_i).sum() == 0:
            continue
        H_y_x += xbins[x_i] * est_entro_MLE_F(Y[np.argwhere(X == x_i).reshape(-1)].reshape(-1,1))

    return np.maximum(0, est_entro_MLE_F(Y) - H_y_x)


def est_MI_logistic_F(X, Y, N,i,j):
    """This function returns our scalar estimate of mutual information I(X;Y)
    when both X and Y are vectors, and returns a row vector consisting
    of the estimate of mutual information between each corresponding column
    of X and Y when they are matrices.
    Input:
    ----- X, Y: two vectors or matrices with the same size, which can only
                contain integers.
    Output:
    ----- est: the estimate of the mutual information between input vectors
               or that between each corresponding column of the input
               matrices. The output data type is double.
    """
    [X, Y, XY] = formalize(X, Y)

    xbins = np.zeros((N))
    for x_i in range(N):
        xbins[x_i] = (X == x_i).sum()

    xbins = xbins / X.shape[0]
    H_y_x = 0
    X = X[:,0] - 1
    Y = Y - 1

    for x_i in range(N):

        if (X == x_i).sum() == 0:
            continue

        t = est_entro_logistic(Y[np.argwhere(X == x_i).reshape(-1)].reshape(-1,1),N,x_i,i,j)
        H_y_x += xbins[x_i] * t

    print(i,j,H_y_x)
    #return np.maximum(0, est_entro_logistic(Y,alphabet=N)- H_y_x)
    return - H_y_x


def est_MI_MLE(X, Y):
    """This function returns the scalar MLE of the mutual information I(X;Y)
    when both X and Y are vectors, and returns a row vector consisting
    of the estimate of mutual information between each corresponding column
    of X and Y when they are matrices.
    Input:
    ----- X, Y: two vectors or matrices with the same size, which can only
                contain integers.
    Output:
    ----- est: the estimate of the mutual information (in bits) between input
               vectors or that between each corresponding column of the input
               matrices. The output data type is double.
    """

    [X, Y, XY] = formalize(X, Y)

    # I(X,Y) = H(X) + H(Y) - H(X,Y)
    return np.maximum(0, est_entro_MLE(X) + est_entro_MLE(Y) - est_entro_MLE(XY))


def formalize(X, Y):
    X = formalize_sample(X)
    Y = formalize_sample(Y)

    if X.shape != Y.shape:
        raise ValueError('Input arguments X and Y should be of the same size.')

    X = X.astype(np.int64, copy=False)
    Y = Y.astype(np.int64, copy=False)

    X = map_int(X)
    Y = map_int(Y)
    XY = (X - 1) * Y.max(axis=0) + Y

    return X, Y, XY

def map_int(samp):
    """Map integer data along each column of X and Y to consecutive integer
    numbers (which start with 1 and end with the total number of distinct
    values in each corresponding column). For example,
                    [  1    6    4  ]        [ 1  3  3 ]
                    [  2    6    3  ] -----> [ 2  3  2 ]
                    [  3    2    2  ]        [ 3  1  1 ]
                    [ 1e5   3   100 ]        [ 4  2  4 ]
    The purpose of this data mapping is to make the effective data range
    as small as possible, minimizing the possibility of overflows.
    """
    id = samp.argsort(axis=0)
    col_index = np.indices(samp.shape)[1]
    samp = samp[id, col_index]
    samp[id, col_index] = np.cumsum(np.r_[np.ones((1, samp.shape[1])), np.diff(samp, axis=0) > 0], axis=0)
    return samp