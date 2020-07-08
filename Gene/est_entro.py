import numpy as np
import scipy.io as sio

poly_entro = None

def Sigmoid(x):

    return 1/(1+np.exp(-x))

def est_entro_JVHW(samp):
    """Proposed JVHW estimate of Shannon entropy (in bits) of the input sample
    This function returns a scalar JVHW estimate of the entropy of samp when
    samp is a vector, or returns a row vector containing the JVHW estimate of
    each column of samp when samp is a matrix.
    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each column
               of the input matrix. The output data type is double.
    """
    samp = formalize_sample(samp)
    [n, wid] = samp.shape
    n = float(n)
    # The order of polynomial is no more than 22 because otherwise floating-point error occurs
    order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
    global poly_entro
    if poly_entro is None:
        poly_entro = sio.loadmat('poly_coeff_entro.mat')['poly_entro']
    coeff = poly_entro[order-1, 0][0]

    f = fingerprint(samp)

    prob = np.arange(1, f.shape[0] + 1) / n

    # Piecewise linear/quadratic fit of c_1
    V1 = np.array([0.3303, 0.4679])
    V2 = np.array([-0.530556484842359, 1.09787328176926, 0.184831781602259])
    f1nonzero = f[0] > 0
    c_1 = np.zeros(wid)

    with np.errstate(divide='ignore', invalid='ignore'):
        if n >= order and f1nonzero.any():
            if n < 200:
                c_1[f1nonzero] = np.polyval(V1, np.log(n / f[0, f1nonzero]))
            else:
                n2f1_small = f1nonzero & (np.log(n / f[0]) <= 1.5)
                n2f1_large = f1nonzero & (np.log(n / f[0]) > 1.5)
                c_1[n2f1_small] = np.polyval(V2, np.log(n / f[0, n2f1_small]))
                c_1[n2f1_large] = np.polyval(V1, np.log(n / f[0, n2f1_large]))

            # make sure nonzero threshold is higher than 1/n
            c_1[f1nonzero] = np.maximum(c_1[f1nonzero], 1 / (1.9 * np.log(n)))

        prob_mat = entro_mat(prob, n, coeff, c_1)

    return np.sum(f * prob_mat, axis=0) / np.log(2)

def entro_mat(x, n, g_coeff, c_1):
    # g_coeff = {g0, g1, g2, ..., g_K}, K: the order of best polynomial approximation,
    K = len(g_coeff) - 1
    thres = 4 * c_1 * np.log(n) / n
    T, X = np.meshgrid(thres, x)
    ratio = np.minimum(np.maximum(2 * X / T - 1, 0), 1)
    q = np.arange(K).reshape((1, 1, K))
    g = g_coeff.reshape((1, 1, K + 1))
    MLE = - X * np.log(X) + 1 / (2 * n)
    polyApp = np.sum(np.concatenate((T[..., None], ((n * X)[..., None]  - q) / (T[..., None] * (n - q))), axis=2).cumprod(axis=2) * g, axis=2) - X * np.log(T)
    polyfail = np.isnan(polyApp) | np.isinf(polyApp)
    polyApp[polyfail] = MLE[polyfail]
    output = ratio * MLE + (1 - ratio) * polyApp
    return np.maximum(output, 0)

def est_entro_MLE(samp):
    """Maximum likelihood estimate of Shannon entropy (in bits) of the input
    sample
    This function returns a scalar MLE of the entropy of samp when samp is a
    vector, or returns a (row-) vector consisting of the MLE of the entropy
    of each column of samp when samp is a matrix.
    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each
               column of the input matrix. The output data type is double.
    """
    samp = formalize_sample(samp)
    [n, wid] = samp.shape


    n = float(n)
    f = fingerprint(samp)

    prob = np.arange(1, f.shape[0] + 1) / n
    prob_mat = - np.log2(prob) * prob
    return prob_mat.dot(f)

def est_entro_MLE_F(samp):
    """Maximum likelihood estimate of Shannon entropy (in bits) of the input
    sample
    This function returns a scalar MLE of the entropy of samp when samp is a
    vector, or returns a (row-) vector consisting of the MLE of the entropy
    of each column of samp when samp is a matrix.
    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each
               column of the input matrix. The output data type is double.
    """
    samp = formalize_sample(samp)
    [n, wid] = samp.shape


    n = float(n)
    f = fingerprint(samp)

    prob = np.arange(1, f.shape[0] + 1) / n
    prob_mat = - np.log2(prob) / n
    return prob_mat.dot(f)

def gradient_logistic(samp,mu,sigma,N):


    sa = samp * (samp != N-1) + (5*N) * (samp == N-1)
    sb = samp * (samp != 0) + (-5*N) * (samp == 0)
    a = np.exp((-sa - 0.5 + mu)/sigma)

    b = np.exp((-sb + 0.5 + mu)/sigma)

    g_mu = 1/((1/(1+a)) -(1/(1+b))) * ((- a/(((1+a) ** 2) * sigma)) + ( b/ (((1+b) ** 2)*sigma)) )
    g_mu = g_mu.sum() / samp.shape[0]

    g_sigma = 1/((1/(1+a)) -(1/(1+b))) * ((- ((sa + 0.5 - mu) * a)/(((1+a) ** 2) * (sigma**2))) + ( ((sb - 0.5 - mu) * b)/ (((1+b) ** 2)*(sigma**2))) )

    g_sigma = g_sigma.sum() / samp.shape[0]

    return g_mu,g_sigma

def forward_logistic(samp,mu,sigma,N):

    sa = samp * (samp != N - 1) + (5*N) * (samp == N - 1)
    sb = samp * (samp != 0) + (-5*N) * (samp == 0)
    a = np.exp((-sa - 0.5 + mu)/sigma)
    b = np.exp((-sb + 0.5 + mu)/sigma)
    #print("sample:",samp,"sa:",sa,"sb:",sb)
    #print(((1/(1+a)) -(1/(1+b))))
    f = np.log(((1/(1+a)) -(1/(1+b))))
    #print(mu, sigma,f)
    return f.sum()/samp.shape[0]


def est_entro_logistic(samp,alphabet,x_i,i,j):

    """Maximum likelihood estimate (logistic distribution) of Shannon entropy (in bits) of the input
        sample
        This function returns a scalar MLE of the entropy of samp when samp is a
        vector, or returns a (row-) vector consisting of the MLE of the entropy
        of each column of samp when samp is a matrix.
        Input:
        ----- samp: a vector or matrix which can only contain integers. The input
                    data type can be any interger classes such as uint8/int8/
                    uint16/int16/uint32/int32/uint64/int64, or floating-point
                    such as single/double.
        Output:
        ----- est: SGA results
        """


    mu = alphabet/2

    dmu = 10
    dsigma = 100
    sigma = 10
    old = 10000
    #new = forward_logistic(samp,mu,sigma,alphabet)
    step_size = 10
    cnt = 0
    while dmu > 1e-2:

        cnt += 1
        dmu,dsigma =  gradient_logistic(samp,mu,sigma,alphabet)
        #print("Forward value:", new, "Gradient Mu:", dmu,"Mu:",mu,"Gradient Sigma:", dsigma,"Sigma:",sigma)
        mu += step_size * dmu
        #sigma += 0.1 * dsigma

        #old = new
        #new = forward_logistic(samp,mu,sigma,alphabet)
        #print(forward_logistic(samp,mu,sigma,alphabet),dmu)
        if cnt > 100:
            break

    #print(mu,sigma)
    #print("Estimated:",mu,"True mu:",true_mu)
    #print("FINISH!!!")

    return -1* forward_logistic(samp,mu,sigma,alphabet)



def formalize_sample(samp):
    samp = np.array(samp)
    if np.any(samp != np.fix(samp)):
        raise ValueError('Input sample must only contain integers.')
    if samp.ndim == 1 or samp.ndim == 2 and samp.shape[0] == 1:
        samp = samp.reshape((samp.size, 1))
    return samp

def fingerprint(samp):
    """A memory-efficient algorithm for computing fingerprint when wid is
    large, e.g., wid = 100
    """
    wid = samp.shape[1]

    d = np.r_[
        np.full((1, wid), True, dtype=bool),
        np.diff(np.sort(samp, axis=0), 1, 0) != 0,
        np.full((1, wid), True, dtype=bool)
    ]


    f_col = []
    f_max = 0

    for k in range(wid):
        a = np.diff(np.flatnonzero(d[:, k]))
        a_max = a.max()
        hist, _ = np.histogram(a, bins=a_max, range=(1, a_max + 1))
        f_col.append(hist)
        if a_max > f_max:
            f_max = a_max

    return np.array([np.r_[col, [0] * (f_max - len(col))] for col in f_col]).T
