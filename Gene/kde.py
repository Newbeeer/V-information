'''Computation of mutual information'''

import numpy as np

from scipy.special import gamma,psi
from scipy.linalg import det
from numpy import pi
from sklearn.neighbors import NearestNeighbors
from scipy.stats.kde import gaussian_kde
import scipy.stats as stats

# __all__=['entropy', 'mutual_information', 'entropy_gaussian']
#
# EPS = np.finfo(float).eps


def _nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def _entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if np.isscalar(C): # C is the variance
        return .5*(1 + np.log(2*pi)) + .5*np.log(C)
    else:
        n = C.shape[0] # dimension
        return .5*n*(1 + np.log(2*pi)) + .5*np.log(abs(det(C)))


def _entropy(X, k=1):
    ''' Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = _nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError(
                "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([_entropy(X, k=k) for X in variables])
            - _entropy(all_vars, k=k))


def mutual_kde(x, y):

    """Mutual information estimated as cumulative sum  of ratio
     P(x)P(y)/P(x,y)
     The probability density functions we estimate with kernel-dencity
     estimator (KDE) using Gaussian kernels.
    :param x: numpy.array, shape (n_samples)
    :param y: numpy.array, shape (n_samples)
    :return: float
        Mutual information, a non-negative value
    Notes:
    Bandwidth make influence on the KDE estimation. We use Scott's rule,
    'scott', that is default parameter in 'gaussian_kde'
    """
    type_bandwidth = 'scott'

    xmin = x.min() - 0.1 * (x.max() - x.min())
    xmax = x.max() + 0.1 * (x.max() - x.min())
    ymin = y.min() - 0.1 * (y.max() - y.min())
    ymax = y.max() + 0.1 * (y.max() - y.min())

    Xm, Ym = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    XYm = np.c_[Xm.ravel(), Ym.ravel()]
    xy = np.c_[x, y]

    bandwidth = 'scott'

    kde_xy = gaussian_kde(xy.T, bw_method=bandwidth)
    kde_x = gaussian_kde(x.ravel(), bw_method=bandwidth)
    kde_y = gaussian_kde(y.ravel(), bw_method=bandwidth)

    dict_bandwidth = {'scott': 1.0, '2scott': 2.0, '05scott': 0.5}
    coef_bandwidth = dict_bandwidth[type_bandwidth]

    kde_x.set_bandwidth(bw_method=bandwidth)
    kde_x.set_bandwidth(bw_method=kde_x.factor * coef_bandwidth)

    kde_xy.set_bandwidth(bw_method=bandwidth)
    kde_xy.set_bandwidth(bw_method=kde_xy.factor * coef_bandwidth)


    kde_y.set_bandwidth(bw_method=bandwidth)
    kde_y.set_bandwidth(bw_method=kde_y.factor * coef_bandwidth)

    # Mutual information
    kde_xy_values = kde_xy(XYm.T)
    a = kde_xy_values

    b = (kde_x(Xm[:, 0])[:, np.newaxis] * kde_y(Ym[0])).ravel()

    if np.sum(b == 0) != 0:
        a *= (b!=0)
        b = b + (b==0) * 1
    mutual_information = np.sum(a * np.log(a/b + 1e-10))

    return mutual_information

def test_MutualInfo(useries, vseries):
    """
    https://en.wikipedia.org/wiki/Mutual_information#Definition
    """

    ukde = stats.gaussian_kde(useries)
    vkde = stats.gaussian_kde(vseries)
    jointkeys = list(zip(useries.tolist(), vseries.tolist()))
    jointvalues = np.vstack([useries, vseries])
    jointkde = stats.gaussian_kde(jointvalues)
    MutualInfo = 0
    for uval, vval in jointkeys:
        uprob = ukde(uval)
        vprob = vkde(vval)
        uvprob = jointkde((uval, vval))
        MutualInfo += uvprob * np.log(uvprob / (uprob * vprob))
    if MutualInfo < 0:
        print("Major Error with Mutual Information Calculation", file=sys.stderr)
    return MutualInfo

