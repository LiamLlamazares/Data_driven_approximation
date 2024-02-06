#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as _np
from scipy.spatial import distance


def identity(x):
    '''
    Identity function.
    '''
    return x


class monomials(object):
    '''
    Computation of monomials in d dimensions.
    '''

    def __init__(self, p, n=None):
        '''
        The parameter p defines the maximum order of the monomials.
        '''
        self.p = p
        self.n = n  # number of monomials, takes first n

    def __call__(self, x, n=None):
        '''
        Evaluate all monomials of order up to p for all data points in x.
        If n is provided, only the first n monomials are evaluated.
        '''
        [d, m
         ] = x.shape  # d = dimension of state space, m = number of test points
        c = allMonomialPowers(
            d, self.p)  # matrix containing all powers for the monomials
        if self.n is None:
            n = c.shape[1]  # number of monomials
        else:
            n = min(self.n, c.shape[1])  # limit to n monomials
        y = _np.ones([n, m])
        for i in range(n):
            for j in range(d):
                y[i, :] = y[i, :] * _np.power(x[j, :], c[j, i])
        return y

    def diff(self, x, n=None):
        '''
        Evaluate the derivative of all monomials of order up to p for all data points in x.
        If n is provided, only the derivative of the first n monomials are evaluated.
        '''
        [d, m
         ] = x.shape  # d = dimension of state space, m = number of test points
        c = allMonomialPowers(
            d, self.p)  # matrix containing all powers for the monomials
        if self.n is None:
            n = c.shape[1]  # number of monomials
        else:
            n = min(self.n, c.shape[1])  # limit to n monomials
        y = _np.zeros([n, d, m])
        for i in range(n):
            for j in range(d):
                y[i, j, :] = c[j, i] * _np.power(x[j, :], c[j, i] - 1)
        return y

    def ddiff(self, x, n=None):
        '''
        Evaluate the second derivative of all monomials of order up to p for all data points in x.
        If n is provided, only the second derivative of the first n monomials are evaluated.
        '''
        [d, m
         ] = x.shape  # d = dimension of state space, m = number of test points
        c = allMonomialPowers(
            d, self.p)  # matrix containing all powers for the monomials
        if self.n is None:
            n = c.shape[1]  # number of monomials
        else:
            n = min(self.n, c.shape[1])  # limit to n monomials
        y = _np.zeros([n, d, d, m])
        for i in range(n):
            for j in range(d):
                for k in range(d):
                    if j == k:
                        y[i, j, k, :] = c[j, i] * (c[j, i] - 1) * _np.power(
                            x[j, :], c[j, i] - 2)
        return y

    def __repr__(self):
        return 'Monomials of order up to %d.' % self.p

    def display(self, alpha, d, name=None, eps=1e-6):
        '''
        Display the polynomial with coefficients alpha.
        '''
        c = allMonomialPowers(
            d, self.p)  # matrix containing all powers for the monomials

        if name != None: print(name + ' = ', end='')

        ind, = _np.where(abs(alpha) > eps)
        k = ind.shape[0]

        if k == 0:  # no nonzero coefficients
            print('0')
            return

        for i in range(k):
            if i == 0:
                print('%.5f' % alpha[ind[i]], end='')
            else:
                if alpha[ind[i]] > 0:
                    print(' + %.5f' % alpha[ind[i]], end='')
                else:
                    print(' - %.5f' % -alpha[ind[i]], end='')

            self._displayMonomial(c[:, ind[i]])
        print('')

    def _displayMonomial(self, p):
        d = p.shape[0]
        if _np.all(p == 0):
            print('1', end='')
        else:
            for j in range(d):
                if p[j] == 0:
                    continue
                if p[j] == 1:
                    print(' x_%d' % (j + 1), end='')
                else:
                    print(' x_%d^%d' % (j + 1, p[j]), end='')

    def length(self):
        '''
        Calculate the number of monomials for all data points in x.
        '''
        p = self.p
        return int((p + 1) * (p + 2) / 2)


class indicators(object):
    '''
    Indicator functions for box discretization Omega.
    '''

    def __init__(self, Omega):
        self.Omega = Omega

    def __call__(self, x):
        [d, m
         ] = x.shape  # d = dimension of state space, m = number of test points
        n = self.Omega.numBoxes()
        y = _np.zeros([n, m])
        for i in range(m):
            ind = self.Omega.index(x[:, i])
            pass
            if ind == -1:
                continue
            y[ind, i] = 1
        return y

    def __repr__(self):
        return 'Indicator functions for box discretization.'


class gaussians(object):
    '''
    Gaussians whose centers are the centers of the box discretization Omega.

    sigma: width of Gaussians
    '''

    def __init__(self, Omega, sigma=1, n=None):
        self.Omega = Omega
        self.sigma = sigma
        self.n = n  # number of Gaussians, all if n is None

    def __call__(self, x, n=None):
        '''
        Evaluate Gaussians for all data points in x.
        '''
        c = self.Omega.midpointGrid(self.n)
        D = distance.cdist(c.T, x.T, 'sqeuclidean')
        y = _np.exp(-1 / (2 * self.sigma**2) * D)
        return y

    def diff(self, x, n=None):
        '''
        Compute partial derivatives for all data points in x.
        '''
        [d, m
         ] = x.shape  # d = dimension of state space, m = number of test points
        if self.n is None:
            n = self.Omega.numBoxes()  # number of basis functions
        else:
            n = min(self.n, self.Omega.numBoxes())
        c = self.Omega.midpointGrid(self.n)
        D = distance.cdist(c.T, x.T, 'sqeuclidean')
        y = _np.zeros([n, d, m])
        for i in range(n):  # for all Gaussians
            for j in range(d):  # for all dimensions
                y[i, j, :] = -2 / (2 * self.sigma**2) * (
                    x[j, :] - c[j, i]) * _np.exp(-1 /
                                                 (2 * self.sigma**2) * D[i, :])

        return y

    def ddiff(self, x):
        '''
        Compute second order derivatives for all data points in x.
        '''
        [d, m
         ] = x.shape  # d = dimension of state space, m = number of test points
        if self.n is None:
            n = self.Omega.numBoxes()  # number of basis functions
        else:
            n = min(self.n, self.Omega.numBoxes())
        c = self.Omega.midpointGrid(self.n)
        D = distance.cdist(c.T, x.T, 'sqeuclidean')
        y = _np.zeros([n, d, d, m])
        for i in range(n):  # for all Gaussians
            for j1 in range(d):  # for all dimensions
                for j2 in range(d):  # for all dimensions
                    if j1 == j2:
                        y[i, j1,
                          j2, :] = (-2 / (2 * self.sigma**2) + 4 /
                                    (4 * self.sigma**4) *
                                    (x[j1, :] - c[j1, i])**2) * _np.exp(
                                        -1 / (2 * self.sigma**2) * D[i, :])
                    else:
                        y[i, j1,
                          j2, :] = (4 / (4 * self.sigma**4) *
                                    (x[j1, :] - c[j1, i]) *
                                    (x[j2, :] - c[j2, i])) * _np.exp(
                                        -1 / (2 * self.sigma**2) * D[i, :])
        return y

    def __repr__(self):
        return 'Gaussian functions for box discretization with bandwidth %f.' % self.sigma

    def length(self):
        return self.Omega.numBoxes()


# auxiliary functions
def nchoosek(n, k):
    '''
    Computes binomial coefficients.
    '''
    return math.factorial(n) // math.factorial(k) // math.factorial(
        n - k)  # integer division operator


def nextMonomialPowers(x):
    '''
    Returns powers for the next monomial. Implementation based on John Burkardt's MONOMIAL toolbox, see
    http://people.sc.fsu.edu/~jburkardt/m_src/monomial/monomial.html.
    '''
    m = len(x)
    j = 0
    for i in range(1, m):  # find the first index j > 1 s.t. x[j] > 0
        if x[i] > 0:
            j = i
            break
    if j == 0:
        t = x[0]
        x[0] = 0
        x[m - 1] = t + 1
    elif j < m - 1:
        x[j] = x[j] - 1
        t = x[0] + 1
        x[0] = 0
        x[j - 1] = x[j - 1] + t
    elif j == m - 1:
        t = x[0]
        x[0] = 0
        x[j - 1] = t + 1
        x[j] = x[j] - 1
    return x


def allMonomialPowers(d, p):
    '''
    All monomials in d dimensions of order up to p.
    '''
    # Example: For d = 3 and p = 2, we obtain
    # [[ 0  1  0  0  2  1  1  0  0  0]
    #  [ 0  0  1  0  0  1  0  2  1  0]
    #  [ 0  0  0  1  0  0  1  0  1  2]]
    n = nchoosek(p + d, p)  # number of monomials
    x = _np.zeros(
        d)  # vector containing powers for the monomials, initially zero
    c = _np.zeros([d, n])  # matrix containing all powers for the monomials
    for i in range(1, n):
        c[:, i] = nextMonomialPowers(x)
    c = _np.flipud(c)  # flip array in the up/down direction
    return c
