#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as _np
from scipy.spatial import distance
from skfem import MeshTri


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
        p = self.p
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


class FEM_2d(object):
    '''
    Finite element basis functions on uniform mesh with 0 boundary. 
    Omega = domain
    n = approximate number of basis functions, equal to number of vertices
    '''

    def __init__(self, Omega):
        self.d = Omega._bounds.shape[0]
        n = Omega.numBoxes()
        # Creates a uniform mesh
        x = _np.linspace(Omega._bounds[0, 0], Omega._bounds[0, 1],
                         int(_np.sqrt(n)) - 1)
        y = _np.linspace(Omega._bounds[1, 0], Omega._bounds[1, 1],
                         int(_np.sqrt(n)) - 1)
        self.mesh = MeshTri.init_tensor(x,
                                        y)  # Create the mesh with N vertices
        #Calculates Jacobian of mappings (linear part)
        self.inverse_mappings = [
            self.__inverse_mapping(i) for i in range(self.mesh.t.shape[1])
        ]
        self.inverse_mapping_jacobians = [
            self.__inverse_mapping_jacobian(i)
            for i in range(self.mesh.t.shape[1])
        ]

    def __x_in_triangle(self, triangle_index, x):
        '''
        Check if point x is inside the triangle 
        '''

        vertices = self.mesh.p[:, self.mesh.t[:, triangle_index]].T

        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (
                p1[1] - p3[1])

        b1 = sign(x, vertices[0], vertices[1]) < 0.0
        b2 = sign(x, vertices[1], vertices[2]) < 0.0
        b3 = sign(x, vertices[2], vertices[0]) < 0.0

        return ((b1 == b2) and (b2 == b3))

    def __get_Triangle(
            self,
            x):  # It's possible to get explicit formula using that mesh is uniform
        '''
        Find the triangle that contains the point x.
        '''
        mesh = self.mesh
        for i in range(mesh.t.shape[1]):
            if self.__x_in_triangle(i, x):
                return i
        return -1

    def __inverse_mapping(self, triangle_index):
        '''
        Returns function that given mesh coordinates of points in triangle returns local (reference) coordinates.
        Sends v0,v1,v2 to (0,0), (1,0), (0,1).
        '''
        mesh = self.mesh

        def inverse_mapping(x):
            vertices = mesh.p[:, mesh.t[:, triangle_index]].T
            A = _np.vstack(
                (vertices[1] - vertices[0], vertices[2] - vertices[0])).T
            x_reference = _np.linalg.solve(A, x - vertices[0])
            return x_reference

        return inverse_mapping

    def __inverse_mapping_jacobian(self, triangle_index):
        '''
        Returns Jacobian of mapping from mesh to reference coordinates.
        Linear part of function that sends v0,v1,v2 to (0,0), (1,0), (0,1).
        '''
        mesh = self.mesh
        vertices = mesh.p[:, mesh.t[:, triangle_index]].T
        A_linear = _np.vstack(
            (vertices[1] - vertices[0], vertices[2] - vertices[0])).T
        return _np.linalg.inv(A_linear)

    def __phi(self, x):
        '''
        Evaluate finite element basis functions for all data points in x.
        '''
        return _np.array([1 - x[0] - x[1], x[0], x[1]])

    def __nabla_phi_ref(self):
        '''
        Compute partial derivatives of the reference basis functions. Size 2x3.
        '''
        return _np.array([[-1, 1, 0], [-1, 0, 1]])

    def calc_G(self, X, f=None, sigma_noise=None):
        '''
        Calculate the mass matrix G given the data points X.
        Sums up phi_i(x_m)phi_j(x_m) for all data points x_m and adds to relevant entries of G.
        '''
        if f is None:
            f = identity
        M = X.shape[1]
        n = self.mesh.p.shape[1]
        G = _np.zeros([n, n])

        if sigma_noise is None:
            for m in range(M):
                triangle_index = self.__get_Triangle(X[:, m])
                if triangle_index == -1:
                    continue
                inverse_mapping = self.inverse_mappings[triangle_index]
                phi = self.__phi(inverse_mapping(X[:, m]))
                phiY = self.__phi(inverse_mapping(f(
                    X[:, m])))  #Used for EDMD only to calculate C matrix
                global_indices = self.mesh.t[:, triangle_index]
                for i in range(3):
                    for j in range(3):
                        G[global_indices[i],
                          global_indices[j]] += phi[i] * phiY[j]
        else:
            for m in range(M):
                triangle_index = self.__get_Triangle(X[:, m])
                if triangle_index == -1:
                    continue
                inverse_mapping = self.inverse_mappings[triangle_index]
                phi = self.__phi(inverse_mapping(X[:, m]))
                phiY = self.__phi(inverse_mapping(f(
                    X[:, m])))  #Used for EDMD only to calculate C matrix
                global_indices = self.mesh.t[:, triangle_index]
                for i in range(3):
                    for j in range(3):
                        G[global_indices[i], global_indices[j]] += (
                            phi[i] + sigma_noise * _np.random.randn()) * (
                                phiY[j] + sigma_noise * _np.random.randn())
        return G

    def calc_C(self, X, b, sigma, f=None, sigma_noise=None):
        '''
        Calculate the structure matrix <b_k partial_k phi_i,phi_j>,  <Sigma_kl d_xk phi i, d_xl phi j> given the data points X.
        Sums up d_xk Sigma_kl phi_i(x_m)d_xl phi_j(x_m) for all data points x_m and over k,l and adds to relevant entries of C.
        Uses nabla_phi_i(x) = J^{-t} nabla_phi_ref (B^{-1}(x)), where J is the Jacobian of the mapping B from reference to mesh coordinates.
        
          * J^{-1}(x) where J is the Jacobian of the mapping from reference to mesh coordinates.
        '''
        if f is None:  #gEDMD
            M = X.shape[1]
            mesh = self.mesh
            n = mesh.p.shape[1]
            d = self.d
            Y = b(X)
            C = _np.zeros([n, n])

            if sigma_noise is None:
                for m in range(M):
                    triangle_index = self.__get_Triangle(X[:, m])
                    if triangle_index == -1:
                        continue
                    inverse_mapping = self.inverse_mappings[triangle_index]
                    nabla_phi_ref = self.__nabla_phi_ref()
                    J_inv = self.inverse_mapping_jacobians[triangle_index]
                    phi = self.__phi(inverse_mapping(
                        X[:, m]))  # Value of basis functions at x_m
                    nabla_phi = _np.dot(
                        J_inv.T,
                        nabla_phi_ref)  # size d x 3 [partial_xi phi_j(x_m)]
                    for k in range(d):
                        for l in range(d):
                            for i in range(3):
                                for j in range(3):
                                    gi, gj = mesh.t[:,
                                                    triangle_index][_np.array(
                                                        [i,
                                                         j])]  # global indices
                                    C[gi,
                                      gj] += Y[k, m] * nabla_phi[k, i] * phi[j]
                                    if sigma is not None:
                                        C[gi, gj] += -0.5 * sigma[
                                            k, l] * nabla_phi[k,
                                                              i] * nabla_phi[l,
                                                                             j]
            else:
                for m in range(M):
                    triangle_index = self.__get_Triangle(X[:, m])
                    if triangle_index == -1:
                        continue
                    inverse_mapping = self.inverse_mappings[triangle_index]
                    nabla_phi_ref = self.__nabla_phi_ref()
                    J_inv = self.inverse_mapping_jacobians[triangle_index]
                    phi = self.__phi(inverse_mapping(
                        X[:, m]))  # Value of basis functions at x_m
                    nabla_phi = _np.dot(
                        J_inv.T,
                        nabla_phi_ref)  # size d x 3 [partial_xi phi_j(x_m)]
                    for k in range(d):
                        for l in range(d):
                            for i in range(3):
                                for j in range(3):
                                    gi, gj = mesh.t[:,
                                                    triangle_index][_np.array(
                                                        [i,
                                                         j])]  # global indices
                                    C[gi, gj] += (
                                        Y[k, m] +
                                        sigma_noise * _np.random.randn()) * (
                                            nabla_phi[k, i] * phi[j] +
                                            sigma_noise * _np.random.randn())
                                    if sigma is not None:
                                        C[gi, gj] += -0.5 * sigma[k, l] * (
                                            nabla_phi[k, i] +
                                            sigma_noise * _np.random.randn()
                                        ) * (nabla_phi[l, j] +
                                             sigma_noise * _np.random.randn())
        else:
            self.calc_G(X, f=f, sigma_noise=sigma_noise)  #EDMD
        return C

    def __repr__(self):
        return '2D Finite element basis functions on uniform mesh.'


class FEM_1d(object):
    '''
    Finite element basis functions in 1D on uniform mesh with 0 boundary. Only correct for second derivative of the form sigma nabla phi_i nabla phi_j.
    If sigma depends on x, the implementation needs to be adjusted.
    n = number of basis functions, equal to number of vertices
    a, b = bounds of the domain
    '''

    def __init__(self, a, b, n):
        self.n = n
        self.a = a
        self.b = b
        self.h = (b - a) / (n + 1)

    def __call__(self, x, n=None):
        '''
        Evaluate finite element basis functions for all data points in x.
        '''
        n = self.n
        m = x.shape[1]
        y = _np.zeros([n, m])
        h = self.h

        for j in range(m):
            if x[0, j] <= self.a or x[0, j] >= self.b:
                # print('Warning: Data point outside of domain.')
                continue
            i = int((x[0, j] - self.a) /
                    h)  # index of the left vertex, belongs to [0, n]
            if i == 0:
                y[i, j] = (x[0, j] - i * h) / h  # next to left boundary
            elif i == n:
                y[i - 1, j] = 1 - (x[0, j] - i * h) / h  #  right boundary
            else:
                y[i - 1, j] = (x[0, j] - i * h) / h  # linear interpolation
                y[i, j] = 1 - y[i - 1, j]

        return y

    def diff(self, x, n=None):
        '''
        Compute partial derivatives for all data points in x.
        '''
        n = self.n
        m = x.shape[1]
        y = _np.zeros(
            [n, 1, m]
        )  # only one dimension. Stored in this shape for compatibility with other observables
        h = self.h

        for j in range(m):
            if x[0, j] <= self.a or x[0, j] >= self.b:
                # print('Warning: Data point outside of domain.')
                continue
            i = int((x[0, j] - self.a) / h)
            if i == 0:
                y[i, 0, j] = 1 / h
            elif i == n:
                y[i - 1, 0, j] = -1 / h
            else:
                y[i - 1, 0, j] = 1 / h
                y[i, 0, j] = -1 / h

        return y

    def __repr__(self):
        return 'Finite element basis functions on uniform mesh.'


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
