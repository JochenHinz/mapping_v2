#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import linalg
from numba import njit, float64, int64, prange


@njit(
    int64[:](
        float64[:],
        float64[:],
        int64
    )
)
def position_in_knotvector( t, x, xlen ):
    """
    Return the position of ``x`` in the knotvector ``t``.
    If x equals t[-1], return the position before the first
    occurence of x in t.

    Parameters
    ----------
    t: knotvector
    x: vector of positions
    xlen: length of x
    """
    ret = np.empty( xlen, dtype=int64 )

    for i in prange( xlen ):
        for j in range( len( t ) - 1 ):

            # if x equals the last knot, return this
            if x[i] == t[-1]:
                ret[i] = np.where( t == x[i] )[0][0] - 1
                break

            if t[j] <= x[i] < t[j+1]:
                ret[i] = j
                break
        else:
            ret[i] = -1

    return ret


@njit(
    float64(
        int64,
        float64,
        float64[:],
        float64[:],
        int64
    )
)
def deBoor(k, x, t, c, p):
    """
    Evaluates S(x).
    Univariate case only.

    Args
    ----
    k: index of knot interval that contains x
    x: position
    t: knotvector
    c: array of control points
    p: degree of B-spline
    """
    if k == -1:
        return 0

    d = np.empty( p+1, dtype=float64 )

    for i in range( p+1 ):
        d[i] = c[i + k -p]

    for r in prange(1, p+1):
        for j in prange(p, r-1, -1):

            alpha = ( x - t[j + k - p] ) \
                    / ( t[j + 1 + k - r] - t[j + k - p] )
            d[j] = (1.0 - alpha) * d[j-1] + alpha * d[j]

    return d[p]


@njit(
    float64[:](
        float64[:],
        int64,
        float64[:],
        float64[:],
        int64
    )
)
def deBoor_vectorized( x, xlen, t, c, p ):
    """
    Vectorized version of deBoor.

    Parameters
    ----------
    x: Vector of positions
    xlen: length of x
    t: knotvector
    c: array of control points
    p: degree of the B-spline
    """

    ret = np.empty( xlen, dtype=float64 )
    k = position_in_knotvector( t, x, xlen )

    for i in prange( xlen ):
        ret[i] = deBoor( k[i], x[i], t, c, p )

    return ret


@njit(
    float64[:](
        int64,
        float64,
        float64[:],
        int64
    )
)
def nonzero_bsplines( mu, x, t, d ):
    """
    Return the value of the d+1 nonzero basis
    functions at position ``x``.

    Parameters
    ----------
    mu: position in ``t`` containing ``x``
    x: position
    t: knotvector
    d: degree of B-spline basis
    """

    b = np.zeros( d + 1, dtype=float64 )
    b[-1] = 1

    if x == t[-1]:
        return b

    for r in range( 1, d + 1 ):

        k = mu - r + 1
        w2 = ( t[k + r] - x ) / ( t[k + r] - t[k] )
        b[d - r] = w2 * b[d - r + 1]

        for i in range( d - r + 1, d ):
            k = k + 1
            w1 = w2
            w2 = ( t[k + r] - x ) / ( t[k + r] - t[k] )
            b[i] = ( 1 - w1 ) * b[i] + w2 * b[i + 1]

        b[d] = ( 1- w2 ) * b[d]

    return b


@njit(
    float64[:, :](
        float64[:],
        int64,
        float64,
        int64
    )
)
def nonzero_bsplines_deriv( kv, p, x, dx ):  # based on Algorithm 2.3 of the NURBS-book
    """
    Return the value of the d+1 nonzero basis
    functions nd their derivatives up to order ``dx`` at position ``x``.

    Parameters
    ----------
    x: position
    kv: knotvector
    p: degree of B-spline basis
    dx: max order of the derivative
    """

    i = position_in_knotvector( kv, np.array([x], dtype=float64), 1 )[0]
    ndu = np.empty( (p + 1, p + 1), dtype=float64 )
    a = np.empty( (p + 1, p + 1), dtype=float64 )
    ders = np.empty( (dx + 1, p + 1) )

    left = np.empty( (p + 1, ), dtype=float64 )
    right = np.empty( (p + 1, ), dtype=float64 )

    ndu[0, 0] = 1.0

    for j in range(1, p+1):
        left[j] = x - kv[i + 1 - j]
        right[j] = kv[i + j] - x
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]

            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    for j in range(p + 1):
        ders[0, j] = ndu[j, p]

    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, dx + 1):
            d = 0.0
            rk, pk = r - k, p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range( j1, j2 + 1 ):
                a[s2, j] = ( a[s1, j] - a[s1, j - 1] ) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            j, s1, s2 = s1, s2, j

    r = p
    for k in range(1, dx + 1):
        for j in range(p+1):
            ders[k, j] *= r
        r *= p - k

    return ders


@njit(
    float64[:](
        float64[:],
        int64,
        int64,
        float64,
        int64
    )
)
def der_ith_basis_fun( kv, p, i, x, dx ):  # based on algorithm A2.5 from the NURBS-book

    """
    Return the N_ip(x) and its derivatives up to ``dx``,
    where N denotes the ``i``-th basis function of order
    ``p`` resulting from knotvector ``kv`` and x the position.

    Parameters
    ----------
    kv: knotvector
    p: degree of the basis
    i: index of the basis function
    x: position
    dx: highest-order derivative

    XXX: For some reason all basis functions that are supported
    on the first element don't seem to work.
    The zeroth-order derivative is fine but all others give wrong
    values compared to the scipy counterpart.
    All other basis functions seem to work.
    """

    basis_len = len(kv) - p - 1

    if x < kv[i] or x >= kv[i + p + 1]:
        if i != basis_len - 1 or x > kv[-1]:
            ''' x lies outside of the support of basis function or domain '''
            return np.zeros( (dx + 1, ), dtype=float64 )
        if i == basis_len - 1 and x == kv[-1]:
            '''
            special case: evaluation of the last basis function
            in the last point of the interval. Return a sequence
            (p / a_0) ** 0, (p / a_1) ** 1, ... (p / a_dx) ** dx
            '''
            # a = 1
            # ret = np.empty( (dx + 1, ), dtype=float64 )
            # for i in range( ret.shape[0] ):
            #     ret[i] = a
            #     if i != ret.shape[0] - 1:
            #         a *= p / ( kv[basis_len - 1 + p - i] - kv[basis_len - 1 - i] ) 
            # return ret
            x -= 1e-15



    ders = np.empty( (dx + 1, ), dtype=float64 )
    N = np.zeros( (p + 1, p + 1), dtype=float64 )

    for j in range(p + 1):
        if ( x >= kv[i + j] and x < kv[i + j + 1] ):
            N[j, 0] = 1.0
        else:
            N[j, 0] = 0.0

    for k in range(1, p + 1):
        saved = 0.0 if N[0, k - 1] == 0.0 else \
            (x - kv[i]) * N[0, k - 1] / (kv[i + k] - kv[i])
        for j in range(p - k + 1):
            Uleft, Uright = kv[i + j + 1], kv[i + j + k + 1]
            if N[j + 1, k - 1] == 0:
                N[j, k], saved = saved, 0
            else:
                temp = N[j + 1, k - 1] / (Uright - Uleft)
                N[j, k] = saved + (Uright - x) * temp
                saved = (x - Uleft) * temp

    ders[0] = N[0, p]
    ND = np.zeros( (k + 1, ), dtype=float64 )
    for k in range(1, dx + 1):
        for j in range(k + 1):
            ND[j] = N[j, p - k]
        for jj in range(1, k + 1):
            saved = 0.0 if ND[0] == 0.0 else ND[0] / (kv[i + p - k + jj] - kv[i])
            for j in range(k - jj + 1):
                Uleft, Uright = kv[i + j + 1], kv[i + j + p + jj + 1]
                if ND[j + 1] == 0.0:
                    ND[j], saved = (p - k + jj) * saved, 0.0
                else:
                    temp = ND[j + 1] / (Uright - Uleft)
                    ND[j] = (p - k + jj) * (saved - temp)
                    saved = temp

        ders[k] = ND[0]

    return ders


@njit(
    float64[:,:](
        float64[:],
        int64,
        int64,
        float64[:],
        int64
    )
)
def der_ith_basis_fun_vectorized( kv, p, i, x, dx ):

    """
    Vectorized version of ``der_ith_basis_fun``
    Returns a matrix of length containing the function
    evaluations of the i-th entry in ret[i, :]
    """

    ret = np.zeros( ( x.shape[0], dx + 1 ), dtype=float64 )

    for j in prange( ret.shape[0] ):
        ret[j, :] = der_ith_basis_fun( kv, p, i, x[j], dx )

    return ret


def call_to_basis( kv, p, x, dx ):

    k = dx.max()

    float64 = np.float64

    basis_len = len(kv) - p - 1

    evaluations = np.empty( (basis_len, len(x), len(dx)), dtype=float64 )

    for i in prange( basis_len ):
        evaluations[i] = der_ith_basis_fun_vectorized( kv, p, i, x, k )[:, dx]

    return evaluations


@njit(
    float64[:](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        int64,
        int64,
        float64[:],
        int64,
        int64
    )
)
def _call( xi, eta, kv0, kv1, p0, p1, x, dx, dy ):
    """
    Return function evaluations at positions (xi, eta).
    ``xi`` and ``eta`` must have equal lengths.

    Parameters
    ----------
    xi: Vector of xi-values
    eta: Vector of eta-values
    kv0: knotvector in xi-direction
    kv1: knotvector in eta-direction
    p0: degree in xi-direction
    p1: degree in eta-direction
    x: vector of control points
    """

    n_eta = kv1.shape[0] - p1 - 1

    ret = np.zeros( xi.shape, dtype=float64 )

    element_indices0 = position_in_knotvector( kv0, xi, len(xi) )
    element_indices1 = position_in_knotvector( kv1, eta, len(eta) )

    for i in prange( len(xi) ):

        mu0, mu1 = element_indices0[i], element_indices1[i]

        xi_calls = nonzero_bsplines_deriv( kv0, p0, xi[i], dx )[dx, :]
        eta_calls = nonzero_bsplines_deriv( kv1, p1, eta[i], dy )[dy, :]

        for j in range( p0 + 1 ):

            a = xi_calls[j]

            for k in range( p1 + 1 ):

                b = eta_calls[k]
                global_index = (mu0 - p0 + j) * n_eta + mu1 - p1 + k

                ret[i] += x[global_index] * a * b

    return ret


def call( g, xi, eta, dx=0, dy=0 ):
    """
    Call to the mapping function of a TensorGridObject.
    Currently limited to bivariate TGO`s.

    Parameters
    ----------
    g: TensorGridObject
    xi: xi-values of the call (flat vector)
    eta: eta-values of the call (flat vector)
    dx: order of the derivative in the xi-direction
    dy: order of the derivative ins the eta-direction

    Returns
    -------
    np.array of dimension ( len(xi), g.targetspace )
    containing the evaluations of the x, y, ... -components
    of the mapping function.

    """

    DX = ( dx, dy )

    xi = np.clip( xi, g.knots[0][0], g.knots[0][-1] )
    eta = np.clip( eta, g.knots[1][0], g.knots[1][-1] )

    assert len(g) == 2, NotImplementedError
    assert all( d_ <= p for d_, p in zip( DX, g.degree ) )

    vecs = np.array_split( g.x, g.targetspace )

    make = lambda *args: np.stack( [ _call(*args, *g.extend_knots(),
        *g.degree, vec, *DX) for vec in vecs ], axis=1 )

    return make( xi, eta )


@njit(
    float64[:, :](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        int64,
        int64,
        float64[:],
        int64,
        int64
    )
)
def _tensor_call( xi, eta, kv0, kv1, p0, p1, x, dx, dy ):
    """
    Return function evaluations at all positions
    (xi_i, eta_i) in the outer product of univariate
    positions ``xi`` and ``eta``.
    Optimized because the bases are only evaluated in
    ``xi`` and ``eta`` once.
    Returns a matrix instead of a flat vector.

    Parameters
    ----------
    xi: Vector of xi-values
    eta: Vector of eta-values
    kv0: knotvector in xi-direction
    kv1: knotvector in eta-direction
    p0: degree in xi-direction
    p1: degree in eta-direction
    x: vector of control points
    dx: order of the derivative in the xi-direction
    dy: order of the derivative ins the eta-direction
    """

    mu0 = position_in_knotvector( kv0, xi, len(xi) )
    mu1 = position_in_knotvector( kv1, eta, len(eta) )

    n_eta = kv1.shape[0] - p1 - 1

    s0, s1 = p0 + 1, p1 + 1

    xi_evals = np.empty( s0 * xi.shape[0], dtype=np.float64 )
    eta_evals = np.empty( s1 * eta.shape[0], dtype=np.float64 )

    for i in prange( len(xi) ):
        xi_evals[ i * s0: (i + 1) * s0 ] = \
                nonzero_bsplines_deriv( kv0, p0, xi[i], dx )[dx, :]

    for i in prange( len(eta) ):
        eta_evals[ i * s1: (i + 1) * s1 ] = \
                nonzero_bsplines_deriv( kv1, p1, eta[i], dy )[dy, :]

    ret = np.zeros( ( len(xi), len(eta) ), dtype=np.float64 )

    for i in prange( len(xi) ):
        for j in prange( len(eta) ):

            local_xi = xi_evals[ i * s0: (i + 1) * s0 ]
            local_eta = eta_evals[ j * s1: (j + 1) * s1 ]

            for k in range( s0 ):

                a = local_xi[k]

                for l in range( s1 ):

                    b = local_eta[l]

                    global_index = (mu0[i] - p0 + k) \
                        * n_eta \
                        + mu1[j] - p1 + l

                    ret[i, j] += x[global_index] * a * b

    return ret


def tcall( g, xi, eta, dx=0, dy=0 ):

    DX = ( dx, dy )

    xi = np.clip( xi, g.knots[0][0], g.knots[0][-1] )
    eta = np.clip( eta, g.knots[1][0], g.knots[1][-1] )

    assert len(g) == 2, NotImplementedError
    assert all( d_ <= p for d_, p in zip( DX, g.degree ) )

    vecs = np.array_split( g.x, g.targetspace )

    make = lambda *args: np.stack( [ _tensor_call(*args, *g.extend_knots(),
        *g.degree, vec, *DX) for vec in vecs ], axis=2 )

    return make( xi, eta )


@njit(
    int64[:](
        float64[:],
        int64,
        float64[:]
    )
)
def element_to_supported_basis_functions( t, p, element_boundaries ):

    indices = []
    indices_length = [0]
    basis_length = t.shape[0] - p - 1
    start = 0

    for i in range( len(element_boundaries) - 1 ):

        left, right = element_boundaries[i], element_boundaries[i+1]
        length = 0
        indicator = 0

        for j in range( start, basis_length ):

            support = set( t[j: j + p + 2] )

            if ( left in support ) and ( right in support ):

                indices.append(j)
                length += 1
                indicator = 1

            else:
                if indicator:
                    break

        indices_length.append( length )
        start = indices[ -indices_length[-1] ]

    indices = np.array( indices, dtype=int64 )
    indices_length = np.array( indices_length, dtype=int64 ).cumsum()

    """
    The vector giving the number of nonzero basisfunctions per element
    is extracted by taking result[ -nelements: ]
    """
    return np.concatenate( ( indices, indices_length ) )


def test():
    from . import std
    g = std.Cube().add_c0( [ [0.5], [] ] )
    g.x += 0.01 * np.random.randn( len( g.x ) )
    xi = np.linspace( 0, 1, 101 )
    eta = np.linspace( 0, 1, 51 )
    Y, X = np.meshgrid( eta, xi )
    test_call = lambda xi, eta:\
            call( xi, eta, *g.extend_knots(), *g.degree, g.x[ :len( g.x )//2 ] )
    X, Y = X.ravel(), Y.ravel()
    return g, xi, eta, X, Y, test_call


@njit(
    float64[:](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        int64,
        int64,
        float64[:, :],
        float64[:],
        float64[:, :]
    )
)
def assemble_from_svd( xi, eta, kv0, kv1, p0, p1, U, S, V ):
    """
    Assemble a flat vector containing the function evaluations
    of (xi, eta) from a SVD of the control points

    Parameters
    ----------
    xi: positions in the xi-direction
    eta: positions in the eta-direction
    kv0: knotvector in the xi-direction
    kv1: knotvector in the eta-direction
    p0: degree in the xi-direction
    p1: degree in the eta-direction
    U: U in X = U @ diag( S ) @ V
    S: S in X = U @ diag( S ) @ V
    V: V in X = U @ diag( S ) @ V
    """

    n = xi.shape[0]
    k = S.shape[0]

    ret = np.zeros( n, dtype=float64 )

    for i in range( k ):
        ret += S[i] * deBoor_vectorized( xi, len(xi), kv0, U[:, i], p0 ) \
                    * deBoor_vectorized( eta, len(eta), kv1, V[i], p1 )

    return ret


def multivariate_evalf( g, xi, eta, k=None ):

    assert len( xi ) == len( eta )
    kvs = g.extend_knots()
    ps = g.degree
    xis = [ xi, eta ]

    X = g.x[ :len( g.x )//2 ].reshape( g.ndims )

    if k:
        U, S, V = linalg.svds( X, k=k )
    else:
        U, S, V = np.linalg.svd( X )

    return assemble_from_svd( *xis, *kvs, *ps, U, S, V  )
