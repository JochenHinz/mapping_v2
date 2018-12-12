#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import abc
from . import ko, go
from numba import jit, njit, int64, float64, prange
from functools import wraps, reduce, partial
from scipy import interpolate, sparse, optimize
from nutils import matrix, log
from collections import OrderedDict

NUMBA_WARNINGS = 1
NUMBA_DEBUG_ARRAY_OPT_STATS = 1


"""
    XXX: O-grid functionality added. Implement the jacobians
    for all methods.
"""


def blockshaped( arr, nrows, ncols ):
    """
        Array ``arr`` of shape ( h, w ) becomes of shape
        ( -1, nrows, ncols ), where arr_reshaped[i] contains the entries of
        the i-th ( h // nrows, w // ncols ) block (going from left to right,
        top to bottom) in ``arr``.
    """
    h, w = arr.shape
    return arr.reshape(h // nrows, nrows, -1, ncols) \
                            .swapaxes(1, 2).reshape(-1, nrows, ncols)


def make_quadrature( g, order ):

    if len( g ) > 2:
        raise NotImplementedError( 'Dimensionalities >2 are currently \
                                                        not supported.' )

    x, y = np.polynomial.legendre.leggauss( order )

    make_quad = lambda k: \
        ( np.kron( k[1:] - k[:-1], x ) +
            np.kron( k[1:] + k[:-1], np.ones(len(x)) ) ) / 2

    quad = [ make_quad(k) for k in g.knots ]

    _weight_vector = lambda k: \
                            np.kron( np.ones( len(k) - 1 ), y ) \
                            * np.repeat( ( k[1:] - k[:-1] ) / 2, len(y) )

    weights = [ _weight_vector(k) for k in g.knots ]

    slices = [ (None,)*i + (slice(None),) + (None,)*( g.targetspace-i-1 )
                                            for i in range(g.targetspace) ]

    weights = np.prod( [ c[s] for c, s in zip( weights, slices ) ] )

    return quad, weights, len( x )


def make_supports( knotvector: ko.KnotObject ):

    """
        Return tuple of tuples containing the indices on which the basis
        functions are nonvanishing (for each coordinate-direction).
        Will also work with ``ko.TensorKnotVector`` input argument.
    """

    p = knotvector.degree

    def _knotspan( knotobject ):
        kv, km = knotobject.knots, knotobject.knotmultiplicities
        ret = np.repeat( np.arange(len(kv) - 1, dtype=int), km[:-1] )

        if knotobject.periodic:
            p = knotobject.degree
            km0 = km[0]
            head = ret[ -p - 1 + km0: ]
            ret = np.concatenate( [ head, ret ] )

        return ret

    # same as g.extend_knots(), however with g.knots replaced by
    # ascending integers
    knotspan = _knotspan( knotvector )
    supports = tuple( np.unique( knotspan[i: i+p+1] ) for i in range(knotvector.dim) )

    return supports


def neighbours( supports, p ):

    """
        Given a list of array-likes containing the indices that
        support all (univariate) basis functions, generate a list
        of lists containing the indices of the basis functions
        that share common elements.
        XXX: possibly make this jit-compiled.
    """

    N = len( supports )

    LoL = []
    for i, s in enumerate( supports ):
        neighbours = []
        inner = np.arange( i-p, i+p+1, dtype=int ) % N
        for j in inner:
            try:
                support = supports[j]
            except IndexError:
                continue
            if len( np.intersect1d( support, s ) ) != 0:
                neighbours.append( j )
        LoL.append( sorted(neighbours) )
    return LoL


def tensor_neighbours( *neighbours ):

    """
        Generate tensor-lil-index from a list of univariate
        lil-indices.
        This is accomplished by building pseudo-matrices
        (containing only ones and zeros) from each univariate
        lil-index and taking the Kronecker-product.
        XXX: possibly make this jit-compiled.
    """

    mats = []
    for neigh in neighbours:
        N = len( neigh )
        mat = sparse.lil_matrix( (N, N) )
        mat.rows = np.array([ n for n in neigh ], dtype=object )
        mat.data = np.array([ np.ones( len(r) ) for r in mat.rows ], dtype=object )
        mats.append( mat )

    M = reduce( sparse.kron, mats ) if len( mats ) > 1 else mat

    '''
        for some reason M.rows() gives the wrong number of nonzeros
        however, M.nonzero() works fine.
        possibly write some jit-compiled functionality for this in
        the long run.
    '''

    rows = [ [] for i in range(M.shape[0]) ]
    for i, j in zip( *M.nonzero() ):
        rows[i].append(j)

    return np.array( rows, dtype=object )


def make_lilrows( g ):

    """
        Generate a list of lists that can be utilized for
        M.rows of the Mass-matrix M corresponding the basis
        in lil-format.
        XXX: possibly make this jit-compiled.
    """

    # make supports and slice them
    supports = [ make_supports(k) for k in g.knotvector ]
    # make list of univariate neighbours
    uneighbours = [ neighbours(s, p)
                                for s, p in zip(supports, g.degree) ]
    # take tensor-product
    tneighbours = tensor_neighbours( *uneighbours )

    return tneighbours


""" Jitted functions """


@jit(
    float64[:](
        int64,
        int64
    ),
    nopython=True
)
def unit_vector( N, i ):
    """ Return i-th unit vector of length N """
    uv = np.zeros( N )
    uv[ i ] = 1
    return uv


@jit(
    int64[:](
        int64[:],
        int64[:]
    ),
    nopython=True
)
def intersection( arr1, arr2 ):
    indices = []
    for i in arr1:
        idx = np.where( arr2 == i )[0]
        if len( idx ) > 0:
            indices.append( idx[0] )
    return arr2[ np.array( indices, dtype=int64 ) ]


@njit(
    float64[:](
        int64,
        float64[:, :, :],
        float64[:, :, :],
        int64[:],
        int64[:]
    ),
    parallel=True,
    nogil=True,
    fastmath=True
)
def jitarray( N, ws, quadweights, elemstart, elems ):
    """
        Compute a jitted array

        Parameters
        ----------
        N: length of the array
        ws: function evaluations (W's) in blockshape-format (see below)
        quadweights: element-wise quadrature weights in blockshape-format
        elemstart: global indices of the elements that contribute to the
            i-th entry are given by elemstart[i]: elemstart[i + 1]
        elems: sorted list of element indices, where
            elems[ elemstart[i]: elemstart[i + 1] ][k] gives the global
            index of the k local element corresponding to the i-th basis
            function.

        Returns
        -------
        np.ndarray

    """
    ret = np.zeros( N, dtype=np.float64 )
    for i in prange( N ):
        """ compute i-th entry """
        start = elemstart[i]
        local_to_global_elements = elems[ start: elemstart[i + 1] ]
        n = len( local_to_global_elements )
        sums = np.empty(n)
        for k in prange(n):
            """
                Loop over all elements that contribute to the i-th entry.
                They are given by the ascending list of element support
                indices of the i-th basis function:
                        elements = elems[ elemstart[i]: elemstart[i+1] ].
            """
            # the quadrature point function evaluations of the i-th basis-function
            # w_i over the k-th local element ( 0, ..., len( support(w_i) ) - 1 ) are
            # given by ws[ elem ] with elem = start + k
            elem = start + k
            # the corresponding quadrature weights ``w`` are given by the quadrature
            # weights ``quadweights`` evaluated in the global index corresponding
            # to local element index k.
            # This is simply given by local_to_global_elements[ k ].
            w = quadweights[ local_to_global_elements[k] ]
            sums[k] = ( ws[elem] * w ).sum()
        ret[i] = sums.sum()
    return ret


@njit(
    float64[:](
        int64,
        int64,
        float64[:, :, :],
        float64[:, :, :],
        float64[:, :, :],
        int64[:],
        int64[:],
        int64[:],
        int64[:]
    ),
    nogil=True,
    parallel=True,
    fastmath=True
)
def jitmass( N, m, ws0, ws1, quadweights, elemstart, elems, lilstart, lils ):
    """
        Compute a jitted mass-matrix-like array.
        Here, mass-matrix-like refers to matrices with the same sparsity-pattern.

        Parameters
        ----------
        N: dimension of the jitted matrix
        m: dimension of the spline-basis in the eta-direction
        ws0: test space in blockshape-format
        ws1: trial space in blockshape-format
        quadweights: quadrature weights in blockshape-format
        elemstart: global indices of the elements that contribute to the
            i-th entry are given by elemstart[i]: elemstart[i + 1]
        elems: sorted list of element indices, where
            elems[ elemstart[i]: elemstart[i + 1] ][k] gives the global
            index of the k-th local element corresponding to the i-th basis
            function.
        lilstart: see below
        lils: the nonzero entries in the i-th row of the matrix correspond to
            lils[ lilstart[i]: lilstart[i+1] ]

        Returns
        -------

        np.ndarray
            suitable for the construction of a matrix in
            scipy.sparse.lil_matrix-format

    """
    ret = np.zeros( len(lils), dtype=np.float64 )
    for i in prange( N ):
        start = lilstart[i]
        current = 0
        inner = lils[ start: lilstart[i + 1] ]
        for j in inner:
            elems0 = elems[ elemstart[i]: elemstart[i + 1] ]
            elems1 = elems[ elemstart[j]: elemstart[j + 1] ]
            inter = intersection( elems0, elems1 )
            res = 0
            for k in inter:
                elem0 = elemstart[i] + np.where( elems0 == k )[0][0]
                elem1 = elemstart[j] + np.where( elems1 == k )[0][0]
                res += ( ws0[elem0] * ws1[elem1] * quadweights[k] ).sum()
            ret[ start + current ] = res
            current += 1
    return ret


""" Various auxilliary functions for the FastSolver class """


class HashAbleArray( np.ndarray ):

    """
        Hashable array for use in ``cache``.
    """

    def __new__( cls, data ):
        assert len( data.shape ) == 1
        return data.view( cls )

    def __hash__( self ):
        return tuple.__hash__( tuple(self) )

    def __eq__( self, other ):
        if not self.shape == other.shape:
            return False
        return np.ndarray.__eq__( self, other ).all()


class InstanceMethodCache:

    def __init__( self, func ):
        self._func = func
        self._cache_name = '_' + func.__name__ + '_cache'

    def __get__( self, obj, objtype=None ):
        if obj is None:
            return self._func
        return partial( self, obj )

    def __call__( self, *args, **kwargs ):
        obj, c, *args = args
        c = HashAbleArray( c )
        try:
            cache = getattr( obj, self._cache_name )
        except AttributeError:
            cache = OrderedDict()
            setattr( obj, self._cache_name, cache )
        try:
            ret = cache[c]
        except KeyError:
            ret = cache[c] = self._func( obj, c, *args, **kwargs )

        if len( cache ) > 2:
            for i in list( cache.keys() )[ :2 ]:
                del cache[ i ]

        return ret


def with_boundary_conditions( f ):

    @wraps( f )
    def wrapper( self, c ):
        g = self._g
        vec = g.cons.copy()
        vec[ g.dofindices ] = c
        ret = f( self, vec )
        try:
            self._feval += 1
        except AttributeError:
            self._feval = 1
        return ret[ g.dofindices ]

    return wrapper


class SecondOrderKrylovJacobian( optimize.nonlin.KrylovJacobian ):

    '''
        Inherited from KrylovJacobian but
        allows for O( h^2 )-accurate FD-approximations.
    '''

    def matvec(self, v):
        nv = np.linalg.norm(v)
        if nv == 0:
            return 0*v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc*v) - self.func(self.x0 - sc*v)) / (2 * sc)
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r


def root( I, init=None, jacobian=1, jac_options=None, **scipyargs ):

    scipyargs.setdefault( 'verbose', True )
    scipyargs.setdefault( 'maxiter', 50 )

    res = I.residual

    if init is None:
        init = I._g.x[ I._g.dofindices ]

    if jac_options is None:
        jac_options = {}

    if jacobian == 1:
        jac = optimize.nonlin.KrylovJacobian( **jac_options )
    elif jacobian == 2:
        jac = SecondOrderKrylovJacobian( **jac_options )
    else:
        jac = jacobian( **jac_options )

    log.info( 'solving system with maxiter={}'.format( scipyargs['maxiter'] ) )

    return optimize.nonlin.nonlin_solve( res, init, jacobian=jac, **scipyargs )


def clip_from_zero( A, eps=1e-6 ):
    A[ A < 0 ] = np.clip( A[ A < 0 ], -np.inf, -eps )
    A[ A > 0 ] = np.clip( A[ A > 0 ], eps, np.inf )
    A[ A == 0 ] = eps
    return A


class WEvalDict( dict ):

    ''' Dictionary that only allows for relevant keys '''

    _keynames = { 'w', 'x', 'y', 'xx', 'xy', 'yy' }

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        assert len( self._keynames - set( self.keys() ) ) <= len( self._keynames )

    def __setitem__( self, key, value ):
        assert key in self._keynames
        dict.__setitem__( self, key, value )


def tck( g ):

    '''
        tck function that allows for periodic gridobjects
        (only in one direction).
    '''

    knots = g.extend_knots().copy()
    periodic, p, s = g.periodic, g.degree, list( g.ndims )

    assert len( periodic ) in (0, 1)

    nperiodicfuncs = [ p_ - g.knotmultiplicities[j][0] + 1
        for j, p_ in enumerate(p) ]
    slicedict = { 0: ( slice( 0, nperiodicfuncs[0] ), slice( None ) ),
           1: ( slice( None ), slice( 0, nperiodicfuncs[1] ) ) }
    for i in periodic:
        sl = slicedict[i]
        # this is necessary at the moment because the expand is not cut
        # on the right side if knot multiplicities are present at xi=0
        # change this in the future.
        knots[i] = knots[i][ : np.where( knots[i] == 1 )[0][0] + p[i] + 1 ]

    def _f( c ):
        if len(periodic):
            i = periodic[0]
            c = \
                np.concatenate( [ c.reshape(s), c.reshape(s)[sl] ],
                    axis=i ).ravel()
        return c

    def func( c ):
        return ( *knots, _f(c), *p )

    return func


class FastSolver( abc.ABC ):

    """ Main class for integration """

    def _tck( self, c ):
        if not hasattr( self, '__tck' ):
            self.__tck = tck( self._g )
        return self.__tck( c )

    def _splev( self, i, quad, **scipyargs ):

        ''' Function evaluation for building the test & trial space tabulation '''

        assert i < self._N
        return interpolate.bisplev( *quad,
                self._tck(unit_vector(self._N, i)), **scipyargs )

    def _w_eval( self, item ):

        _dictmaker = lambda a, b: { 'dx': a, 'dy': b }

        kwargs = { 'w': _dictmaker( 0, 0 ),
                   'x': _dictmaker( 1, 0 ),
                   'y': _dictmaker( 0, 1 ),
                   'xx': _dictmaker( 2, 0 ),
                   'xy': _dictmaker( 1, 1 ),
                   'yy': _dictmaker( 0, 2 ) }[ item ]

        values = \
            (
                blockshaped( w, *self._chunklengths ) for w in
                tuple( self._splev( i, self._chunks[i], **kwargs )
                for i in range(self._N) )
            )

        return np.concatenate( list(values) )

    def _set_LoL( self ):

        """
            Generate list of lists ``LoL``,
            where LoL[i] contains the global indices of the
            elements supporting the i-th basis function.
            Assigns LoL to self._LoL and a flattened version
            to self._LoL_flat
        """

        supports = self._supports
        ndims = self._g.ndims
        n, m = [ len(k) - 1 for k in self._g.knots ]
        f = lambda i, j: i * m + j  # global element index
        LoL = []
        for i in range( self._N ):
            L = []
            # select the univariate supports of the i-th basis function
            xi, eta = [ s[i_] for s, i_ in zip(supports, np.unravel_index(i, ndims)) ]
            for x in xi:
                for y in eta:
                    L.append( f(x, y) )
            LoL.append( sorted(L) )
        self._LoL = LoL
        self._LoL_flat = np.concatenate( LoL )

    def __init__( self, g, order ):

        if len( g ) != 2:
            raise NotImplementedError

        assert order >= 2

        self._g = g
        self._N = len( self._g.basis )
        self._feval = 0  # counts the number of residual evaluations

        self._supports = tuple( make_supports(k) for k in g.knotvector )
        self._quad, self._weights, c = make_quadrature( g, order )
        self._chunklengths = [c] * len(g)
        self._quadweights = blockshaped( self._weights, *self._chunklengths )

        # _chunks[i] corresponds to all nonzero quadrature points of w_i
        # *_chunks[i] can be utilized in the self._tck function
        # XXX: try to replace this using numpy strides

        # reshape the i-th univariate quadrature points into shape (-1, c)
        # and select the rows that correspond to the univariate supports
        # of the i-th basis function.

        def chunk(i):
            ndims = g.ndims

            localindex = ( s[i_] for s, i_
                in zip(self._supports, np.unravel_index(i, ndims)) )

            return \
                tuple(
                    q.reshape([-1, c])[ locindex ].ravel() for q, locindex in
                    zip( self._quad, localindex )
                )

        self._chunks = tuple( chunk(i) for i in range(self._N) )

        self._w = WEvalDict()
        self._set_LoL()  # sets self._LoL and self._LoL_flat

        self._lilrows = make_lilrows( g )

        self._lilstart = \
            np.array( [0] + [ len(l) for l in self._lilrows ] ).cumsum()

        self._lilrows_flat = np.concatenate( self._lilrows ).astype( np.int64 )

        self._jitsupportlength = \
            np.array( [0] + [ len(l) for l in self._LoL ] ).cumsum()

    @property
    def M( self ):
        if not hasattr( self, '__M' ):
            self.__M = self.jitmass()
        return self.__M

    """ Magic functions """

    def __getitem__( self, key ):

        '''
            Return blockshaped function evaluations (for test and trial spaces).
        '''
        try:
            ret = self._w[key]
        except KeyError:
            ret = self._w_eval(key)
            log.info( key + ' has been tabulated.' )
            self._w[key] = ret
        except Exception as ex:
            raise Exception( 'Failed with unknown exception {}.'.format(ex) )
        finally:
            return ret

    def __call__( self, c, **kwargs ):

        '''
            Call is overloaded to simply yield the mapping evaluated in c
            (or its derivatives).
        '''

        return interpolate.bisplev( *self._quad, self._tck(c), **kwargs )

    """ Various evaluations of the mapping function """

    @InstanceMethodCache
    def zderivs( self, c ):
        ''' zeroth order derivative '''
        return self( c )

    @InstanceMethodCache
    def fderivs( self, c ):
        ''' first derivatives '''
        return [ self( c, dx=1 ), self( c, dy=1 ) ]

    @InstanceMethodCache
    def all_fderivs( self, c ):
        cs = np.array_split( c, 2 )
        return [ self.fderivs(c_) for c_ in cs ]

    @InstanceMethodCache
    def sderivs( self, c ):
        ''' second derivatives '''
        kwargs = ( { 'dx': 2 }, { 'dx': 1, 'dy': 1 }, { 'dy': 2 } )
        return [ self( c, **k ) for k in kwargs ]

    @InstanceMethodCache
    def metric( self, c ):
        ''' [ g11, g12, g22 ] '''
        cs = np.array_split( c, 2 )
        (x_xi, x_eta), (y_xi, y_eta) = [ self.fderivs(c_) for c_ in cs ]
        return \
            [
                x_xi ** 2 + y_xi ** 2,
                x_xi * x_eta + y_xi * y_eta,
                x_eta ** 2 + y_eta ** 2
            ]

    @InstanceMethodCache
    def jacdet( self, c ):
        cs = np.array_split( c, 2 )
        (x_xi, x_eta), (y_xi, y_eta) = [ self.fderivs(c_) for c_ in cs ]
        return x_xi * y_eta - x_eta * y_xi

    """ Transformations for lil-format """

    def tolil( self, vec ):
        return np.array( [ arr.tolist() for arr in
            np.array_split( vec, self._lilstart[1: -1] ) ], dtype=object )

    def vec_to_mat( self, vec ):
        mat = sparse.lil_matrix( (self._N,) * 2 )
        mat.data = vec
        mat.rows = self._lilrows
        return mat.tocsr()

    """ Jitted arrays """

    def jitmass( self, mul=None, test=None, trial=None ):

        if test is None:
            test = 'w'

        if trial is None:
            trial = 'w'

        if mul is not None:
            weights = self._quadweights * blockshaped( mul, *self._chunklengths )
        else:
            weights = self._quadweights

        v = self[test]
        w = self[trial]

        m = len( self._g.knots[1] ) - 1
        arr = jitmass( self._N, m, v, w, weights, self._jitsupportlength,
                self._LoL_flat, self._lilstart, self._lilrows_flat )
        return self.vec_to_mat( self.tolil(arr) )

    def jitarray( self, mul=None, w='w' ):
        if mul is not None:
            weights = self._quadweights * blockshaped( mul, *self._chunklengths )
        else:
            weights = self._quadweights
        w = self[w]
        return jitarray( self._N, w, weights, self._jitsupportlength, self._LoL_flat )

    """ Additional functionality """

    def project( self, mul, cons=None ):

        nutilsargs = { 'constrain': cons } if cons is not None else {}

        M = matrix.ScipyMatrix(self.M)
        rhs = self.jitarray( mul=mul )

        return M.solve( rhs, **nutilsargs )

    """ Abstract Methods """

    @abc.abstractmethod
    def residual( self, *args, **kwargs ):
        pass

    """ System solve """

    solve = root  # overwrite this in case of MixedFEM


""" The following needs some further tweaking, reducing the total lines of code """


class Elliptic_unscaled( FastSolver ):

    @with_boundary_conditions
    def residual( self, c ):

        cs = np.array_split( c, 2 )

        g11, g12, g22 = self.metric(c)
        dx, dy = [ self.sderivs(c_) for c_ in cs ]

        mul0 = g22 * dx[0] - 2 * g12 * dx[1] + g11 * dx[2]
        mul1 = g22 * dy[0] - 2 * g12 * dy[1] + g11 * dy[2]

        return np.concatenate( [self.jitarray(mul=mul0), self.jitarray(mul=mul1)] )


class Elliptic( FastSolver ):

    def __init__( self, *args, eps=0.001, rhs=None, **kwargs ):
        super().__init__( *args, **kwargs )
        self._eps = eps

        if rhs is None:
            rhs = np.zeros_like( self._g.x[ self._g.dofindices ] )

        rhs_ = np.zeros( len( self._g.x ) )
        rhs_[ self._g.dofindices ] = rhs
        self._rhs = rhs_

    @with_boundary_conditions
    def residual( self, c ):

        cs = np.array_split( c, 2 )

        g11, g12, g22 = self.metric(c)
        dx, dy = [ self.sderivs(c_) for c_ in cs ]

        scale = g11 + g22 + self._eps

        mul0 = ( g22 * dx[0] - 2 * g12 * dx[1] + g11 * dx[2] ) / scale
        mul1 = ( g22 * dy[0] - 2 * g12 * dy[1] + g11 * dy[2] ) / scale

        return np.concatenate( [self.jitarray(mul=mul0), self.jitarray(mul=mul1)] ) \
            + self._rhs


class Elliptic_partial( FastSolver ):

    @with_boundary_conditions
    def residual( self, c ):

        arr = self.jitarray

        jacdet = clip_from_zero( self.jacdet(c) )
        g11, g12, g22 = self.metric(c)

        ret = \
            np.concatenate([
                arr( g22 / jacdet, 'x' ) - arr( g12 / jacdet, 'y' ),
                -arr( g12 / jacdet, 'x' ) + arr( g11 / jacdet, 'y' )
            ])

        return ret


class EllipticControl( Elliptic, abc.ABC ):

    @abc.abstractmethod
    def _tablulate_control_mapping( self ):
        pass

    def __init__( self, g, order, f=None, **kwargs ):
        if order < 6:
            log.warning(
                '''
                    Warning, for this method a Gauss-scheme of at least
                    order 6 is recommended.
                '''
            )
        super().__init__( g, order, **kwargs )

        if isinstance( f, go.TensorGridObject ):

            if len( f ) != 2:
                raise NotImplementedError

            if not ( f.knotvector <= g.knotvector ).all():
                log.warning(
                    '''
                        Warning, using a control mapping whose knotvector is
                        not a subset of the target GridObject knotvector
                        or that does not have the same periodicity properties
                        is not recommended, since Gaussian quadrature is
                        ill-defined in this case.
                    '''
                )

            if any( (p in km) for p, km in zip( f.degree, f.knotmultiplicities ) ):
                log.warning(
                    '''
                        Warning, f contains C^0-continuities, however, strictly
                        speaking, the control mapping needs to be at least
                        C^1-continuous. Consider using f.to_c([1, 1]) instead.
                    '''
                )

            self._f = f.toscipy()

        else:
            '''
                So far, we only allow for instantiations of the class
                ``go.TensorGridObject`` as control mapping.
                XXX: allow for other means of instantiation, for instance
                via a list / dictionary implementing functions for the
                control mapping and all its relevant derivatives.
            '''
            raise NotImplementedError

        self._tablulate_control_mapping()


class EllipticPhysicalControl( EllipticControl ):

    """
        Elliptic Grid Generation with physical (not parametric) control mapping.
        Based on the article `Generation of Structured Difference
        Grids in Two-Dimensional Nonconvex Domains Using Mappings`.
    """

    def _tablulate_control_mapping( self ):

        '''
            Tabulate the control mapping and all its required derivatives
            in the quadrature points.
            XXX: see if we can make this look prettier
                (possibly using some symbolic approach).
        '''

        # by the time this is called, _f and _quad have been set
        # in the __init__ method
        _f = self._f
        _q = self._quad

        # all required first and second derivatives
        X_xi = _f( *_q, dx=1 )
        X_xi_eta = _f( *_q, dx=1, dy=1 )
        X_xi_xi = _f( *_q, dx=2 )
        X_eta = _f( *_q, dy=1 )
        X_eta_eta = _f( *_q, dy=2 )

        _0 = ( Ellipsis, 0 )
        _1 = ( Ellipsis, 1 )

        # Jacobian determinant and its derivatives
        jacdet = X_xi[_0] * X_eta[_1] - X_eta[_0] * X_xi[_1]
        jacsq = jacdet ** 2
        jacdet_xi = X_xi_xi[_0] * X_eta[_1] + X_xi[_0] * X_xi_eta[_1] \
            - X_xi_eta[_0] * X_xi[_1] - X_eta[_0] * X_xi_xi[_1]
        jacdet_eta = X_xi_eta[_0] * X_eta[_1] + X_xi[_0] * X_eta_eta[_1] \
            - X_eta_eta[_0] * X_xi[_1] - X_eta[_0] * X_xi_eta[_1]

        if not ( jacdet > 0 ).all():
            log.warning(
                '''
                    Warning, the control mapping is defective on at least one
                    quadrature point. Proceeding with this control mapping may
                    lead to failure of convergence and / or a defective mapping.
                '''
            )

        # metric tensor of the control mapping divided by the Jacobian determinant
        self._G11 = ( X_xi ** 2 ).sum(-1) / jacdet
        self._G12 = ( X_xi * X_eta ).sum(-1) / jacdet
        self._G22 = ( X_eta ** 2 ).sum(-1) / jacdet

        # derivatives of _G11
        self._G11_xi = \
            2 * ( X_xi * X_xi_xi ).sum(-1) / jacdet \
            - jacdet_xi * (X_xi ** 2).sum(-1) / jacsq
        self._G11_eta = \
            2 * ( X_xi * X_xi_eta ).sum(-1) / jacdet \
            - jacdet_eta * (X_xi ** 2).sum(-1) / jacsq

        # derivatives of _G12
        self._G12_xi = \
            (X_xi_xi * X_eta + X_xi * X_xi_eta).sum(-1) / jacdet  \
            - jacdet_xi * (X_xi * X_eta).sum(-1) / jacsq
        self._G12_eta = \
            (X_xi_eta * X_eta + X_xi * X_eta_eta).sum(-1) / jacdet \
            - jacdet_eta * (X_xi * X_eta).sum(-1) / jacsq

        # derivatives of _G22
        self._G22_xi = \
            2 * ( X_eta * X_xi_eta ).sum(-1) / jacdet \
            - jacdet_xi * (X_eta ** 2).sum(-1) / jacsq
        self._G22_eta = \
            2 * ( X_eta * X_eta_eta ).sum(-1) / jacdet \
            - jacdet_eta * (X_eta ** 2).sum(-1) / jacsq

        log.info( 'The control mapping has been tabulated.' )

    def __new__( cls, g, order, f=None, **kwargs ):

        '''
            If the control mapping ``f`` is None return an instantiation
            of standard EGG.
            Else, return an instantiation of this class.
        '''

        if f is None:
            log.info( 'No control mapping passed, proceeding with standard EGG' )
            return Elliptic( g, order, **kwargs )

        return Elliptic.__new__(cls)

    @with_boundary_conditions
    def residual( self, c ):
        cs = np.array_split( c, 2 )

        (x_xi, x_eta), (y_xi, y_eta) = [ self.fderivs(c_) for c_ in cs ]
        g11, g12, g22 = self.metric(c)

        ddx, ddy = [ self.sderivs(c_) for c_ in cs ]

        scale = g11 + g22 + self._eps

        P = g11 * self._G12_eta - g12 * self._G11_eta
        Q = 0.5 * ( g11 * self._G22_xi - g22 * self._G11_xi )
        S = g12 * self._G22_xi - g22 * self._G12_xi
        R = 0.5 * ( g11 * self._G22_eta - g22 * self._G11_eta )

        mul0 = ( g22 * ddx[0] - 2 * g12 * ddx[1] + g11 * ddx[2] )
        mul1 = ( g22 * ddy[0] - 2 * g12 * ddy[1] + g11 * ddy[2] )

        mul0 += \
            - x_xi * ( self._G22*(P - Q) + self._G12*(S - R) ) \
            - x_eta * ( self._G11*(R - S) + self._G12*(Q - P) )
        mul1 += \
            - y_xi * ( self._G22*(P - Q) + self._G12*(S - R) ) \
            - y_eta * ( self._G11*(R - S) + self._G12*(Q - P) )

        mul0 /= scale
        mul1 /= scale

        return np.concatenate( [self.jitarray(mul=mul0), self.jitarray(mul=mul1)] ) \
            + self._rhs


class EllipticParametricControl( EllipticControl ):

    '''
        Elliptic Grid Generator with parametric control mapping.
        Based on the fourth chapter from the Handbook of Grid Generation.
    '''

    def _tablulate_control_mapping( self ):

        '''
            Tabulate the control mapping and all its required derivatives
            in the quadrature points.
            XXX: see if we can make this look prettier
                (possibly using some symbolic approach).
        '''

        # by the time this is called, _f and _quad have been set
        # in the __init__ method
        _f = self._f
        _q = self._quad

        _0 = ( Ellipsis, 0 )
        _1 = ( Ellipsis, 1 )

        T_xi = _f( *_q, dx=1 )
        s_xi, t_xi = T_xi[_0], T_xi[_1]

        T_eta = _f( *_q, dy=1 )
        s_eta, t_eta = T_eta[_0], T_eta[_1]

        T_xi_xi = _f( *_q, dx=2 )
        s_xi_xi, t_xi_xi = T_xi_xi[_0], T_xi_xi[_1]

        T_xi_eta = _f( *_q, dx=1, dy=1 )
        s_xi_eta, t_xi_eta = T_xi_eta[_0], T_xi_eta[_1]

        T_eta_eta = _f( *_q, dy=2 )
        s_eta_eta, t_eta_eta = T_eta_eta[_0], T_eta_eta[_1]

        jacdet = s_xi * t_eta - s_eta * t_xi

        self._P11 = \
            [
                -1/jacdet * (t_eta * s_xi_xi - s_eta * t_xi_xi),
                -1/jacdet * (-t_xi * s_xi_xi + s_xi * t_xi_xi)
            ]
        self._P12 = \
            [
                -1/jacdet * (t_eta * s_xi_eta - s_eta * t_xi_eta),
                -1/jacdet * (-t_xi * s_xi_eta + s_xi * t_xi_eta)
            ]
        self._P22 = \
            [
                -1/jacdet * (t_eta * s_eta_eta - s_eta * t_eta_eta),
                -1/jacdet * (-t_xi * s_eta_eta + s_xi * t_eta_eta)
            ]

        if not ( jacdet > 0 ).all():
            log.warning(
                '''
                    Warning, the control mapping is defective on at
                    least one quadrature point. Proceeding with this
                    control mapping may lead to failure of convergence
                    and / or a defective mapping.
                '''
            )

        log.info( 'The control mapping has been tabulated.' )

    @with_boundary_conditions
    def residual( self, c ):

        cs = np.array_split( c, 2 )

        P11 = self._P11
        P12 = self._P12
        P22 = self._P22

        g11, g12, g22 = self.metric(c)
        dx, dy = [ self.fderivs(c_) for c_ in cs ]
        ddx, ddy = [ self.sderivs(c_) for c_ in cs ]

        mul0 = g22 * ddx[0] - 2 * g12 * ddx[1] + g11 * ddx[2]
        mul1 = g22 * ddy[0] - 2 * g12 * ddy[1] + g11 * ddy[2]

        S = g22 * P11[0] - 2 * g12 * P12[0] + g11 * P22[0]
        T = g22 * P11[1] - 2 * g12 * P12[1] + g11 * P22[1]

        mul0 += S * dx[0] + T * dx[1]
        mul1 += S * dy[0] + T * dy[1]

        scale = g11 + g22 + self._eps

        return \
            np.concatenate([
                self.jitarray(mul=mul0/scale),
                self.jitarray(mul=mul1/scale)
            ])


class NamedArray( np.ndarray ):

    _varnames = ( 'u', 'v', 'x', 'y' )

    def __new__( cls, arr ):
        assert len(arr.shape) == 1
        assert len(arr) % len(cls._varnames) == 0
        ret = np.array(arr).view( dtype=cls )
        return ret

    def __init__( self, *args, **kwargs ):
        self._N = len(self) // len(self._varnames)
        s = [ i * self._N for i in range( len(self._varnames) + 1 ) ]
        slices = [ slice( n, m, 1 ) for n, m in zip( s[:-1], s[1:] ) ]
        self._slices = dict( zip(self._varnames, slices) )

    def __getitem__( self, key ):
        try:
            return np.array( self.data[ self._slices[key] ] )
        except KeyError:
            return np.ndarray.__getitem__( self, key )

    def __setitem__( self, key, value ):
        if key in self._varnames:
            key = self._slices[key]
        np.ndarray.__setitem__( self, key, value )

    def __iter__( self ):
        return ( self[key] for key in self._varnames )

    __hash__ = HashAbleArray.__hash__

    def tolist( self ):
        return [ self[name] for name in self._varnames ]


def mixed_FEM_BC( f ):

    @wraps( f )
    def wrapper( *args, **kwargs ):

        self, c, *args = args
        g = self._g

        _ret = g.cons.copy()
        _ret[g.dofindices] = c[len(_ret):]

        _c = np.concatenate( [ c[:len(_ret)], _ret ] )
        ret = f( self, NamedArray(_c), *args, **kwargs )

        self._feval += 1

        return ret

    return wrapper


def c0( g ):

    '''
        Return the coordinate directions that contain
        C^0-continuities.
    '''

    return tuple( i for i in range(2)
            if g.degree[i] in g.knotmultiplicities[i] )


def to_named_array( f ):

    '''
        If the second argument (the one next to self)
        is not an instance of ``NamedArray``, turn it
        into an instance of ``NamedArray``.
    '''

    @wraps(f)
    def wrapper( self, c, *args, **kwargs ):
        if not isinstance( c, NamedArray ):
            c = NamedArray(c)
        return f( self, c, *args, **kwargs )

    return wrapper


class MixedFEM( FastSolver ):

    def __init__(
                    self,
                    *args,
                    eps=0.001,
                    **kwargs ):

        g, *args = args

        coordinate_directions = c0(g)

        assert len( coordinate_directions ) in (1, 2) and \
            all( i in (0, 1) for i in coordinate_directions )

        if len( coordinate_directions ) == 2:
            raise NotImplementedError

        super().__init__( g, *args, **kwargs )

        self._eps = eps
        self._coordinate_directions = coordinate_directions
        self._feval = 0

    @property
    def dindices( self ):
        if not hasattr( self, '_dindices' ):
            N = self._N
            self._dindices = \
                np.concatenate( [np.arange(2 * N),
                                self._g.dofindices + 2 * N] )
        return self._dindices

    @property
    def M_x( self ):
        if not hasattr( self, '_M_x' ):
            trial = { 0: 'x', 1: 'y' }[ self._coordinate_directions[0] ]
            self._M_x = self.jitmass( test='w', trial=trial )
        return self._M_x

    @property
    def M_inv( self ):
        if not hasattr( self, '_M_inv' ):
            self._M_inv = sparse.linalg.splu( self.M.tocsc() )
        return self._M_inv

    @to_named_array
    def linear_residual( self, c: NamedArray ):

        u, v, x, y = list(c)

        M_x = self.M_x
        M = self.M
        return \
            np.concatenate([
                M.dot(n) - M_x.dot(m) for n, m in zip([u, v], [x, y])
            ])

    @to_named_array
    def nonlinear_residual( self, c: NamedArray ):

        self._feval += 1

        u, v, x, y = list(c)

        f, s = self.fderivs, self.sderivs
        g11, g12, g22 = self.metric( np.concatenate( [x, y] ) )

        # index in (0, 1) for now, index == (0, 1) is not supported yet
        index = self._coordinate_directions[0]

        scale = g11 + g22 + self._eps
        arr = self.jitarray

        if index == 0:
            mul0 = g22 * f(u)[0] - g12 * f(u)[1] - g12 * s(x)[1] + g11 * s(x)[2]
            mul1 = g22 * f(v)[0] - g12 * f(v)[1] - g12 * s(y)[1] + g11 * s(y)[2]
        else:
            mul0 = g22 * s(x)[0] - g12 * f(u)[0] - g12 * s(x)[1] + g11 * f(u)[1]
            mul1 = g22 * s(y)[0] - g12 * f(v)[0] - g12 * s(y)[1] + g11 * f(v)[1]

        return np.concatenate( [ arr(mul0 / scale), arr(mul1 / scale) ] )

    @mixed_FEM_BC
    def residual( self, c: NamedArray ):

        '''
            Residual for Newton-Krylov.
            The first two block-entries are scaled by the inverse
            of the mass matrix for better convergence.
        '''

        u, v, x, y = list(c)
        f = self.fderivs

        arr = self.jitarray
        index = self._coordinate_directions[0]

        proj = lambda n, m: n - self.M_inv.solve( arr( f(m)[index] ) )
        res0 = np.concatenate([ proj(n, m) for n, m in zip( [u, v], [x, y] ) ])

        res1 = self.nonlinear_residual(c)
        return np.concatenate([ res0, res1[self._g.dofindices] ])

    @mixed_FEM_BC
    def jacresidual( self, c: NamedArray ):

        '''
            Residual for a matrix-based Newton-approach.
            The difference to ``self.residual`` is that
            the linear part of the residual is not preconditioned
            with the mass inverse of the mass matrix.
        '''

        res0 = self.linear_residual(c)
        res1 = self.nonlinear_residual(c)
        return np.concatenate( [ res0, res1 ] )[ self.dindices ]

    @mixed_FEM_BC
    def jacobian( self, c: NamedArray ):

        u, v, x, y = list(c)
        f = self.fderivs

        g11, g12, g22 = self.metric( np.concatenate( [x, y] ) )

        scale = g11 + g22 + self._eps
        B = scale

        index = self._coordinate_directions[0]
        _M = self.jitmass

        dindices = self.dindices

        if index == 0:

            M_xi = self.M_x
            M = self.M

            dR0_dcx = sparse.block_diag( [M_xi] * 2 )
            dR0_dcu = - sparse.block_diag( [M] * 2 )

            dR1_dcu = sparse.block_diag(
                [
                    self.jitmass( mul=g22/scale, test='w', trial='x' ) +
                    self.jitmass( mul=-g12/scale, test='w', trial='y' )
                ] * 2
            )

            x_xi, x_eta = f(x)
            y_xi, y_eta = f(y)

            u_xi, u_eta = f(u)
            v_xi, v_eta = f(v)

            x_xi_eta, x_eta_eta = self( x, dx=1, dy=1 ), self( x, dy=2 )
            y_xi_eta, y_eta_eta = self( y, dx=1, dy=1 ), self( y, dy=2 )

            extra_term = _M(mul=-g12/B, trial='xy') + _M(mul=g11/B, trial='yy')

            prefac0 = \
                ( g22 * u_xi - g12 * u_eta - g12 * x_xi_eta + g11 * x_eta_eta )
            prefac1 = \
                ( g22 * v_xi - g12 * v_eta - g12 * y_xi_eta + g11 * y_eta_eta )

            prefac0 /= scale ** 2
            prefac1 /= scale ** 2

            ################################

            dR10_dcxx = \
                _M(
                    mul=(2*x_xi*(x_eta_eta/B - prefac0) - x_eta/B*(u_eta + x_xi_eta)),
                    trial='x' ) + \
                _M(
                    mul=(2*x_eta*(u_xi/B - prefac0) - x_xi/B*(u_eta + x_xi_eta)),
                    trial='y' ) + extra_term

            dR11_dcxx = \
                _M(
                    mul=(2*x_xi*(y_eta_eta/B - prefac1) - x_eta/B*(v_eta + y_xi_eta)),
                    trial='x' ) + \
                _M(
                    mul=(2*x_eta*(v_xi/B - prefac1) - x_xi/B*(v_eta + y_xi_eta)),
                    trial='y' )

            dR1_dcxx = sparse.vstack( [ dR10_dcxx, dR11_dcxx ] )

            ################################

            dR10_dcxy = \
                _M(
                    mul=(2*y_xi*(x_eta_eta/B - prefac0) - y_eta/B*(u_eta + x_xi_eta)),
                    trial='x' ) + \
                _M(
                    mul=(2*y_eta*(u_xi/B - prefac0) - y_xi/B*(u_eta + x_xi_eta)),
                    trial='y' ) \

            dR11_dcxy = \
                _M(
                    mul=(2*y_xi*(y_eta_eta/B - prefac1) - y_eta/B*(v_eta + y_xi_eta)),
                    trial='x' ) + \
                _M(
                    mul=(2*y_eta*(v_xi/B - prefac1) - y_xi/B*(v_eta + y_xi_eta)),
                    trial='y' ) + extra_term

            dR1_dcxy = sparse.vstack( [ dR10_dcxy, dR11_dcxy ] )

            dR1_dcx = sparse.hstack( [ dR1_dcxx, dR1_dcxy ] )

        else:
            raise NotImplementedError

        # minus in front of the linear part to be compatible with self.jacresidual
        return \
            sparse.vstack([
                sparse.hstack( [-dR0_dcu, -dR0_dcx] ),
                sparse.hstack( [dR1_dcu, dR1_dcx] )
            ]).tolil()[:, dindices][dindices, :].tocsc() # ugly, find better solution.

    def init( self ):
        '''
            Generate the canonical initial guess.
        '''
        x, y = np.array_split( self._g.x, 2 )
        index = self._coordinate_directions[0]
        return \
            np.concatenate(
                [ self.project( self.fderivs(k)[index] ) for k in (x, y) ] +
                [ self._g.x[self._g.dofindices] ]
            )

    def solve( self, jacobian='Schur', **kwargs ):

        if 'init' not in kwargs:
            kwargs[ 'init' ] = self.init()

        # add an empty dict ``jac_options`` if it doesn't already exist
        kwargs.setdefault( 'jac_options', {} )

        # this value is chosen heuristically
        kwargs['jac_options'].setdefault( 'inner_atol', 1e-12 )

        if jacobian in ('Schur', 'schur'):
            kwargs[ 'jacobian' ] = MixedFEMSchurKrylovJacobian
            kwargs['jac_options'][ 'solver' ]= self
            # overwrite the residual
            self.residual = self.jacresidual

        ret = root( self, **kwargs )

        log.info( 'Converged after {} function evaluations.'
                .format( self._feval ) )

        # discard the auxilliary variables from the solution
        return ret[ -len(self._g.dofindices): ]


class MixedFEMSchurKrylovJacobian( optimize.nonlin.KrylovJacobian ):

    '''
        KrylovJacobian that is capable of solving problems
        of the ``MixedFEM`` type. At each Newton-iteration,
        the system of equations

                        | A, B | |du| = |a|
                        | C, D | |dx| = |b|,

        where the matrices A and B correspond to the linear part
        of the MixedFEM-problem, is reduced to a Schur-complement
        type problem with the goal of only computing |dx|:

                (D - C A_inv B) * |dx| = |b| - C A_inv |a|.   (1)

        (1) is approximately solved using a Newton-Krylov approach,
        where D * |dx| and C * ( A_inv B |dx| ) are approximated
        by first order finite-differences on the nonlinear part of
        the residual function. Here, A and B are separable mass-matrix-like
        matrices that are assembled explicitly.
        Upon completion, |du| is computed by solving

                    |du| = A_inv * (|a| - B * |dx|),

        after which a line-search procedure estimates the optimal value of
        eps such that ( |u| + eps * |du|, |x| + eps * |dx| )^T minimizes
        the residual norm over eps in (0, 1].

        The advantage with respect to running Newton-Krylov on the full matrix
        rather than the Schur-complement is that (1) is a better-scaled
        problem in which finite-differences tend to work better and
        convergence is reached in fewer iterations.
        Both versions don't require the assmebly of the nonlinear parts
        of the Jacobian, C and D.

        XXX: A and B are not assembled from a kronecker-product of their
             univariate constituents. Change this in the long run.
        XXX: A is inverted by computing a sparse LU-factorization, not
             taking advantage of its separable nature. And implementation
             that utilizes univariate LU- (or Cholesky-) factorizations
             of the constituents of A should speed things up even further.
        XXX: Add support for MixedFEM in both directions.
    '''

    def __init__( self, solver: MixedFEM, **kwargs ):

        self._solver = solver
        self._g = solver._g

        self.M_inv = solver.M_inv
        self.M_x = \
            sparse.block_diag(
                [solver.M_x]*2
            ).tolil()[:, self._g.dofindices].tocsc()

        self._Nlinear = 2 * solver._N
        self._Nnonlinear = len( self._g.dofindices )
        self._n = self._Nlinear + self._Nnonlinear

        super().__init__( **kwargs )

    def add_bc( self, c ):

        '''
            Add the boundary condition to the nonlinear
            part of the vector of unknowns.
        '''

        v = self._g.cons.copy()
        v[ self._g.dofindices ] = c
        return v

    def add_bc_full( self, c ):

        '''
            Add the boundary condition to the full vector
            of unknowns. Note that the linear part of the
            vector of unknowns is generally unaltered by this
            operations since it is not subject to boundary conditions.
        '''

        u_, x_ = self.split(c)
        return np.concatenate([ u_, self.add_bc(x_) ])

    def Bvec( self, c ):
        return -self.M_x.dot(c)

    def collapse_rhs( self, rhs ):

        '''
            Turn a vector of the form ( |a|, |b| )^T into
            |b| - C A_inv |a|.
            Here, C |x| is approximated through finite-differences.
        '''

        a, b = self.split( rhs )
        return \
            - self.Cvec(
                np.concatenate([
                    self.M_inv.solve(i) for i in np.array_split(a, 2)
                ]),
                deriv='u'
            ) + b

    def split( self, c ):

        '''
            Split a the vector ``c`` into
            parts correponding to auxilliary and main variables.
        '''

        N = self._Nlinear
        return c[ :N ], c[ N: ]

    def Cvec( self, x, sc=None, deriv='u' ):

        '''
            Return an approximation of C * |x|
            or D * |x|, where deriv='u' refers to
            C and deriv='x' to D.
        '''

        assert deriv in ( 'u', 'x' )

        if sc is None:
            sc = self.omega / np.linalg.norm(x)

        f = self._solver.nonlinear_residual

        X0 = self.add_bc_full( self.x0 )

        if deriv == 'u':
            n = len(X0) - len(x)
            X = np.concatenate( [ x, np.zeros(n) ] )
        else:
            X = np.zeros_like( X0 )
            X[ self._Nlinear + self._g.dofindices ] = x

        X = X0 + sc * X

        return \
            (f(X)[self._g.dofindices] - self.f0[-self._Nnonlinear:]) / sc

    def matvec( self, v ):

        '''
            Return an approximation of (D C A_inv B) * |v|.
        '''

        M_inv = self.M_inv

        Bv = self.Bvec(v)

        c_vec = \
            np.concatenate([
                M_inv.solve(i) for i in np.array_split(Bv, 2) ])

        f = self._solver.nonlinear_residual
        f_vec = np.concatenate( [-c_vec, v] )

        # compute finite-difference step size.
        nf = np.linalg.norm(f_vec)
        if nf == 0:
            return 0 * v
        sc = self.omega / nf

        f_vec = self.add_bc_full( self.x0 + sc * f_vec )

        return \
            ( f( f_vec )[self._g.dofindices] - self.f0[-self._Nnonlinear:] ) / sc

    @InstanceMethodCache
    def solve( self, rhs, tol=0 ):

        # Solve the Schur-complement problem.
        rhs_ = self.collapse_rhs(rhs)
        dx = optimize.nonlin.KrylovJacobian.solve( self, rhs_, tol=tol )

        # Solve for |du|.
        du = np.concatenate([ self.M_inv.solve(i) for i in
            np.array_split(self.split( rhs )[0] - self.Bvec(dx), 2)
        ])
        return np.concatenate([ du, dx ])

    def setup( self, *args, **kwargs ):
        if self.preconditioner is not None:
            raise NotImplementedError
        optimize.nonlin.KrylovJacobian.setup( self, *args, **kwargs )

        '''
            overwrite self.op set in ``optimize.nonlin.Krylovjacobian.setup
            since the Schur-complement is of dimension (n, n) rather than
            (self._N, self._N). If this is not done, ``self.solve`` throws
            an error.
        '''

        n = self._Nnonlinear
        self.op = sparse.linalg.LinearOperator( shape=(n, n), matvec=self.matvec )


class MixedFEMParametricControl( EllipticParametricControl ):

    """
        Same as EllipticParametricControl, but allows for
        C^0-continuities.
    """

    def __new__( cls, g, order, f=None, **kwargs ):

        '''
            If the control mapping ``f`` is None return an instantiation
            of standard EGG.
            Else, return an instantiation of this class.
        '''

        if f is None:
            log.info( 'No control mapping passed, proceeding with standard MixedFEM' )
            return MixedFEM( g, order, **kwargs )

        return Elliptic.__new__(cls)

    def __init__( self, g, *args, f=None, **kwargs ):

        '''
            Unfortunately, we need to repeat the MixedFEM
            initialization because we should not inherit from
            MixedFEM and EllipticControl.
        '''

        coordinate_directions = c0( g )

        assert len( coordinate_directions ) in (1, 2) and \
            all( i in (0, 1) for i in coordinate_directions )

        if len( coordinate_directions ) == 2:
            raise NotImplementedError

        super().__init__( g, *args, f=f, **kwargs )

        self._coordinate_directions = coordinate_directions

    @property
    def M_inv( self ):
        if not hasattr( self, '_M_inv' ):
            self._M_inv = sparse.linalg.splu( self.M.tocsc() )
        return self._M_inv

    @mixed_FEM_BC
    def residual( self, c: NamedArray ):

        '''
            Residual for Newton-Krylov.
            The first two block-entries are scaled by the inverse
            of the mass matrix for better convergence.
        '''

        u, v, x, y = list(c)
        f, s = self.fderivs, self.sderivs
        g11, g12, g22 = self.metric( np.concatenate( [x, y] ) )

        # index in (0, 1) for now, index == (0, 1) is not supported yet
        index = self._coordinate_directions[0]

        scale = g11 + g22 + self._eps
        arr = self.jitarray

        proj = lambda n, m: n - self.M_inv.solve( arr( self.fderivs(m)[index] ) )
        res0 = np.concatenate( [ proj(n, m) for n, m in zip( [u, v], [x, y] ) ] )

        if index == 0:
            mul0 = g22 * f(u)[0] - g12 * f(u)[1] - g12 * s(x)[1] + g11 * s(x)[2]
            mul1 = g22 * f(v)[0] - g12 * f(v)[1] - g12 * s(y)[1] + g11 * s(y)[2]
        else:
            mul0 = g22 * s(x)[0] - g12 * f(u)[0] - g12 * s(x)[1] + g11 * f(u)[1]
            mul1 = g22 * s(y)[0] - g12 * f(v)[0] - g12 * s(y)[1] + g11 * f(v)[1]

        P11 = self._P11
        P12 = self._P12
        P22 = self._P22

        S = g22 * P11[0] - 2 * g12 * P12[0] + g11 * P22[0]
        T = g22 * P11[1] - 2 * g12 * P12[1] + g11 * P22[1]

        mul0 += S * f(x)[0] + T * f(x)[1]
        mul1 += S * f(y)[0] + T * f(y)[1]

        res1 = np.concatenate( [ arr(mul0 / scale), arr(mul1 / scale) ] )
        return np.concatenate( [ -res0, res1[self._g.dofindices] ] )

    solve = MixedFEM.solve
    init = MixedFEM.init


def fastsolve( g, method='Elliptic', ischeme=None, intargs=None, **solveargs ):

    if len(g) != 2:
        raise NotImplementedError

    if ischeme is None:
        ischeme = g.ischeme

    if intargs is None:
        intargs = {}

    f = { 'Elliptic': Elliptic,
          'Elliptic_unscaled': Elliptic_unscaled,
          'Elliptic_partial': Elliptic_partial,
          'Elliptic_control': EllipticControl,
          'MixedFEM': MixedFEM,
          'Mixed-FEM': MixedFEM,
          'Mixed-Fem': MixedFEM}[ method ]

    Int = f( g, ischeme, **intargs )
    sol = Int.solve( **solveargs )

    g.x[g.dofindices] = sol
