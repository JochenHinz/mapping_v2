#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numba
import types
import ABC

from functools import wraps
from scipy import interpolate, sparse, optimize
from nutils import function, matrix, log
from collections import OrderedDict


def blockshaped( arr, nrows, ncols ):
    h, w = arr.shape
    return ( arr.reshape( h // nrows, nrows, -1, ncols ).swapaxes( 1, 2 ).reshape( -1, nrows, ncols ) )


def make_quadrature( g, order ):
    assert len( g ) == 2
    x, y = np.polynomial.legendre.leggauss( order )
    quad = [ ( np.kron( k[ 1: ] - k[ :-1 ], x ) + np.kron( k[ 1: ] + k[ :-1 ], np.ones( len( x ) ) ) ) / 2 for k in g.knots ]
    weights = [ np.kron( np.ones( len( k ) - 1 ), y ) * np.repeat( ( k[ 1: ] - k[ :-1 ] ) / 2, len( y ) ) for k in g.knots ]
    slices = [ ( None, ) * i + ( slice(None), ) + ( None, ) * ( g.targetspace - i - 1 ) for i in range( g.targetspace ) ]
    weights = np.prod( [ c[ s ] for c, s in zip( weights, slices ) ] )
    return quad, weights, len( x )


def make_supports( g ):
    assert len( g ) <= 2
    K = np.arange( len( g.basis ), dtype=int )
    IJ = zip( *np.unravel_index( K, g.ndims ) )
    knotspan = [ np.repeat( np.arange( len( kv ), dtype=int ), km ) for kv, km in zip( g.knots, g.knotmultiplicities ) ]
    supports = tuple( tuple( ks[ i: i + d + 2 ] for i, ks, d in zip( ij, knotspan, g.degree ) ) for ij in IJ )
    return supports


class SecondOrderKrylovJacobian( optimize.nonlin.KrylovJacobian ):

    def matvec(self, v):
        nv = np.linalg.norm(v)
        if nv == 0:
            return 0*v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc*v) - self.func(self.x0 - sc*v)) / ( 2 * sc )
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r


""" Jitted functions """


@numba.jit( numba.void( numba.int64, numba.int64, numba.int64[:], numba.int64[:] ), nopython=True )
def localsupport( firststart, secondstart, suppunion, buff ):
    buff[ 0 ] = suppunion[ 0 ] - firststart
    buff[ 1 ] = suppunion[ 1 ] - firststart
    buff[ 2 ] = suppunion[ 0 ] - secondstart
    buff[ 3 ] = suppunion[ 1 ] - secondstart


@numba.jit( numba.void( numba.int64[:], numba.int64[:], numba.int64[:] ), nopython=True )
def supportunion( a, b, buff ):
    buff[ 0 ] = max( a[ 0 ], b[ 0 ] )
    buff[ 1 ] = min( a[ 1 ], b[ 1 ] )


@numba.jit( numba.int64( numba.int64, numba.int64, numba.int64 ), nopython=True )
def local_to_global( i, j, m ):
    return ( i - 1 ) * m + j


@numba.jit( numba.int64[ : ]( numba.int64, numba.int64 ), nopython=True )
def global_to_local( k, m ):
    ret = np.empty( 2, np.int64 )
    ret[ 1 ] = k % m
    ret[ 0 ] = ( k - ret[ 1 ] ) / m
    return ret


@numba.jit( numba.int64( numba.int64, numba.int64[:] ), nopython=True )
def index( glob, locs ):
    for i in range( len( locs ) ):
        if locs[ i ] == glob:
            return i
    else:
        return -1


@numba.jit( numba.int64[ : ]( numba.int64[:], numba.int64[:] ), nopython=True )
def intersection( arr1, arr2 ):
    indices = []
    for i in arr1:
        idx = np.where( arr2 == i )[ 0 ]
        if len( idx ) > 0:
            indices.append( idx[ 0 ] )
    return arr2[ np.array( indices, dtype=numba.int64 ) ]


@numba.jit( numba.float64[:]( numba.int64, numba.int64, numba.float64[:, :, :], numba.float64[:, :, :], numba.float64[ :, :, : ],
        numba.int64, numba.int64[ : ], numba.int64[ : ], numba.int64[ : ], numba.int64[ : ] ), nopython=True, nogil=False, cache=False, parallel=False )
def jitmass( N, m, ws0, ws1, quadweights, cl, elemstart, elems, lilstart, lils ):
    ret = np.zeros( len( lils ), dtype=np.float64 )
    current = 0
    for i in numba.prange( N ):
        for j in lils[ lilstart[ i ]: lilstart[ i+1 ] ]:
            elems0 = elems[ elemstart[ i ]: elemstart[ i + 1 ] ]
            elems1 = elems[ elemstart[ j ]: elemstart[ j + 1 ] ]
            inter = intersection( elems0, elems1 )
            res = 0
            for k in inter:
                elem0 = elemstart[ i ] + index( k, elems0 )
                elem1 = elemstart[ j ] + index( k, elems1 )
                res += ( ws0[ elem0 ] * ws1[ elem1 ] * quadweights[ k ] ).sum()
            ret[ current ] = res
            current += 1
    return ret


@numba.jit( numba.float64[:]( numba.int64, numba.int64, numba.float64[:, :, :], numba.float64[:, :, :], numba.int64, numba.int64[:], numba.int64[:] ), nopython=True, parallel=True )
def jitarray( N, m, ws, quadweights, cl, elemstart, elems ):
    ret = np.zeros( N, dtype=np.float64 )
    for i in numba.prange( N ):
        start = elemstart[ i ]
        elements = elems[ elemstart[ i ]: elemstart[ i + 1 ] ]
        for k in elements:
            elem = start + index( k, elements )
            w = quadweights[ k ]
            ret[ i ] += ( ws[ elem ] * w ).sum()
    return ret


""" Various auxilliary functions for Integrator class """


def cache( f ):
    cache = OrderedDict()

    @wraps( f )
    def wrapper( self, c ):

        if tuple( c ) in cache:
            ret = cache[ tuple( c ) ]
        else:
            ret = f( self, c )
            cache[ tuple( c ) ] = ret

        if len( cache ) > 1:
            del cache[ list( cache.keys() )[ 0 ] ]

        return ret
    return wrapper


def with_boundary_conditions( f ):
    @wraps( f )
    def wrapper( self, c ):
        g = self._g
        vec = g.cons.copy()
        vec[ g.dofindices ] = c
        ret = f( self, vec )
        try:
            self._feval += 1
        except:
            self._feval = 1
        return ret[ g.dofindices ]
    return wrapper


def _root( I, **scipyargs ):

    scipyargs.setdefault( 'method', 'krylov' )

    res = I.residual
    init = I._g.x[ I._g.dofindices ]

    return optimize.root( res, init, **scipyargs )


def root( I, order=1, jac_options={}, **scipyargs ):

    assert order in ( 1, 2 )

    scipyargs.setdefault( 'verbose', True )

    res = I.residual
    init = I._g.x[ I._g.dofindices ]

    if order == 1:
        jac = optimize.nonlin.KrylovJacobian( **jac_options )
    else:
        jac = SecondOrderKrylovJacobian( **jac_options )

    return optimize.nonlin.nonlin_solve( res, init, jacobian=jac, **scipyargs )


def anderson( I, **scipyargs ):

    scipyargs.setdefault( 'verbose', True )
    scipyargs.setdefault( 'f_tol', 1e-5 )

    res = I.residual
    init = I._g.x[ I._g.dofindices ]

    return optimize.anderson( res, init, **scipyargs )


def clip_from_zero( A, eps=1e-6 ):
    A[ A < 0 ] = np.clip( A[ A < 0 ], -np.inf, -eps )
    A[ A > 0 ] = np.clip( A[ A > 0 ], eps, np.inf )
    return A


class Integrator:

    """ Main class for integration """

    tck = lambda self, x: ( *self._g.extend_knots(), x, * self._g.degree )
    splev = lambda self, i, quad, **scipyargs: interpolate.bisplev( *quad, self.tck( self._I[ i ] ), **scipyargs )

    def _set_w( self ):

        def setter( **kwargs ):
            return np.concatenate( [
                blockshaped( w, *self._chunklengths ) for w in
                tuple( self.splev( i, self._chunks[ i ], **kwargs ) for i in range( self._N ) )
                ] )

        w = setter()
        w_xi = setter( dx=1 )
        w_eta = setter( dy=1 )
        w_xi_xi = setter( dx=2 )
        w_eta_eta = setter( dy=2 )
        w_xi_eta = setter( dx=1, dy=1 )
        self._w = { 'w': w, 'w_x': w_xi, 'w_y': w_eta, 'w_xx': w_xi_xi, 'w_xy': w_xi_eta, 'w_yy': w_eta_eta }

    def _set_elements( self ):
        n, m = [ len( k ) - 1 for k in self._g.knots ]
        f = lambda i, j: i * m + j  # global element index
        LIL = []
        supports = self._supports
        for i in range( self._N ):
            L = []
            xi, eta = supports[ i ]
            for x in np.arange( xi[ 0 ], xi[ -1 ] ):
                for y in np.arange( eta[ 0 ], eta[ -1 ] ):
                    L.append( f( x, y ) )
            LIL.append( sorted( L ) )
        self._LIL = LIL

    def __init__( self, g, order ):

        if len( g ) != 2:
            raise NotImplementedError

        self._g = g
        self._N = len( self._g.basis )
        self._I = np.eye( self._N )
        self._p = order

        self._supports = make_supports( g )
        self._quad, self._weights, chunklength = make_quadrature( g, order )
        self._chunklengths = [ chunklength ] * 2
        self._quadweights = blockshaped( self._weights, *self._chunklengths )
        self._chunks = tuple(
                            tuple( q[ s[0] * c: s[-1] * c ] for q, s, c in zip( self._quad, self._supports[ i ], self._chunklengths ) )
                            for i in range( self._N )
                            )
        self._set_w()
        self._set_elements()

        self._lilindex = self._g.integrate( function.outer( self._g.basis ) ).toscipy().tolil().rows  # try to write a function for this
        self._lilstart = np.array( [ 0 ] + [ len( l ) for l in self._lilindex ], dtype=np.int64 ).cumsum()
        self._lilindex_flat = np.concatenate( self._lilindex ).astype( np.int64 )
        self._jitelements_flat = np.concatenate( self._LIL ).astype( np.int64 )
        self._jitsupportlength = np.array( [ 0 ] + [ len( l ) for l in self._LIL ], dtype=np.int64 ).cumsum()

        self._M = self.jitmass()

        if not self.__class__ == Integrator:
            if not hasattr( self, 'residual' ):
                raise AttributeError( "Derived classes of the 'Integrator' base-class must at least implement the residual function" )

        if hasattr( self, 'residual' ):
           # def f( self, **kwargs ):
           #     return root( self, **kwargs )

            self.solve = types.MethodType( root, self )
            self._feval = 0

    """ Magic functions """

    def __getitem__( self, key ):
        return self._w[ key ]

    """ Various evaluations of the mapping function """

    @cache
    def zderivs( self, c ):
        """ zeroth order derivative """
        return interpolate.bisplev( *self._quad, self.tck( c ) )

    @cache
    def fderivs( self, c ):
        """ first derivatives """
        f = lambda **kwargs: interpolate.bisplev( *self._quad, self.tck( c ), **kwargs )
        return [ f( dx=1 ), f( dy=1 ) ]

    @cache
    def sderivs( self, c ):
        """ second derivatives """
        f = lambda **kwargs: interpolate.bisplev( *self._quad, self.tck( c ), **kwargs )
        kwargs = [ { 'dx': 2 }, { 'dx': 1, 'dy': 1 }, { 'dy': 2 } ]
        return [ f( **k ) for k in kwargs ]

    @cache
    def metric( self, c ):
        """ [ g11, g12, g22 ] """
        N = self._N
        assert len( c ) == 2 * N
        x_xi, x_eta = self.fderivs( c[ :N ] )
        y_xi, y_eta = self.fderivs( c[ N: ] )
        return [ x_xi ** 2 + y_xi ** 2, x_xi * x_eta + y_xi * y_eta, x_eta ** 2 + y_eta ** 2 ]

    def jacdet( self, c ):
        N = self._N
        x_xi, x_eta = self.fderivs( c[ :N ] )
        y_xi, y_eta = self.fderivs( c[ N: ] )
        return x_xi * y_eta - x_eta * y_xi

    """ Transformations for LIL-format """

    def tolil( self, vec ):
        return np.array( [ arr.tolist() for arr in np.array_split( vec, self._lilstart[ 1: -1 ] ) ], dtype=object )

    def vec_to_mat( self, vec ):
        mat = sparse.lil_matrix( ( self._N, ) * 2 )
        mat.data = vec
        mat.rows = self._lilindex
        return mat.tocsr()

    """ Jitted arrays """

    def jitmass( self, mul=None, w='w' ):
        weights = self._quadweights.copy()
        if mul is not None:
            weights *= blockshaped( mul, *self._chunklengths )
        w = self[ w ]
        m = len( self._g.knots[ 1 ] ) - 1
        arr = jitmass( self._N, m, w, w, weights, self._chunklengths[ 0 ], self._jitsupportlength,
                self._jitelements_flat, self._lilstart, self._lilindex_flat )
        return self.vec_to_mat( self.tolil( arr ) )

    def jitarray( self, mul=None, w='w' ):
        weights = self._quadweights.copy()
        if mul is not None:
            weights *= blockshaped( mul, *self._chunklengths )
        w = self[ w ]
        return jitarray( self._N, len( self._g.knots[1] ) - 1, w, weights, self._chunklengths[0], self._jitsupportlength, self._jitelements_flat )

    """ Additional functionality """

    def project( self, mul, cons=None ):

        nutilsargs = { 'constrain': cons } if cons is not None else {}

        try:
            M = self._M
        except AttributeError:
            self._M = self.jitmass()
            M = self._M

        M = matrix.ScipyMatrix( M )
        rhs = self.jitarray( mul=mul )

        return M.solve( rhs, **nutilsargs )


class Elliptic_unscaled( Integrator ):

    @with_boundary_conditions
    def residual( self, c ):
        N = self._N
        g11, g12, g22 = self.metric( c )
        x_xi_xi, x_xi_eta, x_eta_eta = self.sderivs( c[ :N ] )
        y_xi_xi, y_xi_eta, y_eta_eta = self.sderivs( c[ N: ] )
        mul0 = g22 * x_xi_xi - 2 * g12 * x_xi_eta + g11 * x_eta_eta
        mul1 = g22 * y_xi_xi - 2 * g12 * y_xi_eta + g11 * y_eta_eta
        return np.concatenate( [ self.jitarray( mul=mul0 ), self.jitarray( mul=mul1 ) ] )


class Elliptic( Integrator ):

    def __init__( self, *args, eps=0.001, **kwargs ):
        super().__init__( *args, **kwargs )
        self._eps = eps

    @with_boundary_conditions
    def residual( self, c ):
        N = self._N
        g11, g12, g22 = self.metric( c )
        scale = g11 + g22 + self._eps
        x_xi_xi, x_xi_eta, x_eta_eta = self.sderivs( c[ :N ] )
        y_xi_xi, y_xi_eta, y_eta_eta = self.sderivs( c[ N: ] )
        mul0 = ( g22 * x_xi_xi - 2 * g12 * x_xi_eta + g11 * x_eta_eta ) / scale
        mul1 = ( g22 * y_xi_xi - 2 * g12 * y_xi_eta + g11 * y_eta_eta ) / scale
        return np.concatenate( [ self.jitarray( mul=mul0 ), self.jitarray( mul=mul1 ) ] )


class Elliptic_partial( Integrator ):

    @with_boundary_conditions
    def residual( self, c ):
        jacdet = clip_from_zero( self.jacdet( c ) )
        g11, g12, g22 = self.metric( c )
        arr = self.jitarray
        ret = np.concatenate(
                                [ 
                                    arr( g22 / jacdet, 'w_x' ) - arr( g12 / jacdet, 'w_y' ),
                                    -arr( g12 / jacdet, 'w_x' ) + arr( g11 / jacdet, 'w_y' )
                                ]
                            )
        return ret


class NamedArray( np.ndarray ):

    def __new__( cls, arr ):
        assert len( arr.shape ) == 1
        assert len( arr ) % 4 == 0
        ret = np.array( arr ).view( dtype=cls )
        N = int( len( arr ) / 4 )
        ret._N = N
        ret._varnames = [ 'u', 'v', 'x', 'y' ]
        s = [ 0, N, 2*N, 3*N, 4*N ]
        ret._slices = dict( zip( ret._varnames, [ slice( n, m, 1 ) for n, m in zip( s[ :-1 ], s[ 1: ] ) ] ) )
        return ret

    def __getitem__( self, key ):
        assert key in self._varnames
        return np.array( self.data[ self._slices[ key ] ] )

    def __setitem__( self, key, value ):
        if key in self._varnames:
            key = self._slices[ key ]
        np.ndarray.__setitem__( self, key, value )

    def list( self ):
        return [ self[ name ] for name in self._varnames ]


def mixed_FEM_BC( f ):

    @wraps( f )
    def wrapper( *args, **kwargs ):
        self, c, *args = args
        g = self._g
        ret1 = g.cons.copy()
        ret1[ g.dofindices ] = c[ len( ret1 ): ]
        ret = f( self, NamedArray( np.concatenate( [ c[ :len( ret1 ) ], ret1 ] ) ), *args, **kwargs )
        try:
            self._feval += 1
        except:
            self._feval = 1
        return ret

    return wrapper


class MixedFEM:

    def __init__( self, *args, eps=0.000, coordinate_directions=None, **kwargs ):

        g, *args = args

        def c0( g ):
            return tuple( i for i in range( 2 ) if g.degree[ i ] in g.knotmultiplicities[ i ] )

        if coordinate_directions is None:
            coordinate_directions = c0( g )

        assert len( coordinate_directions ) in ( 1, 2 ) and all( i in ( 0, 1 ) for i in coordinate_directions )

        if len( coordinate_directions ) == 2:
            raise NotImplementedError

        self._coordinate_directions = coordinate_directions
        self._I = Integrator( g, *args, **kwargs )
        self._g = self._I._g
        self._N = len( self._g.basis )
        self._M = self._I._M.tocsc()
        self._M_inv = sparse.linalg.splu( self._M )
        self._eps = eps

        inheritnames = ( 'zderivs', 'fderivs', 'sderivs', 'metric', 'jitmass', 'jitarray', 'project' )

        for name in inheritnames:
            setattr( self, name, getattr( self._I, name ) )

    @mixed_FEM_BC
    def residual( self, c ):
        u, v, x, y = c.list()
        arr = self.jitarray
        f, s = self.fderivs, self.sderivs
        index = self._coordinate_directions[ 0 ]

        g11, g12, g22 = self.metric( np.concatenate( [ x, y ] ) )
        scale = g11 + g22 + self._eps

        proj = lambda n, m: n - self._M_inv.solve( arr( self.fderivs( m )[ index ] ) )
        res0 = np.concatenate( [ proj( n, m ) for n, m in zip( [ u, v ], [ x, y ] ) ] )

        if index == 0:
            mul0 = g22 * f( u )[ 0 ] - g12 * f( u )[ 1 ] - g12 * s( x )[ 1 ] + g11 * s( x )[ 2 ]
            mul1 = g22 * f( v )[ 0 ] - g12 * f( v )[ 1 ] - g12 * s( y )[ 1 ] + g11 * s( y )[ 2 ]
        else:
            mul0 = g22 * s( x )[ 0 ] - g12 * f( u )[ 0 ] - g12 * s( x )[ 1 ] + g11 * f( u )[ 1 ]
            mul1 = g22 * s( y )[ 0 ] - g12 * f( v )[ 0 ] - g12 * s( y )[ 1 ] + g11 * f( v )[ 1 ]

        res1 = np.concatenate( [ arr( mul0 / scale ), arr( mul1 / scale ) ] )
        return np.concatenate( [ -res0, res1[ self._g.dofindices ] ] )

    def solve( self, order=1, print_feval=True, jac_options={}, **scipyargs ):

        assert order in ( 1, 2 )

        try:
            feval = self._feval
        except AttributeError:
            feval = 0
            self._feval = feval

        scipyargs.setdefault( 'verbose', True )

        N = self._N
        x, y = self._g.x[ :N ], self._g.x[ N: ]
        index = self._coordinate_directions[ 0 ]
        init = np.concatenate( [ -self.project( self.fderivs( k )[ index ] ) for k in ( x, y ) ] + [ self._g.x[ self._g.dofindices ] ] )

        if order == 1:
            jac = optimize.nonlin.KrylovJacobian( **jac_options )
        else:
            jac = SecondOrderKrylovJacobian( **jac_options )

        ret = optimize.nonlin.nonlin_solve( self.residual, init, jacobian=jac, **scipyargs )

        if print_feval:
            feval = self._feval - feval
            entries_computed = 2 * feval * self._N
            entries_jacobian = 4 * self._M.nnz  # this is wrong for mixed-FEM !
            log.info( 'Reached convergence with after {} function evaluations'.format( feval ) )
            log.info( 'This corresponds to {} evaluations of the jacobian'.format( entries_computed / entries_jacobian ) )

        return ret


def fastsolve( g, method='Elliptic', ischeme=None, **scipyargs ):

    assert len( g ) == 2

    if len( g.periodic ) != 0:
        raise NotImplementedError

    if ischeme is None:
        ischeme = g.ischeme

    f = { 'Elliptic': Elliptic, 'Elliptic_unscaled': Elliptic_unscaled,
                                'Elliptic_partial': Elliptic_partial,
                                'MixedFEM': MixedFEM,
                                'Mixed-FEM': MixedFEM,
                                'Mixed-Fem': MixedFEM }[ method ]
    Int = f( g, ischeme )
    sol = Int.solve( **scipyargs )

    if method in ( 'Elliptic', 'Elliptic_unscaled', 'Elliptic_partial' ):
        g.x[ g.dofindices ] = sol

    else:
        g.x[ g.dofindices ] = sol[ len( g.x ): ]

    del Int
