#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse, interpolate
from collections import ChainMap
from nutils import log, numeric, topology, transform, function
from inspect import signature
from . import ko
import numba


def isincreasing(array):
    array = np.asarray(array)
    return np.all(array[1:] - array[:-1] > 0)


def uisincreasing( vec ):
    ''' strictly increasing on the unit interval '''
    return vec[0] == 0 and vec[-1] == 1 and isincreasing(vec)


def withperiodicrepeat( f ):
    '''
        every function call not in the unit interval will be restricted
        to the unit interval with an offset given by x//1
    '''
    def wrapper(x):
        return f( x % 1 ) + x // 1
    return wrapper


def roll_function( f, amount ):
    def ret( x ):
        g = withperiodicrepeat( f )
        offset = g( -amount )
        return g( x - amount ) - offset
    return ret


def unit_vec(length, i):
    ret = np.zeros(length)
    ret[i] = 1
    return ret


def tck(kv, i):
    ''' tck tuple for sp.interpolate.splev'''
    p = kv.degree
    knots = kv.extend_knots()
    vec = unit_vec(kv.dim, i)
    return (knots, vec, p)


def string_to_range(s, l):
    ''' turn a string into a range 'a:b:c' -> range(a,b,c) '''
    s = (s.split(':') + ['', '', ''])[:3]
    f = lambda x: int(x[1]) if not x[1] == '' else dict( [ ((0, ''), 0), ((1, ''), l), ((2, ''), 1) ] )[x]
    return range(*[ f(item) for item in enumerate(s) ])


@log.title
def rectilinear( richshape, periodic=(), name='rect', bnames=None ):
    'rectilinear mesh'

    ndims = len(richshape)
    shape = []
    offset = []
    scale = []
    uniform = True
    for v in richshape:
        if numeric.isint( v ):
            assert v > 0
            shape.append( v )
            scale.append( 1 )
            offset.append( 0 )
        elif np.equal( v, np.linspace(v[0],v[-1],len(v)) ).all():
            shape.append( len(v)-1 )
            scale.append( (v[-1]-v[0]) / float(len(v)-1) )
            offset.append( v[0] )
        else:
            shape.append( len(v)-1 )
            uniform = False

    if isinstance( name, str ):
        wrap = tuple( sh if i in periodic else 0 for i, sh in enumerate(shape) )
        root = transform.RootTrans( name, wrap )
    else:
        assert all( ( name.take(0,i) == name.take(2,i) ).all() for i in periodic )
        root = transform.RootTransEdges( name, shape )

    axes = [ topology.DimAxis(0,n,idim in periodic) for idim, n in enumerate(shape) ]
    topo = topology.StructuredTopology( root, axes, bnames=bnames )

    if uniform:
        if all( o == offset[0] for o in offset[1:] ):
            offset = offset[0]
        if all( s == scale[0] for s in scale[1:] ):
            scale = scale[0]
        geom = function.rootcoords(ndims) * scale + offset
    else:
        funcsp = topo.splinefunc( degree=1, periodic=() )
        coords = numeric.meshgrid( *richshape ).reshape( ndims, -1 )
        geom = ( funcsp * coords ).sum( -1 )

    return topo, geom


@numba.jit(
            numba.float64[ :, : ]
            (
                numba.float64[ : ],
                numba.float64[ : ],
                numba.int64,
                numba.int64,
                numba.int64,
            ),
            nopython=True
        )
def jit_prolongation_matrix( kvnew, kvold, n, m, p ):
    T = np.zeros( ( n, m ), dtype=np.float64 )

    for j in range( m ):
        T[ np.where( np.logical_and( kvold[ j ] <= kvnew[ :n ],
            kvnew[ :n ] < kvold[ j + 1 ] ) )[ 0 ], j ] = 1

    for q in range( 1, p + 1 ):
        T_new = np.zeros( ( n - q, m - q ), dtype=np.float64 )
        for i in range( n - q ):
            for j in np.where(
                    np.logical_or( T[ i, : m - q ] != 0, T[ i, 1: m - q + 1 ] != 0 )
                            )[ 0 ]:
                fac1 = ( kvnew[i + q] - kvold[j] ) / ( kvold[j+q] - kvold[j] ) \
                        if kvold[j+q] != kvold[j] else 0
                fac2 = ( kvold[j + 1 + q] - kvnew[i + q]) / ( kvold[j + q + 1] - kvold[j + 1] ) \
                        if kvold[j + q + 1] != kvold[j + 1] else 0
                T_new[i, j] = fac1 * T[i, j] + fac2 * T[i, j + 1]
        T = T_new

    return T


def prolongation_matrix(kvold, kvnew):
    assert all( isinstance(kv, ko.KnotObject) for kv in (kvold, kvnew) )
    assert_params = [kvnew <= kvold, kvold <= kvnew]
    assert any(assert_params), 'The kvs must be nested'  # check for nestedness
    if all(assert_params):
        return np.eye(kvold.dim)
    p = kvnew.degree
    kv_new, kv_old = [ k.extend_knots() for k in (kvnew, kvold) ]  # repeat first and last knots
    if assert_params[0]:  # kv_new <= kv_old, reverse order
        kv_new, kv_old = kv_old, kv_new
    n = len(kv_new) - 1
    m = len(kv_old) - 1
    T = jit_prolongation_matrix( kv_new, kv_old, n, m, p )
    if kvnew.periodic:  # some additional tweaking in the periodic case
        T_ = T
        T = T[:n-2*p, :m-2*p]
        T[:, 0:p] += T_[:n-2*p, m-2*p: m-2*p+p]
    # return T if kv_new >= kv_old else the restriction
    return T if not assert_params[0] else np.linalg.inv(T.T.dot(T)).dot(T.T)


def smoothen_discrete( vec0, dt, method='finite_difference', stop={ 'T': 0.01 }, minit=10 ):
    '''
        Smoothen vec[1: -1] using `method' until stopping criterion has been reached,
        while vec[0] and vec[-1] are held fixed.
        stop = {
                'T': (finite difference) if t > T, terminate
                'maxiter': (finite difference) if i > maxiter, terminate
                'vec': if not stop[ 'vec' ]( vec ) terminate
               }
    '''
    t, i = 0, 0
    vec = vec0.copy()
    d = ChainMap( stop, {'T': np.inf, 'maxiter': 100, 'vec': lambda x: True} )
    if method == 'finite_difference':
        N = len( vec )
        dx = 1 / ( N - 1 )
        fac = dt / (dx ** 2)
        A = sparse.diags( [ (-fac*np.ones(N-2)).tolist() + [0],
            [1] + ( (1 + 2*fac)*np.ones(N - 2) ).tolist() + [1],
            [0] + ( -fac*np.ones(N-2) ).tolist() ], [-1, 0, 1] ).tocsc()
        A = sparse.linalg.splu( A )
        while True:
            if not all( [ t < d['T'], i < d['maxiter'], d['vec'](vec) ] ):
                if i <= minit:  # timestep too big
                    log.info( 'Initial timestep too big, reducing to {}'.format( dt/10 ) )
                    return smoothen_discrete( vec0, dt/10, method=method, stop=stop )
                break
            vec_n = A.solve( vec )
            vec = vec_n
            t += dt
            i += 1
    else:
        raise "Unknown method '{}'".format( method )

    if d['vec'](vec):
        log.warning( 'Failed to reach the termination criterion' )
    else:
        log.info( 'Criterion reached at t={} in {} iterations'.format( t, i ) )

    return vec


def bnamestonumber(bnames):
    return dict( zip( bnames, np.repeat( np.arange( len(bnames)//2 ), 2  ) ) )


def goal_boundaries_to_corners( goal_boundaries ):
    """ Works only if goal_boundaries.keys() == ( 'left', 'right', 'bottom', 'top' ) """
    sides = ( 'left', 'right', 'bottom', 'top' )
    assert all( [ side in goal_boundaries.keys() for side in sides ] )
    return { (0, 0): goal_boundaries['bottom'].points[0], (1, 0): goal_boundaries['bottom'].points[-1], \
            (0, 1): goal_boundaries[ 'top' ].points[0], (1, 1): goal_boundaries[ 'top' ].points[-1] }


def with_instantiation_dictionary( cls ):
    params = list( signature( cls.__init__ ).parameters )[1:]  # remove self
    # assert not hasattr( cls, 'copy' )
    cls.instdict = lambda self: dict( zip( params, [ self.__dict__[ '_' + i ] for i in params ] ) )
    cls.copy = lambda self: self.__class__( **self.instdict() )
    return cls


def X( kv, verts ):
    """ X for the normal equation """

    assert isinstance( kv, ko.KnotObject )

    n = kv.dim
    knots = kv.extend_knots()
    p = kv.degree

    if kv.periodic:
        nperiodicfuncs = p - kv.knotmultiplicities[0] + 1
        knots = knots[ :np.where( knots == 1 )[0][0] + p + 1 ]
        n += nperiodicfuncs

    Id = np.eye( n )

    def tck( controlpoints ):
        return ( kv.extend_knots(), controlpoints, p )

    X = np.array( [ interpolate.splev( verts, tck( Id[i] ) ) for i in range( n ) ] ).T

    if kv.periodic:
        X[ :, :nperiodicfuncs ] += X[ :, kv.dim: ]
        X = X[ :, :kv.dim ]

    return X


# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
