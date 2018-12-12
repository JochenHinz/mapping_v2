#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from itertools import product
from collections import ChainMap
import inspect
from scipy import interpolate, sparse
from nutils import log
from matplotlib import pyplot as plt
from .aux import uisincreasing, isincreasing


_ = np.newaxis


def indexArray( I, J ):
    """ Return array that satisfies arr[i,j] = [i,j] """
    return np.array( list( product( range(I), range(J) ) ) ).reshape( [I, J, 2] )


def distance( x, y ):
    if len(x.shape) == 1:
        x = x[_]
    if len( y.shape ) == 1:
        y = y[_]
    assert len( x.shape ) == len( y.shape ) == 2 and x.shape[1] == y.shape[1], NotImplementedError
    return ( (x[:, _, :] - y[_, :, :])**2 ).sum(-1)**0.5


def chord_length( arr ):
    return np.concatenate( [ [0], ( ( ( arr[1:] - arr[:-1] )**2 ).sum(1)**0.5 ).cumsum() ] )


def euclidean_norm( arr ):
    return ( arr ** 2 ).sum( 1 ) ** 0.5


def clparam( pc ):
    """ Return chord-length parameterization of an array ``pc`` """
    ret = chord_length( pc )
    return ret / ret[-1]


def rotation_matrix( theta ):
    return np.array( [ [ np.cos( theta ), -np.sin( theta ) ], [ np.sin( theta ), np.cos( theta ) ] ] )


def smoothen_vector( vec0, dt, method='finite_difference', stop={ 'T': 0.01 }, miniter=10 ):
    """
        Smoothen vec[1: -1] using `method' until stopping criterion has been reached,
        while vec[0] and vec[-1] are held fixed.

        Parameters
        ----------
        vec0 : to-be-smoothened vector
        dt : (initial) timestep, is reduced if the convergence criterion is reached
             before #iterations >= ``miniter``
        method : smoothing method
        stop = {
                'T': (finite difference) if t > T, terminate
                'maxiter': (finite difference) if i > maxiter, terminate
                'vec': if not stop[ 'vec' ]( vec ) terminate
               }
        miniter : minimum number of iterations
    """

    t, i = 0, 0
    vec = vec0.copy()
    d = ChainMap( stop, {'T': np.inf, 'maxiter': 100, 'vec': lambda x: True} )
    if method == 'finite_difference':
        N = len( vec )
        dx = 1 / ( N - 1 )
        fac = dt / (dx ** 2)
        A = sparse.diags( [ ( -fac*np.ones(N - 2) ).tolist() + [0],
            [1] + ( ( 1 + 2*fac )*np.ones(N - 2) ).tolist() + [1],
            [0] + ( -fac*np.ones(N - 2) ).tolist() ], [-1, 0, 1] ).tocsc()
        A = sparse.linalg.splu( A )
        while True:
            if not all( [ t < d['T'], i < d['maxiter'], d['vec'](vec) ] ):
                if i <= miniter:  # timestep too big
                    log.info( 'Initial timestep too big, reducing to {}'.format( dt/10 ) )
                    return smoothen_vector( vec0, dt/10, method=method, stop=stop )
                break
            vec = A.solve( vec )
            t += dt
            i += 1
    else:
        raise "Unknown method '{}'".format( method )

    if d['vec'](vec):
        log.warning( 'Failed to reach the termination criterion' )
    else:
        log.info( 'Criterion reached at t={} in {} iterations'.format( t, i ) )

    return vec


def circle( npoints ):
    x = np.linspace( 0, 2*np.pi, npoints )[:-1]
    ret = np.array( [ np.cos( x ), np.sin( x ) ] ).T
    return np.vstack( [ ret, ret[0] ] )


def angle( pc ):
    """ Return mutual angle between points in PointCloud ``pc`` """

    ''' Maybe make this a staticmethod of PointCloud '''

    norm = lambda x: ( x ** 2 ).sum( 1 ) ** 0.5

    if not isinstance( pc, PointCloud ):
        pc = PointCloud( pc, verts=np.linspace( 0, 1, pc.shape[0] ) )

    points = pc._points if not pc.periodic else np.vstack( [ pc.points[ [ -1 ] ], pc._points ] )
    forward = points[ 2: ] - points[ 1: -1 ]
    back = points[ :-2 ] - points[ 1: -1 ]
    expr = np.clip( ( -forward * back ).sum( 1 ) / ( norm( forward ) * norm( back ) ), -1, 1 )

    return np.arccos( expr )


class PointCloud:
    """ Docstring forthcoming """

    def __init__( self, points, verts=None ):
        assert len( points.shape ) == 2, NotImplementedError
        assert points.shape[ 1 ] == 2, NotImplementedError
        periodic = ( points[0] == points[-1] ).all()
        if verts is None:
            verts = clparam( points )
        assert points.shape[0] == len( verts )
        if not periodic:
            assert uisincreasing( verts )
        else:
            verts %= 1
            assert verts[0] == verts[-1]
            assert ( verts >= 0 ).all() and ( verts < 1 ).all()
            verts_ = verts[:-1]
            xi0 = np.argmin( verts_ )
            assert isincreasing( np.roll( verts_, -xi0 ) )
        self._points = points
        self._verts = verts
        self._periodic = periodic

    @property
    def points( self ):
        return self._points if not self.periodic else self._points[ :-1 ]

    @property
    def verts( self ):
        return self._verts if not self.periodic else self._verts[ :-1 ]

    @property
    def periodic( self ):
        return self._periodic

    @property
    def shape( self ):
        return self.points.shape

    def toInterpolatedUnivariateSpline( self, k=3, c0_thresh=0.4, **scipyargs ):
        return InterpolatedUnivariateSpline( self, k=k, c0_thresh=c0_thresh, **scipyargs )

    interpolate = toInterpolatedUnivariateSpline

    def flip( self ):
        verts = 1 - np.flip( self._verts, 0 )
        return self.__class__( np.flip( self._points, 0 ), verts=verts )

    def roll( self, shift ):
        assert self.periodic, 'Rolling non-periodic PointClouds is not allowed'
        verts, points = np.roll( self.verts, shift ), np.roll( self.points, shift, axis=0 )
        verts = ( verts - verts[0] ) % 1
        verts = np.concatenate( [ verts, [ verts[0] ] ] )
        points = np.concatenate( [ points, points[0][_, :] ] )
        return self.__class__( points, verts=verts )

    def reparameterize( self, func ):
        from .rep import ReparameterizationFunction
        if not isinstance( func, ReparameterizationFunction ):
            func = ReparameterizationFunction( func )
        return self.__class__( self._points, verts=func( self._verts ) )

    def is_arclength_parameterized( self ):
        return np.allclose( clparam( self._points )[1:-1], self._verts[1:-1] )

    def angle( self ):
        return angle( self )

    def c0_indices( self, thresh ):
        angle = self.angle()
        ''' in the periodic case the 1 can be omitted because the first point can be C^0 too '''
        return np.where( np.abs( angle ) > thresh )[ 0 ] + { False: 1, True: 0 }[ self.periodic ]

    def plot( self, show=True, **plotkwargs ):
        plotkwargs.setdefault( 's', 0.5 )
        plt.scatter( *self.points.T, **plotkwargs )
        if show:
            plt.show()

    def __len__( self ):
        return self.points.shape[ 0 ]


def roll_to( pc, center=np.array( [0, 0] ), to='right' ):
    assert pc.periodic
    D = pc.points - center
    if to == 'right':
        index = D[:, 0].argmax()
    elif to == 'left':
        index = D[:, 0].argmin()
    elif to == 'top':
        index = D[:, 1].argmax()
    elif to == 'bottom':
        index = D[:, 1].argmin()
    else:
        raise 'Unknown position {}'.format( to )
    return pc.roll( -index )


def roll_to_closest( pc, point ):
    assert pc.periodic
    D = distance( pc.points, point )
    index = D.argmin()
    return pc.roll( -index )


def to_closest_match( x, y, smoothargs={ 'dt': 1e-1, 'dx': 1e-4, 'droptol': 1e-4 }, return_func=False, interpolateargs=None, **reparamargs ):

    assert x.periodic and y.periodic
    assert x.is_arclength_parameterized() and y.is_arclength_parameterized()

    if interpolateargs is None:
        interpolateargs = {}

    D = distance( x.points, y.points )
    i, j = np.unravel_index( D.argmin(), D.shape )
    D = np.roll( np.roll( D, -i, axis=0 ), -j, axis=1 )
    deta = x.verts[i]
    x, y = x.roll( -i ), y.roll( -j )
    from .rep import onesided_reparam
    repfuncs = onesided_reparam( x, y, D=D, **reparamargs )
    x = x.roll( i )
    spl = y.toInterpolatedUnivariateSpline( k=3, **interpolateargs ).shift( -repfuncs[1].invert( -deta % 1 ) )
    y0 = spl(0)
    y = roll_to_closest( y, y0 )
    if distance( spl(0.9999), y.points[0] ) < distance( y0, y.points[0] ):  # not quite happy with this
        y = y.roll( -1 )
    if not np.allclose( y.points[0], y0 ):
        newpoints = np.concatenate( [ y0, y.points, y0 ], axis=0 )  # add ghost point x---x-o-x---x
    else:
        newpoints = y._points
    f = repfuncs[1].smoothwhile( **smoothargs )
    f = f.roll( -f.invert( -deta % 1 ) % 1 )
    y = PointCloud( newpoints )
    if return_func:
        return x, y, f
    return x, y.reparameterize( f )


class UnivariateFunction:
    """ Docstring forthcoming """

    def __init__( self, f, periodic=False ):
        assert inspect.isfunction( f )
        self._f = f
        self._periodic = periodic
        assert self(0).shape[1] == 2, NotImplementedError

    @property
    def periodic( self ):
        return self._periodic

    def __call__( self, x ):
        if np.isscalar( x ):
            x = [x]
        x = np.asarray( x )
        assert len( x.shape ) == 1
        assert ( 0 <= x ).all() and ( x <= 1 ).all()
        return self._f( x )

    def shift( self, dx ):
        return self.__class__( lambda x: self._f(x - dx), periodic=self.periodic )

    def reparameterize( self, func ):
        from .rep import ReparameterizationFunction
        if not isinstance( func, ReparameterizationFunction ):
            func = ReparameterizationFunction( func )
        assert isinstance( func, ReparameterizationFunction )
        return self.__class__( lambda x: self._f( func( x ) ), periodic=self.periodic )

    def translate( self, vec ):
        assert vec.shape == (2,)
        return self.__class__( lambda x: self._f(x) + vec[_], periodic=self.periodic )

    def rotate( self, theta, center=np.array([ 0, 0 ]) ):
        assert center.shape == (2,)
        ret = self.translate( - center )
        def f(x):
            points = ret._f( x )
            return rotation_matrix( theta ).dot( points.T ).T
        dummy = self.__class__( f, periodic=self.periodic )
        return dummy.translate( center )

    def qplot( self, x=np.linspace( 0, 1, 1000 ), show=True, **plotargs ):
        plt.plot( *self(x).T, **plotargs )
        if show:
            plt.show()

    plot = qplot

    def toPointCloud( self, x, verts=None ):
        if self._periodic:
            if x[-1] != x[0]:
                log.warning( 'Warning, a periodic function is evaluated over' + \
                    ' a non-periodic set of points, this is usually a bug' )
        return PointCloud( self(x), verts=verts )

    def restrict( self, x0, x1 ):
        assert 0 <= x0 < x1 <= 1
        if self.periodic:
            if [ x0, x1 ] == [ 0, 1 ]:
                return self
        return self.__class__( lambda x: self._f( ( 1 - x ) * x0 + x * x1 ), periodic=False )

    def stack( self, other, xi ):
        assert 0 < xi < 1
        assert not self.periodic and not other.periodic

        def stacked_func( x ):
            x = np.asarray( x )
            assert len( x.shape ) == 1
            out = np.empty( x.shape + ( 2, ) )
            indices = x < xi
            out[ indices ] = self._f( x[ indices ] / xi )
            out[ ~indices ] = other._f( ( x[ ~indices ] - xi ) / ( 1 - xi ) )
            return out

        return self.__class__( stacked_func, periodic=self.periodic )

    @staticmethod
    def stack_multiple( funcs, xis, periodic=False ):
        assert len( funcs ) == len( xis ) + 1
        if not isinstance( xis, list ):
            xis = xis.tolist()
        xis = [ 0 ] + xis + [ 1 ]

        def stacked_funcs( x ):
            x = np.asarray( x )
            assert len( x.shape ) == 1
            if periodic:
                x = x % 1
            out = np.empty( x.shape + ( 2, ) )
            for i in range( len( xis ) - 1 ):
                if not i == len( xis ) - 2:
                    indices = np.logical_and( x >= xis[ i ], x < xis[ i+1 ] )
                else:
                    indices = np.logical_and( x >= xis[ -2 ], x <= xis[ -1 ] )
                vals = ( x[ indices ] - xis[ i ] ) / ( xis[ i + 1 ] - xis[ i ] )
                out[ indices ] = funcs[ i ]._f( vals )
            return out

        return UnivariateFunction( stacked_funcs, periodic=periodic )


def InterpolatedUnivariateSpline( pc, k=3, c0_thresh=None, **scipyargs ):
    if isinstance( pc, np.ndarray ):
        pc = PointCloud( pc )
    assert isinstance( pc, PointCloud )

    if k > len( pc ) + 1:
        log.warning( 'Warning, the number of points is too small for a {}-th order interpolations, \
                it will be clipped.'.format( k ) )
        k = min( k, len( pc ) + 1 )

    if c0_thresh is None:
        periodic = pc.periodic

        def f( x ):
            verts, points = pc.verts, pc.points
            if periodic:  # repeat points to acquire (nearly) continuous derivative at x = 0
                verts = np.concatenate( [ verts - 1, verts, verts + 1 ] )
                points = np.vstack( [ points ] * 3 )
            splines = ( interpolate.InterpolatedUnivariateSpline( verts, i, k=k, **scipyargs ) for i in points.T )
            return np.stack( [ s(x % 1 if periodic else x) for s in splines ], axis=1 )

        return UnivariateFunction( f, periodic=periodic )

    else:
        c0_indices = pc.c0_indices( thresh=c0_thresh )

        if len( c0_indices ) == 0:
            return InterpolatedUnivariateSpline( pc, c0_thresh=None, k=k, **scipyargs )

        shift_amount = 0

        if pc.periodic:  # roll pointcloud to first C0-index, then it can be treated as an unperiodic Pointcloud
            shift_amount = pc.verts[ c0_indices[ 0 ] ]
            pc = pc.roll( -c0_indices[ 0 ] )
            c0_indices = ( c0_indices - c0_indices[ 0 ] )[ 1: ]  # first index is gonna be added anyways

        c0_indices = [ 0 ] + c0_indices.tolist() + [ pc._points.shape[ 0 ] - 1 ]  # pc._points because in the periodic case we repeat the last

        funcs = [ InterpolatedUnivariateSpline( pc._points[ i: j+1 ], c0_thresh=None, k=k, **scipyargs ) for i, j in zip( c0_indices[ :-1 ], c0_indices[ 1: ] ) ]

        return UnivariateFunction.stack_multiple( funcs, pc._verts[ c0_indices[ 1: -1 ] ], periodic=pc.periodic ).shift( shift_amount )


line = lambda p0, p1: UnivariateFunction( lambda x: p0[ _ ] * ( 1 - x[:, _] ) + p1[ _ ] * x[:, _], periodic=False )

# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
