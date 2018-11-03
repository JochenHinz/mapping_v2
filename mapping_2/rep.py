#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, sparse
from .aux import isincreasing, uisincreasing, roll_function
from .pc import distance, indexArray, clparam, smoothen_vector, PointCloud
from .go import TensorGridObject
import pynverse
from nutils import util, matrix, log
from functools import wraps


""" One-dimensional Reparameterization """

def monotone_cubic_interpolation(verts, y, repeat=True):
    args = [ np.asarray(i) for i in ( verts, y ) ]
    assert all( [ uisincreasing(vec) for vec in args ] )
    if repeat:
        verts, y = [ np.concatenate( [ v[:-1] - 1, v, v[1:] + 1] ) for v in args ]
    return interpolate.pchip( verts, y )


def approximate_inverse(f, y=np.linspace(0, 1, 100) ):
    slowinverse = pynverse.inversefunc( f, domain=[0, 1] )
    x = slowinverse(y)
    return monotone_cubic_interpolation( y, x )


class ReparameterizationFunction:
    """ Docstring forthcoming """

    ''' Import '''

    @classmethod
    def fromverts(cls, from_verts, to_verts ):
        return cls( monotone_cubic_interpolation( from_verts, to_verts ) )

    @classmethod
    def fromFourier( cls, coeffs ):
        y = np.fft.irfft( coeffs )
        y[0], y[-1] = 0, 1
        return cls( monotone_cubic_interpolation( np.linspace(0, 1, len(coeffs)), y ) )

    def __init__(self, f):
        assert f(0) == 0 and f(1) == 1
        self._f = f

    def __call__(self, x):
        x = np.asarray(x)
        assert np.min(x) >= 0 and np.max(x) <= 1, 'Reparameterization is restricted to the unit interval'
        return self._f(x)

    ''' Inversion '''

    def approximate_inverse(self, nverts=1000):
        ''' return approximate inverse of ReparameterizationFunction '''
        return self.__class__( approximate_inverse( self._f, y=np.linspace( 0, 1, nverts ) ) )

    def inverse( self ):
        ''' Return exact inverse of self '''
        log.warning( 'Computing the exact inverse is slow, use approximate_inverse instead')
        return self.__class__( pynverse.inversefunc( self._f, domain=[0, 1] ) )

    def invert( self, y ):
        ''' return x s.t. self( x ) = y '''
        f = pynverse.inversefunc( self, domain=[0, 1] )
        return f( y )

    ''' Operations returning instances of self.__class__ '''

    def roll(self, amount):
        ''' periodically shift (or roll) the ReparameterizationFunction '''
        assert 0 <= amount <= 1, 'The shift must be between 0 and 1'
        return self.__class__( roll_function( self._f, amount ) )

    def restrict(self, x0, x1):
        ''' restrict self to [x0, x1] '''
        assert 0 <= x0 < x1 <= 1
        y0, y1 = self( [x0, x1] )
        dx, dy = x1 - x0, y1 - y0
        return self.__class__( lambda x: ( self._f( x0 + x*dx ) - y0 ) / dy )

    def split(self, vals):
        if np.isscalar( vals ):
            vals = [vals]
        vals = np.unique( np.concatenate( [[0], np.asarray( vals ), [1]] ) )
        offsets = self( vals )
        return [ self.restrict( i, j ) for i, j in zip( offsets[:-1], offsets[1:] ) ]

    def smoothwhile(self, dx=0.001, dt=1e-5, Tmax=np.inf, maxiter=100, droptol=0.01):
        verts = np.arange(0, 1+dx, dx)
        vec0 = self( verts )
        stop = {'T': Tmax, 'maxiter': maxiter, 'vec': lambda x: ( np.abs( x - vec0 ) <= droptol)[1:-1].all() }
        vec = smoothen_vector(vec0.copy(), dt, stop=stop)
        return self.__class__.fromverts( verts, np.concatenate( [ [0], vec[1:-1], [1] ] ) )

    ''' Export '''

    def toFourier( self, dx=0.01 ):
        return np.fft.rfft( self( np.arange(0, 1+dx, dx) ) )

    ''' Plotting '''

    def plot(self, verts=None):
        if verts is None:
            verts = np.linspace(0, 1, 1000)
        plt.plot(verts, self(verts))
        plt.show()


class MatchedTuple:
    """ Docstring forthcoming """

    def __init__( self, i, j, distance ):
        assert all( [ k >= 0 for k in (i, j, distance) ] )
        self._i, self._j, self._d = i, j, distance

    @property
    def index( self ):
        return ( self._i, self._j )

    @property
    def d( self ):
        return self._d


class Matching( tuple ):
    """ Docstring forthcoming """ 

    ''' Auxilliary functions '''

    def appendfirstlast( self, L ):
        L = tuple( L )
        first, last = ( self[0], ), ( self[-1], )
        return sorted( list( set( first + L + last  ) ), key=lambda x: x.index[0] )

    def keepfirstlast( f ):
        def wrapper( *args, **kwargs ):
            self, *args = args
            self_ = f( self, *args, **kwargs )
            return self.__class__( appendfirstlast( self_ ), self._length )
        return wrapper

    ''' Initialization '''

    def __new__( self, args, length ):
        length = tuple( length )
        assert len( length ) == 2
        assert all( [ i >= 2 for i in length ] )
        self._length = length
        assert args[0].index == (0,0) and args[-1].index == length
        args = sorted( args, key=lambda x: x.index[0] )
        assert isincreasing( [ m.index[1] for m in args ] )
        return tuple.__new__( Matching, sorted( args, key=lambda x: x.index[0] ) )

    ''' Various functions ''' 
           
    def mindist( self ):
        return min( [m.d for m in self] )

    @property
    def indices( self ):
        return [ [ i.index[j] for i in self ] for j in range(2) ]

    ''' Filtering '''

    def delete( self, index ):
        assert index not in [0,len( self ) - 1]
        L = self.appendfirstlast( [ self[j] for j in range( len(self) ) if not j == ( index % len(self) ) ] )
        return self.__class__( L , length = self._length )

    def maxdist( self, maxdist ):
        L =  list( filter( lambda x: x.d <= maxdist, self ) )
        return self.__class__( self.appendfirstlast( L ), self._length )

    def maxfac( self, fac ):
        assert fac > 0
        thresh = fac*self.mindist()
        return self.maxdist( thresh )
    
    def indexdist( self, mindist ):
        assert isinstance( mindist, int )
        assert mindist > 1
        ret = self
        dist = lambda x, y: min( [ i - j for i, j in zip( x.index, y.index ) ] )
        while True:
            if len( ret ) > 2:
                for i in range( 1, len( ret ) - 1 ):
                    if dist( ret[i], ret[i - 1] ) < mindist:
                        ret = ret.delete( i )
                        break
                else:
                    if dist( ret[-1], ret[-2] ) < mindist:
                        ret = ret.delete( -2 )
                    return ret
            else:
                return ret

    def filter( self, fac=16, maxdist=np.inf, indexdist=4):
        return self.maxfac( fac ).maxdist( maxdist ).indexdist( indexdist )

    ''' Exporting '''

    def tofunction( self, fromverts, toverts ):
        fromverts, toverts = [ i[j] for i,j in zip( [ fromverts, toverts ], self.indices ) ]
        return ReparameterizationFunction.fromverts( fromverts, toverts )

    ''' Magic functions '''

    def __getitem__( self, index ):
        ret = super( self.__class__, self ).__getitem__( index )
        if isinstance( index, slice ):
            ret = self.__class__(ret, self._length)
        return ret

    def __getslice__( self, i, j ):
        return self.__getitem__( slice(i, j) )


def match( x, y, D=None, method='hierarchical' ):
    assert isinstance( x, PointCloud ) and isinstance( y, PointCloud )
    assert x.periodic == y.periodic
    if D is None:
        D = distance( x.points, y.points )
    if x.periodic:
        assert D.argmin() == 0, 'Please roll input point clouds to closest match'
    idxArray = indexArray( x.points.shape[0], y.points.shape[0] )
    assert idxArray.shape[:-1] == D.shape
    matches = [ MatchedTuple( *idxArray[0, 0], D[0, 0] ) ]
    lastdist = D[-1, -1] if not x.periodic else D[0, 0]
    matches += [ MatchedTuple( x._points.shape[0] - 1, y._points.shape[0] - 1, lastdist ) ]
    if method == 'hierarchical':
        sl = [ slice( 1, None, None ) ]*2 if x.periodic else [ slice( 1, -1, None) ]*2
        return sorted( matches + match_hierarchical( D[sl], idxArray[sl] ), key=lambda x: x.index[0] )
    raise "Unknown method '{}'".format( method )


def match_hierarchical(D, idxArray):
    """ Hierarchically match closest points

    Parameters
    ---------
    D : an array of distances between points {x}_alpha in point cloud 1
        and {y}_beta in point cloud 2
        D_ij = || x_alpha - y_beta ||
    idxArray : array containing the index information.
               idxArray[i, j] = alpha, beta means
               that D_ij = || x_alpha - y_beta ||

    Returns
    -------
    list of MatchedTuples

    """
    matches = []
    try:
        assert 0 not in D.shape  # no points left to match on at least one of the point clouds
        i, j = np.unravel_index( D.argmin(), D.shape)
        matches.append( MatchedTuple( *idxArray[i, j], np.min(D) ) )
        matches += match_hierarchical( D[:i, :j], idxArray[:i, :j] )
        matches += match_hierarchical( D[i+1:, j+1:], idxArray[i+1:, j+1:] )
    except AssertionError:
        pass
    return matches


def divided_reparam(x, y, weights, fac=16, indexdist=4, **matchargs):
    """ Build repararmeterization functions by matching PointCloud ``x``
        and PointCloud ``y``.

    Parameters
    ----------
    x : first input PointCloud
    y : second input PointCloud
    weights : matched tuples (i,j) receive the parametric value 
              weights[0] * x.verts[i] + weights[1] * y.verts[j], should add up to 1
    fac : matches are kept whenever they are fac * ( minimum distance between x and y ) apart
    indexdist : filters the matching such that consecutive tuples (i0, j0) and (i1, j1) satisfy
                min( [i1 - i0, j1- j0] ) >= indexdist

    Returns
    ------
    [ ReparameterizationFunction, ReparameterizationFunction ]
    """
    assert all( [ isinstance(pc, PointCloud) for pc in (x, y) ] )
    alpha, beta = weights
    assert alpha + beta == 1
    assert alpha >= 0 and beta >= 0
    matches = match( x, y, **matchargs )
    matches = Matching( matches, matches[-1].index )
    matches = matches.filter( fac=fac, indexdist=indexdist )
    idx = matches.indices
    xverts, yverts = clparam( x._points ), clparam( y._points )
    avg = alpha * xverts[ idx[0] ] + beta * yverts[ idx[1] ]
    return [ ReparameterizationFunction.fromverts( verts[i], avg ) for verts, i in zip( [ xverts, yverts ], matches.indices ) ]


def averaged_reparam( x, y, **kwargs ):
    """
        Equivalent to divided_reparam( x, y, ( 0.5, 0.5 ), **kwargs ).
    """
    repfuncs = divided_reparam( x, y, (0.5, 0.5), **kwargs) 
    return repfuncs


def onesided_reparam( leader, follower, **kwargs ):
    """ 
        Equivalent to divided_reparam( leader, follower , ( 1, 0 ), **kwargs ) 
    """
    repfuncs = divided_reparam( leader, follower, (1, 0), **kwargs )
    return repfuncs


""" Bivariate Reparameterization """


class BivariateReparameterizationFunction( TensorGridObject ):

    @classmethod
    def fromReparameterizationFunctions( cls, xis, repfuncs, eta ):
        assert len( xis ) == len( repfuncs ) > 1
        assert all( [ isinstance( rep, ReparameterizationFunction ) for rep in repfuncs ] )
        assert eta > 1

        xis = np.asarray( xis )

        xis_ = xis - xis[ 0 ]
        xis_ /= xis_[ -1 ]

        assert xis_[ 0 ] == 0 and xis_[ -1 ] == 1

        eta = np.linspace( 0, 1, eta )

        from . import ko
        knotvector = [ ko.KnotObject( knotvalues=i, degree=1 ) for i in ( xis_, eta ) ]

        g = cls( knotvector=knotvector )
        g._start, g._end = xis[ 0 ], xis[ -1 ]

        g.x[ :len( g.basis ) ] = xis_.repeat( g.ndims[1] )

        for i in range( len( xis_ ) ):
            g.x[ g.index[ i, : ][ g.ndims[1]: ] ] = repfuncs[ i ]( eta )
        return g

    def __init__( self, *args, **kwargs ):  # make self._start, self._end part of __init__

        super().__init__( *args, **kwargs )

        def _constructor( *args, **kwargs ):
            ret = self.__class__( *args, **kwargs )
            ret._start, ret._end = self._start, self._end
            return ret

        self._constructor = _constructor

    def __call__( self, value, npoints=100 ):
        assert isinstance( value, ( float, int ) )
        xi = np.linspace( 0, 1, npoints )
        value = ( value - self._start ) / ( self._end - self._start )
        return ReparameterizationFunction.fromverts( xi, np.round( self.toscipy()( [ value ], xi )[ ..., 1 ], decimals=10 ) )


# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
