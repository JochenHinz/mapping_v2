#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from nutils import *
from collections import defaultdict

sidetonumber_reversed = lambda g: dict( [ (side, i//2) for side, i in 
                            zip( reversed( g.sides ), range( len( g.sides ) ) ) ] )

sidetonumber = lambda g: dict( [ (side, i//2) for side, i in 
                            zip( g.sides, range( len( g.sides ) ) ) ] )

bundledsides = lambda g: list( zip( g.sides[::2], g.sides[1::2] ) )

gauss = lambda i: 'gauss{}'.format(i)


class NutilsSpline( function.Array ):
    def __init__( self, splines, position, shape, nderivs=0 ):
        self._splines = splines
        self._position = position
        self._shape = shape
        self._nderivs = nderivs
        super().__init__( args=[position], shape=position.shape + shape, dtype=float )

    def evalf( self, position ):
        assert position.ndim == self.ndim
        shape = position.shape + self._shape
        position = position.ravel()
        ret = self._splines( position, der=self._nderivs ).reshape( shape )
        return ret

    def _derivative( self, var, seen ):
        return NutilsSpline( self._splines, self._position, self._shape, nderivs=self._nderivs+1 ) \
                [ (...,) + (None,)*len( self._shape ) ] \
                * function.derivative( self._position[ (...,) + (None,)*len( self._shape ) ], var, seen=seen )

    def _derivative_( self, var, axes, seen ):
        return NutilsSpline( self._splines, self._position, self._shape, nderivs=self._nderivs+1 ) \
                [ (...,) + (None,)*len( axes ) ] \
                * function.derivative( self._position[ (...,) + (None,)*len( self._shape ) ], var, axes, seen )

    def _edit( self, op ):
        return NutilsSpline( self._splines, function.edit( self._position, op ), self._shape, nderivs=self._nderivs )


def TensorGridObject_to_NutilsSpline( g ):
    if len( g ) == 2:
        if len( g.periodic ) == 0:
            stn = sidetonumber_reversed( g )  # spl['bottom'] = f( geom[0] )
        elif len( g.periodic ) == 1:
            stn = defaultdict( lambda: g.periodic[0] )  # spl = f( geom[ where periodic ] )
        else:
            raise 'Fully periodic TensorGridObject has no sides'
        return dict( [ ( side, NutilsSpline( g(side).toscipy(),
                        g.geom[ stn[side] ], ( g.targetspace, ), nderivs=0 ) )
                for side in g.sides] )
    else:
        raise NotImplementedError


def transfinite_interpolation( g ):
    assert len( g ) == 2
    assert g.targetspace == 2
    basis = g.basis.vector( 2 )
    splines = TensorGridObject_to_NutilsSpline( g )
    bsides = bundledsides( g )
    stn = sidetonumber( g )
    expression = 0
    for l, r in bsides:
        geom = g.geom[ stn[l] ]
        expression += (1 - geom)*splines[l] + geom*splines[r]
    if len( bsides ) > 1:  # corners
        corner = lambda i, j: g.x[ g.index[i, j] ]
        geom = g.geom
        expression += -(1 - geom[0])*(1 - geom[1])*corner(0,0) - geom[0]*geom[1]*corner(-1,-1) \
                -geom[0]*(1 - geom[1])*corner(-1,0) - (1 - geom[0])*geom[1]*corner(0,-1)
    g.x = g.domain.project( expression, onto=basis, geometry=g.geom,
            ischeme=gauss( g.ischeme ), constrain=g.cons )


def jacobian( g, c ):
    return g.basis.vector( 2 ).dot( c ).grad( g.geom )


def metric_tensor( g, c ):
    jacT = function.transpose( jacobian( g, c ) )
    return function.outer( jacT, jacT ).sum( -1 )


def method_library( g, method='Elliptic', degree=None):
    assert len( g ) == 2

    if degree is None:
        degree = g.ischeme * 4

    target = function.Argument( 'target', [ len( g.x ) ] )

    if method in [ 'Elliptic', 'Elliptic_unscaled' ]:
        G = metric_tensor( g, target )
        J = jacobian( g, target )
        ( ( g11, g12 ), ( g21, g22 ) ) = G
        s = function.stack
        G = s( [ s( [ g22, -g12 ] ), s( [ -g12, g11 ] ) ], axis=1 )
        lapl = ( J.grad( g.geom ) * G[ None ] ).sum( [1, 2] )
        res = ( g.basis.vector(2) * lapl ).sum( -1 )
        res /= ( 1 if method == 'Elliptic_unscaled' else ( g11 + g22 ) )
        res = g.domain.integral( res, geometry=g.geom, degree=degree )

    elif method == 'Elliptic_forward':
        res = function.trace( metric_tensor( g, target ) )
        res = g.domain.integral( res, geometry=g.geom, degree=degree ).derivative( 'target' )

    elif method == 'Winslow':
        res = function.trace( metric_tensor( g, target ) ) / function.determinant( jacobian( g, target ) )
        res = g.domain.integral( res, geometry=g.geom, degree=degree ).derivative( 'target' )

    elif method == 'Elliptic_partial':
        x = g.basis.vector( 2 ).dot( target )
        res = ( g.basis.vector( 2 ).grad( x ) * ( g.geom.grad( x )[None] ) ).sum( [1, 2] )
        res = g.domain.integral( res, geometry=x, degree=degree )

    elif method == 'Liao':
        G = metric_tensor( g, target )
        ( ( g11, g12 ), ( g21, g22 ) ) = G
        res = g.domain.integral( g11 ** 2 + 2 * g12 **2 + g22 ** 2, geometry=g.geom, degree=degree ).derivative( 'target' )

    else:
        raise 'Unknown method {}'.format( method )

    return res


def mixed_fem( g, ltol=1e-5, coordinate_directions=None, **solveargs ):
    assert len( g ) == 2
    n = len ( g.x )
    basis = g.basis

    def c0( g ):
        return tuple( [ i for i in range( 2 ) if g.degree[ i ] in g.knotmultiplicities[ i ] ] )

    if coordinate_directions is None:
        coordinate_directions = c0( g )

    assert len( coordinate_directions ) in ( 1, 2 ) and all( [ i in ( 0, 1 ) for i in coordinate_directions ] )

    s = function.stack

    veclength = { 1: 4, 2: 6 }[ len( coordinate_directions ) ]
    target = function.Argument( 'target', [ veclength * len( basis ) ] )

    U = g.basis.vector( veclength ).dot( target )

    G = metric_tensor( g, target[ -n: ] )
    ( ( g11, g12 ), ( g21, g22 ) ) = G
    scale = g11 + g22

    if len( coordinate_directions ) == 2:
        u, v, x_ = U[ : 2 ], U[ 2: 4 ], U[ 4: ]
        expr = function.concatenate( x_.grad( g.geom ) - s( [ u, v ], axis=1 ) )
        res1 = ( basis.vector(4) * expr ).sum( -1 )
        res2 = g22 * u.grad( g.geom )[ :, 0 ] - g12 * u.grad( g.geom )[ :, 1 ] - g12 * v.grad( g.geom )[ :, 0 ] + g11 * v.grad( g.geom )[ :, 1 ]

    if len( coordinate_directions ) == 1:
        index = coordinate_directions[ 0 ]
        u, x_ = U[ : 2 ], U[ 2: 4 ]
        expr = x_.grad( g.geom )[ :, index ] - u
        res1 = ( basis.vector( 2 ) * expr ).sum( -1 )
        if index == 0:
            x_eta = x_.grad( g.geom )[ :, 1 ]
            res2 = g22 * u.grad( g.geom )[ :, 0 ] - g12 * u.grad( g.geom )[ :, 1 ] - g12 * x_eta.grad( g.geom )[ :, 0 ] + g11 * x_eta.grad( g.geom )[ :, 1 ]
        if index == 1:
            x_xi = x_.grad( g.geom )[ :, 0 ]
            res2 = g22 * x_xi.grad( g.geom )[ :, 0 ] - g12 * u.grad( g.geom )[ :, 0 ] - g12 * x_xi.grad( g.geom )[ :, 1 ] + g11 * u.grad( g.geom )[ :, 1 ]

    res = function.concatenate( [ res1, ( basis.vector( 2 ) * res2 ).sum( -1 ) / scale ] )
    res = g.domain.integral( res, geometry=g.geom, degree=g.ischeme*4 )

    mapping0 = g.basis.vector( 2 ).dot( g.x )
    
    cons = util.NanVec( n * ( len( coordinate_directions ) + 1 ) )
    cons[ -n: ] = g.cons
    init = np.concatenate( [ g.domain.project( mapping0.grad( g.geom )[:, i], geometry=g.geom, onto=g.basis.vector( 2 ), ischeme='gauss12' ) for i in coordinate_directions ] + [ g.x ] )

    lhs = solver.newton( 'target', res, lhs0=init , constrain=cons.where ).solve( ltol, **solveargs )

    g.x = lhs[ -n: ]


def solve_nutils( g, method='Elliptic', ltol=1e-5, **solveargs ):
    assert len( g ) == 2

    target = function.Argument( 'target', [ len( g.x ) ] )

    res = method_library( g, method=method )

    init, cons = g.x, g.cons
    lhs = solver.newton( 'target', res, lhs0=init , constrain=cons.where ).solve( ltol, **solveargs )

    g.x = lhs


def compute_residual( g, method='Elliptic', **kwargs ):
    assert len( g ) == 2

    res = method_library( g, method=method, **kwargs )

    from nutils.solver import Integral

    return Integral.multieval( res, arguments={'target': g.x} )[0]


def plot_residual( g, method='Elliptic', thresh=0.1, **kwargs ):
    res = compute_residual( g, method=method, **kwargs )

    res[ g.consindices ] = 0

    g.plot_function( func=[ function.heaviside( ( function.heaviside( g.basis ).vector( g.targetspace )
        .dot( np.abs( res ) ) ** 2 ).sum() ** 0.5 - thresh ) ] )

# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
