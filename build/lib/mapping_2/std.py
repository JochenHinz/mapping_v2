#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This script is intended as a library for standard instantiations of
    various classes for quick testing
"""

import numpy as np
from . import ko, go, rep
from nutils import *

KnotObject = ko.KnotObject(knotvalues=np.linspace(0, 1, 11), degree=3)
PeriodicKnotObject = ko.KnotObject(knotvalues=np.linspace(0, 1, 11), degree=3, periodic=True)

TensorKnotObject = KnotObject * KnotObject
PeriodicTensorKnotObject = KnotObject * PeriodicKnotObject  # periodic in eta-direction
FullPeriodicTensorKnotObject = PeriodicKnotObject * PeriodicKnotObject

TensorGridObject1D = go.TensorGridObject(knotvector=[KnotObject])
TensorGridObject2D = go.TensorGridObject(knotvector=TensorKnotObject)
TensorGridObject3D = go.TensorGridObject(knotvector=np.prod([KnotObject]*3))

ReparameterizationFunction = rep.ReparameterizationFunction( lambda x: x )


def Cube( dim=2 ):
    g = globals()['TensorGridObject{}D'.format(dim)]
    g.x = g.project(g.geom)
    g.set_cons_from_x()
    return g


def circle( inner=1, outer=2 ):
    g = go.TensorGridObject( knotvector=KnotObject * PeriodicKnotObject )
    def circle( R ):
        return lambda gr: R*function.stack( [function.cos( 2*np.pi*gr[1] ), function.sin( 2*np.pi*gr[1] )] )
    R1, R2 = inner, outer
    func = ( 1 - g.geom[0] )*circle(R1)( g.geom ) + g.geom[0]*circle(R2)( g.geom )
    g.x = g.project( func )
    return g


def wedge():
    g = Cube().copy()
    geom = g.geom
    proj = lambda dom, f: dom.project( f, geometry=geom, onto=g.basis.vector(2), ischeme='gauss12' )
    g.cons = proj( g.domain.boundary['top'], function.stack( [ geom[0], 1 + geom[0]**2 ] ) )
    g.cons |= proj( g.domain.boundary['right'], function.stack( [ 1, 2*geom[1] ] ) )
    g.cons |= proj( g.domain.boundary, geom )
    g.x = g.cons | 0
    from .sol import transfinite_interpolation
    transfinite_interpolation( g )
    return g


def L():

    g = Cube().copy().add_c0( [ [0.5], [] ] )

    x, y = g.geom

    s, p = function.stack, function.piecewise
    bottom = s( [ p( x, [0.5], 0, 4 * ( x - 0.5 ) ), p( x, [0.5], 2*(1 - 2*x), 0 ) ] )
    top = s( [ p( x, [0.5], 1, 1 + 2 * (x - 0.5) ), p( x, [0.5], 2 - 2*x, 1 ) ] )
    left = s( [ y, 2 ] )
    right = s( [ 2, y ] )

    g[ 'right' ] = g( 'right' ).project( right )
    g[ 'left' ] = g( 'left' ).project( left )
    g[ 'bottom' ] = g( 'bottom' ).project( bottom )
    g[ 'top' ] = g( 'top' ).project( top )
    g.set_cons_from_x()

    from . import sol
    sol.transfinite_interpolation( g )

    return g


def horse_shoe( a, b ):
    assert all( i > 1 for i in (a, b) )

    g = Cube().copy()
    xi, eta = g.geom

    s = function.stack

    def reversed_arc( a, b ):
        return s( [ a * function.cos(np.pi * (1 - xi)), b * function.sin(np.pi * (1 - xi)) ] )

    bottom = reversed_arc( 1, 1 )
    top = reversed_arc( a, b )
    left = s( [ 0, -1 - ( a - 1 ) * eta ] )
    right = s( [ 0, 1 + ( a - 1 ) * eta ] )

    g[ 'right' ] = g( 'right' ).project( right )
    g[ 'left' ] = g( 'left' ).project( left )
    g[ 'bottom' ] = g( 'bottom' ).project( bottom )
    g[ 'top' ] = g( 'top' ).project( top )
    g.set_cons_from_x()

    from . import sol
    sol.transfinite_interpolation( g )

    return g
