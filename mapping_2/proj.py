#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .prep import *
from .aux import uisincreasing
from . import pc
from collections import defaultdict
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from nutils import *


def fit_UnivariateFunction( g, funcs, targets, corners={}, btol=1e-2, maxref=8, **options ):

    assert len( g ) == 2

    assert all( [ isinstance( val, pc.UnivariateFunction ) for key, val in funcs.items() ] )

    assert all( [ uisincreasing( val ) for key, val in targets.items() ] )

    targets = { side: funcs[ side ].toPointCloud( targets[ side ] ) for side in targets.keys() }

    sides = funcs.keys()
    if not isinstance( btol, dict ):
        btol = dict( zip( sides, [btol]*len(sides) ) )

    assert sides == btol.keys()

    gsides = list( reversed( g.sides ) )
    if not g.periodic:
        sidedict = dict( [ ( side, gsides.index( side ) // 2 ) for side in gsides ] )
    else:  # O-grid, only two sides
        sidedict = defaultdict( lambda: g.periodic[0] )

    proj = lambda g: univariate_boundary_projection( g, { side: funcs[ side ].toPointCloud( g.greville()[ sidedict[ side ] ] ) for side in sides }, corners=corners, mu=0 )

    gs = [ proj( g ) ]
    i = 0

    while True:
        refvertices = {}
        res = pc_residual( gs[-1], targets )
        log.info( '\n'.join( [ "Largest residual at side '{}' is {}, assumed at index {}."
                .format( side, max( r ), np.argmax( r ) ) for side, r in res.items() ] ) )
        if i > maxref:
            log.warning( 'Failed to reach convergence threshold after {} iterations'.format(i-1) )
            break
        for side in res.keys():
            index = sidedict[ side ]
            verts = targets[ side ].verts[ res[ side ] > btol[ side ] ].tolist()
            refvertices[ index ] = list( set( refvertices.get( index, [] ) + verts ) )
        refvertices = [ refvertices[i] for i in range(2) ]
        if all( [ len( i ) == 0 for i in refvertices ] ):
            break
        tempgo = gs[-1].empty_copy().ref_by_vertices( refvertices )
        gs.append( proj( tempgo ) )
        i += 1

    return gs

# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
