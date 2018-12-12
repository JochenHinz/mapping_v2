#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import ko, go, pc
import numpy as np
from scipy import interpolate, sparse
import scipy as sp
from matplotlib import pyplot as plt
from nutils import *
from collections import defaultdict, ChainMap
from .aux import X


def univariate_regression( g, pointcloud, mu=1e-6 ):
    assert g.targetspace == pointcloud.shape[1]
    assert len( g ) == 1, NotImplementedError

    repeat = g.targetspace

    verts, points = pointcloud.verts, pointcloud.points

    if repeat > 1:
        points = [ pc for pc in points.T ]
    else:
        points = [ points.flatten ]

    X0 = sp.sparse.csr_matrix( X( g.knotvector[0], verts  ) )  # normal equation X

    M = sp.sparse.csr_matrix( X0.T.dot( X0 ) )
    rhs = np.concatenate( [ X0.T.dot( pc ) for pc in points ] )

    if mu not in [ 0, None ]:
        log.info( 'project > Stabilizing via least-distance penalty method' )
        A = g.integrate( function.outer( g.basis.grad( g.geom ) ).sum(-1) ).toscipy()
        log.info( 'project > Building stabilization matrix' )
        M = M + mu*A

    M = sparse.csr_matrix( sparse.block_diag( [M]*repeat ) )

    M = matrix.ScipyMatrix( M )

    g.x = M.solve( rhs, constrain=g.cons )

    return g


def univariate_boundary_projection( g, goal_boundaries, corners={}, **regoptions ):
    assert len( g ) == 2

    if corners:
        assert len( g.periodic ) == 0

        for (i, j), p in corners.items():
            xi, eta = g.sides[:2], g.sides[2:]
            index = ( dict( zip( [0, 1], xi ) )[i], dict( zip( [0, 1], eta ) )[j] )
            g.cons[ g.index.boundary( *index ).flatten ] = p

    for side, goal_boundary in goal_boundaries.items():
        assert isinstance( goal_boundary, pc.PointCloud ), NotImplementedError
        index = g.index.boundary( side ).flatten
        tempgo = g( side ).empty_copy()
        tempgo.cons = g.cons[ index ]  # in case corners have been set
        tempgo = univariate_regression( tempgo, goal_boundaries[ side ], **regoptions )
        g.x[ index ], g.cons[ index ] = [ tempgo.x ]*2

    return g


def pc_residual( g, goals ):
    assert len( g ) == 2, NotImplementedError
    resdict = {}

    for side, goal in goals.items():
        assert isinstance( goal, pc.PointCloud ), NotImplementedError
        tempgo = g( side ).toscipy()
        resdict[ side ] = pc.euclidean_norm( tempgo( goal.verts ) - goal.points )

    return resdict


def withrefinement( func ):
    def wrapped( g, goal_boundaries, corners={}, btol=1e-2, maxref=8, **options ):
        assert len( g ) == 2

        sides = goal_boundaries.keys()
        if not isinstance( btol, dict ):
            btol = dict( zip( sides, [btol]*len(sides) ) )

        assert all( [ isinstance( goal_boundaries[ side ], pc.PointCloud ) for side in sides ] ), \
                    'withrefinement currently only allows for PointCloud inputs'

        assert sides == btol.keys()

        gsides = list( reversed( g.sides ) )
        if not g.periodic:
            sidedict = dict( [ ( side, gsides.index( side ) // 2 ) for side in gsides ] )
        else:  # O-grid, only two sides
            sidedict = defaultdict( lambda: g.periodic[0] )

        gs = [ func( g, goal_boundaries, corners, sidedict, **options ) ]
        i = 0

        while True:
            refvertices = defaultdict( list )
            res = pc_residual( gs[-1], goal_boundaries )
            log.info( '\n'.join( [ "Largest residual at side '{}' is {}, assumed at index {}."
                    .format( side, max( r ), np.argmax( r ) ) for side, r in res.items() ] ) )
            if i > maxref:  # add and residual is larger than threshold
                log.warning( 'Failed to reach convergence threshold after {} iterations'.format(i-1) )
                break
            for side in res.keys():
                index = sidedict[ side ]
                verts = goal_boundaries[ side ].verts[ res[ side ] > btol[ side ] ].tolist()
                refvertices[ index ] = list( set( refvertices[ index ] + verts ) )
            refvertices = [ refvertices[i] for i in range(2) ]
            if all( [ len( i ) == 0 for i in refvertices ] ):
                break
            tempgo = gs[-1].empty_copy().ref_by_vertices( refvertices )
            gs.append( func( tempgo, goal_boundaries, corners, sidedict, **options ) )
            i += 1
        return gs
    return wrapped


@withrefinement
def bivariate_fit(g, goal_boundaries, corners, sidedict, **options ):
    return univariate_boundary_projection( g, goal_boundaries, corners=corners, **options )


@withrefinement
def greville_fit( g, goal_boundaries, corners, sidedict, **options ):
    options.setdefault( 'mu', 0 )

    def splines_to_gb( g ):
        splines = dict( [ ( side, pc.toInterpolatedUnivariateSpline( k=3 ) ) for side, pc in goal_boundaries.items() ] )

        def side_to_pc( side ):
            verts = g.greville()[ sidedict[side] ]
            if not verts[0] == verts[-1]:  # periodic case: repeat last vert
                verts = np.concatenate( [ verts, [ verts[0] ] ] )
            return splines[ side ].toPointCloud( verts )

        gb = dict( [ ( side, side_to_pc(side) ) for side, spl in splines.items() ] )
        return gb

    return univariate_boundary_projection( g, splines_to_gb(g), corners=corners, **options )

# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
