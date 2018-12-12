#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, inspect

from mapping import sep

from mapping_2 import *

from load_xml import twin_screw
import numpy as np

"""
l, r = sep.separator(0)[2:]

l = l.T
r = r.T

L = rep.match( pc.PointCloud( l ), pc.PointCloud( r ) )
M = rep.Matching( L, L[-1].index )
f = M.maxfac( 16 ).tofunction( rep.clparam( l ), rep.clparam( r ) )


g = std.TensorGridObject2D

left, right = [ pc.PointCloud( r*( pc.rotation_matrix(t).dot(pc.circle(100).T) ).T ) for r, t
    in zip( [1, 2], [0, 0] ) ]

goal_boundaries = { 'left': left, 'right': right }

g = go.TensorGridObject( knotvector=std.KnotObject*std.PeriodicKnotObject )
gs = prep.bivariate_fit( g, goal_boundaries )
g = gs[-1]
sol.transfinite_interpolation( g )
g.qplot()
"""

casing = pc.PointCloud( 36.1*pc.rotation_matrix( 1 ).dot( pc.circle( 4001 ).T ).T )
female = pc.roll_to_closest( pc.PointCloud( twin_screw()[1] - np.array( [56.52, 0] ) ), casing.points[0] )
male = pc.roll_to_closest( pc.PointCloud( twin_screw()[0] - np.array( [0, 0] ) ), casing.points[0] )

casing = casing.flip()
female = female.flip()
male = male.flip()

casing, male = pc.to_closest_match( casing, female, indexdist=4, interpolateargs={ 'c0_thresh': 0.4 } )

goal_boundaries = { 'left': casing, 'right': male }

kv = ko.KnotObject( knotvalues=np.linspace(0, 1, 11), degree=3 )
kv_periodic = ko.KnotObject( knotvalues=np.linspace(0, 1, 11), degree=3, periodic=True )


g = go.TensorGridObject( knotvector=kv*kv_periodic, ischeme=12 )

gs = prep.bivariate_fit( g, goal_boundaries, maxref=12, mu=1*1e-5, btol=1e-2 )

g = gs[-1]

sol.transfinite_interpolation( g )

g.qplot( ischeme='bezier10' )
