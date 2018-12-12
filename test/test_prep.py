#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, inspect

from mapping import sep

from mapping_2 import *

pcs = sep.separator(0)

pcs = [ pc.PointCloud( p.T ) for p in pcs ]

goal_boundaries = dict( zip( [ 'bottom', 'top', 'left', 'right' ], pcs ) )

bottom = pcs[ 0 ]
top = pcs[ 1 ]

corners = aux.goal_boundaries_to_corners( goal_boundaries )

g = std.TensorGridObject2D

g = prep.univariate_boundary_projection( g, goal_boundaries, corners=corners )

res = prep.residual( g, goal_boundaries )

gs = prep.bivariate_fit( g.empty_copy(), goal_boundaries, corners, mu=1e-7, btol=1e-4 )
