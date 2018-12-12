#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, inspect

from mapping import sep
from mapping_2 import *

import numpy as np
from nutils import *

pcs = sep.separator(0)

bottom, top, left, right = [ pc.PointCloud( p.T ) for p in pcs ]

repfuncs = rep.averaged_reparam( left, right )

# repfuncs = [ rep.smoothwhile() for rep in repfuncs ]

left, right = left.reparameterize( repfuncs[0] ), right.reparameterize( repfuncs[1] )

pcs = [ bottom, top, left, right ]

goal_boundaries = dict( zip( [ 'bottom', 'top', 'left', 'right' ], pcs ) )

corners = aux.goal_boundaries_to_corners( goal_boundaries )

g = std.TensorGridObject2D
# 
gs = prep.bivariate_fit( g.empty_copy(), goal_boundaries, corners, mu=1e-7, btol=5e-2 )

# g = std.circle()

# g = g.add_c0( [[], [ 0.15 ]] )

# g.x[ g.dofindices ] += 0.00*np.random.randn( len( g.dofindices ) )

#g = g.ref( [1,1] )
# g.set_cons_from_x()

g = gs[-1]
# g = g.add_c0( [[], [ 0.15 ]] )
g.set_cons_from_x()
# 
sol.transfinite_interpolation( g )
# 
# g.qplot()

#from test_rep import g

sol.solve_nutils( g, method='Elliptic' )

# g.qplot()

# sol.solve_nutils( g, method='Mixed_FEM')

# g_ = g.copy()

# sol.mixed_fem( g )
# sol.solve_nutils( g, method='Elliptic_forward' )
# sol.solve_nutils( g, method='Modified_Winslow' )


g.qplot()
# g.plot( 'test', ref=1 )
