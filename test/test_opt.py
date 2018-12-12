#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nutils import *
import opt

from test_rep import g

import sys, os, inspect

from mapping import sep

from mapping_2 import *

from load_xml import twin_screw
import numpy as np

# g = std.wedge()
# g.cons = util.NanVec( len( g.x ) )
# g.set_cons_from_x()
# g.x[ g.dofindices ] += 0.0*np.random.randn( len( g.dofindices ) )

# g = g.ref( [1, 0] )
g.set_cons_from_x()

sol.solve_nutils( g )

# sol.solve_nutils( g )

g.qplot()

dual_basis = 'exact'

optimizer = opt.Optimizer( g, dual_basis=dual_basis )

solution = opt.solve_scipy( optimizer, method='SLSQP' )

# g.x[ g.dofindices ] = solution[:-1]  # solution.xStar[ 'xvars' ][:-1]

# g.qplot()

optimizer = opt.UnconstrainedOptimizer( g, penalty='logsigmoid', mu=-0.01 )

solution = opt.solve_unconstrained_scipy( optimizer, method='SLSQP' )

g.x[ g.dofindices ] = solution

g.qplot()
