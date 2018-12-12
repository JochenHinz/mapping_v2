#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This module contains a selection of methods that compute
    control maps from ``go.TensorGridObject`` instantiations.
"""

import numpy as np
from nutils import function, log
from . import sol


def geometry_to_orthogonality( g, sides=None ):

    '''
        Compute a control mapping that improves the orthogonality
        by the boundaries.

        Parameters
        ----------

        g - go.TensorGridObject containing the target boundaries
        sides - Tuple containing the sides at which to orthogonalize
    '''

    allsides = g.sides

    if sides is None:
        sides = allsides

    x = g.mapping
    domain = g.domain
    basis = g.basis
    stiff = \
        g.integrate(
            function.outer( basis.grad(x, ndims=2) ).sum(2),
            geometry=x
        )

    stdargs = { 'onto': basis, 'geometry': x, 'ischeme': 'gauss6' }

    cons = g.domain.boundary['left'].project( 0, **stdargs )
    cons |= g.domain.boundary['right'].project( 1, **stdargs )

    s = stiff.solve( constrain=cons )

    cons = g.domain.boundary['bottom'].project( 0, **stdargs )
    cons |= g.domain.boundary['top'].project( 1, **stdargs )

    t = stiff.solve( constrain=cons )

    f = g.empty_copy()
    f.x = np.concatenate( [s, t] )

    sidedict = sol.TensorGridObject_to_NutilsSpline( f )

    nspline = sol.NutilsSpline
    stn = sol.sidetonumber_reversed(f)

    for side in allsides:
        if side not in sides:
            g_ = g(side).copy()
            g_.x = \
                np.concatenate( [g_.project( g_.geom[0] )] * 2 )
            spl = \
                nspline(
                        g_.toscipy(),
                        g.geom[ stn[side] ],
                        ( g.targetspace, ),
                        nderivs=0 )
            sidedict[side] = spl

    xi, eta = f.geom

    # hermite interpolation
    f0 = sidedict['bottom'][0] * (1 + 2 * eta) * (1 - eta) ** 2 + \
        sidedict['top'][0] * (3 - 2 * eta) * eta ** 2
    f1 = sidedict['left'][1] * (1 + 2 * xi) * (1 - xi) ** 2 + \
        sidedict['right'][1] * (3 - 2 * xi) * xi ** 2

    f.x = f.project( function.stack( [f0, f1] ), onto=basis.vector(2) )

    return f
