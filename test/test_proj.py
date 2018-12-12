#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from mapping import sep
from mapping_2 import proj, pc, std, aux


def main():

    pcs = sep.separator( 0 )

    sides = [ 'bottom', 'top', 'left', 'right' ]

    pcs = dict( zip( sides, [ pc.PointCloud( p.T ) for p in pcs ] ) )

    corners = aux.goal_boundaries_to_corners( pcs )

    goal_boundaries = { side: p.interpolate( k=3 ) for side, p in pcs.items() }

    targets = dict( zip( sides, [ np.linspace( 0, 1, 100 ) ] * 4 ) )

    g = std.Cube().empty_copy()

    import ipdb
    ipdb.set_trace()

    gs = proj.fit_UnivariateFunction( g, goal_boundaries, targets, corners=corners, maxref=8 )

    gs[-1].qbplot()

    gs[-1].qgplot()


if __name__ == '__main__':
    main()
# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
