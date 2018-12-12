#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mapping_2 import std, fsol, go
import numpy as np


def main( g: go.TensorGridObject= std.Cube().ref( [ 2, 2 ] ) ):

    g.set_cons_from_x()

    g_ = g.copy()
    g_.x[ g_.dofindices ] += 0.1 * np.random.randn( len( g_.dofindices ) )
    fsol.fastsolve( g_ )
    g_.qplot()

    g_ = std.horse_shoe(2, 10).add_c0( [ [0.5], [] ] ).ref( [1, 1] )
    g_.set_cons_from_x()
    # g_.x[ g_.dofindices ] += 0.1 * np.random.randn( len( g_.dofindices ) )

    fsol.fastsolve( g_, method='MixedFEM' )
    g_.qplot()

    # test symbolic jacobian
    Int = fsol.MixedFEM( g_, 7 )
    init = Int.init()
    eps = 1e-7
    res = Int.jacresidual
    f0 = res( init )
    jac_approx = \
        np.stack(
            [ (res( init + eps * i ) - f0)/eps for i in np.eye(init.shape[0]) ],
            axis=1 )
    jac_exact = Int.jacobian( init )

    from matplotlib import pyplot as plt
    plt.spy( jac_approx - jac_exact.todense(), precision=1e-6 )
    plt.show()


if __name__ == '__main__':
    main()
