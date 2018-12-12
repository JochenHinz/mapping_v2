#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mapping_2 import fsol, std
from nutils import log
import numpy as np
import time


def timed_func( f ):
    def wrapper( *args, **kwargs ):
        start = time.time()
        ret = f( *args, **kwargs )
        end = time.time()
        diff = end - start
        log.info( 'It took {} seconds'.format(diff) )
        return ret, diff
    return wrapper


def main():

    g = std.circle().ref([2, 2])
    g.set_cons_from_x()

    fsol.fastsolve( g )

    Int = fsol.Elliptic( g, 40 )

    mul = Int.jacdet(g.x)
    res = timed_func( lambda: Int.jitarray( mul=Int.jacdet(g.x) ) )

    log.info( 'timing Elliptic residual computation' )

    vec = g.x[g.dofindices]

    arr, diff = res()
    arr, diff = res()
    arr, diff = res()
    arr, diff = res()
    arr, diff = res()

    log.info( 'timing Elliptic jacdet computation' )
    jacdet = timed_func( Int.jacdet )

    arr, diff = jacdet( g.x )
    arr, diff = jacdet( g.x )
    arr, diff = jacdet( g.x )
    arr, diff = jacdet( g.x )
    arr, diff = jacdet( g.x )

    mas = timed_func( lambda: Int.jitmass( mul=Int.jacdet(g.x) ) )

    log.info( 'timing mass matrix computation' )

    mass = mas()
    mass = mas()
    mass = mas()
    mass = mas()
    mass, diff_ = mas()

    print( diff_ / mass.nnz, diff / len(arr) )

    g = g.add_c0( [ [0.5], [] ] )
    g.set_cons_from_x()

    Int = fsol.MixedFEM( g, 6 )
    KrylovJac = fsol.MixedFEMSchurKrylovJacobian(Int)

    matvec = timed_func( KrylovJac.matvec )

    init = Int.init()
    init_ = init[ -len(g.dofindices): ]

    KrylovJac.setup( init, Int.jacresidual(init), Int.jacresidual )

    log.info( 'timing MixedFEM Schur matvec computation' )

    mv, diff = matvec(init_)
    mv, diff = matvec(init_)
    mv, diff = matvec(init_)
    mv, diff = matvec(init_)
    mv, diff = matvec(init_)

    res = timed_func( Int.nonlinear_residual )

    log.info( 'timing MixedFEM nonlinear_residual computation' )

    _init = np.concatenate( [ init[ :len(g.x) ], g.x ] )

    arr, diff = res( _init )
    arr, diff = res( _init )
    arr, diff = res( _init )
    arr, diff = res( _init )
    arr, diff = res( _init )


if __name__ == '__main__':
    main()
