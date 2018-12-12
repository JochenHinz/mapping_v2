#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import ko, go
import numpy as np
from scipy import interpolate
import scipy as sp
from matplotlib import pyplot as plt
from nutils import *


def main():

    n = 20
    verts = np.linspace( 0, 1, n )
    # p = np.sin( 50*verts )
    p = np.random.randn( n )
    
    def X( kv, verts ):
        n = kv.dim
        I = np.eye( n )
        def tck( i ):
            return ( kv.extend_knots()[0], I[:, i], kv.degree[0] )
        f = lambda x: np.array( [ interpolate.splev( x, tck( i ) ) for i in range( n ) ] ).T
        return f( verts )

    kv = ko.KnotObject( knotvalues=np.linspace(0, 1, 201), degree=3 )

    g = go.TensorGridObject( knotvector=[kv], targetspace=1 )
    
    X0 = X( g.knotvector, verts )

    I0 = np.array( [ i for i in range( g.dim ) if np.linalg.norm( X0[:, i], np.inf ) < 1e-5 ], dtype=int )
    I_ = np.array( [ i for i in range( g.dim ) if i not in I0 ] )

    print( len(I_) )

    M, rhs = sp.sparse.csr_matrix( X0.T.dot( X0 ) ), X0.T.dot( p )
    M = matrix.ScipyMatrix( M )

    basis0 = g.domain.basis('spline', degree=0)

    zvec = np.zeros( g.dim )
    zvec[ I0 ] = 1
    # zvec[ (I0 + 1)[:-1] ] = 1
    # zvec[ (I0 - 1)[1:] ] = 1
    zvec_ = np.zeros( g.dim )
    zvec_[ I_ ] = 1

    mu = 3 - g.basis.dot( zvec ) + 1*1e-8 * ( g.basis.dot( zvec_ ) + 1e-2*function.heaviside( g.basis.dot( zvec_ ) ) )
    # mu = 1

    A = g.domain.integrate( mu*function.outer( g.basis.grad( g. geom ) ).sum(-1), geometry=g.geom, ischeme='gauss12' )

    cons = util.NanVec( len( g.cons ) )
    # cons[ I0 ] = 0
    cons[0], cons[-1] = p[0], p[-1]

    c = (M + 1e-10*A).solve( rhs, constrain=cons )

    # print( cons, c )


    cons = util.NanVec( len( g.cons ) )
    cons[ I_ ] = c[ I_ ]

    # c = A.solve( constrain=cons )

    x = np.linspace( 0, 1, 100000 )
    h = lambda x: interpolate.splev( x, ( g.extend_knots()[0], c, g.degree[0] ) )
    print( np.linalg.norm( p - h(verts), np.inf ) )
    print( np.abs( p, h(verts) ).argmax() )
    plt.plot( x, h(x) )
    plt.scatter( verts, p )
    plt.show()



if __name__ == '__main__':
    main()
