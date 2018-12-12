#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, os, sys, inspect
import itertools

from mapping_2 import *
import numpy as np
from nutils import mesh
from matplotlib import pyplot as plt

KnotObject = ko.KnotObject

SimpleKnotOject = tb.UniformKnotObject(10, 3, periodic=False)
PeriodicKnotObject = tb.UniformKnotObject(10, 3, periodic=True)

sidedict = ('left', 'right', 'bottom', 'top', 'front', 'back')


def validsidecombinations(length):
    bundledsidedict = list( zip(sidedict[::2], sidedict[1::2]) )[:2*length]
    return [ tuple(bundledsidedict[i][j] for i, j in zip( range(length), k ) ) for k in itertools.product(*[[0, 1]]*length) ]


class testTensorGridObject( unittest.TestCase ):

    def test_init(self):
        # knotvector needs to be provided
        self.assertRaises(AssertionError, go.TensorGridObject, knotvector=None)

        # incompatible knotvector and domain, geom
        halfperiodic = SimpleKnotOject*PeriodicKnotObject

        domain, geom = mesh.rectilinear( halfperiodic.knots, periodic=halfperiodic.periodic )

        self.assertRaises( AssertionError, go.TensorGridObject, knotvector=SimpleKnotOject * SimpleKnotOject, domain=domain, geom=geom )

        # domain provided but not geom
        self.assertRaises(AssertionError, go.TensorGridObject, knotvector=halfperiodic, domain=domain)

        # initialization with faulty knotvector
        kv = [SimpleKnotOject, 'bla']
        self.assertRaises(ValueError, go.TensorGridObject, knotvector=kv)
        self.assertRaises(AssertionError, go.TensorGridObject, knotvector=SimpleKnotOject * SimpleKnotOject * SimpleKnotOject, domain=domain, geom=geom)

        # targetspace < len(self)
        self.assertRaises(AssertionError, go.TensorGridObject, knotvector=SimpleKnotOject * SimpleKnotOject, targetspace=1)

        # knotvector too large
        kv = [SimpleKnotOject]*4
        self.assertRaises(AssertionError, go.TensorGridObject, knotvector=kv)

    def test_props(self):

        tkv = SimpleKnotOject * SimpleKnotOject.ref(1)

        '''knotmultiplicities'''
        g = go.TensorGridObject( knotvector=tkv )
        self.assertTrue( all((i==j).all() for i, j
            in zip(g.knotmultiplicities, [SimpleKnotOject.knotmultiplicities, SimpleKnotOject.ref(1).knotmultiplicities]) ) )

        '''knots'''
        self.assertTrue( all((i==j).all() for i, j
            in zip(g.knots, [SimpleKnotOject.knots, SimpleKnotOject.ref(1).knots]) ) )

        '''degree'''
        self.assertTrue( g.degree == [3, 3] )

        '''periodic'''
        self.assertTrue( g.periodic == () )
        halfperiodic = PeriodicKnotObject * SimpleKnotOject
        self.assertTrue( go.TensorGridObject( knotvector=SimpleKnotOject*PeriodicKnotObject ).periodic == (1,) )
        self.assertTrue( go.TensorGridObject( knotvector=halfperiodic ).periodic == (0,) )
        self.assertTrue( go.TensorGridObject( knotvector=PeriodicKnotObject * PeriodicKnotObject ).periodic == (0, 1) )

        '''ndims'''
        self.assertTrue( go.TensorGridObject( knotvector=SimpleKnotOject * SimpleKnotOject * SimpleKnotOject.ref(1) ).ndims == [13]*2 + [23] )
        self.assertTrue( go.TensorGridObject( knotvector=SimpleKnotOject * PeriodicKnotObject * SimpleKnotOject.ref(1) ).ndims == [13] + [10] + [23] )

    def test_withinheritfuncs(self):
        kv = SimpleKnotOject * SimpleKnotOject
        halfperiodic = SimpleKnotOject * PeriodicKnotObject
        g = lambda x: go.TensorGridObject( knotvector=x )
        knots = np.concatenate([ [0.]*3, np.linspace(0, 1, 11), [1.]*3 ])
        knots_periodic = np.linspace(-0.3, 1.3, 17)
        self.assertTrue( all( np.isclose(i, j).all() for i, j in zip( g(kv).extend_knots(), [ knots, knots ] ) ) )
        self.assertTrue( all( np.isclose(i, j).all() for i, j in zip( g(halfperiodic).extend_knots(), [ knots, knots_periodic ] ) ) )

    def test_basis_dim(self):

        # test basis, dim
        knotvector = SimpleKnotOject * SimpleKnotOject
        g = go.TensorGridObject( knotvector=knotvector )
        self.assertTrue( g.dim, knotvector.dim*2 )

    def test_indices(self):
        # test ndofs, dofindices, consindices
        n, m = 10, 13
        tkv = np.prod([ ko.KnotObject( knotvalues=np.linspace(0, 1, k), degree=3 ) for k in (n, m) ])
        g = go.TensorGridObject( knotvector=tkv )
        repeat = lambda x: np.concatenate([ x, x + g.dim // g.targetspace  ])
        self.assertTrue( g.ndofs, g.dim  )
        self.assertTrue( ( g.dofindices == np.arange(g.dim) ).all() )
        g.set_cons_from_x()
        self.assertTrue( g.ndofs, g.dim - 2*tkv.ndims[0] - 2*tkv.ndims[1] + 4 )
        tensorindices = g.tensorindices[ 1:-1, 1:-1 ].flatten()
        self.assertTrue( ( g.dofindices == repeat( tensorindices.flatten() ) ).all() )
        self.assertTrue( ( set( g.consindices ) == ( set( list( range(g.dim) ) ) - set( list(g.dofindices) ) ) ) )

    def test_getitem(self):

        dims = 10, 13
        tkv = np.prod([ ko.KnotObject( knotvalues=np.linspace(0, 1, k), degree=3 ) for k in dims ])
        g2D = go.TensorGridObject( knotvector=tkv )
        g = g2D
        g2D.x = np.arange( g.dim )
        repeat = lambda x: np.concatenate([ x, x + g.dim//2 ])
        self.assertTrue( ( g['left'] == repeat( np.arange(g.ndims[1]) ) ).all() )
        self.assertTrue( ( g['left', 'top'] == repeat( np.array([ g.ndims[1] - 1 ]) ) ).all() )
        self.assertTrue( ( g['left', 'bottom'] == repeat( np.array([ 0 ]) ) ).all() )
        self.assertTrue( ( g['bottom'] == repeat( np.arange(0, g.dim//2 - g.ndims[1] + 1, g.ndims[1]) ) ).all() )
        self.assertTrue( ( g['right'] == repeat( np.arange(g.dim//2 - g.ndims[1], g.dim//2) ) ).all() )
        self.assertTrue( ( g[:g.ndims[1]] == g['left'][:g.ndims[1]] ).all() )
        for i in validsidecombinations(2):
            self.assertTrue( (g.__getitem__(i) == g.__getitem__( tuple( reversed(i) ) ) ).all() )

        dims = 10, 13, 15
        repeat = lambda x: np.concatenate([ x, x + g.dim//3, x + 2*g.dim//3 ])
        tkv = np.prod([ ko.KnotObject( knotvalues=np.linspace(0, 1, k), degree=3 ) for k in dims ])
        g3D = go.TensorGridObject( knotvector=tkv )
        g = g3D
        g3D.x = np.arange( g.dim )
        for i in validsidecombinations(3):
            self.assertTrue( (g.__getitem__(i) == g.__getitem__( tuple( reversed(i) ) ) ).all() )
        self.assertTrue( ( g['front'] == repeat( np.arange(g.dim//3).reshape(g.ndims)[..., 0].flatten()  )).all() )
        self.assertTrue( ( g['left'] == repeat( np.arange( g.ndims[1]*g.ndims[2] ) )).all() )
        self.assertTrue( ( g['left'] == repeat( np.arange(g.dim//3).reshape(g.ndims)[0].flatten()  )).all() )

        # Periodic

        dims = 13, 14
        tkv = ko.KnotObject( knotvalues=np.linspace(0, 1, dims[0]), degree=3 ) \
            * ko.KnotObject( knotvalues=np.linspace(0, 1, dims[1]), degree=3, periodic=True )
        g = go.TensorGridObject( knotvector=tkv )

        self.assertRaises( IndexError, g.__getitem__, 'top' )
        self.assertRaises( IndexError, g.__getitem__, 'bottom' )
        self.assertRaises( IndexError, g.__getitem__, ('bottom', 'left') )
        self.assertRaises( IndexError, g.__getitem__, ('top', 'right') )
        self.assertRaises( IndexError, g.__getitem__, ('left', 'top') )

    def test_setitem(self):

        dims = 10, 13
        tkv = np.prod([ ko.KnotObject( knotvalues=np.linspace(0, 1, k), degree=3 ) for k in dims ])
        g2D = go.TensorGridObject( knotvector=tkv )

        # assuming that __getitem__ works the following tests make sense

        g2D.x = np.arange(g2D.dim)
        g_test = g2D.empty_copy()
        g_test['left'] = g2D['left']
        self.assertTrue( (g_test['left'] == g2D['left']).all() )
        g_test = g_test.empty_copy()

        for side in validsidecombinations(2):
            g_test.__setitem__( side, g2D.__getitem__(side) )
            self.assertTrue( ( g_test.__getitem__(side) == g2D.__getitem__(side) ).all() )

        dims = 10, 13, 17
        tkv = np.prod([ ko.KnotObject( knotvalues=np.linspace(0, 1, k), degree=3 ) for k in dims ])
        g3D = go.TensorGridObject( knotvector=tkv )
        g3D.x = np.arange( g3D.dim )
        g_test = g3D.empty_copy()

        for side in validsidecombinations(3):
            g_test.__setitem__( side, g3D.__getitem__(side) )
            self.assertTrue( ( g_test.__getitem__(side) == g3D.__getitem__(side) ).all() )

        # Periodic

        g2D = go.TensorGridObject( knotvector=SimpleKnotOject * PeriodicKnotObject )
        repeat = lambda x: np.concatenate([ x, x + g2D.dim//2 ])
        self.assertRaises( IndexError, lambda x: g2D.__setitem__('bottom', x), repeat( np.arange(SimpleKnotOject.dim) ) )
        self.assertRaises( IndexError, lambda x: g2D.__setitem__('top', x), repeat( np.arange(SimpleKnotOject.dim) ) )

    def test_toscipy( self ):

        ''' 1D '''
        x = np.linspace(0, 1, 70)

        g = go.TensorGridObject( knotvector=SimpleKnotOject * SimpleKnotOject ).ref_by( [ [1, 2, 3, 5], []] )
        g.x = np.random.randn( len( g.x ) )
        g_ = g( 'bottom' )
        _g_ = g_ #g_.ref_by( [ [ 0, 5, 6 ] ] )
        self.assertTrue( np.allclose( g_.toscipy()( x ), _g_.toscipy()( x ) )  )
        domain_ = _g_.domain.locate( g.geom[0], _g_.knots[0], eps=1e-7 )
        self.assertTrue( np.allclose( domain_.elem_eval( _g_.mapping, ischeme='vertex' ), _g_.toscipy()( _g_.knots[0] ) ) )

        g_ = go.TensorGridObject( knotvector=SimpleKnotOject )
        g_.x = np.random.randn( len( g_.x ) )
        _g_ = g_.ref_by( [ [ 0, 3, 7 ] ] )
        self.assertTrue( np.allclose( g_.toscipy()( x ), _g_.toscipy()( x ) )  )

        g_ = go.TensorGridObject( knotvector=PeriodicKnotObject, targetspace=3 )
        g_.x = np.random.randn( len( g_.x ) )
        _g_ = g_.ref_by( [ [ 0, 3, 7 ] ] ).raise_multiplicities( [2], [ [0] ] )
        self.assertTrue( np.allclose( g_.toscipy()( x ), _g_.toscipy()( x ) )  )

        g_ = go.TensorGridObject( knotvector=PeriodicKnotObject, targetspace=3 )
        g_.x = np.random.randn( len( g_.x ) )
        _g_ = g_.ref_by( [ [ 0, 3, 7 ] ] ).raise_multiplicities( [2], [ [1] ] )
        self.assertTrue( np.allclose( g_.toscipy()( x ), _g_.toscipy()( x ) )  )

        ''' 2D '''

        y = np.linspace(0, 1, 100)

        def plot( gr ):
            f = gr.toscipy()( x, y )
            plt.scatter( f[..., 0].flatten(), f[..., 1].flatten(), s=0.2 )
            # plt.show()

        g = go.TensorGridObject( knotvector=SimpleKnotOject * SimpleKnotOject )
        g.x = np.random.randn( len( g.x ) )
        g_ = g.ref_by( [ [ 1, 4, 6 ], [2, 7, 8] ] )
        self.assertTrue( np.allclose( g.toscipy()( x, y ), g_.toscipy()( x, y ) ) )

        g = go.TensorGridObject( knotvector=SimpleKnotOject * SimpleKnotOject )
        g.x = np.random.randn( len( g.x ) )
        g_ = g.ref_by( [[ 1, 4, 6 ], [2, 7, 8]] ).raise_multiplicities( [ 3, 3 ], [ [4], [6] ] )
        self.assertTrue( np.allclose( g.toscipy()( x, y ), g_.toscipy()( x, y ) ) )

        g = go.TensorGridObject( knotvector=PeriodicKnotObject * SimpleKnotOject )
        g.x = np.random.randn( len( g.x ) )
        g_ = g.ref_by( [[ 1, 4, 6 ], [2, 7, 8]] ).raise_multiplicities( [ 2, 2 ], [ [0], [1] ] )
        self.assertTrue( np.allclose( g.toscipy()( x, y ), g_.toscipy()( x, y ) ) )

        g = std.circle()
        g_ = g.ref_by( [[ 1, 4, 6 ], [0, 7, 8]] )
        self.assertTrue( np.allclose( g.toscipy()( x, y ), g_.toscipy()( x, y ) ) )
        g_ = g_.raise_multiplicities( [0, 1], [[], [0]] )
        self.assertTrue( np.allclose( g.toscipy()( x, y ), g_.toscipy()( x, y ) ) )

        g = go.TensorGridObject( knotvector=PeriodicKnotObject * PeriodicKnotObject )
        g.x = np.random.randn( len( g.x ) )
        g_ = g.ref_by( [ [ 1, 4, 6 ], [2, 7, 8] ] ).raise_multiplicities( [ 2, 2 ], [ [0], [0] ] )
        self.assertTrue( np.allclose( g.toscipy()( x, y ), g_.toscipy()( x, y ) ) )
        plt.show()

        g = go.TensorGridObject( knotvector=PeriodicKnotObject * PeriodicKnotObject )
        g.x = np.random.randn( len( g.x ) )

if __name__ == '__main__':
    unittest.main()


# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
