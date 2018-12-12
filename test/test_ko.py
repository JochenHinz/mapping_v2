#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest, os, sys, inspect

from mapping_2 import *
import numpy as np
from nutils import *

KnotObject = ko.KnotObject

SimpleKnotOject = tb.UniformKnotObject(10, 3, periodic=False)
PeriodicKnotObject = tb.UniformKnotObject(10, 3, periodic=True)


class TestKnotObject(unittest.TestCase):
    '''Test the functionality of the KnotObject'''

    def test_nelems(self):
        self.assertEqual(SimpleKnotOject.nelems, 10)
        self.assertEqual(PeriodicKnotObject.nelems, 10)

    def test_dim(self):
        ''' Test if KO.dim yields the same as Nutils'''

        def TestAgainstNutils(KO):
            domain, geom = mesh.rectilinear([KO.knots], periodic=(0,) if KO.periodic else ())
            basis = domain.basis('spline', degree=KO.degree, knotmultiplicities=[KO.knotmultiplicities])
            return KO.dim == len(basis)

        # simple uniform Knotobject
        self.assertTrue(TestAgainstNutils(SimpleKnotOject))

        # simple uniform periodic Knotobject
        self.assertTrue(TestAgainstNutils(PeriodicKnotObject))

        # more advanced Knotobject with repetitions
        KO = KnotObject(knotvalues=[0, 0.2, 0.4, 0.5, 0.55, 0.65, 0.7, 0.8, 1], knotmultiplicities=[4, 1, 3, 3, 2, 2, 1, 1, 4], degree=3)
        self.assertTrue(TestAgainstNutils(KO))

        #more advanced periodic Knotobject
        KO = KnotObject(knotvalues=[0, 0.2, 0.4, 0.5, 0.55, 0.65, 0.7, 0.8, 1], knotmultiplicities=[2, 1, 3, 3, 2, 2, 1, 1, 2], degree=3, periodic=True)
        self.assertTrue(TestAgainstNutils(KO))

    def test_expand_knots(self):
        ''' test if expand_knots works '''
        vec = np.concatenate([ [0]*3, np.linspace(0, 1, 11), [1]*3 ])
        self.assertTrue(np.allclose(SimpleKnotOject.expand_knots(), vec))

        # Periodic Knotobject
        vec = np.linspace(0, 1, PeriodicKnotObject.nelems+1)
        self.assertTrue(np.allclose(PeriodicKnotObject.expand_knots(), vec))

    def test_extend_knots(self):
        ''' test if knots get extended properly '''
        vec = np.concatenate([ [0]*3, np.linspace(0, 1, 11), [1]*3 ])
        self.assertTrue(np.allclose(SimpleKnotOject.extend_knots(), vec))

        # Periodic Knotobject
        vec = np.concatenate([ [-0.3, -0.2, -0.1],
                np.linspace(0, 1, PeriodicKnotObject.nelems+1), [1.1, 1.2, 1.3] ])
        self.assertTrue(np.allclose(PeriodicKnotObject.extend_knots(), vec))

    def test_greville( self ):
        ''' Test the greville abscissae '''
        vec = np.array( [ 0, 0.1/3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9666666667, 1 ] )
        self.assertTrue( np.allclose( SimpleKnotOject.greville(), vec ) )

        self.assertTrue( np.allclose( PeriodicKnotObject.greville(), np.arange( -0.1, 0.9, 0.1 ) % 1 ) )

        self.assertTrue( np.allclose( PeriodicKnotObject.raise_multiplicities( PeriodicKnotObject.degree, [0] ).greville(), vec ) )

    def test_to_c(self):
        ''' Test whether to_c gives right knotmultiplicity'''
        # Simple KnotObject
        self.assertTrue(np.allclose(SimpleKnotOject.to_c(2).knotmultiplicities,
                SimpleKnotOject.knotmultiplicities))
        self.assertRaises(ValueError, SimpleKnotOject.to_c, 3)
        self.assertRaises(ValueError, SimpleKnotOject.to_c, 4)

        # Periodic KnotObject
        self.assertTrue(np.allclose(PeriodicKnotObject.to_c(2).knotmultiplicities,
                PeriodicKnotObject.knotmultiplicities))

        # KnotObject with repetitions
        knotvalues = np.array([0, 0.3, 0.4, 0.65, 0.7, 0.9, 1])
        knotmultiplicities = np.array([4, 1 ,2, 3, 1, 3, 4])
        ko = KnotObject(knotvalues=knotvalues, knotmultiplicities=knotmultiplicities, degree=3)
        self.assertTrue(np.allclose(ko.to_c(1).knotmultiplicities,
                np.array([4, 1, 2, 2, 1, 2, 4])))

        # Periodic with repetitions
        knotmultiplicities = np.array([1, 1, 2, 3, 1, 3, 1])
        ko = KnotObject(knotvalues=knotvalues, knotmultiplicities = knotmultiplicities, degree = 3, periodic = True)
        self.assertTrue(np.allclose(ko.to_c(1).knotmultiplicities,
                np.array([1, 1, 2, 2, 1, 2, 1])))

    def test_unify(self):
        ''' test if KnotObjects get unified properly '''
        self.assertTrue( KnotObject.unify(SimpleKnotOject, SimpleKnotOject) == SimpleKnotOject )
        kos = tb.UnrelatedKnotObjects(8, 10, 4)
        self.assertFalse( KnotObject.unify(*kos) == kos[0] )

        # With knotrepetitions
        instlib = SimpleKnotOject.lib
        instlib['knotmultiplicities'][4] = 3
        instlib['knotvalues'][4] -= 0.05
        unified = KnotObject(knotvalues=np.array([0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
                knotmultiplicities=[4, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 4], degree=3)
        self.assertTrue( KnotObject.unify(KnotObject(**instlib), SimpleKnotOject) == unified  )

        instlib = PeriodicKnotObject.lib
        instlib['knotmultiplicities'][4] = 3
        instlib['knotvalues'][4] -= 0.05
        unified = KnotObject(knotvalues=np.array([0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
                knotmultiplicities=[1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1], degree=3, periodic=True)
        self.assertTrue( KnotObject.unify(KnotObject(**instlib), PeriodicKnotObject) == unified  )

    def test_leq(self):
        ''' test if <= operator works properly'''
        self.assertTrue(SimpleKnotOject <= SimpleKnotOject)
        self.assertFalse(SimpleKnotOject <= PeriodicKnotObject)

        # more advanced ones
        self.assertTrue(SimpleKnotOject <= tb.UniformKnotObject(20, 3))
        self.assertFalse(tb.UniformKnotObject(20, 3) <= SimpleKnotOject)

        self.assertTrue(PeriodicKnotObject <= tb.UniformKnotObject(20, 3, periodic=True))
        self.assertFalse(tb.UniformKnotObject(20, 3, periodic=True) <= PeriodicKnotObject)

        ko_p1 = KnotObject(knotvalues=np.linspace(0, 1, 11), degree=1)
        ko_p2 = KnotObject(knotvalues=np.linspace(0, 1, 11), degree=2, knotmultiplicities=
                [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3])

        self.assertTrue(ko_p1 <= ko_p2)

        # changing knotmultiplicity slightly and it should fail
        ko_p2 = KnotObject(knotvalues=np.linspace(0, 1, 11), degree=2, knotmultiplicities=
                [3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 3])
        self.assertFalse(ko_p1 <= ko_p2)

        # unrelated KnotObjects
        kos = tb.UnrelatedKnotObjects(12, 17, 4)
        self.assertFalse( kos[0] <= kos[1] )
        self.assertTrue( kos[0] <= KnotObject.unify(*kos)  )
        self.assertFalse( KnotObject.unify(*kos) <= kos[0]  )

    def test_ref_by(self):
        # Simple refinement
        self.assertTrue( SimpleKnotOject.ref_by([0, 3, 5]) == KnotObject(knotvalues=
                [0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1], degree=3) )
        self.assertTrue( PeriodicKnotObject.ref_by([0, 3, 5]) == KnotObject(knotvalues=
                [0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1], degree=3, periodic=True) )

        # Numpy-like refinement
        self.assertTrue( SimpleKnotOject.ref_by( [0, 4, -2, -1] ) == KnotObject(knotvalues=
                [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1], degree=3)  )
        self.assertTrue( SimpleKnotOject.ref_by( '::2' ) == SimpleKnotOject.ref_by( [0, 2, 4, 6, 8] )  )
        self.assertTrue( SimpleKnotOject.ref_by( '2::2' ) == SimpleKnotOject.ref_by( [2, 4, 6, 8] )  )
        self.assertTrue( SimpleKnotOject.ref_by( range(2, 10, 2) ) == SimpleKnotOject.ref_by( [2, 4, 6, 8] )  )

    def test_ref(self):
        # Simple single uniform refinement
        self.assertTrue( SimpleKnotOject.ref() == tb.UniformKnotObject(20, 3)  )
        self.assertTrue( PeriodicKnotObject.ref() == tb.UniformKnotObject(20, 3, periodic=True)  )

        # More advanced case
        self.assertTrue( SimpleKnotOject.ref(ref=3) == tb.UniformKnotObject(80, 3)  )
        self.assertTrue( PeriodicKnotObject.ref(ref=3) == tb.UniformKnotObject(80, 3, periodic=True)  )

    def test_ref_by_vertices(self):
        # Simple test cases
        self.assertTrue( SimpleKnotOject.ref_by_vertices( [0.05, 0.25, 0.999] ) ==
                SimpleKnotOject.ref_by( [0, 2, -1] ) )
        self.assertTrue( PeriodicKnotObject.ref_by_vertices( [0.05, 0.25, 0.999] ) ==
                PeriodicKnotObject.ref_by( [0, 2, -1] ))

    def test_raise_multiplicities(self):
        # Simple test cases
        km = [4, 3, 1, 3, 1, 3, 1, 1, 1, 1, 4]
        self.assertTrue( SimpleKnotOject.raise_multiplicities( 2, indices=[1, 3, 5]  ) ==
                    KnotObject(knotvalues=SimpleKnotOject.knots, knotmultiplicities=km, degree=3) )
        self.assertTrue( PeriodicKnotObject.raise_multiplicities( 2, indices=[1, 3, 5] ) ==
                KnotObject(knotvalues=PeriodicKnotObject.knots, knotmultiplicities=np.concatenate([ [1], km[1:-1], [1] ]), degree=3, periodic=True))

        # via knotvalues
        self.assertTrue( SimpleKnotOject.raise_multiplicities( 2, knotvalues=[0.1, 0.3, 0.5] ) ==
                    KnotObject(knotvalues=SimpleKnotOject.knots, knotmultiplicities=km, degree=3))
        self.assertTrue( PeriodicKnotObject.raise_multiplicities( 2, knotvalues=[0.1, 0.3, 0.5] ) ==
                KnotObject(knotvalues=PeriodicKnotObject.knots, knotmultiplicities=np.concatenate([ [1], km[1:-1], [1] ]), degree=3, periodic=True))

        # mixed
        self.assertTrue( SimpleKnotOject.raise_multiplicities( 2, knotvalues=[0.1, 0.3, 0.5], indices=[1] ) ==
                    KnotObject(knotvalues=SimpleKnotOject.knots, knotmultiplicities=km, degree=3))
        self.assertTrue( SimpleKnotOject.raise_multiplicities( 2, knotvalues=[0.3, 0.5], indices=[1, 3, 5] ) ==
                    KnotObject(knotvalues=SimpleKnotOject.knots, knotmultiplicities=km, degree=3))
        self.assertTrue( PeriodicKnotObject.raise_multiplicities( 2, knotvalues=[0.1, 0.3, 0.5], indices=[0, -2]) ==
                KnotObject(knotvalues=PeriodicKnotObject.knots, knotmultiplicities=np.concatenate([ [3], km[1:-2], [3], [3] ]), degree=3, periodic=True))

        # raise ValueError if no matching indices are found
        self.assertRaises(ValueError, SimpleKnotOject.raise_multiplicities, 2, knotvalues=[1.1, 1.45, 1.9])
        self.assertRaises(ValueError, PeriodicKnotObject.raise_multiplicities, 2, knotvalues=[1.1, 1.45, 1.9])

        # numpy-like indices
        self.assertTrue( SimpleKnotOject.raise_multiplicities( 2, indices='1:6:2' ) ==
                    KnotObject(knotvalues=SimpleKnotOject.knots, knotmultiplicities=km, degree=3))
        self.assertTrue( SimpleKnotOject.raise_multiplicities( 2, indices=range(1, 6, 2) ) ==
                    KnotObject(knotvalues=SimpleKnotOject.knots, knotmultiplicities=km, degree=3))

    def test_add_c0(self):
        # Simple test cases
        self.assertTrue( SimpleKnotOject.add_c0([0.05, 0.95]) == KnotObject(knotvalues=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                knotmultiplicities=[4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4], degree=3)  )
        self.assertTrue( PeriodicKnotObject.add_c0([0.05, 0.95]) == KnotObject(knotvalues=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                knotmultiplicities=[1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1], degree=3, periodic=True)  )
        self.assertTrue( SimpleKnotOject.add_c0([]) == SimpleKnotOject )

        # Advanced test cases
        self.assertTrue( PeriodicKnotObject.add_c0([0]) == KnotObject(knotvalues=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                knotmultiplicities=[3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3], degree=3, periodic=True)  )
        self.assertTrue( SimpleKnotOject.add_c0([0]) == SimpleKnotOject )
        self.assertTrue( SimpleKnotOject.add_c0([1]) == SimpleKnotOject )
        self.assertTrue( SimpleKnotOject.add_c0([0,1]) == SimpleKnotOject )

    def test_add(self):
        # Simple test cases
        self.assertTrue( (SimpleKnotOject + SimpleKnotOject) == SimpleKnotOject )
        unique = lambda x: np.unique( np.concatenate(x) )
        for i in range(3):
            kos = tb.UnrelatedKnotObjects(6, 12, 3)
            self.assertTrue( np.sum(kos) == KnotObject(knotvalues=unique([k.knots for k in kos ]), degree=3 ) )

        self.assertTrue( (PeriodicKnotObject + PeriodicKnotObject) == PeriodicKnotObject )

    def test_add_knots(self):
        # Simple test cases
        addknots = np.array([0.05, 0.25, 0.55, 0.95])
        self.assertTrue( SimpleKnotOject.add_knots(addknots), KnotObject(knotvalues=\
                [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1], degree=3) )
        self.assertTrue( PeriodicKnotObject.add_knots(addknots), KnotObject(knotvalues=\
                [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1], degree=3, periodic=True) )

    def test_mul(self):
        self.assertTrue( isinstance( SimpleKnotOject * PeriodicKnotObject, ko.TensorKnotObject ))


class TestTensorKnotObject(unittest.TestCase):
    ''' Test the functionality of the TestTensorKnotObject 
        All attributes that are created with _vectorize or _prop_wrapper
        don't have to be tested as they carry out the same functionality
        the one-dimensional KnotObject '''

    def test_init(self):

        tkv = ko.TensorKnotObject( tb.UnrelatedKnotObjects(8, 12, 3) )
        self.assertRaises(ValueError, ko.TensorKnotObject, [SimpleKnotOject, ''])

    def test_dim(self):

        tkv = ko.TensorKnotObject(tb.UnrelatedKnotObjects(8, 12, 3))

        def TestAgainstNutils(tkv):
            domain, geom = mesh.rectilinear( tkv.knots, periodic=tkv.periodic)
            return len( domain.basis('spline', knotmultiplicities=tkv.knotmultiplicities, degree=tkv.degree) ) == tkv.dim

        self.assertTrue( TestAgainstNutils( tkv ) )
        self.assertTrue( TestAgainstNutils( tkv.add_c0([ [], [0.05, 0.82, 0.99] ]) ) )
        self.assertTrue( TestAgainstNutils( tkv.ref_by([ '::2', [4, 5, 6] ]) ) )
        self.assertTrue( TestAgainstNutils( tkv.raise_multiplicities( [3, 3], indices=[ '3:6:2', [3, 5, 8] ] ) ) )
        self.assertTrue( TestAgainstNutils( tkv.raise_multiplicities( [2, 1], indices=[ [], '2:4' ],
                knotvalues=[ tkv[0].knots[1:4], tkv[1].knots[::2] ] ) ) )

    def test_periodic(self):

        self.assertTrue( (SimpleKnotOject * PeriodicKnotObject).periodic == (1,) )
        self.assertTrue( (PeriodicKnotObject * SimpleKnotOject).periodic == (0,) )
        self.assertTrue( (PeriodicKnotObject * PeriodicKnotObject).periodic == (0, 1) )
        self.assertTrue( (SimpleKnotOject * SimpleKnotOject).periodic == () )

    def test_at(self):

        tkv = SimpleKnotOject * PeriodicKnotObject

        self.assertTrue( (SimpleKnotOject * PeriodicKnotObject).at(1) == ko.TensorKnotObject([ PeriodicKnotObject ]) )
        self.assertRaises( AssertionError, tkv.at, 2 )
        self.assertRaises( AssertionError, tkv.at, -3 )
        self.assertTrue( tkv.at(-2) == ko.TensorKnotObject([SimpleKnotOject]) )

    def test_ref_by( self ):
        
        tkv = SimpleKnotOject * SimpleKnotOject
        self.assertTrue( all( tkv.ref_by( [ [ 1, 2, 3 ], [ 2, 4, 9 ] ] ) == 
                SimpleKnotOject.ref_by( [ 1, 2, 3 ] ) * SimpleKnotOject.ref_by( [ 2, 4, 9 ] ) ) )



if __name__ == '__main__':
    unittest.main()

# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
