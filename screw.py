#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET

from scipy import optimize
from functools import partial, lru_cache

from mapping_2 import pc, rep, sol, prep
from collections import defaultdict


def invert_UnivariateFunction( func, point, initial_guess=None, **minimizeargs ):

    minimizeargs.setdefault( 'tol', 1e-12 )
    minimizeargs.setdefault( 'bounds', [ ( 0, 1 ) ] )

    assert isinstance( func, pc.UnivariateFunction )

    if not len( point.shape ) == 1:
        point = point.ravel()

    assert point.shape == func( 0 ).ravel().shape

    f = lambda x: ( ( func._f( x ) - point[ None ] ) ** 2 ).sum( 1 )

    if initial_guess is None:
        xi = np.linspace( 0, 1, 500 )
        initial_guess = xi[ f( xi ).argmin() ]

    eta = optimize.minimize( lambda x: f( x )[ 0 ], initial_guess, **minimizeargs ).x

    assert np.isclose( f( eta ), 0 ), 'The solver has converged to the wrong local minimum, or the given point does not lie on the curve.'
    return eta


def CUSP( a, b, c ):
    """ solve  x ** 2 + y ** 2 = a ** 2 and ( x - c ) ** 2 + y ** 2 = b ** 2 for x and y """

    x = ( a ** 2 - b ** 2 + c ** 2 ) / ( 2 * c )
    y = np.sqrt( a ** 2 - x ** 2 )

    return np.array( [ x, -y ] ), np.array( [ x, y ] )


def CUSP_eta( casings, centers, radii ):

    LeftCasing, RightCasing = casings
    left_center, right_center = centers
    left_radius, right_radius = radii

    assert all( [ isinstance( c, pc.UnivariateFunction ) for c in ( LeftCasing, RightCasing ) ] )
    assert left_center[1] == 0 and right_center[1] == 0

    CUSP_points = CUSP( left_radius, right_radius, ( right_center - left_center )[0] )
    CUSP_points = [ np.array( [ c[0] + left_center[0], c[1] ] ) for c in CUSP_points ]  # shift the CUSP points back to old coordinate system
    eta_left = [ invert_UnivariateFunction( LeftCasing, i ) for i in CUSP_points ]
    eta_right = [ invert_UnivariateFunction( RightCasing, i ) for i in CUSP_points ]

    return eta_left, eta_right


def load_screws():

    """ Load the screw point clouds, roll them to the standard positions and generate casing point clouds """

    xml = ET.parse( 'xml/SRM4_6_gap0.1mm.xml' ).getroot()
    male, female = [ np.array( xml[ i ].text.split(), dtype=float ).reshape( [ 2, -1 ] ).T for i in range( 2 ) ]
    male, female = [ pc.PointCloud( np.vstack( [ p, p[ 0 ][ None ] ] ) ) for p in ( male, female ) ]

    female = female.flip()

    centers = [ np.array( [ 0, 0 ] ), np.array( [ 56.52, 0 ] ) ]

    radii = [ 36.1, 36 ]

    return { 'male': male, 'female': female, 'centers': centers, 'radii': radii }


class ReparameterizedFunction:

    @classmethod
    def fromPointcloud( cls, pc, repfunc=None, orientation=1, center=None, k=3 ):
        return cls( pc.interpolate( k=k ), repfunc=repfunc, orientation=orientation, center=center )

    def __init__( self, ufunc, repfunc=None, orientation=1, center=None ):
        if center is None:
            center = np.array( [ 0, 0 ] )
        assert center.shape == ( 2, )
        # assert ufunc.periodic
        assert orientation in [ -1, 1 ]
        if repfunc is None:
            repfunc = rep.ReparameterizationFunction( lambda x: x )
        assert isinstance( repfunc, rep.ReparameterizationFunction )
        self.func = ufunc
        self.repfunc = repfunc
        self.orientation = orientation
        self.center = center

    def __call__( self, x ):
        return self.func._f( x )

    def pcall( self, x ):
        return self.func.toPointCloud( x )

    def rcall( self, x ):
        return self.func.toPointCloud( x, verts=self.repfunc( x ) )

    def invcall( self, x ):
        return self.func.reparameterize( self.repfunc.inverse() )( x )

    def rotate( self, theta ):
        deta_prime = self.orientation * theta / ( 2 * np.pi )
        deta = -self.repfunc.invert( -deta_prime % 1 )
        repfunc = self.repfunc.roll( deta % 1 )
        func = self.func.rotate( theta, center=self.center ).shift( deta )
        return self.__class__( func, repfunc=repfunc, orientation=self.orientation, center=self.center )


to_closest_match = pc.to_closest_match


def with_ReparameterizedFunction_inheritance( funcnames=() ):
    def decorator( cls ):
        for name in funcnames:
            setattr( cls, name, lambda self, *args, **kwargs:
                    [ getattr( p, name )( *args, **kwargs ) for p in self ] )
        return cls
    return decorator


@with_ReparameterizedFunction_inheritance( funcnames=( '__call__', 'rcall', 'pcall', 'invcall' ) )
class RotorCasingPair:

    @classmethod
    def fromPointClouds( cls, rotor, casing, side, orientation=1, k=3, center=np.array( [ 0, 0 ] ), **matchingargs ):
        assert center.shape == ( 2, )
        assert side in ( 'left', 'right' )
        casing, rotor = [ pc.roll_to( p, to=side ) for p in ( casing, rotor ) ]
        casing, rotor, repfunc = to_closest_match( casing, rotor, return_func=True, indexdist=4, **matchingargs )
        return cls( *[ ReparameterizedFunction.fromPointcloud( f, repfunc=rep, center=center, orientation=orientation, k=k ) for f, rep, center in
            zip( [ rotor, casing ], [ repfunc, None ], [ center ] * 2 ) ] )

    @classmethod
    def fromReparameterizedPointClouds( cls, rotor, casing, repfunc, orientation=1, k=3, center=np.array( [ 0, 0 ] ) ):
        assert center.shape == ( 2, )
        return cls( *[ ReparameterizedFunction.fromPointcloud( f, repfunc=rep, center=center, orientation=orientation, k=k ) for f, rep, center in
            zip( [ rotor, casing ], [ repfunc, None ], [ center ] * 2 ) ] )

    def __init__( self, rotor, casing ):
        self.rotor, self.casing = rotor, casing
        assert all( [ isinstance( p, ReparameterizedFunction ) for p in self ] )

    def __iter__( self ):
        return iter( [ self.rotor, self.casing ] )

    def rotate( self, theta ):
        return self.__class__( *[ p.rotate( theta ) for p in self ] )

    def copy( self ):
        return self.__class__( **self.__dict__.copy() )

    def plot( self, npoints=200, show=True ):
        xi = np.linspace( 0, 1, npoints )
        plt.axis( 'equal' )
        plt.plot( *self.rotor( np.linspace( 0, 1, 5*npoints ) ).T )
        plt.plot( *self.casing( np.linspace( 0, 1, 5*npoints ) ).T )
        pnts = self.invcall( xi )
        for i in range( len( xi ) ):
            p = np.vstack( [ j[i] for j in pnts ] )
            plt.plot( *p.T )
        if show:
            plt.show()


screws = load_screws()

sides = ( 'male', 'female' )


class TwinScrew:

    def set_pointclouds( self, male, female, casingpoints ):
        male_casing, female_casing = [ pc.roll_to( pc.PointCloud(
            self._radii[ i ] * pc.circle( casingpoints ) + self._centers[ i ] ), to=to )
            for i, to in zip( range( 2 ), ( 'left', 'right' ) ) ]

        female_casing = female_casing.flip()
        self._male_casing, self._male, repfunc0 = \
                pc.to_closest_match( male_casing, male, return_func=True, indexdist=4 )
        self._female_casing, self._female, repfunc1 = \
                pc.to_closest_match( female_casing, female, return_func=True, indexdist=4 )
        self._repfuncs = [ repfunc0, repfunc1 ]

    def set_funcs( self, k=3 ):
        Male, Female, MaleCasing, FemaleCasing = [ getattr( self, j ).interpolate( k=k )
                for j in ( '_male', '_female', '_male_casing', '_female_casing' ) ]

        self._Male, self._Female = [ ReparameterizedFunction( f, repfunc=rep, orientation=o, center=c )
                for f, rep, o, c in zip( [ Male, Female ], self._repfuncs, [ 1, -1 ], self._centers ) ]

        self._MaleCasing, self._FemaleCasing = [ ReparameterizedFunction( f, orientation=o, center=c )
                for f, o, c in zip( [ MaleCasing, FemaleCasing ], [ 1, -1 ], self._centers ) ]

        self._Left = RotorCasingPair( self._Male, self._MaleCasing )
        self._Right = RotorCasingPair( self._Female, self._FemaleCasing )

    def __init__( self, male, female, radii=[], centers=[], casingpoints=4001, loberatio=-4/6 ):
        assert all( [ len( c ) == 2 for c in ( radii, centers ) ] )
        assert all( [ isinstance( p, pc.PointCloud ) for p in ( male, female ) ] )
        assert all( [ p.periodic for p in ( male, female ) ] )

        self._radii = radii
        self._centers = centers
        self._loberatio = loberatio

        self.set_pointclouds( male, female, casingpoints )
        self.set_funcs()

        self._CUSP_eta = dict( zip( [ 'male', 'female' ],
                CUSP_eta( [ self._MaleCasing.func, self._FemaleCasing.func ], self._centers, self._radii ) ) )

        self._SeparatorRepfunc = dict( zip( [ 'male', 'female' ], 
            [ lambda theta: rep.ReparameterizationFunction( lambda x: x ) ] * 2 ) )

    @property
    def Casings( self ):
        return { 'male': self._MaleCasing, 'female': self._FemaleCasing }

    @property
    def Rotors( self ):
        return { 'male': self._Male, 'female': self._Female }

    @property
    def casings( self ):
        return { 'male': self._male_casing, 'female': self._female_casing }

    @property
    def rotors( self ):
        return { 'male': self._male, 'female': self._female }

    @property
    def CUSPeta( self ):
        return self._CUSP_eta

    @property
    def Pairs( self ):
        return { 'male': self._Left, 'female': self._Right }

    def set_ReparameterizationFunction( self, xis, eta, evalpoints=500, matchargs={}, smoothargs={} ):
        matchargs.setdefaults( 'fac', 16 )
        matchargs.setdefaults( 'indexdist', 4 )

        smoothargs.setdefaults( 'p', 2 )
        smoothargs.setdefaults( 'c', 1 )

        assert smoothargs[ 'c' ] < smoothargs[ 'p' ]

        repfuncs = defaultdict( list )
        for xi in range( len( xis ) ):
            x, y = [ self.StraightSidedSinglePatchSeparator( xi, evaluation_points=
                ( 50, 50, evalpoints ) )[ side ] for side in ( 'male', 'female' ) ]
            repfunc = rep.averaged_reparam( x, y, **matchargs )
            repfuncs[ 'male' ] += repfunc[ 0 ]
            repfuncs[ 'female' ] += repfunc[ 1 ]

        male = rep.BivariateReparameterizationFunction.fromReparameterizationFunctions \
                ( xis, repfuncs[ 'male' ], eta )
        female = rep.BivariateReparameterizationFunction.fromReparameterizationFunctions \
                ( self._loberatio * xis, repfuncs[ 'female' ], eta )

        self._SeparatorRepfunc = { 'male': male, 'female': female }

    @lru_cache( maxsize=10 )
    def make_C( self, side, theta ):
        """ Description forthcoming """

        assert side in ( 'male', 'female' )
        eta_p0, eta_p1 = self.CUSPeta[ side ]
        assert eta_p1 > eta_p0

        orientation = { 'male': 1, 'female': -1 }[ side ]

        rotor = self.Pairs[ side ].rotate( theta ).rotor
        repfunc = rotor.repfunc

        r = lambda f, **kwargs: ReparameterizedFunction( f, orientation=orientation, center=rotor.center, **kwargs )

        eta_0, eta_1 = [ repfunc.invert( _eta_p ) for _eta_p in ( eta_p0, eta_p1 ) ]
        assert eta_1 > eta_0

        CasingC = r( self.Casings[ side ].func.shift( -eta_p1 ).restrict( 0, 1 - ( eta_p1 - eta_p0 ) ) )

        RotorC = rotor.func.shift( -eta_1 ).restrict( 0, 1 - ( eta_1 - eta_0 ) )
        repfuncC = repfunc.roll( -eta_1 % 1 ).restrict( 0, 1 - ( eta_1 - eta_0 ) )
        RotorC = r( RotorC, repfunc=repfuncC )

        C = { 1: { 'right': CasingC, 'left': RotorC }, -1: { 'right': RotorC, 'left': CasingC } }[ orientation ]
        sep_bot_sep_top = [ pc.line( C[ 'left' ]( i ).ravel(), C[ 'right' ]( i ).ravel() ) for i in ( 1, 0 ) ]
        sep_side = rotor.func.restrict( eta_0, eta_1 )

        sep = dict( zip( [ 'bottom', 'top' ], sep_bot_sep_top ) )
        sep[ { 1: 'left', -1: 'right' }[ orientation ] ] = sep_side

        return { 'C': C, 'separator': sep }

    def MaleC( self, theta ):
        C = self.make_C( 'male', theta )[ 'C' ]
        return RotorCasingPair( C[ 'left' ], C[ 'right' ] )

    def FemaleC( self, theta ):
        C = self.make_C( 'female', theta )[ 'C' ]
        return RotorCasingPair( C[ 'right' ], C[ 'left' ] )

    def Cs( self, theta ):
        return { 'male': self.MaleC( theta ), 'female': self.FemaleC( self._loberatio * theta ) }

    @lru_cache( maxsize=2 )
    def StraightSidedSinglePatchSeparator( self, theta, evaluation_points=( 50, 50, 200 ), c0_center=True, reparameterize=True ):
        """ evaluation_points = ( xi_left, xi_right, eta ) """

        assert len( evaluation_points ) == 3

        for i in range( len( evaluation_points ) ):
            val = evaluation_points[ i ]
            if isinstance( val, int ):
                evaluation_points[ i ] = np.linspace( 0, 1, val )

        theta = [ theta, self._loberatio * theta ]

        sep_left, sep_right = [ self.make_C( side, th )[ 'separator' ] for side, th in zip( [ 'male', 'female'], theta ) ]

        xi_left, xi_right, eta = evaluation_points

        for x in ( xi_left, xi_right, eta ):
            assert x[ 0 ] == 0 and x[ -1 ] == 1

        if c0_center:
            verts = np.concatenate( [ np.linspace( 0, 0.5, len( evaluation_points[0] ) ),
                                np.linspace( 0.5, 1, len( evaluation_points[1] ) )[1:] ] )
        else:
            verts = None

        d = {}
        d[ 'top' ] = pc.PointCloud( np.vstack( [ sep_left['top'].toPointCloud( xi_left ).points,
                            sep_right['top'].toPointCloud( xi_right ).points[1:] ] ), verts=verts )
        d[ 'bottom' ] = pc.PointCloud( np.vstack( [ sep_left['bottom'].toPointCloud( xi_left ).points,
                            sep_right['bottom'].toPointCloud( xi_right ).points[1:] ] ), verts=verts )

        d[ 'left' ] = sep_left[ 'left' ].toPointCloud( eta )
        d[ 'right' ] = sep_right[ 'right' ].toPointCloud( eta )

        if reparameterize:
            for side, th in zip( sides, theta ):
                d[ side ] = d[ side ].reparameterize( self._SeparatorRepfunc[ side ]( th ) )

        return d

    def plot( self, theta ):
        Cs = self.Cs( theta )
        for C in Cs.values():
            C.plot( show=False )
        separator = self.StraightSidedSinglePatchSeparator( theta )
        for side in separator.values():
            side.plot( show=False )
        plt.show()


screw = TwinScrew( **screws )


# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
