#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The GridObject module defines the :class:`TensorGridObject` which
is the core class that represents a mapping with tensor-product B-splines.
It can be manipulated in very much the same way as :class:`ko.TensorKnotObject`,
while the mapping is always preserved, unless an operation leads to a coarser
knot-vector, in which case it is preserved in the least-squares sense.
It also links the :mod:`solver` which allows for the quick generation of
mappings using the command TensorGridObject.solve(**kwargs). The boundary conditions
have to be specified using the TensorGridObject.cons attribute and the initial guess,
unless specified differently, defaults to transfinite interpolation.
"""

import numpy as np
import scipy as sp
from scipy import linalg, interpolate
from nutils import mesh, util, function, plot, log
from . import ko, idx, aux
from matplotlib import pyplot
from collections import defaultdict, ChainMap
import functools

gauss = lambda x: 'gauss{}'.format(x)

_ = np.newaxis


def refine_x( g, to: ko.TensorKnotObject ):
    Ts = [ aux.prolongation_matrix( kvold, kvnew ) for kvold, kvnew in zip(g.knotvector, to) ]
    T = functools.reduce( np.kron, Ts ) if len(Ts) > 1 else Ts[0]
    return sp.linalg.block_diag( *[T] * g.targetspace ).dot( g.x )


def refine_GridObject( g, to: ko.TensorKnotObject ):
    assert isinstance( to, ko.TensorKnotObject )
    d = g.__dict__.copy()
    d['knotvector'] = to
    d['domain'], d['geom'] = aux.rectilinear( to.knots, periodic=to.periodic, bnames=g.domain._bnames )
    ret = g._constructor( **d )
    ret.x = refine_x( g, to )
    return ret


def withrefinement(fn):
    def wrapper( *args, **kwargs ):
        self, *args = args
        return refine_GridObject( self, fn( self, *args, **kwargs ) )
    return wrapper


def withinheritance( parent, properties=(), funcs=(), refinefuncs=() ):
    ''' Decorator to inherit certain properties (not functions) from self.parent '''

    def decorator(cls):
        def _inheritprop(name):
            def wrapper(self):
                return getattr(getattr(self, parent), name)
            return property( fget=wrapper )

        def _inheritfunc(name):
            def wrapper(self, *args, **kwargs):
                return getattr(getattr(self, parent), name)(*args, **kwargs)
            return wrapper

        def _inheritrefinefunc(name):
            @withrefinement
            def wrapper(self, *args, **kwargs):
                return getattr(getattr(self, parent), name)(*args, **kwargs)
            return wrapper

        for name in properties:
            setattr(cls, name, _inheritprop(name))
        for name in funcs:
            setattr(cls, name, _inheritfunc(name))
        for name in refinefuncs:
            setattr(cls, name, _inheritrefinefunc(name))
        return cls
    return decorator


def withsideslicing( fn ):
    """
        Allows function to be called with keywords like
        'left', 'top', returning the corresponding indices.
        For __setitem__ and __getitem__ functions. 
        If function gets called with string or tuple of strings carry out side slicing.
        I.e. ( 'left', 'top' ) -> self.index.boundary( 'left', 'top' ) etc.
        If function gets called by tuple of integers and slices call with
        self.index[ key ].
        If function gets called with list, array or integer simply return f( self, key )
    """
    def wrapper( self, key ):
        if isinstance( key, str ):  # turn single string into tuple
            key = ( key, )
        if isinstance( key, tuple ):  # key is tuple
            if all( [ isinstance( k, str ) for k in key ] ):  # ( 'left', 'top' )
                key = self.index.boundary( *key ).flatten
            else:  # ( slice, slice ) or ( slice, int )
                key = self.index[ key ]
        else:  # list, array or integer
            pass
        return fn(self, key)
    return wrapper


props = ('knots', 'knotmultiplicities', 'degree', 'ndims', 'periodic')
funcs = ('extend_knots', 'expand_knots', 'greville')
refinefuncs = ('ref_by', 'ref_by_vertices', 'ref', 'add_c0', \
                'to_c', 'add_knots', 'raise_multiplicities')


@functools.total_ordering
@withinheritance('knotvector', properties=props, funcs=funcs, refinefuncs=refinefuncs)
class TensorGridObject:

    """ Various auxilliary functions """

    def _defaultvec(name, default=np.zeros):
        """
            if vec is None this defaults to default(appropriate length)
            else vec is rejected if len(vec) != appropriate length
        """
        def wrapper(self, vec=None):
            if vec is None:
                vec = default(self.dim)
            assert len(vec) == self.dim
            setattr(self, '_' + name, vec)
        return wrapper

    set_x = _defaultvec('x')
    set_cons = _defaultvec( 'cons', default=util.NanVec )

    def _getprivate(name):
        def wrapper(self):
            return getattr(self, '_' + name)
        return wrapper

    def set_basis(self):
        self._basis = self.domain.basis('spline', degree=self.degree, knotmultiplicities=self.knotmultiplicities, knotvalues=self.knotvector.knots)

    def set_cons_from_x(self):
        for side in self.sides:
            self._cons[ self.index.boundary(side).flatten ] = self[ side ]

    _constructor = lambda self, *args, **kwargs: self.__class__( *args, **kwargs )  # default constructor can be overwritten
    
    """ classmethods """

    @classmethod
    def from_parent(cls, parent, *key):
        assert isinstance(parent, cls)
        if len(key) == 1:
            side, = key
            d = parent.__dict__.copy()
            d['_domain'] = d['_domain'].boundary[side]

            # flatten appropriate axis
            bnames = parent.domain._bnames
            sidetovalue = dict( zip( bnames, range( len(bnames) ) ) )
            deleteindex = sidetovalue[side] // 2
            deleteaxis = d[ 'axesnames' ][ deleteindex ]
            d['knotvector'] = np.delete( d['knotvector'], deleteindex )
            geom = d[ 'geom' ]
            d[ 'geom' ] = function.stack( [ geom[i] for i in range( len( parent ) ) \
                    if not i == deleteindex ] )
            d[ 'axesnames' ] = tuple( filter( lambda x: x != deleteaxis, d[ 'axesnames' ] ) )
            if len(d['knotvector']) == 0:
                log.warning('The GridObject has not been implemented with an empty knotvector yet')
                raise NotImplementedError

            ret = cls(**d)
            ret.x = parent[ side ]
            for name in ret.domain._bnames:
                ret._cons[ ret.index.boundary(name).flatten ] = ret[ name ]
            return ret
        ret = parent
        for side in key:
            ret = ret.from_parent(ret, side)
        return ret

    """ Initializer """

    def __init__(self, knotvector=None, ischeme=6, targetspace=None, domain=None, geom=None, axesnames=None, **kwargs):

        if isinstance( knotvector, ko.KnotObject ):
            knotvector = [ knotvector ]
        if isinstance( knotvector, list ):  # instantiation via [KnotObject, KnotObject, ...]
            knotvector = ko.TensorKnotObject( knotvector )
        assert isinstance( knotvector, ko.TensorKnotObject )

        self.knotvector = knotvector
        n = len( self )
        assert n <= 3, NotImplementedError

        if not axesnames:
            axesnames = ('xi', 'eta', 'zeta')[ :n ]
        self.axesnames = tuple( axesnames )
        assert len( self.axesnames ) == len( self )

        if targetspace is None:
            targetspace = n
        assert targetspace >= n

        self.ischeme = ischeme
        self.targetspace = targetspace

        if domain is not None:
            assert geom is not None
        # else:
        #     domain, geom = aux.rectilinear( self.knotvector.knots, periodic=self.knotvector.periodic )
        self._domain, self._geom = domain, geom

        assert self.domain.ndims == len( knotvector.ndims ) 
        self.set_basis()
        assert knotvector.dim == len( self.basis )  # hereby it's tested if knotvector and domain are compatible

        self.set_x()
        self.set_cons()
        self.index = idx.TensorIndex.fromTensorGridObject( self )

    """ Various properties """

    x = property(_getprivate('x'), set_x)
    cons = property(_getprivate('cons'), set_cons)
    basis = property( fget=_getprivate('basis') )

    @property
    def domain( self ):
        if self._domain is None:
            self._domain, self._geom = aux.rectilinear( 
                        self.knotvector.knots, periodic=self.knotvector.periodic
                    )
        return self._domain

    @property
    def geom( self ):
        if self._geom is None:
            self._domain, self._geom = aux.rectilinear( 
                        self.knotvector.knots, periodic=self.knotvector.periodic
                    )
        return self._geom


    @property
    def dim(self):
        return len(self.basis) * self.targetspace

    @property
    def ndofs(self):
        return np.sum(~self.cons.where)

    @property
    def dofindices(self):
        return np.arange(self.dim)[~self.cons.where]

    @property
    def consindices(self):
        return np.arange(self.dim)[self.cons.where]

    @property
    def tensorindices(self):
        return self.index._indices

    @property
    def sides(self):
        return self.domain._bnames

    @property
    def mapping(self):
        return self.basis.vector(self.targetspace).dot(self.x)

    @property
    def jacdet(self):
        grd = self.mapping.grad(self.geom)
        return function.determinant( grd ) if len(self) > 1 else function.sqrt( grd.sum(-2) )

    """ Magic functions """

    @withsideslicing
    def __getitem__(self, key):
        return self._x.__getitem__( key )

    def __setitem__(self, key, value):
        f = withsideslicing( lambda x, y: y )
        self._x.__setitem__( f( self, key ), value )

    def __le__(self, other):
        return ( self.knotvector <= other.knotvector ).all()

    def __len__(self):
        return len( self.ndims )

    def __call__(self, *side):
        return self.__class__.from_parent(self, *side)

    """ Functions """

    def copy(self):
        ret = self._constructor(**self.__dict__.copy())
        ret.x = self.x.copy()
        ret.cons = self.cons.copy()
        return ret

    def empty_copy(self):
        d = self.__dict__.copy()
        return self._constructor(**d)

    def project(self, func, **kwargs):
        try:
            length = len(func)
            basis = self.basis.vector(length)
        except TypeError:
            basis = self.basis
        kwargs.setdefault('onto', basis)
        kwargs.setdefault('geometry', self.geom)
        kwargs.setdefault( 'ischeme', gauss(self.ischeme) )
        return self.domain.project(func, **kwargs)

    def integrate( self, funcs, boundary=False, **intargs ):
        intargs = ChainMap( intargs, { 'ischeme': gauss(self.ischeme), 'geometry': self.geom } )        
        domain = self.domain if not boundary else self.domain.boundary
        return domain.integrate( funcs, **intargs )

    def to_p( self, p ):
        assert len( p ) == len( self )

        if p == self.degree:
            return self

        knotvector = self.knotvector.to_p( p )

        basis = self.domain.basis( 'spline', degree=p, knotvalues=self.knots,
                knotmultiplicities=knotvector.knotmultiplicities ).vector( self.targetspace )
        x = self.domain.project( self.mapping, ischeme=gauss( self.ischeme * 2 ), geometry=self.geom, onto=basis )
        g = self._constructor( knotvector=knotvector, periodic=self.periodic, targetspace=self.targetspace, axesnames=self.axesnames )
        g.x = x
        return g

    """ Refinement """

    @withrefinement
    def ref_by( self, *args, **kwargs ):
        return self.knotvector.ref_by( *args, **kwargs )

    """ Defect detection """

    def defects_discrete( self, ischeme=None, ref=0 ):
        if ischeme is None:
            ischeme = self.ischeme

        return ( self.domain.refine(ref).elem_eval( self.jacdet, geometry=self.geom, ischeme=gauss( ischeme ) ) > 0 ).all()

    """ Operator overloading """

    @staticmethod
    def are_nested( leader, follower ):
        return any( [ leader <= follower, follower <= leader ] )

    @staticmethod
    def unify( *args ):
        """
            unify all input TensorGridObjects to one unified grid
        """
        assert all( isinstance( g, TensorGridObject ) for g in args )
        if len( args ) == 1:
            return args
        unified = functools.reduce( lambda x, y: refine_GridObject( x, x.knotvector + y.knotvector ), args )
        return [ unified ] + [ refine_GridObject( g, unified.knotvector ) for g in args[1:] ]

    @staticmethod
    def fastunify( *args ):
        """
            unify all input TensorGridObjects but only return the prolonged
            x-vectors and the resulting knotvector
        """
        if len( args ) == 1:
            g = args[ 0 ]
            return [ g.x ], g.knotvector
        unified_knotvector = functools.reduce( lambda x, y: x + y, [ g.knotvector for g in args ] )
        xs = [ refine_x( g, unified_knotvector ) for g in args ]
        return xs, unified_knotvector

    @staticmethod
    def grid_interpolation( verts, grids, **scipyargs ):
        assert len( grids ) > 1
        assert len( verts ) == len( grids )
        assert aux.isincreasing( verts )

        if len( verts ) == 1:
            return lambda x: grids[0]
        
        scipyargs.setdefault( 'k', min( 5, len( grids ) - 1 ) )
        scipyargs.setdefault( 'ext', 0 )

        assert scipyargs[ 'k' ] <= len( grids ) - 1

        unified = grids[0].__class__.unify( *grids )
        intpfunc = lambda A, s: np.array( [ interpolate.InterpolatedUnivariateSpline( verts, a, **scipyargs)( s ) for a in A ] )

        def ret( s ):
            A = np.stack( [ g.x for g in unified ], axis=1 )
            _g = unified[0].empty_copy()
            intpfunc_ = lambda s: intpfunc( A, s )
            _g.x = intpfunc_( s )
            return _g

        return ret

    ''' Function requirements decorator '''

    def requires_dependence( *requirements, operator=all ):
        def decorator( fn ):
            def decorated( *args ):
                if operator( [ req( *args ) for req in requirements ] ):
                    return fn( *args )
                raise Exception( 'Cannot perform requested operation with given arguments' )
            return decorated
        return decorator

    sameclass = lambda x, y: type( x ) == type( y )
    subclass = lambda x, y: issubclass( type( y ), type( x ) )
    superclass = lambda x, y: issubclass( type( x ), type( y ) )
    samedim = lambda x, y: len( x ) == len( y )
    subdim = lambda x, y: len( y ) == len( x ) - 1
    same_degree = lambda x, y: x.degree == y.degree
    not_periodic = lambda x, y: all( [ len( x.periodic ), len( y.periodic ) == 0 ] )

    @requires_dependence( samedim, sameclass, same_degree )
    def __add__( self, other ):
        unified = self.knotvector + other.knotvector

        return [ refine_GridObject( self, unified ), refine_GridObject( other, unified ) ]

    @requires_dependence( samedim, sameclass, same_degree )
    def __mod__( self, other ):
        fromgrid = refine_GridObject( other, self.knotvector + other.knotvector )
        ret = self.copy()
        ret.x = np.asarray( ret.cons | refine_GridObject( fromgrid, self.knotvector ).x )
        return ret

    """ Plotting """

    def plotting_requirements(fn):
        def decorated(*args, **kwargs):
            self, *args = args
            if len(self) <= self.targetspace:
                return fn(self, *args, **kwargs)
            raise NotImplementedError
        return decorated

    @plotting_requirements
    def plot_function(self, func=[], ischeme='bezier5', ref=0, boundary=False, **plotkwargs):
        assert self.targetspace == 2, NotImplementedError
        plt = plot.PyPlot( 'I am a dummy', **plotkwargs )
        domain = self.domain if not boundary else self.domain.boundary
        func = [ self.mapping ] + func
        points = domain.refine( ref ).elem_eval( func, ischeme=ischeme, separate=True )

        if domain.ndims == 2:
            plt.mesh(*points)
        elif domain.ndims == 1:
            assert len( func ) == 1
            plt.segments( np.asarray(*points) )
            plt.aspect( 'equal' )
            plt.autoscale( enable=True, axis='both', tight=True )
        else:
            raise NotImplementedError

        plt.show()

    def qplot( self, **kwargs ):
        self.plot_function( **kwargs )

    def qbplot( self, **kwargs ):
        self.qplot( boundary=True, **kwargs )

    def qgplot( self, **plotkwargs ):
        assert len( self ) == 2, NotImplementedError
        plt = plot.PyPlot( 'I am a dummy', **plotkwargs )
        domain = self.domain
        points = domain.elem_eval( self.geom, ischeme='bezier5', separate=True )
        plt.mesh(points)
        plt.show()

    def plot( self, name, funcs={}, ref=0, boundary=False ):
        # assert len( self ) == 2
        domain = self.domain if not boundary else self.domain.boundary
        points, det, *fncs = domain.refine( ref ).elem_eval( [ self.mapping, self.jacdet ] + [ funcs[ item ] for item in funcs.keys() ], ischeme='vtk', separate=True )
        with plot.VTKFile( name ) as vtu:
            vtu.unstructuredgrid( points, npars=2 )
            vtu.pointdataarray( 'Jacobian_Determinant', det )
            for name, func in zip( funcs.keys(), fncs ):
                vtu.pointdataarray( name, func )


    ''' Exporting '''

    def _toscipy( self ):
        assert len( self ) in (1, 2)
        knots = self.extend_knots()
        cpoints = [ pc for pc in self.x.reshape( [ self.targetspace, len( self.basis ) ] ) ]
        periodic, p, s = self.periodic, self.degree, list( self.ndims )

        for i in periodic:
            nperiodicfuncs = [ p[j] - self.knotmultiplicities[j][0] + 1 \
                               for j in range( len( self ) ) ]
            sl = { 0: ( slice( 0, nperiodicfuncs[0] ), slice( None ) ), \
                   1: ( slice( None ), slice( 0, nperiodicfuncs[1] ) ) }[i] if len( self ) == 2 \
                   else slice( 0, nperiodicfuncs[0] )
            knots[i] = knots[i][ : np.where( knots[i] == 1 )[0][0] + p[i] + 1 ]
            cpoints = [ np.concatenate( [ w.reshape( s ), w.reshape( s )[ sl ] ], \
                    axis=i ).flatten() for w in cpoints ]
            s[i] += nperiodicfuncs[i]
        
        if len( self ) == 1:

            def f( x, **scipyargs ):
                splines = [ interpolate.splev( x, \
                        ( knots[0], w, self.degree[0] ), \
                                **scipyargs ) for w in cpoints ]
                return splines

        elif len( self ) == 2:

            def f( *args, **scipyargs ):
                splines = [interpolate.bisplev( *args, \
                        ( *knots, w, *self.degree ), \
                        **scipyargs) for w in cpoints ]
                return splines

        else:
            raise NotImplementedError

        return f

    def toscipy( self ):
        f = self._toscipy()

        if len( self ) == 1:
            return lambda x, **scipyargs: np.hstack( [ y[:, _] for y in f( x, **scipyargs ) ] )
        elif len( self ) == 2:
            return lambda *args, **scipyargs: np.stack( [ y for y in f( *args, **scipyargs ) ], axis=-1 )
        else:
            raise NotImplementedError


    del _defaultvec
    del _getprivate


def z_stack( gos, verts, p=1 ):

    if p != 1:
        raise NotImplementedError

    assert aux.isincreasing( verts )

    xs, unified_knotvector = TensorGridObject.fastunify( *gos )

    z = np.linspace( 0, 1, len( gos ) )
    z_knotvector = ko.KnotObject( knotvalues=z, degree=p )

    g = TensorGridObject( knotvector=unified_knotvector*z_knotvector )
    n = np.prod( unified_knotvector.ndims )

    for j in range( len( z ) ):
        g.x[ g.index[ :, :, j ][ : 2*n ] ] = xs[ j ]
        g.x[ g.index[ :, :, j ][ -n: ] ] = verts[ j ]

    return g

# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
