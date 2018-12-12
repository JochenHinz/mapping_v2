#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyoptsparse
import matplotlib.pyplot as plt
import pyOpt
import numpy as np
from mapping_2 import go, ko
from nutils import function
from nutils.solver import Integral
from scipy import sparse


class DefectGridObject( go.TensorGridObject ):

    def __init__( self, g, dual_basis='exact' ):
        assert len( g ) == 2, NotImplementedError

        if dual_basis=='exact':
            p, km = g.degree, g.knotmultiplicities
            begin = [ 2*p[i] if i not in g.periodic else p[i] + km[i][0] for i in range(2) ]
            knotmultiplicities = [ [ begin[i] ] + [ p[i] + j for j in km[i][1:-1] ] + [ begin[i] ] for i in range(2) ]
            knotvector = np.prod( [
                                    ko.KnotObject( knotvalues=k.knotvalues, degree=2*k.degree-1, knotmultiplicities=i, periodic=k.periodic )
                                    for k, i in zip( g.knotvector, knotmultiplicities )
                                    ] )
        else:
            assert isinstance( dual_basis, int )
            p = ( dual_basis, )*2
            knotvector = np.prod( [ k.to_p( dual_basis ) for k in g.knotvector ] )

        super().__init__( knotvector=knotvector, ischeme=g.ischeme*3, domain=g.domain, geom=g.geom )
        M = self.integrate( function.outer( self.basis ) ).toscipy().tocsc()
        self._M = M
        self._M_inv = sparse.linalg.splu( M )
        self._g = g

    def project_array( self, arr ):
        assert arr.shape[0] == self._M.shape[0]
        return self._M_inv.solve( arr )


def cost_func_library( name, g, target ):
    x = g.basis.vector( 2 ).dot( target )
    jac = x.grad( g.geom )
    det = function.determinant( jac )
    g11, g22 = [ ( jac[:, i]**2 ).sum() for i in range(2) ]
    g12 = ( jac[:, 0]*jac[:, 1] ).sum()
    if name == 'Liao':
        return g11**2 + 2*g12**2 + g22**2
    elif name == 'Area-Orthogonality' or 'AO':
        return g11 * g22
    elif name == 'Winslow':
        return ( g11 + g22 ) / det
    else:
        raise 'Unknown name {}'.format( name )


def with_boundary_condition( addone=False ):
    def wrapper( f ):
        def wrapped( *args, **kwargs ):
            self, c = args
            x = self._g.cons.copy()
            indices = self._g.dofindices
            if addone:
                indices = np.array( np.concatenate( [ indices, [ len( x ) ] ] ), dtype=int )
                x = np.concatenate( [ x, [0] ])
            x[ indices ] = c
            return f( self, x, **kwargs )
        return wrapped
    return wrapper


def withsparse( eps=1e-12 ):
    def wrapper( f ):
        def wrapped( *args, **kwargs ):
            ret = f( *args, **kwargs )
            ret[ np.abs( ret < eps ) ] = 0
            return sparse.csr_matrix( ret )
        return wrapped
    return wrapper


class Initializer:

    def __init__( self, g, eps=5, **dgokwargs ):
        self._g = g
        self._dgo = DefectGridObject( g, **dgokwargs )
        self._target = function.Argument( 'target', [ len( g.basis.vector(2) ) ] )
        self._jacdet = function.determinant( g.basis.vector(2).dot( self._target ).grad( self._g.geom ) )
        self._b = self._g.domain.integral( self._dgo.basis*self._jacdet, geometry=self._g.geom, degree=self._dgo.ischeme )
        self._dbdc = self._b.derivative( 'target' )
        unitvec = np.zeros( len( self._g.dofindices ) + 1 )
        unitvec[-1] = 1
        self._unitvec = unitvec
        x0 = np.concatenate( [ self._g.x[ self._g.dofindices ], [0] ] )
        z0 = np.min( self.constraint( x0 ) )
        z0 = 1.5*z0 if z0 <= 0 else 0.6*z0
        x0[-1] = z0
        self.x0 = x0
        assert ( self.constraint( x0 ) > 0 ).all()
        self.eps = eps

    @with_boundary_condition( addone=True )
    def constraint( self, c ):
        x, c = c[ :-1 ], c[ -1 ]
        ret = self._dgo.project_array( Integral.multieval( self._b, arguments={'target': x} )[0] ) - c
        return ret

    @with_boundary_condition( addone=True )
    def constraint_jacobian( self, c ):
        x, c = c[ :-1 ], c[ -1 ]
        L = self._dgo.project_array( Integral.multieval( self._dbdc, arguments={'target': x} )[0].toarray() )[:, self._g.dofindices]
        ret = np.hstack( [ L, -np.ones( L.shape[0] )[:, None] ] )
        return ret

    def func( self, c ):
        ret = c[-1]
        print( ret, 'func' )
        return 0.5*( self.eps - ret ) ** 2

    def grad( self, c ):
        return -( self.eps -c[-1] ) * self._unitvec

    def hess( self, c ):
        return np.zeros( [ len( c ) ] * 2 )


class Optimizer:

    def __init__( self, g, func='Winslow', **dgokwargs ):
        self._g = g
        self._dgo = DefectGridObject( g, **dgokwargs )
        self._target = function.Argument( 'target', [ len( g.basis.vector(2) ) ] )
        self._jacdet = function.determinant( g.basis.vector(2).dot( self._target ).grad( self._g.geom ) )
        self._b = self._g.domain.integral( self._dgo.basis*self._jacdet, geometry=self._g.geom, degree=self._dgo.ischeme )
        self._dbdc = self._b.derivative( 'target' )
        f = cost_func_library( func, g, self._target )
        self._func = self._g.domain.integral( f, geometry=self._g.geom, degree=self._dgo.ischeme )
        self._grad = self._func.derivative( 'target' )
        self._hess = self._grad.derivative( 'target' )
        self.x0 = self._g.x[ self._g.dofindices ]

    @with_boundary_condition()
    def constraint( self, c ):
        ret = self._dgo.project_array( Integral.multieval( self._b, arguments={'target': c} )[0] )
        return ret

    @with_boundary_condition()
    def constraint_jacobian( self, c ):
        ret = self._dgo.project_array( Integral.multieval( self._dbdc, arguments={'target': c} )[0].toarray() )[:, self._g.dofindices]
        return ret

    @with_boundary_condition()
    def func( self, c ):
        ret = Integral.multieval( self._func, arguments={'target': c} )[0]
        print( ret )
        return ret

    @with_boundary_condition()
    def grad( self, c ):
        ret = Integral.multieval( self._grad, arguments={'target': c} )[0]
        return ret[ self._g.dofindices ]

    @with_boundary_condition()
    def hess( self, c ):
        ret = Integral.multieval( self._hess, arguments={'target': c} )[0].toarray()
        return ret[ self._g.dofindices ][:, self._g.dofindices]


class UnconstrainedOptimizer:

    def __init__( self, g, func='Liao', penalty='exp', mu=-10 ):
        self._g = g
        self._target = function.Argument( 'target', [ len( g.basis.vector(2) ) ] )
        self._jacdet = function.determinant( g.basis.vector(2).dot( self._target ).grad( self._g.geom ) )
        f = cost_func_library( func, g, self._target )

        if penalty == 'exp':
            f += function.Exp( mu * self._jacdet )
        elif penalty == 'logsigmoid':
            f += -function.Log( 1 / ( 1 + function.Exp( -20*self._jacdet ) ) - 0.5 + 0.0001 )

        self._func = self._g.domain.integral( f, geometry=self._g.geom, degree=self._g.ischeme*3 )
        self._grad = self._func.derivative( 'target' )
        self._hess = self._grad.derivative( 'target' )
        self.x0 = self._g.x[ self._g.dofindices ]

    @with_boundary_condition()
    def func( self, c ):
        ret = Integral.multieval( self._func, arguments={'target': c} )[0]
        if ret > 1e10:
            ret = 1e10
        print( ret )
        return ret

    @with_boundary_condition()
    def grad( self, c ):
        ret = Integral.multieval( self._grad, arguments={'target': c} )[0]
        return np.clip( ret[ self._g.dofindices ], -1e10, 1e10 )

    @with_boundary_condition()
    def hess( self, c ):
        ret = Integral.multieval( self._hess, arguments={'target': c} )[0].toarray()
        return ret[ self._g.dofindices ][:, self._g.dofindices]


def jacdet( g, x ):
    return function.determinant( g.basis.vector(2).dot( x ).grad( g.geom ) )


class Improved_Initializer:

    def __init__( self, g, eps=5, **dgokwargs ):
        self._g = g
        self._dgo = DefectGridObject( g, **dgokwargs )
        self._target = function.Argument( 'target', [ len( g.basis.vector(2) ) ] )
        self._jacdet = jacdet( g, self._target )
        self._b = self._g.domain.integral( self._dgo.basis*self._jacdet, geometry=self._g.geom, degree=self._dgo.ischeme )
        self._dbdc = self._b.derivative( 'target' )
        self._indextuple = ( 0, len( self._g.dofindices ), len( self._g.dofindices ) + len( self._dgo.basis ) )
        unitvec = np.zeros( len( self._g.dofindices ) + 1 + len( self._dgo.basis ) )
        unitvec[ self._indextuple[1] ] = 1
        self._unitvec = unitvec
        x0 = np.concatenate( [ self._g.x[ self._g.dofindices ], [0], [0] * len( self._dgo.basis ) ] )
        M_inv_b0 = self._dgo._M_inv.solve( self.equality_constraint( x0 ) )
        z0 = np.min( M_inv_b0 )
        z0 = 1.5*z0 if z0 <= 0 else 0.6*z0
        d0 = M_inv_b0 - z0 * np.ones( len( self._dgo.basis ) )
        self.x0 = np.concatenate( [ x0[ : self._indextuple[1] ], [ z0 ], d0 ] )
        assert np.allclose( self.equality_constraint( self.x0 ), 0 ) and ( self.inequality_constraint( self.x0 ) >= 0 ).all()
        self.eps = eps

    def func( self, c ):
        ret = c[ self._indextuple[1] ]
        print( ret )
        return 0.5*( self.eps - ret ) ** 2

    def grad( self, c ):
        z = c[ self._indextuple[1] ]
        return - ( self.eps - z ) * self._unitvec

    def b( self, c ):
        k = self._indextuple[1]
        c, z, d = c[ :k  ], c[ k ], c[ k + 1: ]
        x = self._g.cons.copy()
        x[ self._g.dofindices ] = c
        b = Integral.multieval( self._b, arguments={ 'target': x } )[0]
        return b

    def equality_constraint( self, c ):
        k = self._indextuple[1]
        c, z, d = c[ :k  ], c[ k ], c[ k + 1: ]
        x = self._g.cons.copy()
        x[ self._g.dofindices ] = c
        b = Integral.multieval( self._b, arguments={ 'target': x } )[0]
        ret = b - self._dgo._M.dot( z*np.ones( len( d ) ) + d )
        print( np.allclose( ret, 0 ) )
        return ret

    def inequality_constraint( self, c ):
        k = self._indextuple[1]
        c, z, d = c[ :k  ], c[ k ], c[ k + 1: ]
        print( ( d >= 0  ).all() )
        return d

    def equality_constraint_jacobian( self, c ):
        k = self._indextuple[1]
        c, z, d = c[ :k  ], c[ k ], c[ k + 1: ]
        x = self._g.cons.copy()
        x[ self._g.dofindices ] = c
        dbdc = Integral.multieval( self._dbdc, arguments={ 'target': x } )[0].toscipy().tolil()[:, self._g.dofindices].tocsr()
        M = self._dgo._M
        M_one = sparse.csr_matrix( M.dot( np.ones( len( d ) )[:, None] ) )
        return sparse.hstack( [ dbdc, -M_one, -M ] )

    def inequality_constraint_jacobian( self, c ):
        k = self._indextuple[1]
        c, z, d = c[ :k  ], c[ k ], c[ k + 1: ]
        # return np.hstack( [ np.zeros( [ len( d ), len( c ) + 1 ] ), np.eye( len( d ) ) ] )
        return sparse.hstack( [ sparse.csr_matrix( ( len( d ), len( c ) + 1 ) ), sparse.identity( len( d ) )  ] )


def solve_pyopt( optimizer, method='IPOPT' ):
    assert isinstance( optimizer, ( Initializer, Optimizer ) )

    def objfunc( x ):
        f = optimizer.func( x )
        g = optimizer.constraint( x )
        return f, g, False

    def sensfunc( x, f, g, *args, **kwargs ):
        x = np.asarray( x )
        f_g, g_g = optimizer.grad( x ), optimizer.constraint_jacobian( x )
        return f_g[ None ], g_g, False

    optProb = pyOpt.Optimization( 'Grid Generation', objfunc )

    value = optimizer.x0
    lower = [ -1e10 ] * len( value )
    upper = [ 1e10 ] * len( value )
    optProb.addVarGroup( 'xvars', len(value), lower=lower, upper=upper, value=value )

    lower_ = [0.0001]*len( optimizer._dgo.basis )
    upper_ = [ 1e10 ]*len( optimizer._dgo.basis )

    optProb.addConGroup( 'con', len(lower_), lower=lower_, upper=upper_)

    optProb.addObj('obj')

    try:
        prob = getattr( pyOpt, method )()
    except:
        raise Exception( "Error loading the '{}' optimizer".format( method ) )

    return prob( optProb, sens_type=sensfunc )[1]


def solve_pyoptsparse( optimizer, method='SLSQP' ):
    assert isinstance( optimizer, ( Initializer, Optimizer ) )

    def csr_to_mat( A ):
        return { 'csr': [ A.indptr, A.indices, A.data ], 'shape': A.shape}

    def objfunc( xdict ):
        x = xdict[ 'xvars' ]

        funcs = {}
        funcs[ 'obj' ] = optimizer.func( x )
        funcs[ 'con' ] = optimizer.constraint( x )

        return funcs, False

    def sensfunc( xdict, funcs ):
        x = xdict[ 'xvars' ]

        funcsSens = {}
        funcsSens[ 'obj', 'xvars' ] = optimizer.grad( x )
        funcsSens[ 'con', 'xvars' ] = optimizer.constraint_jacobian( x )

        return funcsSens, False

    optProb = pyoptsparse.Optimization( 'Grid Generation', objfunc )

    value = optimizer.x0
    lower = [ None ] * len( value )
    upper = [ None ] * len( value )

    optProb.addVarGroup( 'xvars', len(value), lower=lower, upper=upper, value=value )

    lower = [0.0000001] * len( optimizer._dgo.basis )
    upper = [ 1e20 ] * len( optimizer._dgo.basis )

    optProb.addConGroup( 'con', len(lower), lower=lower, upper=upper ) #, wrt=[ 'xvars' ], jac={ 'xvars': jac } )

    optProb.addObj( 'obj' )
    optProb.printSparsity()
    opt = pyoptsparse.OPT( method, options={} )

    return opt( optProb, sens=sensfunc ).xStar[ 'xvars' ]


def solve_scipy( optimizer, constrain=True, **ignoreargs ):
    assert isinstance( optimizer, ( Initializer, Optimizer ) )

    f = optimizer.func
    g = optimizer.grad

    from scipy import optimize

    constraints = [ { 'type': 'ineq', 'fun': lambda x: optimizer.constraint( x ) - 0.01*np.ones( len( optimizer._dgo.basis ) ), 'jac': optimizer.constraint_jacobian } ] if constrain else ()

    method = 'SLSQP' if constrain else 'Newton-CG'

    hess = None if constrain else optimizer.hess

    x = optimize.minimize( f, optimizer.x0, jac=g, hess=hess, constraints=constraints, method=method )

    print( x.success )

    return x.x


def solve_scipy_improved( optimizer, **ignoreargs ):
    assert isinstance( optimizer, Improved_Initializer )

    f = optimizer.func
    g = optimizer.grad

    from scipy import optimize

    constraints = [ { 'type': 'eq', 'fun': optimizer.equality_constraint, 'jac': optimizer.equality_constraint_jacobian } ]
    constraints += [ { 'type': 'ineq', 'fun': optimizer.inequality_constraint, 'jac': optimizer.inequality_constraint_jacobian } ]

    x = optimize.minimize( f, optimizer.x0, jac=g, constraints=constraints, method='SLSQP' )

    return x.x[ :optimizer._indextuple[1] ]


def solve_pyoptsparse_improved( optimizer, method='SLSQP' ):
    assert isinstance( optimizer, Improved_Initializer )

    def csr_to_mat( A ):
        return { 'csr': [ A.indptr, A.indices, A.data ], 'shape': A.shape}

    def objfunc( xdict ):
        x = xdict[ 'xvars' ]

        funcs = {}
        funcs[ 'obj' ] = optimizer.func( x )
        funcs[ 'con' ] = np.concatenate( [ optimizer.equality_constraint( x ), optimizer.inequality_constraint( x ) ] )

        return funcs, False

    def sensfunc( xdict, funcs ):
        x = xdict[ 'xvars' ]

        funcsSens = {}
        funcsSens[ 'obj', 'xvars' ] = optimizer.grad( x )
        funcsSens[ 'con', 'xvars' ] = csr_to_mat(  sparse.vstack( [ optimizer.equality_constraint_jacobian( x ), optimizer.inequality_constraint_jacobian( x ) ] ).tocsr() )

        return funcsSens, False

    optProb = pyoptsparse.Optimization( 'Grid Generation', objfunc )

    value = optimizer.x0
    lower = [ None ] * len( value )
    upper = [ None ] * len( value )

    optProb.addVarGroup( 'xvars', len(value), lower=lower, upper=upper, value=value )

    lower = [ -1e-8 ] * len( optimizer._dgo.basis ) + [0] * len( optimizer._dgo.basis )
    upper = [ 1e-8 ] * len( optimizer._dgo.basis ) + [ 1e15 ] * len( optimizer._dgo.basis )

    x0 = optimizer.x0

    jac = csr_to_mat(  sparse.vstack( [ optimizer.equality_constraint_jacobian( x0 ), optimizer.inequality_constraint_jacobian( x0 ) ] ).tocsr() )

    optProb.addConGroup( 'con', len(lower), lower=lower, upper=upper, wrt=[ 'xvars' ], jac={ 'xvars': jac } )

    optProb.addObj( 'obj' )
    optProb.printSparsity()
    opt = pyoptsparse.OPT( method, options={} )

    return opt( optProb, sens=sensfunc ).xStar[ 'xvars' ][ :optimizer._indextuple[1] ]


def solve_unconstrained_scipy( optimizer, method='SLSQP', maxiter=10 ):
    assert isinstance( optimizer, UnconstrainedOptimizer )

    f = optimizer.func
    g = optimizer.grad
    hess = optimizer.hess

    from scipy import optimize

    x = optimize.minimize( f, optimizer.x0, jac=g, hess=hess, method=method, options={ 'maxiter': maxiter } )

    print( x.success )

    return x.x




# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
