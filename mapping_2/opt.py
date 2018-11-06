#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, float64, int32, int64, prange
from . import fsol, jitBSpline, aux
from scipy.interpolate import bisplev
from scipy import sparse
from nutils import log
from functools import wraps


# XXX: possibly make this function part of the .aux module
bsplev = lambda xi, eta, g, x, **kwargs: \
    bisplev( xi, eta, ( *g.extend_knots(), x, *g.degree ), **kwargs ).ravel()


tcall = lambda xi, eta, g, x, **kwargs: \
    jitBSpline._tensor_call(
        xi, eta, *g.extend_knots(), *g.degree, x, **kwargs
    ).ravel()


FastSolver = fsol.FastSolver


try:
    import cyipopt
except ImportError:
    CYIPOPT_INSTALLED = False
else:
    CYIPOPT_INSTALLED = True
    del cyipopt


def constraint( optimizer, xi, eta, c ):

    '''
        Return a vector of Jacobian Determinant
        function evaluations over the tensor-product
        points resulting from ``xi`` and ``eta``.
        The knotvector(s) are taken from ``g`` and the
        control points are given by ``c``
        (hence not necessarily g.x).
    '''

    g = optimizer._g

    assert len( g ) == g.targetspace == 2, NotImplementedError

    x, y = np.array_split( c, 2 )

    _bsplev = lambda c, **kwargs: bisplev( xi, eta, optimizer._tck(c), **kwargs ).ravel()

    x_xi = _bsplev( x, dx=1 )
    x_eta = _bsplev( x, dy=1 )
    y_xi = _bsplev( y, dx=1 )
    y_eta = _bsplev( y, dy=1 )

    return x_xi * y_eta - x_eta * y_xi


@njit(
    float64[:, :](
        float64[:],
        int64,
        int32[:],
        int32[:],
        float64[:]
    )
)
def assemble_sparse_times_dense( Dense, m, indices0, indices1, data ):

    '''
        Given a dense vector ``Dense`` and a sparse matrix A of shape
        ( Dense.shape[0], ``m`` ), whose data is given by ``data``
        and stored in ``indices0`` and ``indices1``, return a dense
        matrix whose entries are given by Dense[:, None] * A.
        The assembly is faster because only the nonzero entries of
        A are operated on.
        Required for the assembly of the Constraint Jacobian.

        XXX: possibly return a fully sparse matrix in the future,
        i.e., only the .data attribute and the positions of the
        nonzero entries.
    '''

    ret = np.zeros( (Dense.shape[0], m), dtype=float64 )
    for i in prange( len(data) ):
        j, k = indices0[i], indices1[i]
        ret[j, k] = data[i] * Dense[j]

    return ret


def constraint_jacobian( optimizer, xi, eta, c, w_xi, w_eta ):

    '''
        Assemble the discrete constraint jacobian
        corresponding to the discrete constraint function.
        The arguments are the same as in ``jacobian( ... )``
        with the two additional arguments ``w_xi`` and ``w_eta``
        that are sparse matrices containing the function evaluations
        over all constraint abscissae of the first derivatives of all
        the basis functions (per row).
    '''

    g = optimizer._g
    
    assert len( g ) == g.targetspace == 2, NotImplementedError

    x, y = np.array_split( c, 2 )

    # XXX: this is a repetition from the ``jacobian`` function.
    # Find a more compact solution

    _bsplev = lambda c, **kwargs: bisplev( xi, eta, optimizer._tck(c), **kwargs ).ravel()

    x_xi = _bsplev( x, dx=1 )
    x_eta = _bsplev( x, dy=1 )
    y_xi = _bsplev( y, dx=1 )
    y_eta = _bsplev( y, dy=1 )

    indices_xi = w_xi.nonzero()
    data_xi = w_xi.data

    indices_eta = w_eta.nonzero()
    data_eta = w_eta.data

    m = w_xi.shape[1]

    # XXX: make more compact (this is kinda ugly).
    x_eta_w_xi = assemble_sparse_times_dense( x_eta, m, *indices_xi, data_xi )
    y_xi_w_eta = assemble_sparse_times_dense( y_xi, m, *indices_eta, data_eta )
    x_xi_w_eta = assemble_sparse_times_dense( x_xi, m, *indices_eta, data_eta )
    y_eta_w_xi = assemble_sparse_times_dense( y_eta, m, *indices_xi, data_xi )

    return np.hstack( [y_eta_w_xi - y_xi_w_eta, x_xi_w_eta - x_eta_w_xi] )


def with_constraint_boundary_conditions( f ):

    '''
        Calls ``f`` with a  vector of inner DOFs
        augmented with the boundary DOFs resulting
        from the Dirichlet boundary condition of
        ``g``.
    '''

    @wraps( f )
    def wrapper( self, c, *args, **kwargs ):
        g = self._g
        vec = g.cons.copy()
        vec[ g.dofindices ] = c
        return f( self, vec, *args, **kwargs )

    return wrapper


class ConstrainedMinimizer( FastSolver ):

    '''
        Derived class from ``fsol.Fastsolver`` that has two additional
        instance methods ``constraint`` and ``constraint_gradient``.
        Can be used to define optimization problems for IPOPT.
        The ``constraint`` function constitutes a discrete evaluation
        over a tensor-product of the ``xi`` and ``eta`` points passed to
        the __init__ function via ``absc``.
        XXX: implement additional constraint functionality such as
             projection onto a duality basis or more complex constraint
             functions such as ones that impose constraint(c) >= F(c)
             where F is some positive function (possibly of the control
             points c).
        XXX: Make everything symbolic using, for instance, SymPy.
             The long term goal is that we can add several Grid-Quality
             functionals with weighting functions and the gradient /
             Hessian and computed automatically and assembled via the
             JIT functionality from the .fsol module.
    '''

    def __init__( self, g, order, absc=None ):

        # any problems with ``g`` will be caught by the parent __init__
        super().__init__( g, order )

        self._dindices = self._g.dofindices

        # The Jacobian-Determinant is evaluated in
        # the tensor-product points of xi and eta
        # as a constraint to warrant bijectivity of the result
        if isinstance( absc, int ):
            # if an integer is passed to absc, we construct
            # abscissae that correspond to and ``absc``-order
            # Gauss quadrature scheme over all elements
            absc = fsol.make_quadrature( g, absc )[0]

        self._absc = absc

        if absc is not None:
            assert all(
                all( [aux.isincreasing(x_), x_[0] >= 0, x_[-1] <= 1] )
                                                    for x_ in absc ), \
                'Invalid constraint abscissae received.'

            _bsplev = lambda i, **kwargs: sparse.csr_matrix(
                    self._splev( i, self._absc, **kwargs ).ravel()[:, None]
            )

            structure = sparse.hstack( [ _bsplev(i) for i in range(self._N) ] )
            self._w_xi = sparse.hstack( [ _bsplev(i, dx=1) for i in range(self._N) ] )
            self._w_eta = sparse.hstack( [ _bsplev(i, dy=1) for i in range(self._N) ] )

            # sparsity structure of the constraint jacobian
            self.jacobianstructure = \
                sparse.hstack( [ structure, structure ] ).\
                tolil()[:, self._dindices ].nonzero()

        log.info( 'Constraint jacobian sparsity pattern stored.' )

        if not ( self.constraint( self._g[self._dindices] ) >= 0 ).all():
            log.warning(
                '''
                    Warning, the initial guess corresponding to the passed
                    GridObject is not feasible. Unless another initial guess
                    is specified, the optimizer will most likely fail.
                '''
            )

    @with_constraint_boundary_conditions
    def constraint( self, c ):

        g = self._g

        if self._absc is None:
            raise AssertionError('No constraints set.')

        log.info( 'Computing constraint ...' )
        ret = constraint( self, *self._absc, c )
        log.info( 'Completed' )

        return ret

    @with_constraint_boundary_conditions
    def constraint_gradient( self, c, sprs=False ):

        if self._absc is None:
            raise AssertionError('No constraints set.')

        d = self._g.dofindices

        log.info( 'Computing constraint gradient ...' )

        ret = constraint_jacobian(
            self, *self._absc, c, self._w_xi, self._w_eta )[:, d]

        log.info( 'Completed' )

        if not sprs:
            return ret

        return sparse.csr_matrix( ret )


def with_scalar_boundary_conditions( f ):

    @wraps( f )
    def wrapper( self, c ):
        g = self._g
        vec = g.cons.copy()
        vec[ g.dofindices ] = c
        ret = f( self, vec )
        try:
            self._feval += 1
        except AttributeError:
            self._feval = 1
        return ret

    return wrapper


class Liao( ConstrainedMinimizer ):

    @with_scalar_boundary_conditions
    def residual( self, c ):

        g11, g12, g22 = self.metric( c )
        ret = ( self._weights * ( g11 ** 2 + 2 * g12 ** 2 + g22 ** 2 ) ).sum()

        log.info( ret )

        return ret

    @fsol.with_boundary_conditions
    def gradient( self, c ):

        (x_xi, x_eta), (y_xi, y_eta) = self.all_fderivs(c)
        g11, g12, g22 = self.metric( c )

        mul00 = 4 * g11 * x_xi + 4 * g12 * x_eta
        mul01 = 4 * g12 * x_xi + 4 * g22 * x_eta

        mul10 = 4 * g11 * y_xi + 4 * g12 * y_eta
        mul11 = 4 * g12 * y_xi + 4 * g22 * y_eta

        arr = self.jitarray
        return \
            np.concatenate( [
                arr( mul=mul00, w='x' ) + arr( mul=mul01, w='y' ),
                arr( mul=mul10, w='x' ) + arr( mul=mul11, w='y' )
            ] )


class Winslow( ConstrainedMinimizer ):

    @fsol.InstanceMethodCache
    def metric_trace( self, c ):
        """ g11 + g22 """
        (x_xi, x_eta), (y_xi, y_eta) = self.all_fderivs(c)
        return x_xi ** 2 + y_xi ** 2 + x_eta ** 2 + y_eta ** 2

    @fsol.InstanceMethodCache
    def jacdet( self, c ):
        J = FastSolver.jacdet( self, c )
        J = np.clip( J, 1e-8, np.inf )
        return J

    @with_scalar_boundary_conditions
    def residual( self, c ):
        J = self.jacdet(c)
        (x_xi, x_eta), (y_xi, y_eta) = self.all_fderivs(c)
        return ( self._weights *
            (x_xi ** 2 + y_xi ** 2 + x_eta ** 2 + y_eta ** 2) / J ).sum()

    @fsol.with_boundary_conditions
    def gradient( self, c ):
        J = self.jacdet(c)
        (x_xi, x_eta), (y_xi, y_eta) = self.all_fderivs(c)
        trace = self.metric_trace(c)

        mul00, mul01 = 2 * x_xi / J - trace * y_eta / (J ** 2), \
            ( 2 * x_eta / J + trace * y_xi / ( J ** 2 ) )
        mul10, mul11 = ( 2 * y_xi / J + trace * x_eta / ( J ** 2 ) ),\
            2 * y_eta / J - trace * x_xi / (J ** 2)

        arr = self.jitarray
        return \
            np.concatenate( [
                arr( mul=mul00, w='x' ) + arr( mul=mul01, w='y' ),
                arr( mul=mul10, w='x' ) + arr( mul=mul11, w='y' )
            ] )


class AO( ConstrainedMinimizer ):

    @with_scalar_boundary_conditions
    def residual( self, c ):
        g11, g12, g22 = self.metric( c )
        ret = ( self._weights * ( g11 * g22 ) ).sum()
        log.info( ret )
        return ret

    @fsol.with_boundary_conditions
    def gradient( self, c ):

        log.info( 'Computing gradient' )

        (x_xi, x_eta), (y_xi, y_eta) = self.all_fderivs(c)
        g11, g12, g22 = self.metric( c )

        mul00 = 2 * x_xi * g22
        mul01 = 2 * x_eta * g11

        mul10 = 2 * y_xi * g22
        mul11 = 2 * y_eta * g11

        arr = self.jitarray
        return \
            np.concatenate( [
                arr( mul=mul00, w='x' ) + arr( mul=mul01, w='y' ),
                arr( mul=mul10, w='x' ) + arr( mul=mul11, w='y' )
            ] )


class Area( ConstrainedMinimizer ):

    @with_scalar_boundary_conditions
    def residual( self, c ):
        J = self.jacdet(c)
        ret = ( self._weights * J ** 2 ).sum()
        log.info( ret )
        return ret

    @fsol.with_boundary_conditions
    def gradient( self, c ):
        J = self.jacdet(c)
        (x_xi, x_eta), (y_xi, y_eta) = self.all_fderivs(c)

        mul00, mul01 = 2 * J * y_eta, - 2 * J * y_xi
        mul10, mul11 = - 2 * J * x_eta, 2 * J * x_xi

        arr = self.jitarray
        return \
            np.concatenate( [
                arr( mul=mul00, w='x' ) + arr( mul=mul01, w='y' ),
                arr( mul=mul10, w='x' ) + arr( mul=mul11, w='y' )
            ] )


def minimize_ipopt_sparse( fun, x0, jacobianstructure, args=(), kwargs=None,
        method=None, jac=None, hess=None, hessp=None, bounds=None,
        constraints=(), tol=None, callback=None, options=None ):

    if not CYIPOPT_INSTALLED:
        raise ImportError(
            ''' IPOPT functionality unavailable because
            cyipopt is not installed properly. Please visit
            https://github.com/matthias-k/cyipopt. The package is best installed
            using the Anaconda environment.'''
        )

    import cyipopt
    from ipopt.ipopt_wrapper import IpoptProblemWrapper, get_constraint_bounds, \
        replace_option, convert_to_bytes, get_bounds

    class SparseIpoptProblemWrapper( IpoptProblemWrapper ):

        def __init__( self, jacobianstructure, *args, **kwargs ):
            super().__init__( *args, **kwargs )
            self.jacobianstructure = lambda: jacobianstructure

        def jacobian( self, x ):
            return self._constraint_jacs[0]( x, *self._constraint_args[0] ).data

    _x0 = np.atleast_1d(x0)
    problem = SparseIpoptProblemWrapper(
        jacobianstructure, fun, args=args, kwargs=kwargs, jac=jac, hess=hess,
        hessp=hessp, constraints=constraints)
    lb, ub = get_bounds(bounds)

    cl, cu = get_constraint_bounds(constraints, x0)

    if options is None:
        options = {}

    nlp = cyipopt.problem(n=len(_x0),
                          m=len(cl),
                          problem_obj=problem,
                          lb=lb,
                          ub=ub,
                          cl=cl,
                          cu=cu)

    # python3 compatibility
    convert_to_bytes(options)

    # Rename some default scipy options
    replace_option(options, b'disp', b'print_level')
    replace_option(options, b'maxiter', b'max_iter')
    if b'print_level' not in options:
        options[b'print_level'] = 0
    if b'tol' not in options:
        options[b'tol'] = tol or 1e-8
    if b'mu_strategy' not in options:
        options[b'mu_strategy'] = b'adaptive'
    if b'hessian_approximation' not in options:
        if hess is None and hessp is None:
            options[b'hessian_approximation'] = b'limited-memory'
    for option, value in options.items():
        try:
            nlp.addOption(option, value)
        except TypeError as e:
            raise TypeError('Invalid option for IPOPT: {0}: {1} (Original message: "{2}")'.format(option, value, e))

    x, info = nlp.solve(_x0)

    if np.asarray(x0).shape == ():
        x = x[0]

    from scipy.optimize import OptimizeResult

    return OptimizeResult(x=x, success=info['status'] == 0, status=info['status'],
                          message=info['status_msg'],
                          fun=info['obj_val'],
                          info=info,
                          nfev=problem.nfev,
                          njev=problem.njev,
                          nit=problem.nit)


def minimize( g, order=6, constraints=5, inplace=True, method='Liao', **kwargs ):

    if constraints is None:
        raise NotImplementedError

    optargs = { 'g': g, 'order': order, 'absc': constraints }

    optimizer = { 'Liao': Liao, 'Winslow': Winslow,
            'AO': AO, 'Area': Area }[method]( **optargs )

    fun = optimizer.residual
    x0 = optimizer._g.x[ optimizer._dindices ]
    jacobianstructure = optimizer.jacobianstructure
    jac = optimizer.gradient

    constraint_gradient = lambda *args, **kwargs: \
        optimizer.constraint_gradient( *args, sprs=True, **kwargs )
    constraints = [ { 'type': 'ineq', 'fun': optimizer.constraint,
        'jac': constraint_gradient }]

    solution = minimize_ipopt_sparse( fun, x0, jacobianstructure,
        jac=jac, constraints=constraints, **kwargs )

    if not inplace:
        return solution

    g.x[ g.dofindices ] = solution.x
