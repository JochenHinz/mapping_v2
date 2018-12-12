import numpy as np
from timeit import default_timer as timer
from numba import jit, float64, int32
from . import std, ko


@jit( float64[:, :]( float64[:], float64[:], int32 ) )
def make_T( kvn, kvo, p ):
    n = kvn.shape[0] - 1
    m = kvo.shape[0] - 1
    T = np.zeros([n, m])

    for i in range( n ):
        for j in range( m ):
            if kvn[i] >= kvo[j] and kvn[i] < kvo[j+1]:
                T[i, j] = 1

    for q in range(p):
        q = q + 1
        T_new = np.zeros([n - q, m - q])
        for i in range( T_new.shape[0] ):
            for j in range( T_new.shape[1] ):
                fac1 = (kvn[i + q] - kvo[j])/(kvo[j+q] - kvo[j]) if kvo[j+q] != kvo[j] else 0
                fac2 = (kvo[j + 1 + q] - kvn[i + q])/(kvo[j + q + 1] - kvo[j + 1]) if kvo[j + q + 1] != kvo[j + 1] else 0
                T_new[i, j] = fac1*T[i, j] + fac2*T[i, j + 1]
        T = T_new
    return T


def prolongation_matrix(kvold, kvnew):
    assert all( [isinstance(kv, ko.KnotObject) for kv in (kvold, kvnew)] )
    assert_params = [kvnew <= kvold, kvold <= kvnew]
    assert any(assert_params), 'The kvs must be nested'  # check for nestedness
    if all(assert_params):
        return np.eye(kvold.dim)
    p = kvnew.degree
    kv_new, kv_old = [k.extend_knots() for k in (kvnew, kvold) ]  # repeat first and last knots
    if assert_params[0]:  ## kv_new <= kv_old, reverse order 
        kv_new, kv_old = list( reversed( [kv_new, kv_old] ) )

    T = make_T( kv_new, kv_old, kvnew.degree )

    if kvnew.periodic:  # some additional tweaking in the periodic case
        n, m = T.shape
        T_ = T
        T = T[:n-2*p, :m-2*p]
        T[:, 0:p] += T_[:n-2*p, m-2*p: m-2*p+p]
    # return T if kv_new >= kv_old else the restriction
    return T if not assert_params[0] else np.linalg.inv(T.T.dot(T)).dot(T.T)

kv_old = std.KnotObject
kv_new = kv_old.ref( 1 )

T = make_T( kv_new.extend_knots(), kv_old.extend_knots(), kv_new.degree )

kv_old = kv_old.ref( 4 )
kv_new = kv_new.ref( 4 )


def tim( f, *args ):
    s = timer()
    f( *args )
    e = timer()
    return e - s

print( tim( make_T, kv_new.extend_knots(), kv_old.extend_knots(), kv_old.degree ) )
