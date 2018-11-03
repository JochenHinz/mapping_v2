from . import aux, ko
import numpy as np


def UniformKnotObject(nelems, p, periodic=False):
    return ko.KnotObject(knotvalues=np.linspace(0, 1, nelems+1), degree=p, periodic=periodic)


def UnrelatedKnotObjects( n0, n1, p ):
    return [ ko.KnotObject( np.unique( np.concatenate([ [0], np.random.uniform(0, 1, n - 2), [1] ]) ), p ) for n in (n0, n1) ]
