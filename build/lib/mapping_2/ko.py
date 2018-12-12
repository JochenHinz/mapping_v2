import numpy as np
import scipy as sp
import functools, itertools
from collections import defaultdict

from nutils import *
from . import aux


_ = (slice(None), None)


@functools.total_ordering
class KnotObject:
    'Basic knot-vector that can be refined etc.'

    @classmethod
    def unify(cls, kv1, kv2):  # make this more readable, add description
        ''' Return knotvalues and knotmultiplicities of unified knot vector '''
        assert kv1.periodic == kv2.periodic, 'Cannot unify knot-vectors of periodic and non-periodic type'
        assert kv1.degree == kv2.degree
        kvkm = lambda x: [x.knots, x.knotmultiplicities]
        dict1, dict2 = [ defaultdict( lambda:1, zip(*item) ) for item in ( kvkm(k) for k in [kv1, kv2] ) ] # if key does not exist it defaults to 1
        union = np.unique( np.concatenate( [kv1.knots, kv2.knots] ) )
        km = [ max(dict1[i], dict2[i]) for i in union ]
        return cls( knotvalues=union, knotmultiplicities=km, degree=kv1.degree, periodic=kv1.periodic )

    def __init__( self, knotvalues=None, degree=3, knotmultiplicities=None, periodic=False ):
        assert knotvalues is not None, 'The knotsequence has to be provided'
        assert aux.uisincreasing(knotvalues), 'The knot-sequence needs to be strictly increasing between 0 and 1'
        # forthcoming: failswitch in case knot-sequence is too short for specified degree
        knotvalues = np.round(knotvalues, 10)
        self._knots = tuple(knotvalues)
        self._degree = degree
        self._periodic = periodic
        if knotmultiplicities is None:
            start = self.degree + 1 if not self.periodic else 1
            knotmultiplicities = np.asarray([start] + [1]*(self.nelems - 1) + [start], dtype=int)
        self._knotmultiplicities = tuple(knotmultiplicities)
        if self.periodic:
            assert self.knotmultiplicities[0] == self.knotmultiplicities[-1], 'Mismatch between first and last knotmultiplicities'

    def _toslice(self, s, length=None):
        ''' turn input s into appropriate index slice '''
        if length is None:
            length = self.nelems
        if isinstance( s, str ):
            return np.unique( np.fromiter( aux.string_to_range(s, length), dtype=int ) )
        if isinstance( s, range ):
            return np.fromiter( s, dtype=int )
        if isinstance( s, (list, np.ndarray) ):
            if len(s) > 0:
                assert np.max(s) < length and np.min(s) >= -length
            return np.unique( np.asarray(s, dtype=int) % length )
        raise ValueError('Invalid input type')

    @property
    def knots(self):
        return np.asarray(self._knots, dtype=float)

    @property
    def knotvalues(self):
        return self.knots

    @property
    def periodic(self):
        return self._periodic

    @property
    def knotmultiplicities(self):
        return np.asarray(self._knotmultiplicities, dtype=int)

    @property
    def degree(self):
        return self._degree

    @property
    def nelems(self):
        return len(self.knots) - 1

    @property
    def dim(self):
        ''' amount of basis functions resulting from knot vector'''
        return np.sum(self._knotmultiplicities[:-1]) if self.periodic else len(self.extend_knots()) - self.degree - 1 

    @property
    def lib(self):
        ''' return dict that contains all attributes'''
        names = ['knotvalues', 'knotmultiplicities', 'degree', 'periodic']
        f = lambda x: getattr(self, x)
        return dict( zip( names, map( f, names ) ) )

    def copy(self):
        return self.__class__(**self.lib)

    def expand_knots(self):
        '''for .xml output;
           the distinction between periodic = True and periodic = False is not made'''
        kvkm = [self.knots, self._knotmultiplicities]
        vec = lambda x: np.fromiter(itertools.chain.from_iterable(x), float)
        return vec( [ [i]*j for i, j in zip(*kvkm) ] )

    def extend_knots(self):
        ret = self.expand_knots()
        if self.periodic:  # this seems to work
            km = self.knotmultiplicities
            ret = ret[km[0] - 1:]
            p = self.degree
            ret = np.concatenate([ret[-p - 1:-1] - 1, ret, ret[1:p + 1] + 1])
        return ret

    def greville( self ):
        """ Return the Greville abscissae corresponding to KnotObject """
        knots = self.extend_knots()
        p = self.degree
        periodic = self.periodic
        if periodic:
            knots = knots[ : - p - 1 ]
            if self.knotmultiplicities[0] == p + 1:
                periodic = False
        grev = np.array( [ knots[ i + 1: i + p + 1 ] for i in range( self.dim ) ] ).sum( 1 ) / p
        return grev % 1 if periodic else grev

    def to_c(self, n):
        if n > self.degree - 1:
            raise ValueError('Continuity exceeds capabilities of basis degree')
        assert n >= -1
        kv, km = self.knots, self.knotmultiplicities
        if self.periodic:
            km = np.clip(km, 1, self.degree - n)
        else:
            m = self.degree + 1
            km = np.concatenate([ [m], np.clip(km[1:-1], 1, self.degree - n), [m] ])
        return self.__class__(degree=self.degree, knotvalues=kv, knotmultiplicities=km, periodic=self.periodic)

    def one_on(self, val):  # this should be done without the use of scipy eventually
        ''' return the indices of the basis functions that assume the value one on val '''
        assert isinstance(val, (float, int))
        assert not self.periodic, NotImplementedError
        indices = []
        for i in range(self.dim):
            tck = aux.tck(self, i)
            spl = sp.interpolate.splev(val, tck)
            if np.abs(spl - 1) < 1e-4:
                indices.append(i)
        return indices

    def __le__(self, other):  # see if one is subset of other
        if not (set(self.knots) <= set(other.knots) and self.degree <= other.degree and self.periodic == other.periodic):
            # knots no subset or self.p > other.p => return False
            return False
        else:  # check if knotmultiplicities are smaller or equal
            kvkm = lambda x: [x.knots, x.knotmultiplicities]
            dict1, dict2 = [ dict(zip(*item)) for item in map(kvkm, [self, other]) ]
            ''' if order not equal: other.knotmultiplicities >= self.knotmultiplicities + dkm
                must hold in order for span(other) to contain span(self) '''
            dkm = other.degree - self.degree
            # {knotvalues: knotmultiplicity, ... }
            return all( [dict1[i] + dkm <= dict2[i] for i in self.knots] )

    def __ge__(self, other):  # see if one is subset of other
        return other <= self

    def __eq__(self, other):
        return self <= other and self >= other

    def to_p(self, p):  # I am not quite sure what the function of this is exactly
        if p == self.degree:
            return self
        elif p < self.degree:
            km = np.clip( self.knotmultiplicities, 1, p+1 )
        else:
            dp = p - self.degree
            km = self.knotmultiplicities + dp
        if self.periodic:
            km[ 0 ] = min( [ p, km[ 0 ] ] )
            km[ -1 ] = min( [ p, km[ -1 ] ] )
        return self.__class__(degree=p, knotmultiplicities=km, knotvalues=self.knotvalues, periodic=self.periodic)

    def ref_by(self, indices):
        indices = self._toslice(indices, length=self.nelems)
        if len(indices) == 0:
            return self
        assert all([len(indices) <= self.nelems, np.max(indices) < self.nelems])
        # amount of indices is of course smaller than the amount of elements
        indices_ = indices + 1
        new_knots, new_km = self.knots, self.knotmultiplicities
        add = (new_knots[indices_] + new_knots[indices])/2.0
        new_knots = np.insert(new_knots, indices_, add)
        new_km = np.insert(new_km, indices_, [1]*len(indices))
        return self.__class__(degree=self.degree, knotvalues=new_knots, knotmultiplicities=new_km, periodic=self.periodic)

    def ref(self, ref=1):
        assert ref >= 0
        if ref == 0:
            return self
        ret = self.copy()
        for i in range(ref):
            ret = ret.ref_by(':')
        return ret

    def ref_by_vertices(self, vertices):
        ''' refine by providing vertex values.
            look for element that contains v for v in vertices
            and refine it'''
        vertices = np.asarray(vertices)
        if len(vertices) == 0:
            return self
        knots = self.knots
        idx = (knots[:-1][_] <= vertices)*(knots[1:][_] >= vertices) # compute boolean mask
        indices = np.unique( np.where(idx)[0]  )
        return self.ref_by(indices)

    def raise_multiplicities(self, amount, indices=[], knotvalues=[]):
        indices = self._toslice( indices, length=len(self.knots) )
        if all( [len(indices) == 0, len(knotvalues) == 0] ) or amount == 0:
            return self
        knots, new_km = self.knots, self.knotmultiplicities
        indices_ = np.unique( np.where( np.isclose( knots[_] - np.asarray(knotvalues), 0  )  )[0] )  # turn knotvalues into indices
        indices = np.unique( np.concatenate( [indices, indices_] )  )
        if len(indices) == 0:
            raise ValueError('Invalid set of knotvalues or indices supplied')
        if self.periodic:
            last = len(self.knots) - 1
            if 0 in indices:
                indices = np.unique( np.concatenate([ indices, [last] ]) )
            elif last in indices:
                indices = np.unique( np.concatenate([ [0], indices ]) )
        new_km[indices] += amount
        if np.max(new_km) > self.degree + 1:
            log.warning('Warning, knot-repetitions that exceed p+1 detected, they will be clipped')
        new_km = np.clip( new_km, 1, self.degree + 1  )
        return self.__class__(degree=self.degree, knotvalues=knots, knotmultiplicities=new_km, periodic=self.periodic)

    def add_c0(self, knotvalues):
        kv = self.add_knots(knotvalues)
        kv = kv.raise_multiplicities(self.degree, knotvalues=knotvalues).to_c(0)
        return kv

    def __add__(self, other):
        p = max(self.degree, other.degree)
        temp = self.__class__.unify(self, other)
        kv, km = temp.knots, temp.knotmultiplicities
        return self.__class__(degree=p, knotvalues=kv, knotmultiplicities=km, periodic=self.periodic)

    def add_knots(self, knotvalues):
        if len(knotvalues) == 0:
            return self
        assert all([i <= 1 and i >= 0 for i in knotvalues])  # and (0 not in knotvalues and 1 not in knotvalues)
        knotvalues = np.unique( np.concatenate([ [0], knotvalues, [1] ]) )
        dummy = self.__class__(degree=self.degree, knotvalues=knotvalues, periodic=self.periodic)
        return self + dummy

    def __mul__(self, other):
        assert isinstance(other, type(self))
        return TensorKnotObject([self, other])


@functools.total_ordering
class TensorKnotObject(numpy.ndarray):

    def __new__(cls, data):
        if not all( isinstance(d, KnotObject) for d in data ):
            raise ValueError
        return numpy.array(data).view(dtype=cls)
    
    def _vectorize(name, return_type=None):
        def wrapper(*args, **kwargs):
            self, *args = args
            rt = self.__class__ if return_type is None else return_type
            assert all(len(a) <= len(self) for a in itertools.chain(args, kwargs.values())) # length per argument does not exceed len(self)
            return rt( [ getattr(e, name)(*(a[i] for a in args), **{k: v[i] for k, v in kwargs.items()}) for i, e in enumerate(self) ] )
        return wrapper
    
    def _prop_wrapper(name, return_type = list):
        @property
        def wrapper(self):
            return return_type([getattr(e, name) for e in self])
        return wrapper
    
    extend_knots = _vectorize('extend_knots', list)
    expand_knots = _vectorize('expand_knots', list)
    greville = _vectorize( 'greville', list )
    knots = _prop_wrapper('knots')
    degree = _prop_wrapper('degree')
    ndims = _prop_wrapper('dim')
    knotmultiplicities = _prop_wrapper('knotmultiplicities')
    ref_by = _vectorize('ref_by')
    ref_by_vertices = _vectorize('ref_by_vertices')
    ref = _vectorize('ref')
    add_knots = _vectorize('add_knots')
    raise_multiplicities = _vectorize('raise_multiplicities')
    to_c = _vectorize('to_c')
    add_c0 = _vectorize('add_c0')
    to_p = _vectorize('to_p')
    
    @property
    def periodic(self):
        return tuple([i for i,j in enumerate(self) if j.periodic])

    @property
    def dim(self):
        return np.prod([i.dim for i in self])

    def __le__(self, other):
        return np.array([ i<=j for i,j in zip(self, other) ], dtype=bool)
    
    def at(self,n):  ## in __getitem__[n] we return the n-th knot_object, here a new tensor_kv with new._kvs = [self._kvs[n]]
        assert n < len(self) and n >= -len(self)
        return self.__class__( [ self[n % len(self)] ] )
        
    def __mul__(self,other):
        if isinstance(other, KnotObject):
            l = [i for i in self] + [other]
            return self.__class__(l)
        else:
            raise NotImplementedError
    
    del _vectorize
    del _prop_wrapper

# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
