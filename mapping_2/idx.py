#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict

__ = slice(None)

sidetuple = ('left', 'right', 'bottom', 'top', 'front', 'back')
oppositeside = dict( zip( sidetuple[::2] + sidetuple[1::2], sidetuple[1::2] + sidetuple[::2] ) )
bundledsides = tuple( zip( sidetuple[::2], sidetuple[1::2] ) )
mapsidevalue = dict( zip( sidetuple, np.repeat( np.arange(3), 2 ) ) )


slicelist = ( ([0], __, __), ([-1], __, __), (__, [0], __), (__, [-1], __), (__, __, [0]), (__, __, [-1]) )
slicedict = dict( zip(sidetuple, slicelist) )


class TensorIndex:

    @classmethod
    def fromTensorGridObject( cls, go ):
        n = go.knotvector.ndims
        indices = np.arange(go.knotvector.dim, dtype=int).reshape( n )
        return cls(indices, go.domain, length=go.knotvector.dim, targetspace=go.targetspace)

    def __init__( self, indices, domain, length=None, targetspace=1 ):
        if length is None:
            length = len(indices)
        self.domain = domain
        self.length = length
        self._indices = indices
        self.targetspace = targetspace

    @property
    def indices( self ):
        return self._indices

    @property
    def bnames( self ):
        return self.domain._bnames

    @property
    def shape( self ):
        return self.indices.shape

    def repeat( self, x ):
        l, t = self.length, self.targetspace
        return np.concatenate( [ i*l + j for i, j in enumerate( [x]*t ) ] )

    def __getitem__( self, key ):
        indices = self._indices[key]
        if np.isscalar( indices ):
            indices = np.asarray( [indices] )
        return self.repeat( indices ).flatten()

    def boundary( self, *sides ):
        if len(sides) == 1:
            side, = sides
            if side not in self.bnames:
                raise IndexError
            d = self.__dict__.copy()
            d.pop('domain'), d.pop('_indices')
            return self.__class__( self.indices[ slicedict[side][:len(self.shape)] ], self.domain.boundary[side], **d )
        ret = self
        for side in sides:
            ret = ret.boundary(side)
        return ret

    @property
    def flatten( self ):
        return self.repeat( self.indices.ravel() )

    @property
    def ravel( self ):
        return self.flatten

# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
