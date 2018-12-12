#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from xml.etree import ElementTree as ET


def xml_element( g, index=0 ):

    """
        Save ``go.TensorGridObject``-like object as .xml-file.
        ``g`` must implement:
            __len__
            targetspace: int
            knotvector: ko.TensorKnotObject
        and must inherit periodic, expand_knots, ... from knotvector
    """

    geometry = ET.Element( 'Geometry' )
    geometry.set( 'id', str( index ) )
    geometry.set( 'type', 'TensorBSpline' + str( g.targetspace ) )

    Basis = ET.SubElement( geometry, 'Basis' )
    Basis.set( 'type', 'TensorBSplineBasis' + str( g.targetspace ) )

    Bases = [ ET.SubElement( Basis, 'Basis' ) for i in range( len( g ) ) ]
    KVS = [ ET.SubElement( Base, 'KnotVector' ) for Base in Bases ]

    for i in range( len( Bases ) ):
        Bases[ i ].set( 'type', 'BSplineBasis' )
        Bases[ i ].set( 'index', str( i ) )
        KVS[ i ].set( 'degree', str( g.degree[ i ] ) )
        KVS[ i ].set( 'periodic', str( i in g.periodic ) )
        KVS[ i ].text = ' '.join( np.array( g.expand_knots()[ i ], dtype=str ) )

    Coeffs = ET.SubElement( geometry, 'coefs' )
    Coeffs.set( 'geoDim', str( g.targetspace ) )

    c = [ _.reshape( g.ndims ).T.ravel() for _ in np.split( g.x, g.targetspace ) ]
    c = np.stack( [ c_.astype( str ) for c_ in c ], axis=1 ).ravel()

    Coeffs.text = ' '.join( c )

    return geometry


def save_xml( g, name ):

    # assert len( g ) == 2, NotImplementedError

    xml = ET.Element( 'xml' )
    xml.append( xml_element( g, index=0 ) )
    xml = ET.ElementTree( xml )
    xml.write( '{}.xml'.format( name ) )


def load_xml( path, index=0, return_type=None ):
    """
        Load TensorGridObject-like object from .xml-file.
        return_type must accept at least ``knotvector`` and ``targetspace``
        keyword arguments.
        If return_type is None it defaults to ``go.TensorGridObject``.
    """

    xml = ET.parse( path ).getroot()
    xml = xml[ index ]
    basis = xml[0]
    coeffs = xml[1]

    n_kvs = int( xml.attrib[ 'type' ][ -1 ] )

    kvs_kms = [ np.unique( np.array( basis[ i ][0].text.split(), dtype=float ), return_counts=True ) for i in range( 2 ) ]
    degrees = [ int( basis[ i ][ 0 ].attrib[ 'degree' ] ) for i in range( 2 ) ]

    try:
        periodic = [ j for j in range( n_kvs ) if basis[ j ][ 0 ].attrib[ 'periodic' ] == 'True' ]
    except KeyError:
        periodic = ()

    targetspace = int( xml[ 1 ].attrib[ 'geoDim' ] )

    from . import go, ko

    if return_type is None:
        return_type = go.TensorGridObject

    knotvector=[ ko.KnotObject( knotvalues=kvs_kms[j][0],
            knotmultiplicities=kvs_kms[j][1], degree=degrees[j], periodic=j in periodic )
            for j in range( n_kvs ) ]

    g = return_type( knotvector=knotvector, targetspace=targetspace )

    x = np.array( coeffs.text.split(), dtype=float ).reshape( [ -1, targetspace ], order='C' ).T

    for i in range( x.shape[ 0 ] ):
        x[ i ] = x[ i ].reshape( list( reversed( g.ndims ) ) if len( g ) > 1 else -1 ).T.ravel()

    g.x = x.ravel()

    return g

# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=4:ts=4:sts=4:foldignore=#
