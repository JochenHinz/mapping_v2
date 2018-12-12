#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, inspect

from mapping import sep

from mapping_2 import pc

l, r = sep.separator(0)[2:]

l = l.T
r = r.T

F = pc.InterpolatedUnivariateSpline( l )
G = pc.InterpolatedUnivariateSpline( pc.circle( 500 ), periodic=True )
