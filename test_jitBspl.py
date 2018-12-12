#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from mapping_2 import std, jitBSpline

g = std.Cube()


import ipdb
ipdb.set_trace()

jitBSpline.der_ith_basis_fun( g.extend_knots()[0], g.degree[0], 4, 0.14, 3 )
