#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mapping_2 import std, sol, xml
import opt


def main():

    g = xml.load_xml( '74.xml' )
    g.set_cons_from_x()

    initial = opt.Optimizer( g, func='Liao' )

    sol = opt.solve_scipy( initial )

    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    main()
