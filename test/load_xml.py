#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xml.etree.ElementTree as ET


def twin_screw():
    xml = ET.parse( 'xml/SRM4_6_gap0.1mm.xml' ).getroot()
    male, female = [ np.array( xml[i].text.split(), dtype=float ).reshape( [2, -1] ).T for i in range(2) ]
    male, female = [ np.vstack( [ pc, pc[0][None] ] ) for pc in (male, female) ]
    return male, female
