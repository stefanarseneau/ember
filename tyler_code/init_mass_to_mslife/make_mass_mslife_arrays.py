# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:20:36 2022

@author: Tyler
"""

import numpy
from astropy.table import Table
import glob
from read_mist_models import EEP

filenames = glob.glob("C:\\Users\\Tyler\\Documents\\WD_Research\\MIST_Isochrones\\MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS\\*.track.eep")
masses_grid = numpy.array([int(filenames[i][97:102]) for i in range(len(filenames))])/100.0
lifetimes = []
for filename in filenames:
    eep = EEP(filename, verbose = False)
    lifetime = eep.eeps['star_age'][numpy.where(eep.eeps['phase'] >= 0)][-1] - eep.eeps['star_age'][numpy.where(eep.eeps['phase'] >= 0)][0]
    lifetimes.append(lifetime)

lifetimes = numpy.array(lifetimes)/1e9

numpy.save('mi.npy', masses_grid)
numpy.save('msl.npy', lifetimes)