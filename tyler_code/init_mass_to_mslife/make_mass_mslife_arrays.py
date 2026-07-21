# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:20:36 2022

@author: Tyler
"""

import numpy
from astropy.table import Table
import glob, re, os
from read_mist_models import EEP

filenames = glob.glob("/home/arseneau/observational/software/ember/tyler_code/init_mass_to_mslife/mist_feh_p025/MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_EEPS/*.track.eep")
#masses_grid = numpy.array([int(filenames[i][97:102]) for i in range(len(filenames))])/100.0

pattern = re.compile(r"(\d+)M\.track\.eep$")
masses_grid = numpy.array([
    int(pattern.search(os.path.basename(f)).group(1)) / 100.0
    for f in filenames
])

lifetimes = []
for filename in filenames:
    eep = EEP(filename, verbose = False)
    lifetime = eep.eeps['star_age'][numpy.where(eep.eeps['phase'] >= 0)][-1] - eep.eeps['star_age'][numpy.where(eep.eeps['phase'] >= 0)][0]
    lifetimes.append(lifetime)

lifetimes = numpy.array(lifetimes)/1e9
print(lifetimes)
numpy.save('mist_feh_p025/mi.npy', masses_grid)
numpy.save('mist_feh_p025/msl.npy', lifetimes)
