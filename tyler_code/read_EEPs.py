# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:47:03 2019

@author: tmhei
"""
from read_mist_models import EEP
import numpy
import glob
# ===================================================================================
def main():
    eep = EEP("00400M.track.eep")
#    print(eep.eeps['star_age'][numpy.where(eep.eeps['phase'] > 0)])
    eep.plot_HR(fignum = 1)
    filenames = glob.glob("*.track.eep")
    masses = numpy.array([int(filenames[i][0:5]) for i in range(len(filenames))])/100.0
    print(masses[numpy.where((masses <= 9) & (masses >= 0.9))])
    
    
# ===================================================================================
if __name__ == '__main__':
    main()