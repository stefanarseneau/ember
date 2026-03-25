#!/bin/bash

module load miniconda
conda activate structure

ember summarize /projectnb/mesaelm/ember/omnidwarf/thick /projectnb/mesaelm/ember/omnidwarf/thick_mcmc.pqt
