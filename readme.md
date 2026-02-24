## EMBER — Estimating Mass and Bolometric properties of Evolved Remnants

`ember` is a tool for fitting the spectral energy distribution of white dwarfs in Gaia. It works on calibrated BP/RP spectra, meaning that it is entirely native to Gaia-- no external survey data needed (unless you want it!). It also uses monte carlo sampling of MIST white dwarf cooling tracks to measure cooling ages and theoretical hydrogen layer masses.

**Installation:**

To install, first run
```
pip install git+https://github.com/stefanarseneau/ember
```

This will make `ember` a callable command on your system. However, in order to use it, one environment variable is required which points the scripts to the Bauer+2025 (ApJS In Prep) cooling tracks. This is not packaged with the CLI to keep it somewhat lightweight. Navigate to [here](https://zenodo.org/records/15242047) and follow the instructions to unpack `default_grids_full.tgz`. Then run the following command:

```
export COOLING_PATH=/path/to/default_grids_full
```

Now `ember` should be fully functional.

