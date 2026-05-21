#!/bin/bash

echo "              .    *   .    *   .     ╔══════════════════════════╗"
echo "        *   .   .-\"\"\"\"-. .   *        ║      WD+MS Figures       ║"
echo "           .  .'  . * .  '.  .        ╚══════════════════════════╝"
echo "        *   /  * HHH * HH  \   *"
echo "       .   | . HHHHHHHHHHH . |   .                                     "
echo "     *     | * HHHHHHHHHHH * |     *    ~ Featuring Data From ~        "
echo "       .   | . HHHHHHHHHHH . |   .                                     "
echo "        *   \  * HH * HHH *  /   *       Kareem! GALAH! APOGEE!  "
echo "           .  '.  . * .  .'  .            ASTRA! LAMOST! NEA!     "
echo "        *   .   '-....-'   .   *          "
echo "              .    *   .    *   .          "
echo "                *       *                 "

cd "$(dirname "$0")" || exit

printf '\nSample properties\n'
python sample_summary.py

printf '\nMass-Teff diagram\n'
python mass_teff.py

printf '\nParallax and mass uncertainty improvement\n'
python uncertainty_improvement.py

printf '\nLower Limit Plots\n'
python lower_limits.py

printf '\nDA/DB metallicity histogram (SKIPPING POWER ANALYSIS)\n'
python da_db_hist.py --skippower

printf '\nAge-metallicity relations\n'
python age_metallicity.py

printf '\nAge-abundance figures (Li, alpha, C, Ba)\n'
python age_abundance.py

printf '\nIFMR comparison\n'
python ifmr_comparison.py

printf '\nMS lifetimes and metallicity sensitivity\n'
python ms_lifetimes.py

printf '\nExoplanet host table\n'
python exoplanet_table.py

printf '\nCool WD velocity distribution\n'
python coolwd_avr.py

printf '\nCool WD velocity distribution with isotropic rvs\n'
python coolwd_avr.py --isotropic-rvs

printf '\nVelocity distribution and AVR age inference\n'
python lb_velocity_dist.py

printf '\nDone. PDFs written to analysis/figures/\n'
