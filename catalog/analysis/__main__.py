"""Central dispatcher for analysis scripts.

Usage:
    python -m catalog.analysis <command> [args]
    python -m catalog.analysis all

Commands are imported lazily — only the selected script's dependencies are loaded.
"""

import importlib
import sys

COMMANDS = {
    "age-abundance":    "catalog.analysis.age_abundance",
    "age-metallicity":  "catalog.analysis.age_metallicity",
    "da-db-hist":       "catalog.analysis.da_db_hist",
    "exoplanet-ages":   "catalog.analysis.exoplanet_ages",
    "exoplanet-table":  "catalog.analysis.exoplanet_table",
    "ifmr-comparison":  "catalog.analysis.ifmr_comparison",
    "mass-teff":        "catalog.analysis.mass_teff",
    "ms-lifetimes":     "catalog.analysis.ms_lifetimes",
    "sample-summary":   "catalog.analysis.sample_summary",
    "uncertainty":      "catalog.analysis.uncertainty_improvement",
    "wdwd-inconsistent": "catalog.analysis.wdwd_inconsistent",
}


def _usage():
    print("Usage: python -m catalog.analysis <command> [args]\n")
    print("Commands:")
    for name in COMMANDS:
        print(f"  {name}")
    print("  all")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _usage()
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd not in COMMANDS and cmd != "all":
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        _usage()
        sys.exit(1)

    # Strip the command name so per-script argparse sees only its own args.
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if cmd == "all":
        for name, module_path in COMMANDS.items():
            print(f"\n{'─' * 40}\n{name}\n{'─' * 40}")
            importlib.import_module(module_path).main()
    else:
        importlib.import_module(COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()
