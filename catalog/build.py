"""Dispatcher: python -m catalog.build [--wdms | --wdwd]

Delegates to the appropriate pipeline package:
  --wdms  (default)  catalog.wdms  — WD+MS combined/metallicity catalogs
  --wdwd             catalog.wdwd  — WD+WD stitch → El-Badry → pairs
"""

import sys


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wdwd", action="store_true")
    parser.add_argument("--wdms", action="store_true")
    parser.add_argument("--correct-ages",       action="store_true")
    parser.add_argument("--check-correct-ages", action="store_true")
    args, _ = parser.parse_known_args()

    kwargs = dict(correct_ages=args.correct_ages,
                  check_correct_ages=args.check_correct_ages)

    if args.wdwd:
        from .wdwd.build import main as _main
        _main(**{k: v for k, v in kwargs.items() if k == "correct_ages"})
    else:
        from .wdms.build import main as _main
        _main(**kwargs)


if __name__ == "__main__":
    main()
