#!/usr/bin/env python3
"""
Entry point for python -m heavytails.

This module allows the heavytails package to be executed as a module:
    python -m heavytails --help
    python -m heavytails sample pareto --params '{"alpha": 2.0, "xm": 1.0}' -n 1000
"""

from heavytails.cli import main

if __name__ == "__main__":
    main()
