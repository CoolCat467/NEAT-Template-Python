#!/bin/bash
# -*- coding: utf-8 -*-
# Programmed by CoolCat467

# Show all commands that we are running
set -xe

# Install meshy's mypy branch
pip install git+https://github.com/meshy/mypy@unexpected-unused-type-ignore

# Install project in edit mode
pip install -e ../NEAT-Template-Python

# Get full path to this file
this_file=$(realpath dmypy_test.sh)

# Get absolute path prefix from path to this file
abs_prefix=$(python3 -c "import os;print(os.path.split('$this_file')[0])")

# abs_prefix = "$HOME/path/to/where/this/is/downloaded/at"

# Dmypy runs
dmypy --status-file="dmypy.json" run --timeout=1800 --log-file="log.txt" --export-types "$abs_prefix/src/neat/neat.py" -- --no-error-summary --warn-unused-ignores --show-error-codes --warn-unreachable --disallow-untyped-defs --soft-error-limit=-1 --warn-redundant-casts --disallow-untyped-calls --no-color-output --no-implicit-reexport --strict --show-absolute-path --cache-fine-grained --hide-error-context --cache-dir="/home/samuel/.idlerc/mypy" --show-traceback --show-column-numbers --show-error-end
dmypy --status-file="dmypy.json" run --timeout=1800 --log-file="log.txt" --export-types "$abs_prefix/src/neat/neat.py" -- --no-error-summary --warn-unused-ignores --show-error-codes --warn-unreachable --disallow-untyped-defs --soft-error-limit=-1 --warn-redundant-casts --disallow-untyped-calls --no-color-output --no-implicit-reexport --strict --show-absolute-path --cache-fine-grained --hide-error-context --cache-dir="/home/samuel/.idlerc/mypy" --show-traceback --show-column-numbers --show-error-end
dmypy --status-file="dmypy.json" run --timeout=1800 --log-file="log.txt" --export-types "$abs_prefix/examples/minimax/checkers_ai.py" -- --no-error-summary --warn-unused-ignores --show-error-codes --warn-unreachable --disallow-untyped-defs --soft-error-limit=-1 --warn-redundant-casts --disallow-untyped-calls --no-color-output --no-implicit-reexport --strict --show-absolute-path --cache-fine-grained --hide-error-context --cache-dir="/home/samuel/.idlerc/mypy" --show-traceback --show-column-numbers --show-error-end
