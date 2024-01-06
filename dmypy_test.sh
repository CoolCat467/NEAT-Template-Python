#!/bin/bash
# -*- coding: utf-8 -*-
# Programmed by CoolCat467

set -xe
pip install -e ../NEAT-Template-Python
# Full path listing might be important, it changes the way dmypy 1.8.0 seems to handle the files.
dmypy --status-file="dmypy.json" run --timeout=1800 --log-file="log.txt" --export-types "src/neat/neat.py" -- --no-error-summary --warn-unused-ignores --show-error-codes --warn-unreachable --disallow-untyped-defs --soft-error-limit=-1 --warn-redundant-casts --disallow-untyped-calls --no-color-output --no-implicit-reexport --strict --show-absolute-path --cache-fine-grained --hide-error-context --cache-dir="/home/samuel/.idlerc/mypy" --show-traceback --show-column-numbers --show-error-end
dmypy --status-file="dmypy.json" run --timeout=1800 --log-file="log.txt" --export-types "src/neat/neat.py" -- --no-error-summary --warn-unused-ignores --show-error-codes --warn-unreachable --disallow-untyped-defs --soft-error-limit=-1 --warn-redundant-casts --disallow-untyped-calls --no-color-output --no-implicit-reexport --strict --show-absolute-path --cache-fine-grained --hide-error-context --cache-dir="/home/samuel/.idlerc/mypy" --show-traceback --show-column-numbers --show-error-end
dmypy --status-file="dmypy.json" run --timeout=1800 --log-file="log.txt" --export-types "examples/minimax/checkers_ai.py" -- --no-error-summary --warn-unused-ignores --show-error-codes --warn-unreachable --disallow-untyped-defs --soft-error-limit=-1 --warn-redundant-casts --disallow-untyped-calls --no-color-output --no-implicit-reexport --strict --show-absolute-path --cache-fine-grained --hide-error-context --cache-dir="/home/samuel/.idlerc/mypy" --show-traceback --show-column-numbers --show-error-end
