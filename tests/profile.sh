#!/bin/bash
python -m cProfile -s tottime test_profile_iteration.py | less
