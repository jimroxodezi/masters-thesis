# Lets "python -m unittest tests.test_x" find tests/helpers.py; "discover -s tests" needs nothing.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
