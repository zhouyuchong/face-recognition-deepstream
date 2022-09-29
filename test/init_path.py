import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

_this_dir = os.path.dirname(__file__)
# add src path
add_path(os.path.join(_this_dir, '../src'))