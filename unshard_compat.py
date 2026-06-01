"""Unshard wrapper with Python 3.13 pathlib compatibility."""
import sys, pathlib, types
if 'pathlib._local' not in sys.modules:
    m = types.ModuleType('pathlib._local')
    m.Path = pathlib.Path; m.PurePath = pathlib.PurePath
    m.PosixPath = pathlib.PosixPath; m.WindowsPath = pathlib.WindowsPath
    sys.modules['pathlib._local'] = m
from lm_engine.unshard import main
main()
