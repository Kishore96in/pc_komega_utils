import sys

from ._decimate import make_decimated_power

simdir = sys.argv[1]
make_decimated_power(simdir)
