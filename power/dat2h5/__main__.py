import argparse
import time
from configparser import ConfigParser
from ast import literal_eval
import os

from ._dat2hdf import powerxy_to_hdf5

parser = argparse.ArgumentParser(
	prog = "python -m pc_komega_utils.power.dat2h5",
	formatter_class = argparse.ArgumentDefaultsHelpFormatter,
	)
parser.add_argument(
	'SIMDIR',
	help = "Directory containing the simulation to be processed.",
	type = str,
	)
parser.add_argument(
	'--verbose',
	default = False,
	action = 'store_true',
	)
parser.add_argument(
	'--conf-file',
	help = "Name of the config file (located in the simulation directory)"
	default = "decimate_power.conf",
	type = str,
	)
parser.add_argument(
	'--compress',
	default = False,
	help = "Whether to compress the resulting HDF5 datasets",
	action = 'store_true',
	)

args = parser.parse_args()

if args.verbose:
	t_start = time.time()
	print(f"Processing {args.SIMDIR}")

sim = pc.sim.get(args.SIMDIR)
simdir = sim.path

conf_file = simdir/args.conf_file
if not os.path.isfile(conf_file):
	raise RuntimeError(f"Configuration file not found in {simdir}")

conf = ConfigParser()
conf.optionxform = str #Preserve case of keys
conf.read(conf_file)

z_vals = literal_eval(conf['powerxy_to_hdf5']['z'])
z_vals = np.sort(z_vals)

for file_name in sim.datadir.glob("power*_xy.dat"):
	power_name = file_name.removeprefix("power").removesuffix(".dat")
	
	if args.compress:
		compression = 'gzip'
	else:
		compression = None
	
	powerxy_to_hdf5(
		power_name,
		file_name,
		sim.datadir,
		compression = compression,
		)

if args.verbose:
	print(f"Finished in {time.time() - t_start:.2f}s")
