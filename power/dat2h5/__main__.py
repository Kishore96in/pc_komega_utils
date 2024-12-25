import argparse
import time
import os
import pencil as pc

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
	'--compress',
	default = False,
	help = "Whether to compress the resulting HDF5 datasets",
	action = 'store_true',
	)

args = parser.parse_args()

if args.verbose:
	t_start = time.time()

sim = pc.sim.get(args.SIMDIR, quiet=True)
simdir = sim.path

if args.verbose:
	print(f"Processing {sim.path}")

for f in sim.datadir.glob("power*_xy.dat"):
	file_name = f.name
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
