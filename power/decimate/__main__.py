import argparse
import time

from ._decimate import make_decimated_power

parser = argparse.ArgumentParser(
	prog = "python -m pc_komega_utils.power.decimate",
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

args = parser.parse_args()

if args.verbose:
	t_start = time.time()
	print(f"Decimating {args.SIMDIR}")

make_decimated_power(args.SIMDIR)

if args.verbose:
	print(f"Finished in {time.time() - t_start:.2f}s")
