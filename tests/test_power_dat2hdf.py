import os
import numpy as np
import pencil as pc
import datetime
import pathlib
import h5py

import pc_komega_utils
from pc_komega_utils.power.dat2hdf import powerxy_to_hdf5

def get_datadir():
	module_loc = os.path.dirname(pc_komega_utils.__file__)
	return os.path.join(module_loc, "tests", "data")

def get_cachedir():
	cachedir = pathlib.Path(f"/tmp/test_power_cached-{os.getpid()}-{datetime.datetime.now().isoformat()}")
	
	if not os.path.exists(cachedir):
		os.makedirs(cachedir)
	elif not os.path.isdir(cachedir):
		raise ValueError(f"Cache directory '{cachedir}' is not a directory")
	
	return cachedir

def test_powerxy_to_hdf5():
	datadir = get_datadir()
	cachedir = get_cachedir()
	
	grid = pc.read.grid(datadir=datadir)
	
	p_src = pc.read.power(
		datadir = datadir,
		quiet = True,
		)
	
	out_file = cachedir/"poweruz_xy.hdf5"
	powerxy_to_hdf5(
		power_name = "uz_xy",
		file_name = "poweruz_xy.dat",
		datadir = datadir,
		out_file = out_file,
		)
	
	with h5py.File(out_file) as f:
		assert f['nzpos'][()] == p_src.nzpos
		assert len(f['kx']) == 101
		assert len(f['ky']) == 1
		assert len(f['times/it']) == 2
		assert len(f['times/t']) == 2
		
		assert "uz_xy" in f.keys()
		assert len(f['uz_xy'].keys()) == len(f['times/it'])
		
		it = f['times/it'][0]
		assert np.all(p_src.uz_xy[0] == f['uz_xy'][str(it)])
		
		it = f['times/it'][1]
		assert np.all(p_src.uz_xy[1] == f['uz_xy'][str(it)])
