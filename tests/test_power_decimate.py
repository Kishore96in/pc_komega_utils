import os
import numpy as np
import pencil as pc

import pc_komega_utils
from pc_komega_utils.power.decimate._decimate import _decimate_power_obj

def get_datadir():
	module_loc = os.path.dirname(pc_komega_utils.__file__)
	return os.path.join(module_loc, "tests", "data")

def test_make_decimated_power():
	datadir = get_datadir()
	
	grid = pc.read.grid(datadir=datadir)
	
	p_src = pc.read.power(
		datadir = datadir,
		quiet = True,
		)
	
	z_vals = [1, 0.4, 0.8]
	
	p = _decimate_power_obj(p_src, z_vals, izax=1)
	
	assert len(p.zpos) == len(z_vals)
	assert len(p.kx) == 101
	assert len(p.ky) == 1
	assert len(p.t) == 2
	assert p.uz_xy.shape == (2, len(z_vals), 1, 101)
	
	assert np.all(np.isclose(p.zpos, z_vals, atol=grid.dz))
