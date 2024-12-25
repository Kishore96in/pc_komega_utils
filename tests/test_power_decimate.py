import numpy as np
import pencil as pc

from pc_komega_utils.power.decimate import decimate_power_obj

from .fixtures import datadir

def test_make_decimated_power(datadir):
	grid = pc.read.grid(datadir=datadir)
	
	p_src = pc.read.power(
		datadir = datadir,
		quiet = True,
		)
	
	z_vals = [1, 0.4, 0.8]
	
	p = decimate_power_obj(p_src, z_vals, izax=1)
	
	assert len(p.zpos) == len(z_vals)
	assert len(p.kx) == 101
	assert len(p.ky) == 1
	assert len(p.t) == 2
	assert p.uz_xy.shape == (2, len(z_vals), 1, 101)
	
	assert np.all(np.isclose(p.zpos, z_vals, atol=grid.dz))
