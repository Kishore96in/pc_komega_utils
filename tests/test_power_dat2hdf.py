import numpy as np
import pencil as pc
import h5py

import pc_komega_utils
from pc_komega_utils.power.dat2hdf import (
	powerxy_to_hdf5,
	read_power,
	)

from .fixtures import datadir, cachedir, datadir_tmp

def test_powerxy_to_hdf5(datadir, cachedir):
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

def test_read_power(datadir_tmp):
	datadir=datadir_tmp
	
	grid = pc.read.grid(datadir=datadir)
	
	p_src = pc.read.power(
		datadir = datadir,
		quiet = True,
		)
	
	powerxy_to_hdf5(
		power_name = "uz_xy",
		file_name = "poweruz_xy.dat",
		datadir = datadir,
		)
	
	p = read_power(datadir, z=[1])
	
	assert p.uz_xy.shape[1] == 1
	
	iz = np.argmin(np.abs(p_src.zpos - 1))
	assert np.all(p_src.uz_xy[:,iz] == p.uz_xy[:,0])

def test_read_power2(datadir_tmp):
	datadir=datadir_tmp
	
	grid = pc.read.grid(datadir=datadir)
	
	p_src = pc.read.power(
		datadir = datadir,
		quiet = True,
		)
	
	powerxy_to_hdf5(
		power_name = "uz_xy",
		file_name = "poweruz_xy.dat",
		datadir = datadir,
		)
	
	p = read_power(datadir, z=[0,1])
	
	assert p.uz_xy.shape[1] == 2
	
	iz0 = np.argmin(np.abs(p_src.zpos - 0))
	iz1 = np.argmin(np.abs(p_src.zpos - 1))
	assert np.all(p_src.uz_xy[:,[iz0,iz1]] == p.uz_xy)

def test_read_power3(datadir_tmp):
	datadir=datadir_tmp
	
	grid = pc.read.grid(datadir=datadir)
	
	p_src = pc.read.power(
		datadir = datadir,
		quiet = True,
		)
	
	powerxy_to_hdf5(
		power_name = "uz_xy",
		file_name = "poweruz_xy.dat",
		datadir = datadir,
		)
	
	p = read_power(datadir)
	
	assert np.all(p_src.uz_xy == p.uz_xy)
