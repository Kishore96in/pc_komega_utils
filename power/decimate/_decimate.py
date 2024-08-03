"""
Given the output of the power_xy subroutine, save a HDF5 file which contains power spectra at only a subset of the z values.
"""

import pencil as pc
import os
import sys
from configparser import ConfigParser
from ast import literal_eval
import h5py
import warnings
import numpy as np
import copy

from ..cached import PowerCached

def make_decimated_power(simdir):
	"""
	Arguments:
		simdir: str. Path to the simulation directory.
	
	The list of z values to save is read from a config file (decimate_power.conf) located in simdir. An example of the content of the config file:
	```
	[decimate_power]
	z: [-0.1, 0.5, 0.9, 1]
	```
	"""
	sim = pc.sim.get(simdir, quiet=True)
	
	if isinstance(sim, bool):
		raise RuntimeError(f"Simulation not found in {simdir}")
	
	p = pc.read.power(
		datadir = sim.datadir,
		quiet = True,
		)
	izax = 1 #Index of the axis corresponding to z
	
	conf_file = os.path.join(simdir, "decimate_power.conf")
	if not os.path.isfile(conf_file):
		raise RuntimeError(f"Configuration file not found in {simdir}")
	
	conf = ConfigParser()
	conf.optionxform = str #Preserve case of keys
	conf.read(conf_file)
	
	z_vals = literal_eval(conf['decimate_power']['z'])
	z_vals = np.sort(z_vals)
	
	p_d = copy.copy(p)
	
	if sim.param['lintegrate_z']:
		raise ValueError("Simulation was run with lintegrate_z=T")
	
	for key, arr in p.__dict__.items():
		if key in ['kx', 'ky', 'k', 't']:
			continue
		elif key == "nzpos":
			setattr(p_d, "nzpos", len(z_vals))
			continue
		elif key == "zpos":
			pass
		else:
			#Assume that any array not handled above has a z axis at izax.
			arr = np.moveaxis(arr, izax, 0)
		
		if np.shape(arr)[0] != len(p.zpos):
			raise RuntimeError(f"Attribute {key} has the wrong shape; for axis {izax}, expected size {len(p.zpos)}, but got {np.shape(arr)[0]}")
		
		arr_d = np.full(
			(len(z_vals), *arr.shape[1:]),
			np.nan,
			dtype = arr.dtype,
			)
		
		for izv, z_val in enumerate(z_vals):
			arr_d[izv] = arr[np.argmin(np.abs(z_val - p.zpos))]
		
		if np.any(np.isnan(arr_d)):
			raise RuntimeError
		
		if key != "zpos":
			arr_d = np.moveaxis(arr_d, 0, izax)
		setattr(p_d, key, arr_d)
	
	if np.any(np.diff(p_d.zpos) == 0):
		warnings.warn("Requested z values are closer than the grid spacing. Decimated power file will contain duplicated values.")
	
	tmpfile = os.path.join(sim.path, "data", "power_decimated.h5.tmp")
	finalfile = os.path.join(sim.path, "data", "power_decimated.h5")
	
	with h5py.File(tmpfile, 'w') as f:
		for key, val in p_d.__dict__.items():
			f.create_dataset(key, data=val)
	
	os.rename(tmpfile, finalfile)

class m_pxy_decimated():
	"""
	Mixin to be used with pcko.read.dr_pxy_base
	"""
	def read_power(self):
		filename = os.path.join(self.datadir, "power_decimated.h5")
		if not os.path.isfile(filename):
			raise FileNotFoundError(filename)
		return PowerCached(filename)
