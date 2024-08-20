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
import types

from ..cached import PowerCached, h5py_dataset_wrapper

def _decimate_power_obj(p, z_vals, izax):
	"""
	Filter a Pencil Power object or a PowerCached instance such that it only contains the power at the z values specified in z_vals. z_vals is used as-is (no sorting is performed).
	
	Arguments:
		p: Pencil Power object or PowerCached instance
		z_vals: list of float
		izax: int, index of the axis corresponding to z
	"""
	#`p_d = copy.copy(p)` does not seem correct with p is a PowerCached instance.
	p_d = types.SimpleNamespace()
	
	for key, arr in p.__dict__.items():
		arr = np.array(arr) #convert hdf5 datasets to numpy arrays.
		
		if key in ['kx', 'ky', 'k', 't']:
			setattr(p_d, key, arr)
			continue
		elif key == "nzpos":
			setattr(p_d, "nzpos", len(z_vals))
			continue
		elif key.startswith("_"):
			#PowerCached._cache needs to be skipped.
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
	
	return p_d

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
	
	if sim.param['lintegrate_z']:
		raise ValueError("Simulation was run with lintegrate_z=T")
	
	p_d = _decimate_power_obj(p, z_vals, izax)
	
	tmpfile = os.path.join(sim.path, "data", "power_decimated.h5.tmp")
	finalfile = os.path.join(sim.path, "data", "power_decimated.h5")
	
	with h5py.File(tmpfile, 'w') as f:
		for key, val in p_d.__dict__.items():
			f.create_dataset(key, data=val)
	
	os.rename(tmpfile, finalfile)

class m_pxy_decimated():
	"""
	Mixin to be used with pcko.read.dr_pxy_base
	
	DANGER: if you override do_ft provided here, you will need some special handling for the kx and ky attributes.
	"""
	def read_power(self):
		filename = os.path.join(self.datadir, "power_decimated.h5")
		if not os.path.isfile(filename):
			raise FileNotFoundError(filename)
		return PowerCached(filename)
	
	def do_ft(self):
		super().do_ft()
		
		#Convert the following attributes from h5py_dataset_wrapper to numpy arrays (to allow pickling).
		self.kx = self.kx[()]
		self.ky = self.ky[()]
