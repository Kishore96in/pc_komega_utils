"""
Replacement for pc.read.powers.Power that stores the data in a HDF5 file, rather than in memory.
"""

import pencil as pc
import h5py
import os

class read_power():
	"""
	Stores the output of pc.read.power in a HDF5 file. When an attribute is requested, it is looked up in the HDF5 file. This avoids the need to always keep the Power() object in memory (a problem for large simulations).
	"""
	def __init__(
		self,
		*args,
		cachedir=".",
		ignore_cache=False,
		**kwargs,
		):
		if not os.path.exists(cachedir):
			os.makedirs(cachedir)
		elif not os.path.isdir(cachedir):
			raise ValueError(f"Cache directory '{cachedir}' is not a directory")
		
		fname = os.path.join(cachedir, "power_cache.h5")
		if os.path.exists(fname) and not ignore_cache:
			self._cache = h5py.File(fname, 'r')
		else:
			self._cache = h5py.File(fname, 'w')
			
			p = pc.read.power(*args, **kwargs)
			for k in p.__dict__.keys():
				self._h5cache(k, getattr(p,k))
	
	def _h5cache(self, name, value):
		"""
		name : str
		value : numpy array
		"""
		dset = self._cache.create_dataset(name, data=value)
	
	def __getattr__(self, name):
		if name != '_cache' and name in self._cache.keys():
			data = self._cache[name]
			if data.ndim == 0:
				return data[()]
			else:
				return data
		else:
			raise AttributeError
	
	def __del__(self):
		self._cache.close()
	
	def keys():
		return set(self._cache.keys())
