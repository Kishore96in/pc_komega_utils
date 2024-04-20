"""
Replacement for pc.read.powers.Power that stores the data in a HDF5 file, rather than in memory.
"""

import pencil as pc
import h5py
import os

class read_power_cached():
	def __init__(self, *args, cachedir=".", **kwargs):
		if not os.path.exists(cachedir):
			os.makedirs(cachedir)
		elif not os.path.isdir(cachedir):
			raise ValueError(f"Cache directory '{cachedir}' is not a directory")
		
		fname = os.path.join(cachedir, "power_cache.h5")
		self.cache = h5py.File(fname, 'w')
		
		p = pc.read.power(*args, **kwargs)
		for k in p.keys():
			self._h5cache(k, getattr(p,k))
	
	def _h5cache(self, name, value):
		"""
		name : str
		value : numpy array
		"""
		dset = self.cache.create_dataset(name, value.shape, dtype=value.dtype)
		dset[()] = value
	
	def __getattr__(self, name):
		if name != 'cache' and name in self.cache.keys():
			return self.cache[name]
		else:
			raise AttributeError
	
	def __del__(self):
		self.cache.close()
