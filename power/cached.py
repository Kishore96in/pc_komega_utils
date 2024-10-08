"""
Replacement for pc.read.powers.Power that stores the data in a HDF5 file, rather than in memory.

Objects intended for public use:
	InvalidCacheWarning
	read_power
	m_pxy_cached
"""

import pencil as pc
import h5py
import os
import numpy as np
import warnings

class InvalidCacheWarning(RuntimeWarning): pass

class PowerCached():
	"""
	Class which acts as a wrapper around a HDF5 file.
	
	Arguments:
		filename: string. Path to HFD5 file.
	"""
	def __init__(
		self,
		filename,
		):
		self._cache = h5py.File(filename, 'r')
		self._populate_keys_from_cache()
	
	def _populate_keys_from_cache(self):
		for k in self._cache.keys():
			data = self._cache[k]
			if data.ndim == 0:
				setattr(self, k, data[()])
			else:
				setattr(self, k, h5py_dataset_wrapper(data))
	
	def __del__(self):
		if hasattr(self, "_cache"):
			self._cache.close()
	
	def keys(self):
		return set(self._cache.keys())
	
	def __getstate__(self):
		return self._cache.filename
	
	def __setstate__(self, cachename):
		self._cache = h5py.File(cachename, 'r')
		self._populate_keys_from_cache()

class read_power(PowerCached):
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
		
		#Find the datadir
		if len(args) > 0:
			datadir = args[0]
		else:
			datadir = kwargs.get('datadir', "data")
		
		fname = os.path.join(cachedir, "power_cache.h5")
		
		if ignore_cache or not (
			os.path.exists(fname) and
			os.path.getmtime(datadir) < os.path.getmtime(fname)
			):
			warnings.warn(f"Cached power data ({fname}) is nonexistent or invalid; reading anew.", InvalidCacheWarning)
			
			p = pc.read.power(*args, **kwargs)
			
			self._cache = h5py.File(fname, 'w')
			for k in p.__dict__.keys():
				self._h5cache(k, getattr(p,k))
			self._cache.close()
		
		super().__init__(fname)
	
	def _h5cache(self, name, value):
		"""
		Add a key-value pair (name:value) to the HDF5 cache
		
		name : str
		value : numpy array
		"""
		dset = self._cache.create_dataset(name, data=value)

class h5py_dataset_wrapper():
	"""
	Allows mathematical operations to be performed on h5py Dataset instances as if they are numpy arrays.
	"""
	def __init__(self, dset):
		self._dset = dset
	
	def __getattr__(self, name):
		return getattr(self._dset, name)
	
	def __getitem__(self, key):
		return self._dset[key]
	
	def __len__(self):
		return len(self._dset)
	
	def __getstate__(self):
		"""
		Just a more useful error message.
		"""
		raise TypeError(f"Cannot pickle HDF5 dataset (name: {self._dset.name}).")
	
	# List of magic methods: https://python-course.eu/oop/magic-methods.php
	# Binary operators
	def __add__(self, other):
		return np.add(self, other)
	def __radd__(self, other):
		return np.add(other, self)
	def __sub__(self, other):
		return np.subtract(self, other)
	def __rsub__(self, other):
		return np.subtract(other, self)
	def __mul__(self, other):
		return np.multiply(self, other)
	def __rmul__(self, other):
		return np.multiply(other, self)
	def __floordiv__(self, other):
		return np.floor_divide(self, other)
	def __truediv__(self, other):
		return np.true_divide(self, other)
	def __mod__(self, other):
		return np.mod(self, other)
	def __pow__(self, other):
		return np.power(self, other)
	
	# Unary operators
	def __neg__(self):
		return -1*self
	def __abs__(self):
		return np.abs(self)
	
	# Comparison operators
	def __lt__(self, other):
		return np.less(self, other)
	def __le__(self, other):
		return np.less_equal(self, other)
	def __eq__(self, other):
		return np.equal(self, other)
	def __ne__(self, other):
		return np.not_equal(self, other)
	def __ge__(self, other):
		return np.greater_equal(self, other)
	def __gt__(self, other):
		return np.greater(self, other)

class m_pxy_cached():
	"""
	Mixin to be used with dr_pxy_base.
	
	This uses the cached power reader provided by pcko.power.cached.
	
	DANGER: if you override do_ft provided here, you will need some special handling for the kx and ky attributes.
	"""
	def __init__(self, *args, **kwargs):
		self._ignore_cache = kwargs.pop('ignore_cache', False)
		
		super().__init__(*args, **kwargs)
	
	def read_power(self):
		return read_power(
			datadir = self.datadir,
			quiet = True,
			cachedir = os.path.join(self.simdir, "postprocess_power_cache"),
			ignore_cache = self._ignore_cache,
			)
	
	def do_ft(self):
		super().do_ft()
		
		#Convert the following attributes from h5py_dataset_wrapper to numpy arrays (to allow pickling).
		self.kx = self.kx[()]
		self.ky = self.ky[()]
