"""
Wrappers for dr_base instances, that perform various postprocessing tricks (e.g. jackknifing or smoothing) without rereading the underlying simulation data.
"""

import numpy as np

from .utils import smooth_tophat

class wrap_base():
	"""
	Given a dr_base instance, makes a wrapper of it that allows to manipulate the data without rereading it.
	"""
	@property
	def suffix(self):
		raise NotImplementedError
	
	@property
	def _args_map(self):
		"""
		Allows subclasses to define which attribute names should be used for the positional arguments called during instantiation.
		"""
		raise NotImplementedError
	
	def __new__(cls, dr, *args, **kwargs):
		newcls = type(f"{type(dr).__name__}_{cls.suffix}", (cls, dr.__class__,) , {})
		obj = object.__new__(newcls)
		return obj
	
	def __init__(self, *args):
		if len(args) != len(self._args_map):
			raise ValueError(f"Wrong number of arguments (expected { len(self._args_map)}; got {len(args)})")
		if self._args_map['_dr'] != 0:
			#This is assumed in __new__
			raise ValueError("dr should always be first argument")
		
		#We use a list because we want this to be mutable (see __setattr__)
		self._args = list(args)
		
		dr = self._dr
		for attr in [
			"simdir",
			"datadir",
			"param",
			"dim",
			"grid",
			"k_tilde_min",
			"k_tilde_max",
			"omega_tilde_min",
			"omega_tilde_max",
			"fig_savedir",
			"field_name",
			"cbar_label",
			"ts",
			"av_xy",
			]:
			if hasattr(dr, attr):
				setattr(self, attr, getattr(dr, attr))
		
		self.set_t_range(dr.t_min, dr.t_max)
	
	def __getnewargs__(self):
		"""
		Make subclass instances pickle-able
		"""
		return tuple(self._args)
	
	def __getattr__(self, name):
		if name in self._args_map.keys():
			return self._args[self._args_map[name]]
		else:
			raise AttributeError
	
	def __setattr__(self, name, value):
		if name in self._args_map.keys():
			self._args[self._args_map[name]] = value
		else:
			# https://stackoverflow.com/questions/7042152/how-do-i-properly-override-setattr-and-getattribute-on-new-style-classes/7042247#7042247
			super().__setattr__(name, value)

class dr_stat(wrap_base):
	"""
	Given a dr_base instance, divides the time series into subintervals and uses those to estimate the error in the data.
	
	Initialization arguments:
		dr: dr_base instance
		n_intervals: int. Number of subintervals to divide the time range into.
	
	Notable attributes:
		self.data: mean of the data in the subintervals of the source
		self.sigma: estimate of the error in self.data
	"""
	suffix = "stat"
	_args_map = {"_dr": 0, "n_intervals": 1}
	
	def do_ft(self):
		dr = self._dr
		n_intervals = self.n_intervals
		
		t_min_orig = dr.t_min
		t_max_orig = dr.t_max
		dr.set_t_range(self.t_min, self.t_max)
		
		#Choose t_max that ensures all the subintervals have the same number of time points.
		nt = len(dr.omega_tilde)
		dt = (dr.t_max - dr.t_min)/nt
		nt = n_intervals*np.floor(nt/n_intervals)
		t_max = dr.t_min + nt*dt
		
		t_ranges = np.linspace(dr.t_min, t_max, n_intervals + 1)
		
		data_mean = 0
		data2_sum = 0
		for t_min, t_max in zip(t_ranges[:-1], t_ranges[1:]):
			dr.set_t_range(t_min, t_max)
			data_mean += dr.data
			data2_sum += dr.data**2
		
		data_mean = data_mean/n_intervals
		#sqrt(n/(n-1)) is to reduce the bias in the estimate for the standard deviation.
		sigma = np.sqrt(data2_sum/n_intervals - (data_mean)**2)*np.sqrt(n_intervals/(n_intervals-1))
		
		self.omega = dr.omega
		self.data = data_mean
		self.sigma = sigma/np.sqrt(n_intervals)
		
		if hasattr(dr, "kx"):
			self.kx = dr.kx
		if hasattr(dr, "ky"):
			self.ky = dr.ky
		
		dr.set_t_range(t_min_orig, t_max_orig)

class dr_sm(wrap_base):
	"""
	Given a dr_base instance, smooth the data along the frequency axis.
	
	Initialization arguments:
		dr: dr_base instance
		n: int. half width of the smoothing filter, as a multiple of the spacing between different values of omega.
	"""
	suffix = "sm"
	_args_map = {"_dr": 0, "n": 1}
	
	def do_ft(self):
		dr = self._dr
		
		t_min_orig = dr.t_min
		t_max_orig = dr.t_max
		dr.set_t_range(self.t_min, self.t_max)
		
		self.omega = dr.omega
		self.data = smooth_tophat(dr.data, self.n, axis=dr.data_axes['omega_tilde'])
		
		if hasattr(dr, "kx"):
			self.kx = dr.kx
		if hasattr(dr, "ky"):
			self.ky = dr.ky
		
		dr.set_t_range(t_min_orig, t_max_orig)
	
	@property
	def smoothing_width(self):
		"""
		Width of the smoothing filter in the same units as omega_tilde.
		"""
		d_omega = self.omega_tilde[1] - self.omega_tilde[0]
		return (2*self.n+1) * d_omega

class dr_stat_smsig(dr_stat):
	"""
	Like dr_stat, but with the error estimates smoothed wrt. omega.
	"""
	suffix = "stat_smsig"
	
	def do_ft(self):
		dr = self._dr
		n_intervals = self.n_intervals
		
		t_min_orig = dr.t_min
		t_max_orig = dr.t_max
		dr.set_t_range(self.t_min, self.t_max)
		
		#Choose t_max that ensures all the subintervals have the same number of time points.
		nt = len(dr.omega_tilde)
		dt = (dr.t_max - dr.t_min)/nt
		nt = n_intervals*np.floor(nt/n_intervals)
		t_max = dr.t_min + nt*dt
		
		t_ranges = np.linspace(dr.t_min, t_max, n_intervals + 1)
		
		data_mean = 0
		data2_sum = 0
		for t_min, t_max in zip(t_ranges[:-1], t_ranges[1:]):
			dr.set_t_range(t_min, t_max)
			data_mean += dr.data
			data2_sum += dr.data**2
		
		data_mean = data_mean/n_intervals
		#sqrt(n/(n-1)) is to reduce the bias in the estimate for the standard deviation.
		sigma = np.sqrt(data2_sum/n_intervals - (data_mean)**2)*np.sqrt(n_intervals/(n_intervals-1))
		sigma = smooth_tophat(sigma, 1, axis=self.data_axes['omega_tilde'])
		
		self.omega = dr.omega
		self.data = data_mean
		self.sigma = sigma/np.sqrt(n_intervals)
		
		if hasattr(dr, "kx"):
			self.kx = dr.kx
		if hasattr(dr, "ky"):
			self.ky = dr.ky
		
		dr.set_t_range(t_min_orig, t_max_orig)
