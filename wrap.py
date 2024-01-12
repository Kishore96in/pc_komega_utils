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
	
	def __new__(cls, dr, *args, **kwargs):
		newcls = type(f"{type(dr).__name__}_{cls.suffix}", (cls, dr.__class__,) , {})
		obj = object.__new__(newcls)
		return obj
	
	def __post_init__(self):
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
	
	def __init__(self, dr, n_intervals):
		self._dr = dr
		self.n_intervals = n_intervals
		
		self.__post_init__()
	
	def do_ft(self):
		dr = self._dr
		n_intervals = self.n_intervals
		
		t_min_orig = dr.t_min
		t_max_orig = dr.t_max
		if (self.t_min != dr.t_min) or (self.t_max != dr.t_max):
			dr.set_t_range(self.t_min, self.t_max)
		
		#Choose t_max that ensures all the subintervals have the same number of time points.
		nt = len(dr.omega_tilde)
		dt = (dr.t_max - dr.t_min)/nt
		nt = n_intervals*np.floor(nt/n_intervals)
		t_max = dr.t_min + nt*dt
		
		t_ranges = np.linspace(dr.t_min, t_max, n_intervals + 1)
		
		data_sum = 0
		data2_sum = 0
		for t_min, t_max in zip(t_ranges[:-1], t_ranges[1:]):
			dr.set_t_range(t_min, t_max)
			data_sum += dr.data
			data2_sum += dr.data**2
		
		data_mean = data_sum/n_intervals
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
	
	def __init__(self, dr, n):
		self._dr = dr
		self.n = n
		
		self.__post_init__()
	
	def do_ft(self):
		dr = self._dr
		
		t_min_orig = dr.t_min
		t_max_orig = dr.t_max
		if (self.t_min != dr.t_min) or (self.t_max != dr.t_max):
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
