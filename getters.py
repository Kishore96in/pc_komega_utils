"""
Objects intended for public use:
	AbstractGetter
	get_sigma_stdev
	getter_kyeq0
	getter_kxeq0
"""

import numpy as np
import abc
import warnings

from .utils import stdev_central

class AbstractGetter(abc.ABC):
	def __new__(cls, *args, **kwargs):
		self = object.__new__(cls)
		
		if len(args) > 0:
			self.dr = args[0]
		else:
			self.dr = kwargs['dr']
		
		data, omt = self.get_data(*args, **kwargs)
		sigma = self.get_sigma(*args, **kwargs, data=data, omt=omt)
		
		if np.all(sigma == 0):
			"""
			This can only happen if np.all(data == 0), in which case we set sigma=1 to prevent divide-by-zero errors.
			"""
			sigma[:] = 1
		elif np.any(sigma == 0):
			warnings.warn("Estimated error was zero in some bins. Applying floor to estimated error.")
			min_sigma = np.min(np.compress(sigma != 0, sigma)) #smallest nonzero value
			sigma = np.where(sigma == 0, min_sigma, sigma)
		
		return {
			'data_near_target': data,
			'omt_near_target': omt,
			'sigma': sigma,
			}
	
	@abc.abstractmethod
	def get_data(self, dr, k_tilde, z, om_tilde_min, om_tilde_max):
		raise NotImplementedError
	
	@abc.abstractmethod
	def get_sigma(self, *args, **kwargs):
		raise NotImplementedError

class get_sigma_stdev():
	"""
	Frequency-independent estimate of sigma using the standard deviation of the data.
	"""
	def get_sigma(self, *args, data, omt, **kwargs):
		return stdev_central(data, 0.05, adjust=True)

class getter_kyeq0(get_sigma_stdev, AbstractGetter):
	def get_data(self, dr, k_tilde, z, om_tilde_min, om_tilde_max):
		data_near_target, [omt_near_target, *_] = dr.get_slice(
			omega_tilde=(om_tilde_min, om_tilde_max),
			kx_tilde = k_tilde,
			ky_tilde = 0,
			z = z,
			compress = True,
			)
		return data_near_target, omt_near_target

class getter_kxeq0(get_sigma_stdev, AbstractGetter):
	def get_data(self, dr, k_tilde, z, om_tilde_min, om_tilde_max):
		data_near_target, [omt_near_target, *_] = dr.get_slice(
			omega_tilde=(om_tilde_min, om_tilde_max),
			kx_tilde = 0,
			ky_tilde = k_tilde,
			z = z,
			compress = True,
			)
		return data_near_target, omt_near_target
