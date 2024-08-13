"""
Models used to fit mode profiles

Objects intended for public use:
	AbstractModelMaker
	ModelBaselinePoly
	ModelBaselineExp
	ModelLineLorentzian
	ModelLineVoigt
	ModelMakerLorentzian
	ModelMakerVoigt
"""

import numpy as np
import scipy.special
import abc

class AbstractModelMaker(abc.ABC):
	def __init__(self, poly_order, n_lorentz):
		self.poly_order = poly_order
		self.n_lines = n_lorentz
		self.nparams = (1 + self.poly_order) + self.n_lineparams*self.n_lines
	
	def unpack_params(self, args):
		assert len(args) == self.nparams
		params_poly = np.array(args[:self.poly_order+1])
		params_lorentz = np.reshape(args[self.poly_order+1:], (self.n_lines, self.n_lineparams))
		return params_poly, params_lorentz
	
	def pack_params(self, params_poly, params_lorentz):
		"""
		params_poly: numpy array of length poly_order + 1
		params_lorentz: numpy array of shape (n_lines, n_lineparams)
		
		Returns list
		"""
		assert len(params_poly) == self.poly_order + 1
		assert np.shape(params_lorentz) == (self.n_lines, self.n_lineparams)
		return tuple([*params_poly, *np.reshape(params_lorentz, self.n_lineparams*self.n_lines)])
	
	@abc.abstractmethod
	def baseline(self, om, *params_poly):
		raise NotImplementedError
	
	def __call__(self, om, *args):
		assert self.pack_params(*self.unpack_params(args)) == args, f"{args = }\n{self.pack_params(*self.unpack_params(args)) = }"
		
		params_poly, params_lorentz = self.unpack_params(args)
		ret = self.poly(om, *params_poly)
		for i in range(self.n_lines):
			ret += self.line(om, *params_lorentz[i], params_poly = params_poly)
		return ret
	
	@property
	def n_lorentz(self):
		"""
		For backwards compatibility
		"""
		return self.n_lines
	
	def lorentzian(self, *args, **kwargs):
		"""
		For backwards compatibility
		"""
		return self.line(*args, **kwargs)
	
	def poly(self, *args, **kwargs):
		"""
		For backwards compatibility
		"""
		return self.baseline(*args, **kwargs)
	
	@property
	@abc.abstractmethod
	def n_lineparams(self):
		"""
		Number of parameters needed to specify the line profile
		"""
		raise NotImplementedError
	
	@abc.abstractmethod
	def line(self, om, *args, **kwargs):
		"""
		Model for the profile of an individual mode. The first argument is a 1D array of angular frequencies, and the ones that follow are the parameters of the mode profile.
		"""
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def _ind_line_freq(self):
		"""
		Index of the central angular frequency of a line in the tuple of line parameters
		"""
		raise NotImplementedError
	
	def get_line_freq(self, *args):
		"""
		Get the central angular frequency of a line, given the line parameters
		"""
		if len(args) != self.n_lineparams:
			raise ValueError(f"Wrong number of parameters for line (expected {self.n_lineparams}; got {len(args)}")
		
		return args[self._ind_line_freq]
	
	@abc.abstractmethod
	def get_line_hwhm(self, *args):
		"""
		Get the HWHM (half-width at half-maximum) of a line, given the line parameters
		"""
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def _width_like_params(self):
		"""
		Returns a list of indices of the line parameters that can be interpreted as widths. This is used by fit_mode to set bounds on the parameters.
		"""
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def _positive_params(self):
		"""
		Returns a list of indices of the line parameters that need to be positive. This is used by fit_mode to set bounds on the parameters.
		"""
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def _amplitude_like_params(self):
		"""
		Returns a list of indices of the line parameters that should be scaled when the data is scaled. This is used by scale_params
		"""
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def _baseline_amplitude_like_params(self):
		"""
		Returns a list of indices of the baseline parameters that should be scaled when the data is scaled. This is used by scale_params
		"""
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def _baseline_positive_params(self):
		"""
		Returns a list of indices of the baseline parameters that should be constrained to be positive. This is used by fit_mode to set bounds on the parameters.
		"""
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def _baseline_ensure_positive(self):
		"""
		If the function used for the baseline is such that one cannot easily ensure its positivity by constraining the parameters, one can instead set this property to True.
		
		When this property is True, fit_mode uses a modified expression for the least-squares residual that prefers a positive baseline. If False, no such special handling is done.
		"""
		raise NotImplementedError
	
	def scale_params(self, params, factor):
		"""
		Given a vector of params, apply a scaling factor to the polynomial and to the amplitudes of the lines, and return the vector of scaled params. This is useful if you want to pass a scaled version of the data to a fitting routine, but want to finally obtain parameters corresponding to the unscaled data.
		"""
		params_poly, params_lines = self.unpack_params(params)
		
		for i in range(len(params_poly)):
			if i in self._baseline_amplitude_like_params:
				params_poly[i] = params_poly[i]*factor
		
		for i in range(np.shape(params_lines)[1]):
			if i in self._amplitude_like_params:
				params_lines[:,i] = params_lines[:,i]*factor
		
		return self.pack_params(params_poly, params_lines)
	
	def _baseline_guesser(self, omt, data):
		"""
		Given the data, return a guess for the parameters of the baseline.
		"""
		return np.zeros(self.poly_order + 1)

class ModelBaselinePoly():
	_baseline_ensure_positive = True
	
	@property
	def _baseline_amplitude_like_params(self):
		return list(range(self.poly_order + 1))
	
	@property
	def _baseline_positive_params(self):
		return []
	
	def baseline(self, om, *params_poly):
		assert len(params_poly) == self.poly_order + 1
		ret = 0
		for i,a in enumerate(params_poly):
			ret += a*om**i
		return ret

class ModelBaselineExp():
	_baseline_ensure_positive = False
	
	@property
	def _baseline_amplitude_like_params(self):
		return [0]
	
	@property
	def _baseline_positive_params(self):
		return [0,1]
	
	def baseline(self, om, *params_poly):
		if len(params_poly) != 2:
			raise ValueError("ModelBaselineExp requires poly_order = 1")
		
		assert len(params_poly) == self.poly_order + 1
		
		a, b = params_poly
		return a*np.exp(-b*abs(om))

class ModelBaselinePowerLaw():
	_baseline_ensure_positive = False
	
	@property
	def _baseline_amplitude_like_params(self):
		return [0]
	
	@property
	def _baseline_positive_params(self):
		return [0,1]
	
	def baseline(self, om, *params_poly):
		if len(params_poly) != 2:
			raise ValueError("ModelBaselinePowerLaw requires poly_order = 2")
		
		assert len(params_poly) == self.poly_order + 1
		
		a, b = params_poly
		return a*abs(om)**(-b)
	
	def _baseline_guesser(self, omt, data):
		"""
		In the variable names below, prefix l stands for 'log'
		"""
		if not np.all(omt > 0):
			raise ValueError
		if np.any(data < 0):
			raise ValueError
		
		if np.all(data == 0):
			return np.array([0,0])
		else:
			lP1 = np.log(data[0])
			lP2 = np.log(data[-1])
			lo1 = np.log(omt[0])
			lo2 = np.log(omt[-1])
			
			la = (lo2*lP1 - lo1*lP2)/(lo2-lo1)
			b = (la - lP1)/lo1
			
			if b < 0:
				#We impose positivity of b.
				b = 0
			return np.array([np.exp(la), b])
	
	def _get_power_law_slope(self, params_poly):
		"""
		For use with ModelLineLorentzianWithPowerLawForcing
		"""
		return params_poly[1]

class ModelLineLorentzian():
	n_lineparams = 3
	_ind_line_freq = 1
	_width_like_params = [2]
	_positive_params = [0,2]
	_amplitude_like_params = [0]
	
	def line(self, om, A, om_0, gam, **kwargs):
		return (A*gam/np.pi)/((om - om_0)**2 + gam**2)
	
	def get_line_hwhm(self, A, om_0, gam):
		return gam

class ModelLineVoigt():
	n_lineparams = 4
	_ind_line_freq = 1
	_width_like_params = [2,3]
	_positive_params = [0,2,3]
	_amplitude_like_params = [0]
	
	def line(self, om, A, om_0, gam, sigma, **kwargs):
		return A*scipy.special.voigt_profile(om-om_0, sigma, gam)
	
	def get_line_hwhm(self, A, om_0, gam, sigma):
		"""
		An approximation formula from https://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile (accessed 2:39 PM IST, 09-Jun-2024)
		"""
		f_L = 2*gam
		f_G = 2.3548200450309493*sigma
		f = 0.5346*f_L + np.sqrt(0.2166*f_L**2 + f_G**2)
		return f/2

class ModelMakerLorentzian(ModelBaselinePoly, ModelLineLorentzian, AbstractModelMaker):
	"""
	An instance of this class behaves like a function which is the sum of a polynomial of order poly_order and n_lorentz Lorentzians.
	"""
	pass

class ModelMakerVoigt(ModelBaselinePoly, ModelLineVoigt, AbstractModelMaker):
	"""
	An instance of this class behaves like a function which is the sum of a polynomial of order poly_order and n_lorentz Voigt functions.
	"""
	pass
