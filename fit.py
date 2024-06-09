"""
Fit mode amplitudes and profiles given simulation data.

Functions:
	fit_mode
	fit_mode_auto
	get_mode_eigenfunction
"""

import warnings
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.stats

from .utils import stdev_central
from .getters import getter_kyeq0 as _default_getter

class make_model():
	"""
	An instance of this class behaves like a function which is the sum of a polynomial of order poly_order and n_lorentz Lorentzians.
	"""
	def __init__(self, poly_order, n_lorentz):
		self.poly_order = poly_order
		self.n_lorentz = n_lorentz
		self.nparams = (1 + self.poly_order) + 3*self.n_lorentz
	
	def unpack_params(self, args):
		assert len(args) == self.nparams
		params_poly = args[:self.poly_order+1]
		params_lorentz = np.reshape(args[self.poly_order+1:], (self.n_lorentz, 3))
		return params_poly, params_lorentz
	
	def pack_params(self, params_poly, params_lorentz):
		"""
		params_poly: numpy array of length poly_order + 1
		params_lorentz: numpy array of shape (n_lorentz, 3)
		
		Returns list
		"""
		assert len(params_poly) == self.poly_order + 1
		assert np.shape(params_lorentz) == (self.n_lorentz, 3 )
		return tuple([*params_poly, *np.reshape(params_lorentz, 3*self.n_lorentz)])
	
	def lorentzian(self, om, A, om_0, gam):
		return (A*gam/np.pi)/((om - om_0)**2 + gam**2)
	
	def poly(self, om, *params_poly):
		assert len(params_poly) == self.poly_order + 1
		ret = 0
		for i,a in enumerate(params_poly):
			ret += a*om**i
		return ret
	
	def __call__(self, om, *args):
		assert self.pack_params(*self.unpack_params(args)) == args, f"{args = }\n{self.pack_params(*self.unpack_params(args)) = }"
		
		params_poly, params_lorentz = self.unpack_params(args)
		ret = self.poly(om, *params_poly)
		for i in range(self.n_lorentz):
			ret += self.lorentzian(om, *params_lorentz[i])
		return ret

def fit_mode(
	data_near_target,
	omt_near_target,
	poly_order,
	n_lorentz,
	sigma = None,
	om_guess = None,
	gamma_max = None,
	debug = 0,
	identifier = "",
	):
	"""
	Given a dr_yaver_base instance, find the amplitude of a particular mode as a function of depth
	
	Mandatory arguments:
		data_near_target: 1D array. The power spectrum as a function of frequency
		omt_near_target: 1D array. Frequencies corresponding to the entries of data_near_target
		poly_order: int. Order of polynomial to use to fit the continuum.
		n_lorentz: int. Number of Lorentzian profiles to use for fitting.
	
	Optional arguments:
		om_guess: list of float. Guesses for the omega_tilde at which modes are present. Length need not be the same as n_lorentz.
		gamma_max: float. Upper limit on the width of the Lorentzians. By default, om_tilde_max = om_tilde_min.
		sigma: 1D array: absolute error in data_near_target
		debug: int. Set >0 to print debug output.
		identifier: str. Used by wrapper functions to allow more informative error messages.
	
	Returns:
		model: make_model instance. This will have an attribute popt that gives the optimal fit values. To plot the resulting model returned by this function, you can do plt.plot(omt_near_target, model(omt_near_target, *model.popt))
	"""
	om_tilde_min = min(omt_near_target)
	om_tilde_max = max(omt_near_target)
	
	if gamma_max is None:
		gamma_max = om_tilde_max - om_tilde_min
	
	if (len(identifier) > 0 and identifier[0] != '('):
			identifier = f"({identifier})"
	
	model = make_model(poly_order, n_lorentz)
	
	if len(data_near_target) < model.nparams:
		raise RuntimeError("Data series is too short for fit.")
	
	#initial guess for the parameters.
	guess_poly = np.zeros(model.poly_order + 1)
	guess_lor = np.zeros((model.n_lorentz,3))
	
	guess_poly[0] = data_near_target[0]
	
	guess_lor[:,1] = np.linspace(om_tilde_min, om_tilde_max, model.n_lorentz+2)[1:-1]
	if om_guess is not None:
		for i in range(min(model.n_lorentz, len(om_guess))):
			guess_lor[i,1] = om_guess[i]
	
	guess_lor[:,2] = gamma_max
	
	guess = model.pack_params(guess_poly, guess_lor)
	
	#Bounds for the parameters
	lbound_poly = np.full(model.poly_order+1, -np.inf)
	lbound_lor = np.full((model.n_lorentz,3), -np.inf)
	lbound_lor[:,0] = 0
	lbound_lor[:,1] = om_tilde_min
	lbound_lor[:,2] = 0
	lbound = model.pack_params(lbound_poly, lbound_lor)
	
	ubound_poly = np.full(model.poly_order+1, np.inf)
	ubound_lor = np.full((model.n_lorentz,3), np.inf)
	ubound_lor[:,1] = om_tilde_max
	ubound_lor[:,2] = gamma_max
	ubound = model.pack_params(ubound_poly, ubound_lor)
	
	try:
		model.popt, model.pcov, infodict, mesg, _ = scipy.optimize.curve_fit(
			model,
			omt_near_target,
			data_near_target,
			p0 = guess,
			sigma = sigma,
			bounds = (lbound,ubound),
			method='trf',
			maxfev = 1e4*model.nparams,
			ftol = None,
			x_scale='jac',
			jac='3-point',
			absolute_sigma = True,
			full_output = True,
			)
	except Exception as e:
		raise RuntimeError(f"Failed for {om_tilde_min = }, {om_tilde_max = }, {poly_order = }, {n_lorentz = }, {om_guess = }, with error: {e} {identifier}")
	
	if debug > 0:
		print(f"\t{mesg = }\n\t{infodict['nfev'] = }")
	
	return model

def fit_mode_auto(
	data_near_target,
	omt_near_target,
	poly_order,
	sigma,
	n_lorentz_max = 5,
	n_lorentz_min = 0,
	threshold_ratio = None,
	threshold_p = None,
	om_guess = None,
	gamma_max = None,
	debug = 0,
	identifier = "",
	):
	"""
	Keep on increasing n_lorentz in fit_mode until the fit no longer improves.
	
	Arguments:
		data_near_target: 1D array. The power spectrum as a function of frequency
		omt_near_target: 1D array. Frequencies corresponding to the entries of data_near_target
		poly_order: int. Order of the polynomial to use for fitting the continuum.
		n_lorentz_max: int. Maximum number of Lorentzians that can be used in the fit.
		n_lorentz_min: int. Minimum number of Lorentzians that can be used in the fit.
		threshold_ratio: float. Ratio of reduced chi-squared needed to accept addition of a Lorentzian.
		threshold_p: float: if the probability of the obtained chi-squared is less than this value, accept the fit.
		sigma: 1D array. Absolute error estimates for the data.
		om_guess: list of float. Passed to fit_mode.
		gamma_max: float. Passed to fit_mode.
		identifier: str. Used by wrapper functions to allow more informative error messages.
	
	Actual termination condition is determined by the earliest reached of threshold_ratio and threshold_p.
	"""
	if threshold_ratio is None:
		threshold_ratio = 0.8
	if sigma is None:
		#Wrappers such as get_mode_eigenfunction currently allow sigma=None, so keep this error message.
		raise ValueError("fit_mode_auto requires an error estimate")
	
	if (len(identifier) > 0 and identifier[0] != '('):
			identifier = f"({identifier})"
	
	#Function to calculate the reduced chi-square corresponding to a particular fit.
	chi2r = lambda fit: np.sum(((data_near_target - fit(omt_near_target, *fit.popt) )/sigma)**2)/(len(data_near_target) - fit.nparams)
	def chi2p(fit):
		"""
		Uses chi-squared distribution, and returns probability of a chi-squared that large being generated by noise alone.
		"""
		chi2 = np.sum(((data_near_target - fit(omt_near_target, *fit.popt) )/sigma)**2)
		dof = len(data_near_target) - fit.nparams
		return scipy.stats.chi2(dof).cdf(chi2)
		
	
	fit_old = None
	for n_lorentz in range(n_lorentz_min, n_lorentz_max+1):
		fit = fit_mode(
			data_near_target = data_near_target,
			omt_near_target = omt_near_target,
			poly_order = poly_order,
			n_lorentz = n_lorentz,
			sigma = sigma,
			om_guess = om_guess,
			gamma_max = gamma_max,
			debug = debug - 1,
			identifier = identifier,
			)
		
		c = chi2r(fit)
		if debug > 0:
			if (fit_old is None):
				#Just to prevent an undefined variable in the next line.
				c_old = np.inf
			print(f"\tfit_mode_auto: {c = :.2e}, {c/c_old = :.2e}, {chi2p(fit) = :.2e}")
		if (threshold_p is not None) and (chi2p(fit) < threshold_p):
			if debug > 0:
				print("Terminated due to threshold_p")
			return fit
		elif (fit_old is not None) and c/c_old > threshold_ratio:
			if debug > 0:
				print("Terminated due to threshold_ratio")
			return fit_old
		elif c == 0:
			"""
			Usually seems to happen only in regions where the velocity is zero.
			"""
			if not np.all(data_near_target == 0):
				warnings.warn("χ² was 0 even though the data are nonzero.", RuntimeWarning)
			return fit
		
		fit_old = fit
		c_old = c
	
	om_tilde_min = min(omt_near_target)
	om_tilde_max = max(omt_near_target)
	warnings.warn(f"Improvement in fit has not converged even with {n_lorentz = } ({om_tilde_min = }) ({om_tilde_max = }) ({om_guess = }) {identifier}")
	return fit

def get_mode_eigenfunction(
	dr,
	omega_0,
	k_tilde,
	z_list,
	om_tilde_min,
	om_tilde_max,
	poly_order = 1,
	force_n_lorentz = None,
	omega_tol = None,
	mode_mass_method="sum",
	debug = 0,
	getter = _default_getter,
	identifier = "",
	**kwargs,
	):
	"""
	Use fit_mode to get the z-dependent eigenfunction of the mode whose frequency (omega_tilde) is close to omega_0 at k_tilde.
	
	Arguments:
		dr: dr_yaver_base instance.
		omega_0: float
		k_tilde: float.
		z_list: list of float. Values of z at which to get the eigenfunction.
		om_tilde_min: float. Lower limit of the band of omega_tilde in which to fit the data.
		om_tilde_max: float. Upper limit of the band of omega_tilde in which to fit the data.
		poly_order: int. Order of the polynomial to use for fitting the continuum.
		force_n_lorentz: int. Force this many Lorentizans to be used for the fitting (rather than automatically determining based on the data). If this is set to None (default), the number of Lorentzians will be automatically determined.
		omega_tol: float or None. If (not None) and (the distance between the detected mode and omega_0) is greater than or equal to this value, do not consider that mode for computation of the mode mass.
		mode_mass_method: string. How to compute the mode mass.
			Allowed values:
				"sum": sum the fitted Lorentzian closest to omega_0 (along with other Lorentzians that are closer to it than its width) over the frequency axis
				"sum_full": like "sum", but also add the residual part of the data to the mode mass (i.e. the part that is fitted by neither the polynomial nor the Lorentzians)
				"integral": like "sum", but integrate over the frequency axis rather than summing
				"sum_multi": like "sum", but consider all Lorentzians that are within omega_tol of omega_0
		getter: function. Instance of getters.AbstractGetter
		identifier: str. Use to get more informative error messages in wrapped functions.
	
	Other kwargs are passed to either fit_mode or fit_mode_auto depending on the value of force_n_lorentz.
	"""
	if not om_tilde_min < omega_0 < om_tilde_max:
		raise ValueError("Cannot fit mode that is outside search band.")
	
	if mode_mass_method not in [
		"sum",
		"sum_full",
		"integral",
		"sum_multi",
		]:
		raise ValueError(f"Unsupported mode_mass_method: {mode_mass_method}")
	
	if omega_tol is None:
		omega_tol = np.inf
	
	P_list = []
	for z in z_list:
		if debug > 0:
			print(f"get_mode_eigenfunction: {z = }")
		
		d = getter(dr, k_tilde, z, om_tilde_min, om_tilde_max)
		data_near_target = d['data_near_target']
		omt_near_target = d['omt_near_target']
		sigma = d['sigma']
		
		ide = identifier + f"{z = }"
		
		if force_n_lorentz is None:
			fit = fit_mode_auto(
				data_near_target=data_near_target,
				omt_near_target=omt_near_target,
				sigma=sigma,
				poly_order=poly_order,
				om_guess=[omega_0],
				debug=debug-1,
				identifier = ide,
				**kwargs,
				)
		else:
			fit = fit_mode(
				data_near_target=data_near_target,
				omt_near_target=omt_near_target,
				sigma=sigma,
				poly_order=poly_order,
				om_guess=[omega_0],
				n_lorentz=force_n_lorentz,
				debug=debug-1,
				identifier = ide,
				**kwargs,
				)
		
		_, params_lorentz = fit.unpack_params(fit.popt)
		
		if len(params_lorentz) > 0:
			"""
			Among all Lorentzians which are within omega_tol of omega_0, we choose the Lorentzian with the highest mode mass. Lorentzians which are closer to the center of this mode than its width are considered as part of the same mode.
			"""
			selected = params_lorentz[
				np.abs(params_lorentz[:,1] - omega_0) < omega_tol
				]
			
			if len(selected) > 0:
				modes = [fit.lorentzian(omt_near_target, *params) for params in selected]
				if mode_mass_method == "integral":
					mode_masses = np.array([np.trapz(mode, omt_near_target) for mode in modes])
				elif mode_mass_method in ["sum", "sum_full", "sum_multi"]:
					mode_masses = np.array([np.sum(mode) for mode in modes])
				else:
					raise ValueError(f"Unsupported {mode_mass_method = }")
				
				main_mode = np.argmax(mode_masses)
				width = selected[main_mode,2]
				omega_c = selected[main_mode,1]
				
				if debug > 0:
					print(f"get_mode_eigenfunction: {omega_c = :.2e}, {width = :.2e}")
				
				if mode_mass_method == "sum_multi":
					mode_mass = np.sum(mode_masses)
				else:
					mode_mass = np.sum(np.where(
						np.abs(selected[:,1] - omega_c) < width,
						mode_masses,
						0,
						))
			else:
				mode_mass = 0
		elif np.any(data_near_target != 0):
			"""
			u_z is nonzero, but no Lorentzian was fitted.
			Setting these to zero leads to jarring discontinuities in the plot of the mode eigenfunction. It feels dishonest to add an extra lorentzian there by hand and then get a fit, so I shall just set them to nan to indicate that the amplitude of the mode was too close to the noise threshold to say anything.
			"""
			mode_mass = np.nan
		else:
			mode_mass = 0
		
		if mode_mass_method == "sum_full":
			"""
			Nishant suggested that this may be required to get a good-looking fit for the eigenfunction.
			"""
			residuals = data_near_target - fit(omt_near_target, *fit.popt)
			mode_mass += np.sum(residuals)
		
		P_list.append(mode_mass)
	
	return np.array(P_list)

def get_mode_eigenfunction_from_simset(dr_list, *args, **kwargs):
	"""
	Given multiple realizations of the same setup, calculate the mode mass in all of them and return estimates of the mean mode mass and the error in the mean.
	
	Arguments
		dr_list: list of dr_base instances
		Other arguments are the same as get_mode_eigenfunction (except for dr)
	"""
	
	mass_list = []
	for dr in dr_list:
		mass_list.append(get_mode_eigenfunction(
			dr=dr,
			*args,
			**kwargs,
			))
	mass_list = np.array(mass_list)
	
	mean = np.average(mass_list, axis=0)
	if len(dr_list) == 1:
		warnings.warn("Cannot estimate error with only one realization")
		err = np.full_like(mean, np.nan)
	else:
		err = np.std(mass_list, axis=0)/np.sqrt(len(dr_list))
	
	return mean, err
