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
	dr,
	k_tilde,
	z,
	om_tilde_min,
	om_tilde_max,
	poly_order,
	n_lorentz,
	om_guess = None,
	gamma_max = None,
	):
	"""
	Given a dr_yaver_base instance, find the amplitude of a particular mode as a function of depth
	
	Arguments:
		dr: dr_yaver_base instance
		k_tilde: float, wavenumber at which to find the amplitude
		z: float, z-coordinate at which to read the data
		om_tilde_min: float. Consider the part of the data above this frequency
		om_tilde_max: float. Consider the part of the data below this frequency
		poly_order: int. Order of polynomial to use to fit the continuum.
		n_lorentz: int. Number of Lorentzian profiles to use for fitting.
		om_guess: list of float. Guesses for the omega_tilde at which modes are present. Length need not be the same as n_lorentz.
		gamma_max: float. Upper limit on the width of the Lorentzians. By default, om_tilde_max = om_tilde_min.
	
	Returns:
		model: make_model instance. This will have an attribute popt that gives the optimal fit values. To plot the resulting model returned by this function, you can do plt.plot(omt_near_target, model(omt_near_target, *model.popt))
	"""
	if gamma_max is None:
		gamma_max = om_tilde_max - om_tilde_min
	
	data_near_target, [omt_near_target, *_] = dr.get_slice(
		omega_tilde=(om_tilde_min, om_tilde_max),
		kx_tilde = k_tilde,
		ky_tilde = 0,
		z = z,
		compress = True,
		)
	
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
	
	if model.n_lorentz > 0:
		guess_lor[0,0] = 10*np.max(np.abs(data_near_target))*guess_lor[0,2]*np.pi
	
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
	
	sigma = estimate_sigma(data_near_target, gamma_max=gamma_max, omega_tilde=omt_near_target)
	
	try:
		model.popt, model.pcov = scipy.optimize.curve_fit(
			model,
			omt_near_target,
			data_near_target,
			p0 = guess,
			sigma = sigma,
			bounds = (lbound,ubound),
			method='dogbox',
			maxfev=int(1e4),
			xtol=None,
			x_scale='jac',
			absolute_sigma = True,
			)
	except Exception as e:
		raise RuntimeError(f"Failed for {k_tilde = }, {z = }, {om_tilde_min = }, {om_tilde_max = }, {poly_order = }, {n_lorentz = }, {om_guess = }, {gamma_max = } with error: {e}")
	
	return model

def fit_mode_auto(
	dr,
	k_tilde,
	z,
	om_tilde_min,
	om_tilde_max,
	poly_order,
	n_lorentz_max = 5,
	threshold = 0.5,
	threshold_p = None,
	om_guess = None,
	gamma_max = None,
	):
	"""
	Keep on increasing n_lorentz in fit_mode until the fit no longer improves.
	
	Arguments:
		dr: dr_yaver_base instance
		k_tilde: float
		z: float
		om_tilde_min: float
		om_tilde_max: float
		poly_order: int. Order of the polynomial to use for fitting the continuum.
		n_lorentz_max: int. Maximum number of Lorentzians that can be used in the fit.
		threshold: float. Ratio of reduced chi-squared needed to accept addition of a Lorentzian.
		threshold_p: float: if the probability of the obtained chi-squared is less than this value, accept the fit.
		om_guess: list of float. Passed to fit_mode.
		gamma_max: float. Passed to fit_mode.
	
	Actual termination condition is determined by the earliest reached of threshold and threshold_p.
	"""
	
	data_near_target, [omt_near_target, *_] = dr.get_slice(
		omega_tilde=(om_tilde_min, om_tilde_max),
		kx_tilde = k_tilde,
		ky_tilde = 0,
		z = z,
		compress = True,
		)
	
	sigma = estimate_sigma(data_near_target, gamma_max=gamma_max, omega_tilde=omt_near_target)
	
	
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
	for n_lorentz in range(n_lorentz_max):
		fit = fit_mode(
			dr,
			k_tilde=k_tilde,
			z=z,
			om_tilde_min=om_tilde_min,
			om_tilde_max=om_tilde_max,
			poly_order=poly_order,
			n_lorentz=n_lorentz,
			om_guess=om_guess,
			gamma_max=gamma_max,
			)
		
		c = chi2r(fit)
		if (threshold_p is not None) and (chi2p(fit) < threshold_p):
			if debug:
				print("Terminated due to threshold_p")
			return fit
		elif (fit_old is not None) and c/c_old > threshold:
			if debug:
				print("Terminated due to threshold")
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
	
	raise RuntimeError(f"Improvement in fit has not converged even with {n_lorentz = }")

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
	gamma_max = None,
	mode_mass_method="sum",
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
		gamma_max: float. See fit_mode.
		mode_mass_method: string. How to compute the mode mass. Can be "sum" or "integral".
	"""
	if not om_tilde_min < omega_0 < om_tilde_max:
		raise ValueError("Cannot fit mode that is outside search band.")
	
	if omega_tol is None:
		omega_tol = np.inf
	
	P_list = []
	for z in z_list:
		if force_n_lorentz is None:
			fit = fit_mode_auto(
				dr=dr,
				k_tilde=k_tilde,
				z=z,
				om_tilde_min=om_tilde_min,
				om_tilde_max=om_tilde_max,
				poly_order=poly_order,
				om_guess=[omega_0],
				gamma_max=gamma_max,
				)
		else:
			fit = fit_mode(
				dr=dr,
				k_tilde=k_tilde,
				z=z,
				om_tilde_min=om_tilde_min,
				om_tilde_max=om_tilde_max,
				poly_order=poly_order,
				om_guess=[omega_0],
				gamma_max=gamma_max,
				n_lorentz=force_n_lorentz,
				)
		
		_, params_lorentz = fit.unpack_params(fit.popt)
		
		data_near_target, [omt_near_target, *_] = dr.get_slice(
			omega_tilde=(om_tilde_min, om_tilde_max),
			kx_tilde = k_tilde,
			ky_tilde = 0,
			z = z,
			compress = True,
			)
		
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
				elif mode_mass_method == "sum":
					mode_masses = np.array([np.sum(mode) for mode in modes])
				else:
					raise ValueError(f"Unsupported {mode_mass_method = }")
				
				main_mode = np.argmax(mode_masses)
				width = selected[main_mode,2]
				omega_c = selected[main_mode,1]
				
				mode_mass = np.sum(np.where(
					np.abs(selected[:,1] - omega_c) < width,
					mode_masses,
					0,
					))
			else:
				mode_mass == 0
		elif np.any(data_near_target != 0):
			"""
			u_z is nonzero, but no Lorentzian was fitted.
			Setting these to zero leads to jarring discontinuities in the plot of the mode eigenfunction. It feels dishonest to add an extra lorentzian there by hand and then get a fit, so I shall just set them to nan to indicate that the amplitude of the mode was too close to the noise threshold to say anything.
			"""
			mode_mass = np.nan
		else:
			mode_mass = 0
		
		P_list.append(mode_mass)
	
	return np.array(P_list)

def smooth(data, n):
	"""
	data: numpy array
	n: int, such that width of the smoothing filter (top hat) is 2*n+1
	"""
	weight = np.ones(2*n+1)
	weight = weight/np.sum(weight)
	return scipy.signal.convolve(data, weight, mode='same')

def smooth_gauss(data, n):
	"""
	data: numpy array
	n: int, such that width of the smoothing filter (Gaussian) is 2*n+1 and its standard deviation is n/3.
	"""
	weight = scipy.signal.windows.gaussian(2*n+1, std=n/3)
	weight = weight/np.sum(weight)
	return scipy.signal.convolve(data, weight, mode='same')

def stdev_central(arr, frac, adjust=False):
	"""
	Estimate standard derivation of an array arr, considering only values between the frac*100 and (1-frac)*100 percentiles
	
	Arguments:
		arr: 1D numpy array
		frac: float
		adjust: bool. Whether to scale the standard deviation to account for the outliers by assuming the array elements are IID Gaussian.
	
	Returns:
		stdev: scalar of type arr.dtype
	"""
	sort = np.sort(arr)
	n = len(arr)
	i_min = int(np.round(n*frac))
	i_max = int(np.round(n*(1-frac)))
	cut = sort[i_min:i_max]
	if len(cut) < 2:
		raise ValueError(f"{frac = } is too high; not enough values left to estimate standard deviation.")
	std = np.std(cut)
	
	if adjust:
		sol = scipy.optimize.minimize(lambda x: (scipy.stats.norm().cdf(x) - frac)**2, x0=0)
		if not sol.success:
			raise RuntimeError(f"Could not find truncation location corresponding to given percentile for Gaussian distribution. {sol.message}")
		a = sol.x[0]
		scl = scipy.stats.truncnorm(a,-a).std()
		return std/scl
	else:
		return std

def estimate_sigma(data, gamma_max, omega_tilde):
	sigma = np.full_like(data, stdev_central(data, 0.05, adjust=True))
	
	if np.all(sigma == 0):
		"""
		This can only happen if np.all(data == 0), in which case we set sigma=1 to prevent divide-by-zero errors.
		"""
		sigma[:] = 1
	elif np.any(sigma == 0):
		warnings.warn("Estimated error was zero in some bins. Applying floor to estimated error.")
		min_sigma = np.min(np.compress(sigma != 0, sigma)) #smallest nonzero value
		sigma = np.where(sigma == 0, min_sigma, sigma)
	
	return sigma

