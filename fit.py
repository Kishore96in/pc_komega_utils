"""
Fit mode amplitudes and profiles given simulation data.

Objects intended for public use:
	fit_mode
	fit_mode_auto
	get_mode_eigenfunction
	get_mode_eigenfunction_from_simset
"""

import warnings
import numpy as np
import scipy.optimize
import scipy.stats
import scipy.linalg

from .utils import stdev_central
from .getters import getter_kyeq0 as _default_getter
from .models import ModelMakerLorentzian

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
	ModelMaker = ModelMakerLorentzian,
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
		ModelMaker: AbstractModelMaker. Describes what function to use to fit the mode profiles.
	
	Returns:
		model: ModelMaker instance. This will have an attribute popt that gives the optimal fit values. To plot the resulting model returned by this function, you can do plt.plot(omt_near_target, model(omt_near_target, *model.popt))
	"""
	om_tilde_min = min(omt_near_target)
	om_tilde_max = max(omt_near_target)
	
	if gamma_max is None:
		gamma_max = (om_tilde_max - om_tilde_min)/6
	
	if (len(identifier) > 0 and identifier[0] != '('):
			identifier = f"({identifier})"
	
	model = ModelMaker(poly_order, n_lorentz)
	
	if len(data_near_target) < model.nparams:
		raise RuntimeError("Data series is too short for fit.")
	
	#If the values are very small (e.g. 1e-13), the fits get drastically affected (presumably due to accumulation of rounding errors). Before passing the data to the optimization routine, we thus scale it. The inverse of this scaling will later be applied to the returned optimal parameters.
	scale = np.max(data_near_target)
	data_near_target = data_near_target/scale
	
	if sigma is None:
		#This is equivalent to not having any sigma in _residual
		sigma = 1
	else:
		sigma = sigma/scale
	
	#initial guesses for the parameters.
	guess_poly = model._baseline_guesser(omt_near_target, data_near_target)
	guess_lor = np.zeros((model.n_lines,model.n_lineparams))
	
	i_om = model._ind_line_freq
	
	guess_lor[:,i_om] = np.linspace(
		om_tilde_min + gamma_max,
		om_tilde_max - gamma_max,
		model.n_lines,
		)
	if om_guess is not None:
		"""
		If om_guess is given, for each om_guess, find the closest guessed frequency populated above and shift it to om_guess.
		"""
		inds_to_change = []
		marker_val = 1e3*max(om_guess)
		
		for i in range(min(model.n_lines, len(om_guess))):
			i_min = np.argmin(np.abs(guess_lor[:,i_om] - om_guess[i]))
			
			inds_to_change.append(i_min)
			guess_lor[i_min,i_om] = marker_val
		
		for i, ind in enumerate(inds_to_change):
			guess_lor[ind,i_om] = om_guess[i]
	
	for i in model._width_like_params:
		d_omt = omt_near_target[1] - omt_near_target[0]
		guess_lor[:,i] = np.sqrt(d_omt*gamma_max)
	
	guess = model.pack_params(guess_poly, guess_lor)
	
	#Bounds for the parameters
	lbound_poly = np.full(model.poly_order+1, -np.inf)
	for i in model._baseline_positive_params:
		lbound_poly[i] = 0
	lbound_lor = np.full((model.n_lines,model.n_lineparams), -np.inf)
	for i in model._positive_params:
		lbound_lor[:,i] = 0
	lbound_lor[:,i_om] = om_tilde_min + gamma_max
	lbound = model.pack_params(lbound_poly, lbound_lor)
	
	ubound_poly = np.full(model.poly_order+1, np.inf)
	ubound_lor = np.full((model.n_lines,model.n_lineparams), np.inf)
	ubound_lor[:,i_om] = om_tilde_max - gamma_max
	for i in model._width_like_params:
		ubound_lor[:,i] = gamma_max
	ubound = model.pack_params(ubound_poly, ubound_lor)
	
	def _residuals(params):
		total = model(omt_near_target, *params)
		
		#The usual residual
		res = abs((total - data_near_target)/sigma)
		
		if model._baseline_ensure_positive:
			params_poly, _ = model.unpack_params(params)
			poly = model.poly(omt_near_target, *params_poly)
			
			res += np.where(poly < 0, abs(poly/sigma), 0)
		
		return res
	
	try:
		res = scipy.optimize.least_squares(
			_residuals,
			x0 = guess,
			bounds = (lbound,ubound),
			method='trf',
			max_nfev = 1e4*model.nparams,
			ftol = None,
			x_scale='jac',
			jac='3-point',
			)
		
		model.popt = model.scale_params(res.x, scale)
		mesg = res.message
		nfev = res.nfev
		
		"""
		Estimate the errors in the parameters.
		
		The idea is that we don't completely trust the value of sigma given to us by the user, so we assume the actual error is S*sigma. S is fixed by requiring the reduced chi-squared to be 1.
		
		Implementation here is from scipy.optimize.curve_fit (file scipy/optimize/_minpack_py.py)
		
		Note that the expression used below assumes we are doing linear least squares, and thus may not be completely correct in our case.
		"""
		# Do Moore-Penrose inverse discarding zero singular values.
		_, s, VT = scipy.linalg.svd(res.jac, full_matrices=False)
		threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
		s = s[s > threshold]
		VT = VT[:s.size]
		pcov = np.dot(VT.T / s**2, VT)
		
		if np.isnan(pcov).any():
			# indeterminate covariance
			pcov = np.full((len(popt), len(popt)), np.inf, dtype=float)
		
		ysize = len(omt_near_target)
		#In scipy, this is the branch `if not absolute_sigma:`
		if ysize > res.x.size:
			s_sq = np.sum(res.fun**2) / (ysize - model.nparams)
			pcov = pcov * s_sq
		else:
			pcov.fill(inf)
		
		#Errors need to be scaled in the same way as the data
		model.perr =  model.scale_params(np.sqrt(np.diag(pcov)), scale)
		
		assert len(model.popt) == len(model.perr)
	except Exception as e:
		raise RuntimeError(f"Failed for {om_tilde_min = }, {om_tilde_max = }, {poly_order = }, {n_lorentz = }, {om_guess = }, with error: {e} {identifier}")
	
	if debug > 0:
		print(f"\t{mesg = }\n\t{nfev = }")
	
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
	ModelMaker = ModelMakerLorentzian,
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
		ModelMaker: AbstractModelMaker. Passed to fit_mode.
	
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
			ModelMaker = ModelMaker,
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

class Eigenprofile():
	"""
	Holds the result of get_mode_eigenfunction
	
	Properties:
		mass: np.ndarray. Mode mass as a function of z
		error: np.ndarray. Error in the mode mass at each z
		fit: list of AbstractModelMaker instances. The optimal fit at each z
		omega_c: np.ndarray. Central frequency of the most massive mode that was considered for computation of the mode mass, at each z
	"""
	def __init__(self, mass, error, fit, omega_c):
		self.mass = mass
		self.error = error
		self.fit = fit
		self.omega_c = omega_c

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
	error_method="derivative",
	debug = 0,
	getter = _default_getter,
	identifier = "",
	full_output = False,
	omega_0_other = None,
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
		full_output: bool. If True, return an object containing both the mode masses and the errors. If False, return an array of the mode masses.
		omega_0_other: list of float. If multiple lines are being fit, this specifies the guessed frequencies to use for the lines (in addition to omega_0)
	
	Other kwargs are passed to either fit_mode or fit_mode_auto depending on the value of force_n_lorentz.
	"""
	if not om_tilde_min < omega_0 < om_tilde_max:
		raise ValueError("Cannot fit mode that is outside search band.")
	
	if omega_tol is None:
		omega_tol = np.inf
	if omega_0_other is None:
		omega_0_other = []
	
	P_list = []
	P_err_list = []
	fit_list = []
	omega_c_list = []
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
				om_guess=[omega_0, *omega_0_other],
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
				om_guess=[omega_0, *omega_0_other],
				n_lorentz=force_n_lorentz,
				debug=debug-2,
				identifier = ide,
				**kwargs,
				)
		
		mode_mass, mode_info = _get_mode_mass(
			model = fit,
			popt = fit.popt,
			omega_0 = omega_0,
			omega_tol = omega_tol,
			omt_near_target = omt_near_target,
			mode_mass_method = mode_mass_method,
			debug = debug,
			extra_info = True,
			data_near_target = data_near_target,
			)
		P_list.append(mode_mass)
		
		if full_output:
			if error_method == "derivative":
				"""
				Estimate the error in the mode mass from the error in the fit parameters by using the numerical derivative of the former.
				"""
				mder = _get_mode_mass_derivative(
					model = fit,
					popt = fit.popt,
					omega_0 = omega_0,
					omega_tol = omega_tol,
					omt_near_target = omt_near_target,
					data_near_target = data_near_target,
					mode_mass_method = mode_mass_method,
					)
				err = np.sqrt(np.sum((mder*fit.perr)**2))
			elif error_method == "monte-carlo":
				err = _get_mode_mass_err_mc(
					model = fit,
					popt = fit.popt,
					perr = fit.perr,
					omega_0 = omega_0,
					omega_tol = omega_tol,
					omt_near_target = omt_near_target,
					mode_mass_method = mode_mass_method,
					data_near_target = data_near_target,
					)
			else:
				raise ValueError(f"Given error_method ({error_method}) is invalid")
			
			P_err_list.append(err)
			
			fit_list.append(fit)
			omega_c_list.append(mode_info['omega_c'])
	
	if full_output:
		return Eigenprofile(
			mass = np.array(P_list),
			error = np.array(P_err_list).transpose(),
			fit = fit_list,
			omega_c = np.array(omega_c_list),
			)
	else:
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

def _get_mode_mass(
	model,
	popt,
	omega_0,
	omega_tol,
	omt_near_target,
	data_near_target,
	mode_mass_method,
	debug = 0,
	extra_info = False,
	):
	if mode_mass_method not in [
		"sum",
		"sum_full",
		"integral",
		"sum_multi",
		]:
		raise ValueError(f"Unsupported mode_mass_method: {mode_mass_method}")
	
	params_poly, params_lorentz = model.unpack_params(popt)
	i_om = model._ind_line_freq
	
	if len(params_lorentz) > 0:
		"""
		Among all Lorentzians which are within omega_tol of omega_0, we choose the Lorentzian with the highest mode mass. Lorentzians which are closer to the center of this mode than its width are considered as part of the same mode.
		"""
		selected = params_lorentz[np.abs(params_lorentz[:,i_om] - omega_0) < omega_tol]
		
		if len(selected) > 0:
			modes = [model.line(omt_near_target, *params, params_poly = params_poly) for params in selected]
			if mode_mass_method == "integral":
				mode_masses = np.array([np.trapz(mode, omt_near_target) for mode in modes])
			elif mode_mass_method in ["sum", "sum_full", "sum_multi"]:
				mode_masses = np.array([np.sum(mode) for mode in modes])
			else:
				raise ValueError(f"Unsupported {mode_mass_method = }")
			
			main_mode = np.argmax(mode_masses)
			width = model.get_line_hwhm(*selected[main_mode])
			omega_c = model.get_line_freq(*selected[main_mode])
			
			if debug > 0:
				print(f"get_mode_eigenfunction: {omega_c = :.2e}, {width = :.2e}")
			
			if mode_mass_method == "sum_multi":
				mode_mass = np.sum(mode_masses)
			else:
				mode_mass = np.sum(np.where(
					np.abs(selected[:,i_om] - omega_c) < width,
					mode_masses,
					0,
					))
		else:
			mode_mass = 0
			omega_c = np.nan
	elif np.any(data_near_target != 0):
		"""
		u_z is nonzero, but no Lorentzian was fitted.
		Setting these to zero leads to jarring discontinuities in the plot of the mode eigenfunction. It feels dishonest to add an extra lorentzian there by hand and then get a fit, so I shall just set them to nan to indicate that the amplitude of the mode was too close to the noise threshold to say anything.
		"""
		mode_mass = np.nan
		omega_c = np.nan
	else:
		mode_mass = 0
		omega_c = np.nan
	
	if mode_mass_method == "sum_full":
		"""
		Nishant suggested that this may be required to get a good-looking fit for the eigenfunction.
		"""
		residuals = data_near_target - model(omt_near_target, *popt)
		mode_mass += np.sum(residuals)
	
	if extra_info:
		return (
			mode_mass,
			{
				'omega_c': omega_c,
				}
			)
	else:
		return mode_mass

def _get_mode_mass_derivative(popt, **kwargs):
	"""
	Derivative of the mode mass with respect to the fit parameters. This is required in order to propagate errors.
	"""
	der = np.full_like(popt, np.nan)
	
	#Step size that minimizes the sum of roundoff and truncation errors (for a finite difference scheme for the first derivative that has second-order errors)
	dp = np.finfo(float).eps**(1/3)
	
	for i in range(len(der)):
		popt_mdp = np.array(popt)
		popt_mdp[i] -= dp
		
		popt_pdp = np.array(popt)
		popt_pdp[i] += dp
		
		dm = _get_mode_mass(popt = popt_pdp, **kwargs) - _get_mode_mass(popt = popt_mdp, **kwargs)
		der[i] = dm/(2*dp)
	
	return der

def _get_mode_mass_error_der(popt, perr, **kwargs):
	"""
	Estimate the error in the mode mass from the error in the fit parameters by using the numerical derivative of the former.
	"""
	mder = _get_mode_mass_derivative(
				popt = popt,
				**kwargs,
				)
	return np.sqrt(np.sum((mder*perr)**2))

def _get_mode_mass_err_mc(popt, perr, **kwargs):
	"""
	Use a Monte-Carlo method to estimate the error in the mode mass, given the error in the fit parameters.
	
	Returns an array [lower_limit, upper_limit]
	"""
	gen = np.random.default_rng()
	
	mode_masses = []
	for i in range(100):
		par = gen.normal(loc=popt, scale=perr)
		mode_masses.append(_get_mode_mass(popt=par, **kwargs))
	
	mode_masses=np.array(mode_masses)
	
	#The percentiles below correspond to 1-sigma deviation for a Gaussian distribution
	ulim = np.percentile(mode_masses, 84.1)
	llim = np.percentile(mode_masses, 15.9)
	
	return llim, ulim
