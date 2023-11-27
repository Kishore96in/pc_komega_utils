"""
This script assumes it is being run from the simulation directory.
Assumes x and t are equispaced.
"""

import os
import pencil as pc
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
import scipy.optimize

class plot_container():
	def __init__(self, fig, ax, im, savedir="."):
		self.fig = fig
		self.ax = ax
		self.im = im
		self.cbar = colorbar
		self.savedir = savedir
	
	def save(self, name, **kwargs):
		loc = os.path.join(self.savedir, name)
		loc_dir = os.path.dirname(loc)
		if not os.path.exists(loc_dir):
			os.makedirs(loc_dir)
		self.fig.savefig(loc, **kwargs)

class contourplot_container(plot_container):
	def __init__(self, fig, ax, im, colorbar, savedir="."):
		self.fig = fig
		self.ax = ax
		self.im = im
		self.cbar = colorbar
		self.savedir = savedir

class disp_rel_from_yaver():
	def __init__(self,
		simdir=".", #Location of the simulation to be read
		t_min=300, #For all calculations, only use data saved after this time
		t_max=None, #For all calculations, only use data saved before this time
		k_tilde_min = 0, #plot limit
		k_tilde_max = 20, #plot limit
		omega_tilde_min = 0, #plot limit
		omega_tilde_max = 10, #plot limit
		fig_savedir = ".", #Where to save the figures
		field_name = 'uzmxz', #which field to use to plot the dispersion relation
		cbar_label = r"$\tilde{{\omega}} \hat{{u}} / D^2$", #label to use for the colorbar
		):
		
		sim = pc.sim.get(simdir, quiet=True)
		
		self.simdir = simdir
		self.datadir = sim.datadir
		self.param = pc.read.param(datadir=self.datadir)
		self.grid = pc.read.grid(datadir=self.datadir, trim=True, quiet=True)
		self.t_min = t_min
		self.t_max = t_max
		self.k_tilde_min = k_tilde_min
		self.k_tilde_max = k_tilde_max
		self.omega_tilde_min = omega_tilde_min
		self.omega_tilde_max = omega_tilde_max
		self.fig_savedir = fig_savedir
		self.field_name = field_name
		self.cbar_label = cbar_label
		
		self.read()
		self.do_ft()
		self.get_scales()
	
	def read(self):
		self.ts = pc.read.ts(datadir=self.datadir, quiet=True)
		#TODO: handle hdf5 read errors below (just output a warning on each fail and retry)?
		if self.t_max is None:
			t_max = self.ts.t[-1]
		else:
			t_max = self.t_max
		
		self.av_y = pc.read.aver(
			datadir=self.datadir,
			simdir=self.simdir,
			plane_list=['y'],
			var_names=[self.field_name],
			time_range=[self.t_min, t_max],
			)
		self.av_xy = pc.read.aver(
			datadir=self.datadir,
			simdir=self.simdir,
			plane_list=['xy'],
			time_range=[self.t_min, t_max],
			)
		
	def do_ft(self):
		fftshift = scipy.fft.fftshift
		fftfreq = scipy.fft.fftfreq
		x = self.grid.x
		z = self.grid.z
		t = self.av_y.t
		uz = getattr(self.av_y.y, self.field_name)
		
		assert np.shape(uz) == (len(t), len(z), len(x))
		
		uz_fft = scipy.fft.fft2(uz, norm='forward', axes=[0,2])
		uz_fft = fftshift(uz_fft, axes=[0,2])
		self.uz_fft = np.transpose(uz_fft, axes=[0,2,1]) #Move the z-axis to the end.
		n_omega, n_kx, _ = np.shape(self.uz_fft)
		
		self.omega = 2*np.pi*fftshift(fftfreq(n_omega, d = (max(t)-min(t))/n_omega ))
		self.kx = 2*np.pi*fftshift(fftfreq(n_kx, d = (max(x)-min(x))/n_kx ))
	
	def get_scales(self):
		"""
		Calculate quantities used to normalize k, omega, and uz.
		NOTE: currently I am not using a depth-dependent normalization, since we would like to compare amplitudes of modes at different depths.
		"""
		cs_d = np.sqrt(self.param.cs2cool)
		g = np.abs(self.param.gravz)
		self.L_0 = cs_d**2/g
		self.omega_0 = g/cs_d
		
		urms = np.sqrt(np.average(self.av_xy.xy.uz2mz, axis=0))
		urms = np.max(urms) #Choosing the peak urms since I don't want the normalization to be depth-dependent.
		self.D = urms/self.omega_0
	
	def prep_data_for_plot(self, z):
		omega_0 = self.omega_0
		L_0 = self.L_0
		D = self.D
		
		#Find out which segments of the arrays are needed for the plot.
		ikx_max = np.argmin(np.abs(self.k_tilde_max - self.kx*L_0))
		ikx_min = np.argmin(np.abs(self.k_tilde_min - self.kx*L_0))
		iomega_max = np.argmin(np.abs(self.omega_tilde_max - self.omega/omega_0))
		iomega_min = np.argmin(np.abs(self.omega_tilde_min - self.omega/omega_0))
		
		kx = self.kx[ikx_min:ikx_max]
		omega = self.omega[iomega_min:iomega_max]
		uz_fft = self.uz_fft[iomega_min:iomega_max,ikx_min:ikx_max]
		
		iz_surf = np.argmin(np.abs(z - self.grid.z))
		
		data = np.abs(omega[:,None]*uz_fft[:,:,iz_surf]/(omega_0*D**2)) #NOTE: multiplying by omega to take 'running difference'
		data = np.where(data == 0, np.nan, data) #replace 0 with nan so that log scaling works.
		
		return kx*L_0, omega/omega_0, data
	
	def plot_komega(self, z):
		k_tilde, omega_tilde, data = self.prep_data_for_plot(z)
		
		fig,ax = plt.subplots()
		im = ax.contourf(
			k_tilde,
			omega_tilde,
			data,
			levels = np.logspace(
				np.log10(np.nanmin(data)),
				np.log10(np.nanmax(data)),
				1001
				),
			norm = mpl.colors.LogNorm(),
			)
		
		c = plt.colorbar(
			im,
			ax = ax,
			format = mpl.ticker.LogFormatterSciNotation(),
			ticks = mpl.ticker.LogLocator(),
			)
		ax.set_title(f"z = {z:.2f}")
		ax.set_xlabel(r"$\widetilde{{k}}_x$")
		ax.set_ylabel(r"$\widetilde{{\omega}}$")
		c.set_label(r"$\hat{{u}} \widetilde{{\omega}} / D^2$")
		
		fig.tight_layout()
		
		return contourplot_container(fig, ax, im, c, savedir=self.fig_savedir)
	
	def get_data_at_kz(self, k_tilde, z, omega_tilde_min=None, omega_tilde_max=None, abs=True):
		"""
		Get the values of omega_tilde and P(omega_tilde) at specified k_tilde and z.in the range omega_tilde_min < omega_tilde < omega_tilde_max
		
		Arguments:
			k_tilde: float
			z: float
			omega_tilde_min: float
			omega_tilde_max: float
			abs: bool, whether to take the absolute value of the data before returning.
		
		Returns:
			omt_near_target: numpy array of float
			data_near_target: numpy array of float
		"""
		if omega_tilde_min is None:
			omega_tilde_min = self.omega_tilde_min
		if omega_tilde_max is None:
			omega_tilde_max = self.omega_tilde_max
		
		iz = np.argmin(np.abs(z - self.grid.z))
		ik = np.argmin(np.abs(k_tilde - self.kx*self.L_0))
		data = self.omega*self.uz_fft[:,ik,iz]/(self.omega_0*self.D**2)
		
		if abs:
			data = np.abs(data)
		
		om_tilde = self.omega/self.omega_0
		
		if omega_tilde_min < np.min(om_tilde):
			raise ValueError(f"omega_tilde_min ({omega_tilde_min:.2e}) needs to be greater than the minimum value of omega_tilde ({np.min(om_tilde):.2e}).")
		if omega_tilde_max > np.max(om_tilde):
			raise ValueError(f"omega_tilde_max ({omega_tilde_max:.2e}) needs to be less than the maximum value of omega_tilde ({np.max(om_tilde):.2e}).")
		
		i_min = np.argmin(np.abs(om_tilde - omega_tilde_min))
		i_max = np.argmin(np.abs(om_tilde - omega_tilde_max))
		data_near_target = data[i_min:i_max]
		omt_near_target = om_tilde[i_min:i_max]
		
		return omt_near_target, data_near_target

class disp_rel_nonorm_from_yaver(disp_rel_from_yaver):
	def get_scales(self):
		cs_d = np.sqrt(self.param.cs2cool)
		g = np.abs(self.param.gravz)
		self.L_0 = cs_d**2/g
		self.omega_0 = g/cs_d
		
		self.D = 1

def oplot_dr_f(dr, plot=None, ax=None):
	"""
	Overplot the dispersion relation corresponding to the f mode
	
	Arguments:
		dr: disp_rel_from_yaver instance
		plot: contourplot_container instance
	
	Dispersion relation is omega^2 = g*kx (note that the density contrast is not accounted for here)
	"""
	if plot is not None:
		ax = plot.ax
	elif ax is None:
		raise ValueError("Need to pass an Axes instance to plot on.")
	
	g = np.abs(dr.param.gravz)
	k_tilde = np.linspace(*ax.get_xlim(), 100)
	kx = k_tilde/dr.L_0
	omega = np.sqrt(g*kx)
	omega_tilde = omega/dr.omega_0
	return ax.plot(k_tilde, omega_tilde, ls='--', c='k', alpha=0.3)

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
	):
	"""
	Given a disp_rel_from_yaver instance, find the amplitude of a particular mode as a function of depth
	
	Arguments:
		dr: disp_rel_from_yaver instance
		k_tilde: float, wavenumber at which to find the amplitude
		z: float, z-coordinate at which to read the data
		om_tilde_min: float. Consider the part of the data above this frequency
		om_tilde_max: float. Consider the part of the data below this frequency
		poly_order: int. Order of polynomial to use to fit the continuum.
		n_lorentz: int. Number of Lorentzian profiles to use for fitting.
		om_guess: list of float. Guesses for the omega_tilde at which modes are present. Length need not be the same as n_lorentz.
	
	Returns:
		model: make_model instance. This will have an attribute popt that gives the optimal fit values. To plot the resulting model returned by this function, you can do plt.plot(omt_near_target, model(omt_near_target, *model.popt))
	"""
	omt_near_target, data_near_target = dr.get_data_at_kz(k_tilde, z, omega_tilde_min=om_tilde_min, omega_tilde_max=om_tilde_max)
	
	model = make_model(poly_order, n_lorentz)
	
	#initial guess for the parameters.
	guess_poly = np.zeros(model.poly_order + 1)
	guess_lor = np.zeros((model.n_lorentz,3))
	
	if model.n_lorentz > 0:
		guess_lor[0,0] = np.max(data_near_target)
	guess_lor[:,1] = np.linspace(om_tilde_min, om_tilde_max, model.n_lorentz+2)[1:-1]
	if om_guess is not None:
		for i in range(min(model.n_lorentz, len(om_guess))):
			guess_lor[i,1] = om_guess[i]
	
	guess_lor[:,2] = (om_tilde_max - om_tilde_min)/2
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
	ubound_lor[:,2] = om_tilde_max - om_tilde_min
	ubound = model.pack_params(ubound_poly, ubound_lor)
	
	#A crude guess for sigma
	sigma = stdev_central(data_near_target, 0.05)
	
	model.popt, model.pcov = scipy.optimize.curve_fit(
		model,
		omt_near_target,
		data_near_target,
		p0 = guess,
		sigma = np.full_like(data_near_target, sigma),
		bounds = (lbound,ubound),
		)
	
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
	):
	"""
	Keep on increasing n_lorentz in fit_mode until the fit no longer improves.
	
	Arguments:
		dr: disp_rel_from_yaver instance
		k_tilde: float
		z: float
		om_tilde_min: float
		om_tilde_max: float
		poly_order: int. Order of the polynomial to use for fitting the continuum.
		n_lorentz_max: int. Maximum number of Lorentzians that can be used in the fit.
		threshold: float. Ratio of reduced chi-squared needed to accept addition of a Lorentzian.
	"""
	
	omt_near_target, data_near_target = dr.get_data_at_kz(k_tilde, z, omega_tilde_min=om_tilde_min, omega_tilde_max=om_tilde_max)
	
	sigma = stdev_central(data_near_target, 0.05)
	
	#Function to calculate the reduced chi-square corresponding to a particular fit.
	chi2r = lambda fit: np.sum(((data_near_target - fit(omt_near_target, *fit.popt) )/sigma)**2)/(len(data_near_target) - fit.nparams)
	
	fit_old = None
	for n_lorentz in range(n_lorentz_max):
		fit = fit_mode(dr, k_tilde, z, om_tilde_min, om_tilde_max, poly_order, n_lorentz)
		
		if (fit_old is not None) and chi2r(fit)/chi2r(fit_old) > threshold:
			return fit_old
		
		fit_old = fit
	
	raise RuntimeError(f"Improvement in fit has not converged even with {n_lorentz = }")

def get_mode_eigenfunction(dr, omega_0, k_tilde, z_list, om_tilde_min, om_tilde_max, poly_order=1):
	"""
	Use fit_mode to get the z-dependent eigenfunction of the mode whose frequency (omega_tilde) is close to omega_0 at k_tilde.
	"""
	if not om_tilde_min < omega_0 < om_tilde_max:
		raise ValueError("Cannot fit mode that is outside search band.")
	
	P_list = []
	for z in z_list:
		fit = fit_mode_auto(dr, k_tilde, z, om_tilde_min, om_tilde_max, poly_order=poly_order)
		_, params_lorentz = fit.unpack_params(fit.popt)
		
		omt_near_target, _ = dr.get_data_at_kz(k_tilde, z, omega_tilde_min=om_tilde_min, omega_tilde_max=om_tilde_max)
		
		if len(params_lorentz) > 0:
			domega = omt_near_target[1] - omt_near_target[0]
			
			d_from_om0 = np.abs(omega_0 - params_lorentz[:,1]) #how far each detected mode is from omega_0
			imode = np.argmin(d_from_om0)
			
			omega_c = params_lorentz[imode,1]
			d_from_omc = np.abs(omega_c - params_lorentz[:,1])
			
			mode_mass = 0
			#TODO: check with Nishant if summing the peaks as below is the right thing to do.
			for i in range(fit.n_lorentz):
				if np.abs(params_lorentz[i,1] - omega_c) < domega:
					mode = fit.lorentzian(omt_near_target, *params_lorentz[i])
					mode_mass += np.trapz(mode, omt_near_target)
			
		else:
			"""
			NOTE
			Setting these directly to zero leads to jarring discontinuities in the plot of the mode eigenfunction. It feels dishonest to add an extra lorentzian there by hand and then get a fit, so I shall just set them to nan to indicate that the amplitude of the mode was too close to the noise threshold to say anything.
			"""
			mode_mass = np.nan
		
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

def stdev_central(arr, frac):
	"""
	Estimate standard derivation of an array arr, considering only values between the frac*100 and (1-frac)*100 percentiles
	
	Arguments:
		arr: 1D numpy array
		frac: float
	
	Returns:
		stdev: same as arr.dtype
	"""
	sort = np.sort(arr)
	n = len(arr)
	i_min = int(np.round(n*frac))
	i_max = int(np.round(n*(1-frac)))
	cut = sort[i_min:i_max]
	return np.std(cut)

if __name__ == "__main__":
	dr = disp_rel_from_yaver()
	dr.plot_komega(1)
	
	plt.show()
