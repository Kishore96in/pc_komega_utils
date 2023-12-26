"""
This script assumes it is being run from the simulation directory.
Assumes x, y, and t are equispaced.

Class naming scheme
	disp_rel_from_yaver: calculate dispersion relation from y averaged data
	disp_rel_from_dvar: calculate dispersion relation from downsampled snapshots
	
	scalesMixin*: definitions of omega_0 and L_0
	scalesMixin_data*: definitions of how the data should be normalized

Functions:
	fit_mode
	fit_mode_auto
	get_mode_eigenfunction
"""

import os
import warnings
import pencil as pc
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
import scipy.optimize
import numbers
import collections
import abc

from dataclasses import dataclass

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

class disp_rel(metaclass=abc.ABCMeta):
	@property
	@abc.abstractmethod
	def cbar_label_default(self):
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def field_name_default(self):
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def data_axes(self):
		raise NotImplementedError
	
	@abc.abstractmethod
	def read(self):
		raise NotImplementedError
	
	@abc.abstractmethod
	def do_ft(self):
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def omega_0(self):
		raise NotImplementedError
	
	@property
	@abc.abstractmethod
	def L_0(self):
		raise NotImplementedError
	
	def __init__(self,
		simdir=".", #Location of the simulation to be read
		t_min=300, #For all calculations, only use data saved after this time
		t_max=None, #For all calculations, only use data saved before this time
		k_tilde_min = 0, #plot limit
		k_tilde_max = 20, #plot limit
		omega_tilde_min = 0, #plot limit
		omega_tilde_max = 10, #plot limit
		fig_savedir = ".", #Where to save the figures
		field_name = None, #which field to use to plot the dispersion relation
		cbar_label = None, #label to use for the colorbar
		):
		if cbar_label is None:
			cbar_label = self.cbar_label_default
		if field_name is None:
			field_name = self.field_name_default
		
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
		
		#Sanity checks
		for k in self.data_axes.keys():
			if not hasattr(self, k):
				raise AttributeError(f"Key {k} in data_axes is not an attribute.")
	
	def contourplotter(self, x, y, data):
		fig,ax = plt.subplots()
		im = ax.contourf(
			x,
			y,
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
		
		return contourplot_container(fig, ax, im, c, savedir=self.fig_savedir)
	
	@staticmethod
	def _generate_slicer(omega, omega_list):
		"""
		Choose the slice of omega_list corresponding to the given omega. Returns an object which can be used to index an array. Argument omega can be either None (select the entire range), a float, or a tuple or two floats.
		
		Slicing with the return of this function will not reduce the number of axes in the array.
		"""
		omega_list = np.array(omega_list)
		if omega is None:
			return slice(None)
		elif isinstance(omega, collections.abc.Iterable) and len(omega) == 2:
			i_min = np.argmin(np.abs(omega[0] - omega_list))
			i_max = np.argmin(np.abs(omega[1] - omega_list))
			return slice(i_min, i_max)
		elif isinstance (omega, numbers.Number):
			i = np.argmin(np.abs(omega - omega_list))
			return [i]
		else:
			raise ValueError(f"Unable to handle {type(omega) = }")
	
	def get_slice(self, **kwargs):
		"""
		Slice data in terms of physical values
		
		Each arg can be either a float (get the data at that particular value of the specified parameter) or a tuple of two floats (get the data in that range of the specified parameter). Each argument needs to be a keyword, corresponding to the keys of data_axes.
		"""
		data = self.data
		
		coords = [None for i in range(data.ndim)]
		for name, i in self.data_axes.items():
			coord_list = getattr(self, name)
			if name in kwargs.keys():
				coord_val = kwargs[name]
			else:
				coord_val = None
			sl = self._generate_slicer(coord_val, coord_list)
			
			data = np.moveaxis(data, i, 0)
			data = data[sl]
			data = np.moveaxis(data, 0, i)
			
			coords[i] = coord_list[sl]
		
		return data, coords

class scalesMixin_databyD2():
	def scale_data(self, data):
		urms = np.sqrt(np.average(self.av_xy.xy.uz2mz, axis=0))
		urms = np.max(urms) #Choosing the peak urms since I don't want the normalization to be depth-dependent.
		D = urms/self.omega_0
		return np.abs(data)/D**2

class scalesMixin_dataRDbyD2():
	def scale_data(self, data):
		urms = np.sqrt(np.average(self.av_xy.xy.uz2mz, axis=0))
		urms = np.max(urms) #Choosing the peak urms since I don't want the normalization to be depth-dependent.
		D = urms/self.omega_0
		
		data = np.moveaxis(data, self.data_axes['omega_tilde'], -1) # for broadcasting
		#NOTE: multiplying by omega to take 'running difference'
		data = np.abs(self.omega_tilde * data)/D**2
		data = np.moveaxis(data, -1, self.data_axes['omega_tilde'])
		return data

class scalesMixin_SBC15(scalesMixin_dataRDbyD2):
	"""
	Use the length and frequency scales defined by Singh et al, 2015.
	"""
	@property
	def L_0(self):
		cs_d = np.sqrt(self.param.cs2cool)
		g = np.abs(self.param.gravz)
		return cs_d**2/g
	
	@property
	def omega_0(self):
		cs_d = np.sqrt(self.param.cs2cool)
		g = np.abs(self.param.gravz)
		return g/cs_d

class scalesMixin_L0HP():
	"""
	Here, L_0 is set as the pressure scale height
	"""
	@property
	def L_0(self):
		gamma = self.param.gamma
		cs_d = np.sqrt(self.param.cs2cool)
		g = np.abs(self.param.gravz)
		return cs_d**2/(g*gamma)
	
	@property
	def omega_0(self):
		cs_d = np.sqrt(self.param.cs2cool)
		g = np.abs(self.param.gravz)
		return g/cs_d

class disp_rel_from_yaver(scalesMixin_SBC15, disp_rel):
	@property
	def data_axes(self):
		return {'omega_tilde':0, 'kx_tilde':1, 'z':2}
		
	@property
	def cbar_label_default(self):
		return  r"$\tilde{{\omega}} \hat{{u}} / D^2$"
	
	@property
	def field_name_default(self):
		return "uzmxz"
	
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
		Lx = self.grid.Lx
		z = self.grid.z
		t = self.av_y.t
		uz = getattr(self.av_y.y, self.field_name)
		
		assert np.shape(uz) == (len(t), len(z), len(x))
		
		uz_fft = scipy.fft.fftn(uz, norm='forward', axes=[0,2])
		uz_fft = fftshift(uz_fft, axes=[0,2])
		data = np.transpose(uz_fft, axes=[0,2,1]) #Move the z-axis to the end.
		n_omega, n_kx, _ = np.shape(data)
		
		self.omega = 2*np.pi*fftshift(fftfreq(n_omega, d = (max(t)-min(t))/n_omega ))
		self.kx = 2*np.pi*fftshift(fftfreq(n_kx, d = Lx/n_kx ))
		self.data = self.scale_data(data)
	
	def prep_data_for_plot(self, z):
		"""
		Return k_tilde, omega_tilde, and the normalized Fourier-transformed vertical velocity at a given height z.
		"""
		data, [omega_tilde, kx_tilde, _] = self.get_slice(
			kx_tilde = (self.k_tilde_min, self.k_tilde_max),
			omega_tilde = (self.omega_tilde_min, self.omega_tilde_max),
			z = z,
			)
		
		data = data[:,:,0]
		data = np.where(data == 0, np.nan, data) #replace 0 with nan so that log scaling works.
		
		return kx_tilde, omega_tilde, data
	
	def plot_komega(self, z):
		"""
		Plot the k-omega diagram at a given height z.
		"""
		k_tilde, omega_tilde, data = self.prep_data_for_plot(z)
		p = self.contourplotter(k_tilde, omega_tilde, data)
		
		p.ax.set_title(f"$z = {z:.2f}$")
		p.ax.set_xlabel(r"$\widetilde{{k}}_x$")
		p.ax.set_ylabel(r"$\widetilde{{\omega}}$")
		p.cbar.set_label(self.cbar_label)
		
		p.fig.tight_layout()
		return p
	
	def get_data_at_kz(self, k_tilde, z, omega_tilde_min=None, omega_tilde_max=None):
		"""
		Get the values of omega_tilde and P(omega_tilde) at specified k_tilde and z in the range omega_tilde_min < omega_tilde < omega_tilde_max.
		
		Arguments:
			k_tilde: float
			z: float
			omega_tilde_min: float
			omega_tilde_max: float
		
		Returns:
			omt_near_target: numpy array of float
			data_near_target: numpy array of float
		"""
		if omega_tilde_min is None:
			omega_tilde_min = self.omega_tilde_min
		if omega_tilde_max is None:
			omega_tilde_max = self.omega_tilde_max
		
		if omega_tilde_min < np.min(self.omega_tilde):
			raise ValueError(f"omega_tilde_min ({omega_tilde_min:.2e}) needs to be greater than the minimum value of omega_tilde ({np.min(self.omega_tilde):.2e}).")
		if omega_tilde_max > np.max(self.omega_tilde):
			raise ValueError(f"omega_tilde_max ({omega_tilde_max:.2e}) needs to be less than the maximum value of omega_tilde ({np.max(self.omega_tilde):.2e}).")
		
		data_near_target, [omt_near_target, _, _] = self.get_slice(
			omega_tilde=(omega_tilde_min, omega_tilde_max),
			kx_tilde = k_tilde,
			z = z
			)
		
		data_near_target = omt_near_target*data_near_target[:,0,0]
		
		return omt_near_target, data_near_target
	
	@property
	def omega_tilde(self):
		return self.omega/self.omega_0
	
	@property
	def kx_tilde(self):
		return self.kx*self.L_0
	
	@property
	def z(self):
		return self.grid.z

class disp_rel_nonorm_from_yaver(disp_rel_from_yaver):
	@property
	def cbar_label_default(self):
		return  r"$\tilde{{\omega}} \hat{{u}}$"
	
	def scale_data(self, data):
		data = np.moveaxis(data, self.data_axes['omega_tilde'], -1) # for broadcasting
		#NOTE: multiplying by omega to take 'running difference'
		data = np.abs(self.omega_tilde * data)
		data = np.moveaxis(data, -1, self.data_axes['omega_tilde'])
		return data

class disp_rel_from_yaver_L0_HP(scalesMixin_L0HP, disp_rel_from_yaver):
	pass

@dataclass
class fake_grid:
	x: np.ndarray
	y: np.ndarray
	z: np.ndarray

class disp_rel_from_dvar(scalesMixin_dataRDbyD2, scalesMixin_L0HP, disp_rel):
	"""
	Read downsampled snapshots and plot dispersion relations from them.
	"""
	@property
	def data_axes(self):
		return {'omega_tilde':0, 'kx_tilde':1, 'ky_tilde':2, 'z':3}
		
	@property
	def cbar_label_default(self):
		return  r"$\tilde{{\omega}} \hat{{u}} / D^2$"
	
	@property
	def field_name_default(self):
		return "uz"
	
	@property
	def omega_tilde(self):
		return self.omega/self.omega_0
	
	@property
	def kx_tilde(self):
		return self.kx*self.L_0
	
	@property
	def ky_tilde(self):
		return self.ky*self.L_0
	
	@property
	def z(self):
		return self.grid_d.z
	
	def read(self):
		sim = pc.sim.get(self.simdir, quiet=True)
		
		self.ts = pc.read.ts(sim=sim, quiet=True)
		if self.t_max is None:
			t_max = self.ts.t[-1]
		else:
			t_max = self.t_max
		
		vard = []
		t_vard = []
		
		if sim.param['io_strategy'] == "HDF5":
			proc_folder = "allprocs"
			extension = ".h5"
		elif sim.param['io_strategy'] == "dist":
			proc_folder = "proc0"
			extension = ""
		else:
			raise NotImplemented(f"Unsupported io_strategy {sim.param['io_strategy']}")
		
		#Get list of downsampled snapshots with corresponding times.
		snap_list = np.loadtxt(os.path.join(sim.datadir, proc_folder, "varN_down.list"), dtype=np.dtype([('name', str, 20), ('time', float)]))
		snap_list = np.unique(snap_list, axis=0)
		snap_list = np.sort(snap_list, axis=0, order='time')
		
		for varname, t in snap_list:
			if self.t_min < t < t_max:
				print(f"Reading {varname}", end='\r') #progress report
				var = pc.read.var(trimall=True, var_file=f"{varname}{extension}", sim=sim)
				vard.append(getattr(var, self.field_name))
				t_vard.append(var.t)
				
				if not np.isclose(t, var.t):
					raise RuntimeError(f"Snapshot time disagrees with the time in varN_down.list. {varname = }, {t = }")
		
		self.vard = np.array(vard)
		self.t_vard = np.array(t_vard)
		self.grid_d = fake_grid(x=var.x, y=var.y, z=var.z)
		
		self.av_xy = pc.read.aver(
			datadir=self.datadir,
			simdir=self.simdir,
			plane_list=['xy'],
			time_range=[self.t_min, t_max],
			)
	
	def do_ft(self):
		fftshift = scipy.fft.fftshift
		fftfreq = scipy.fft.fftfreq
		x = self.grid_d.x
		y = self.grid_d.y
		z = self.grid_d.z
		Lx = self.grid.Lx
		Ly = self.grid.Ly
		t = self.t_vard
		uz = self.vard
		
		assert np.shape(uz) == (len(t), len(z), len(y), len(x))
		
		uz_fft = scipy.fft.fftn(uz, norm='forward', axes=[0,2,3])
		uz_fft = fftshift(uz_fft, axes=[0,2,3])
		data = np.transpose(uz_fft, axes=[0,3,2,1])
		n_omega, n_kx, n_ky, _ = np.shape(data)
		
		self.omega = 2*np.pi*fftshift(fftfreq(n_omega, d = (max(t)-min(t))/n_omega ))
		self.kx = 2*np.pi*fftshift(fftfreq(n_kx, d = Lx/n_kx ))
		self.ky = 2*np.pi*fftshift(fftfreq(n_ky, d = Ly/n_ky ))
		self.data = self.scale_data(data)
	
	def plot_komega(self, z):
		"""
		Plot the normalized Fourier-transformed vertical velocity vs (kx_tilde, omega_tilde) at a given height z and ky_tilde=0.
		"""
		data, [omega_tilde, kx_tilde, _, _] = self.get_slice(
			kx_tilde = (self.k_tilde_min, self.k_tilde_max),
			omega_tilde = (self.omega_tilde_min, self.omega_tilde_max),
			ky_tilde = 0,
			z = z,
			)
		
		data = data[:,:,0,0]
		data = np.where(data == 0, np.nan, data) #replace 0 with nan so that log scaling works.
		
		p = self.contourplotter(kx_tilde, omega_tilde, data)
		
		p.ax.set_title(f"$z = {z:.2f}$")
		p.ax.set_xlabel(r"$\widetilde{{k}}_x$")
		p.ax.set_ylabel(r"$\widetilde{{\omega}}$")
		p.cbar.set_label(self.cbar_label)
		
		p.fig.tight_layout()
		return p
	
	def plot_kyomega(self, z):
		"""
		Plot the normalized Fourier-transformed vertical velocity vs (ky_tilde, omega_tilde) at a given height z and kx_tilde=0.
		"""
		data, [omega_tilde, kx_tilde, _, _] = self.get_slice(
			kx_tilde = 0,
			omega_tilde = (self.omega_tilde_min, self.omega_tilde_max),
			ky_tilde = (self.k_tilde_min, self.k_tilde_max),
			z = z,
			)
		
		data = data[:,0,:,0]
		data = np.where(data == 0, np.nan, data) #replace 0 with nan so that log scaling works.
		
		p = self.contourplotter(ky_tilde, omega_tilde, data)
		
		p.ax.set_title(f"z = {z:.2f}")
		p.ax.set_xlabel(r"$\widetilde{{k}}_y$")
		p.ax.set_ylabel(r"$\widetilde{{\omega}}$")
		p.cbar.set_label(self.cbar_label)
		
		p.fig.tight_layout()
		return p
	
	def plot_ring(self, z, omega_tilde):
		"""
		Plot the normalized Fourier-transformed vertical velocity vs (kx_tilde, ky_tilde) at a given height z and angular frequency omega_tilde.
		"""
		data, [[om_tilde], kx_tilde, ky_tilde, _] = self.get_slice(
			kx_tilde = (self.k_tilde_min, self.k_tilde_max),
			omega_tilde = omega_tilde,
			ky_tilde = (self.k_tilde_min, self.k_tilde_max),
			z = z,
			)
		
		data = data[0,:,:,0]
		data = np.where(data == 0, np.nan, data) #replace 0 with nan so that log scaling works.
		
		p = self.contourplotter(kx_tilde, ky_tilde, data)
		
		p.ax.set_title(f"z = {z:.2f}")
		p.ax.set_xlabel(r"$\widetilde{{k}}_y$")
		p.ax.set_ylabel(r"$\widetilde{{k}}_x$")
		p.cbar.set_label(self.cbar_label)
		
		p.fig.tight_layout()
		return p

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
	gamma_max = None,
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
		gamma_max: float. Upper limit on the width of the Lorentzians. By default, om_tilde_max = om_tilde_min.
	
	Returns:
		model: make_model instance. This will have an attribute popt that gives the optimal fit values. To plot the resulting model returned by this function, you can do plt.plot(omt_near_target, model(omt_near_target, *model.popt))
	"""
	if gamma_max is None:
		gamma_max = om_tilde_max - om_tilde_min
	
	omt_near_target, data_near_target = dr.get_data_at_kz(k_tilde, z, omega_tilde_min=om_tilde_min, omega_tilde_max=om_tilde_max)
	
	model = make_model(poly_order, n_lorentz)
	
	#initial guess for the parameters.
	guess_poly = np.zeros(model.poly_order + 1)
	guess_lor = np.zeros((model.n_lorentz,3))
	
	if model.n_lorentz > 0:
		guess_lor[0,0] = np.max(np.abs(data_near_target))
	
	guess_lor[:,1] = np.linspace(om_tilde_min, om_tilde_max, model.n_lorentz+2)[1:-1]
	if om_guess is not None:
		for i in range(min(model.n_lorentz, len(om_guess))):
			guess_lor[i,1] = om_guess[i]
	
	guess_lor[:,2] = gamma_max/2
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
	
	#A crude guess for sigma
	sigma = stdev_central(data_near_target, 0.05)
	
	if sigma == 0:
		"""
		This can only happen if np.all(data_near_target == 0), in which case we set sigma=1 to prevent divide-by-zero errors.
		"""
		sigma = 1
	
	model.popt, model.pcov = scipy.optimize.curve_fit(
		model,
		omt_near_target,
		data_near_target,
		p0 = guess,
		sigma = np.full_like(data_near_target, sigma),
		bounds = (lbound,ubound),
		maxfev=int(1e4),
		absolute_sigma = True,
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
	om_guess = None,
	gamma_max = None,
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
		om_guess: list of float. Passed to fit_mode.
		gamma_max: float. Passed to fit_mode.
	"""
	
	omt_near_target, data_near_target = dr.get_data_at_kz(k_tilde, z, omega_tilde_min=om_tilde_min, omega_tilde_max=om_tilde_max)
	
	sigma = stdev_central(data_near_target, 0.05)
	
	if sigma == 0:
		"""
		This can only happen if np.all(data_near_target == 0), in which case we set sigma=1 to prevent divide-by-zero errors.
		"""
		sigma = 1
	
	#Function to calculate the reduced chi-square corresponding to a particular fit.
	chi2r = lambda fit: np.sum(((data_near_target - fit(omt_near_target, *fit.popt) )/sigma)**2)/(len(data_near_target) - fit.nparams)
	
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
		if (fit_old is not None) and c/c_old > threshold:
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
	):
	"""
	Use fit_mode to get the z-dependent eigenfunction of the mode whose frequency (omega_tilde) is close to omega_0 at k_tilde.
	
	Arguments:
		dr: disp_rel_from_yaver instance.
		omega_0: float
		k_tilde: float.
		z_list: list of float. Values of z at which to get the eigenfunction.
		om_tilde_min: float. Lower limit of the band of omega_tilde in which to fit the data.
		om_tilde_max: float. Upper limit of the band of omega_tilde in which to fit the data.
		poly_order: int. Order of the polynomial to use for fitting the continuum.
		force_n_lorentz: int. Force this many Lorentizans to be used for the fitting (rather than automatically determining based on the data). If this is set to None (default), the number of Lorentzians will be automatically determined.
		omega_tol: float or None. If (not None) and (the distance between the detected mode and omega_0) is greater than or equal to this value, do not consider that mode for computation of the mode mass.
		gamma_max: float. See fit_mode.
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
		
		omt_near_target, data_near_target = dr.get_data_at_kz(k_tilde, z, omega_tilde_min=om_tilde_min, omega_tilde_max=om_tilde_max)
		
		if len(params_lorentz) > 0:
			domega = omt_near_target[1] - omt_near_target[0]
			
			imode = np.argmin(np.abs(omega_0 - params_lorentz[:,1]))
			omega_c = params_lorentz[imode,1]
			
			mode_mass = 0
			#TODO: check with Nishant if summing the peaks as below is the right thing to do.
			if np.abs(omega_c - omega_0) < omega_tol:
				for i in range(fit.n_lorentz):
					if np.abs(params_lorentz[i,1] - omega_c) < domega:
						mode = fit.lorentzian(omt_near_target, *params_lorentz[i])
						mode_mass += np.trapz(mode, omt_near_target)
			
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
