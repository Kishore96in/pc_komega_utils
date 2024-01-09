"""
Assumes x, y, and t are equispaced.

Class naming scheme
	dr_*: read simulation data and plot k-omega diagrams
	m_scl_*: mixin classes defining omega_0 and L_0
	m_dscl_*: mixin classes defining how the data should be normalized
"""

import os
import warnings
import pencil as pc
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
import matplotlib as mpl
import numbers
import collections
import abc

from dataclasses import dataclass

from .utils import smooth_tophat

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

class dr_base(metaclass=abc.ABCMeta):
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
	
	@property
	def t_min(self):
		return getattr(self, "_t_min", None)
	
	@t_min.setter
	def t_min(self, _):
		raise AttributeError("Use set_t_range to change t_min")
	
	@property
	def t_max(self):
		return getattr(self, "_t_max", None)
	
	@t_max.setter
	def t_max(self, _):
		raise AttributeError("Use set_t_range to change t_max")
	
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
		self.dim = pc.read.dim(datadir=self.datadir)
		self.grid = pc.read.grid(datadir=self.datadir, trim=True, quiet=True)
		self.k_tilde_min = k_tilde_min
		self.k_tilde_max = k_tilde_max
		self.omega_tilde_min = omega_tilde_min
		self.omega_tilde_max = omega_tilde_max
		self.fig_savedir = fig_savedir
		self.field_name = field_name
		self.cbar_label = cbar_label
		
		self.read()
		self.set_t_range(t_min, t_max)
		
		#Sanity checks
		for k in self.data_axes.keys():
			if not hasattr(self, k):
				raise AttributeError(f"Key {k} in data_axes is not an attribute.")
	
	def contourplotter(self, x, y, data):
		if np.all(data == 0):
			raise RuntimeError("The selected slice is all zeros")
		if np.shape(data) != (len(x), len(y)):
			raise ValueError(f"data array needs to have shape [len(x), len(y)].")
		
		data = data.transpose()
		data = np.where(data == 0, np.nan, data) #replace 0 with nan so that log scaling works.
		
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
			if omega[1] <= omega[0]:
				raise ValueError("Right limit of range is not greater than left limit.")
			
			i_min = np.argmin(np.abs(omega[0] - omega_list))
			i_max = np.argmin(np.abs(omega[1] - omega_list))
			
			if i_max <= i_min:
				raise ValueError(f"Given range ({omega[0]}, {omega[1]}) is less than the grid spacing ({omega_list[1] - omega_list[0]}).")
			if i_min > len(omega_list) - 1:
				raise ValueError("Requested range is outside the coordinate bounds.")
			
			return slice(i_min, i_max)
		elif isinstance (omega, numbers.Number):
			i = np.argmin(np.abs(omega - omega_list))
			return [i]
		else:
			raise ValueError(f"Unable to handle {type(omega) = }")
	
	def get_slice(self, data=None, compress=False, **kwargs):
		"""
		Slice data in terms of physical values
		
		Each arg can be either a float (get the data at that particular value of the specified parameter) or a tuple of two floats (get the data in that range of the specified parameter). Each argument needs to be a keyword, corresponding to the keys of data_axes.
		
		Arguments:
			data: optional, numpy array. Slice this instead of self.data. Needs to be of the same shape as data.
			compress: option, bool. Whether to remove size-one axes from the sliced array.
		"""
		if data is None:
			data = self.data
		elif np.shape(data) != np.shape(self.data):
			raise ValueError("Array to be sliced must be of the same shape as self.data.")
		
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
		
		if compress:
			data = data.reshape(*[i for i in data.shape if i != 1])
		
		return data, coords
	
	def slice_time(self, t, arr):
		"""
		Given times t and values at those times (arr), return a slice of arr between self.t_min and self.t_max.
		
		Arguments:
			t: 1D numpy array
			arr: numpy array whose first axis is the same size as t
		"""
		if np.shape(arr)[0] != len(t):
			raise ValueError("Time axis size mismatch.")
		
		if self.t_max is None:
			t_max = self.ts.t[-1]
		else:
			t_max = self.t_max
		
		if self.t_min < t[0]:
			warnings.warn(f"t_min is not in provided range of t; {self.t_min = }, {t[0] = }")
		if t_max > t[-1]:
			warnings.warn(f"t_max is not in provided range of t; {t_max = }, {t[-1] = }")
		
		it_min = np.argmin(np.abs(t - self.t_min))
		it_max = np.argmin(np.abs(t - t_max))
		
		return arr[it_min:it_max]
	
	def set_t_range(self, t_min, t_max=None):
		"""
		Change the range of t to the given values, and then calculate the Fourier transform of the data in this interval (by calling do_ft).
		"""
		if t_max is None:
			t_max = self.ts.t[-1]
		
		if t_min >= t_max:
			raise ValueError("t_min needs to be less than t_max")
		
		if not (self.t_min == t_min and self.t_max == t_max):
			self._t_min = t_min
			self._t_max = t_max
			self.do_ft()

class dr_yaver_base(dr_base):
	@property
	def data_axes(self):
		return {'omega_tilde':0, 'kx_tilde':1, 'z':2}
	
	@property
	def field_name_default(self):
		return "uzmxz"
	
	def read(self):
		self.ts = pc.read.ts(datadir=self.datadir, quiet=True)
		#TODO: handle hdf5 read errors below (just output a warning on each fail and retry)?
		self.av_y = pc.read.aver(
			datadir=self.datadir,
			simdir=self.simdir,
			plane_list=['y'],
			var_names=[self.field_name],
			)
		self.av_xy = pc.read.aver(
			datadir=self.datadir,
			simdir=self.simdir,
			plane_list=['xy'],
			)
		
	def do_ft(self):
		fftshift = scipy.fft.fftshift
		fftfreq = scipy.fft.fftfreq
		x = self.grid.x
		Lx = self.grid.Lx
		z = self.grid.z
		t = self.slice_time(self.av_y.t, self.av_y.t)
		uz = self.slice_time(self.av_y.t, getattr(self.av_y.y, self.field_name))
		
		assert np.shape(uz) == (len(t), len(z), len(x))
		
		uz_fft = scipy.fft.fftn(uz, norm='forward', axes=[0,2])
		uz_fft = fftshift(uz_fft, axes=[0,2])
		data = np.transpose(uz_fft, axes=[0,2,1]) #Move the z-axis to the end.
		n_omega, n_kx, _ = np.shape(data)
		
		self.omega = 2*np.pi*fftshift(fftfreq(n_omega, d = (max(t)-min(t))/n_omega ))
		self.kx = 2*np.pi*fftshift(fftfreq(n_kx, d = Lx/n_kx ))
		self.data = self.scale_data(data)
	
	def plot_komega(self, z):
		"""
		Plot the k-omega diagram at a given height z.
		"""
		data, [omega_tilde, kx_tilde, _] = self.get_slice(
			kx_tilde = (self.k_tilde_min, self.k_tilde_max),
			omega_tilde = (self.omega_tilde_min, self.omega_tilde_max),
			z = z,
			)
		
		p = self.contourplotter(kx_tilde, omega_tilde, data[:,:,0].transpose())
		
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
			z = z,
			compress = True
			)
		
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

class dr_3d_base(dr_base):
	"""
	Base class for objects which have both kx and ky.
	"""
	@property
	def data_axes(self):
		return {'omega_tilde':0, 'kx_tilde':1, 'ky_tilde':2, 'z':3}
	
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
	@abc.abstractmethod
	def z(self):
		raise NotImplementedError
	
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
		
		p = self.contourplotter(kx_tilde, omega_tilde, data[:,:,0,0].transpose())
		
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
		data, [omega_tilde, _, ky_tilde, _] = self.get_slice(
			kx_tilde = 0,
			omega_tilde = (self.omega_tilde_min, self.omega_tilde_max),
			ky_tilde = (self.k_tilde_min, self.k_tilde_max),
			z = z,
			)
		
		p = self.contourplotter(ky_tilde, omega_tilde, data[:,0,:,0].transpose())
		
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
		
		p = self.contourplotter(kx_tilde, ky_tilde, data[0,:,:,0])
		
		p.ax.set_title(f"z = {z:.2f}")
		p.ax.set_xlabel(r"$\widetilde{{k}}_x$")
		p.ax.set_ylabel(r"$\widetilde{{k}}_y$")
		p.cbar.set_label(self.cbar_label)
		
		p.fig.tight_layout()
		return p

@dataclass
class fake_grid:
	x: np.ndarray
	y: np.ndarray
	z: np.ndarray

class dr_dvar_base(dr_3d_base):
	"""
	Read downsampled snapshots and plot dispersion relations from them.
	"""
	@property
	def field_name_default(self):
		return "uz"
	
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
		t = self.slice_time(self.t_vard, self.t_vard)
		uz = self.slice_time(self.t_vard, self.vard)
		
		assert np.shape(uz) == (len(t), len(z), len(y), len(x))
		
		uz_fft = scipy.fft.fftn(uz, norm='forward', axes=[0,2,3])
		uz_fft = fftshift(uz_fft, axes=[0,2,3])
		data = np.transpose(uz_fft, axes=[0,3,2,1])
		n_omega, n_kx, n_ky, _ = np.shape(data)
		
		self.omega = 2*np.pi*fftshift(fftfreq(n_omega, d = (max(t)-min(t))/n_omega ))
		self.kx = 2*np.pi*fftshift(fftfreq(n_kx, d = Lx/n_kx ))
		self.ky = 2*np.pi*fftshift(fftfreq(n_ky, d = Ly/n_ky ))
		self.data = self.scale_data(data)

class dr_pxy_base(dr_dvar_base):
	"""
	Read the output of power_xy and plot dispersion relations from them. Requires lintegrate_z=F, lintegrate_shell=F, and lcomplex=F in power_spectrum_run_pars.
	"""
	@property
	def field_name_default(self):
		return "uz_xy"
	
	@property
	def z(self):
		return self.pxy.zpos
	
	def read(self):
		#check if the right values were passed to power_spectrum_run_pars
		if not self.param.lcomplex:
			raise ValueError("Need lcomplex=T")
		if self.param.lintegrate_shell:
			raise ValueError("Need lintegrate_shell=F")
		if self.param.lintegrate_z:
			raise ValueError("Need lintegrate_z=F")
		
		#The following two cause problems because the size-one axis is compressed by Power.read.
		if self.dim.nxgrid == 1:
			raise ValueError("Need nxgrid > 1")
		if self.dim.nygrid == 1:
			raise ValueError("Need nygrid > 1")
		
		sim = pc.sim.get(self.simdir, quiet=True)
		
		self.ts = pc.read.ts(sim=sim, quiet=True)
		self.pxy = pc.read.power(datadir=self.datadir, quiet=True)
		self.av_xy = pc.read.aver(
			datadir=self.datadir,
			simdir=self.simdir,
			plane_list=['xy'],
			)
	
	def do_ft(self):
		fftshift = scipy.fft.fftshift
		fftfreq = scipy.fft.fftfreq
		kx = self.pxy.kx
		ky = self.pxy.ky
		z = self.pxy.zpos
		t = self.slice_time(self.pxy.t, self.pxy.t)
		uz = self.slice_time(self.pxy.t, getattr(self.pxy, self.field_name))
		
		assert np.shape(uz) == (len(t), len(z), len(ky), len(kx))
		
		uz_fft = scipy.fft.fftn(uz, norm='forward', axes=[0])
		uz_fft = fftshift(uz_fft, axes=[0])
		data = np.transpose(uz_fft, axes=[0,3,2,1])
		n_omega, _, _, _ = np.shape(data)
		
		self.omega = 2*np.pi*fftshift(fftfreq(n_omega, d = (max(t)-min(t))/n_omega ))
		self.kx = kx
		self.ky = ky
		self.data = self.scale_data(data)

class m_dscl_dbyD2():
	@property
	def cbar_label_default(self):
		return  r"$\left| \hat{{u}} \right|/ \mathcal{{D}}^2$"
	
	def scale_data(self, data):
		urms = np.sqrt(np.average(self.slice_time(self.av_xy.t, self.av_xy.xy.uz2mz), axis=0))
		urms = np.max(urms) #Choosing the peak urms since I don't want the normalization to be depth-dependent.
		D = urms/self.omega_0
		return np.abs(data)/D**2

class m_dscl_rdbyurmsmax():
	@property
	def cbar_label_default(self):
		return  r"$\left| \widetilde{{\omega}} \, \widetilde{{P}} \right|$"
	
	def scale_data(self, data):
		urms = np.sqrt(np.average(self.slice_time(self.av_xy.t, self.av_xy.xy.uz2mz), axis=0))
		urms = np.max(urms) #Choosing the peak urms since I don't want the normalization to be depth-dependent.
		
		data = np.moveaxis(data, self.data_axes['omega_tilde'], -1) # for broadcasting
		#NOTE: multiplying by omega to take 'running difference'
		data = np.abs(self.omega_tilde * data)/urms
		data = np.moveaxis(data, -1, self.data_axes['omega_tilde'])
		return data

class m_dscl_rdbyD2():
	@property
	def cbar_label_default(self):
		return  r"$\left| \widetilde{{\omega}} \, \hat{{u}} \right| / \mathcal{{D}}^2$"
	
	def scale_data(self, data):
		urms = np.sqrt(np.average(self.slice_time(self.av_xy.t, self.av_xy.xy.uz2mz), axis=0))
		urms = np.max(urms) #Choosing the peak urms since I don't want the normalization to be depth-dependent.
		D = urms/self.omega_0
		
		data = np.moveaxis(data, self.data_axes['omega_tilde'], -1) # for broadcasting
		#NOTE: multiplying by omega to take 'running difference'
		data = np.abs(self.omega_tilde * data)/D**2
		data = np.moveaxis(data, -1, self.data_axes['omega_tilde'])
		return data

class m_scl_SBC15(m_dscl_rdbyD2):
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

class m_scl_HP():
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

class m_cpl_imshow():
	"""
	Mixin that overrides dr_base.contourplotter to do imshow instead.
	"""
	def contourplotter(self, x, y, data):
		if np.all(data == 0):
			raise RuntimeError("The selected slice is all zeros")
		if np.shape(data) != (len(x), len(y)):
			raise ValueError(f"data array needs to have shape [len(x), len(y)].")
		
		data = data.transpose()
		data = np.where(data == 0, np.nan, data) #replace 0 with nan so that log scaling works.
		
		dx = x[1] - x[0]
		dy = y[1] - y[0]
		
		fig,ax = plt.subplots()
		im = ax.imshow(
			data,
			origin = 'lower',
			extent = (min(x)-dx/2, max(x)+dx/2, min(y)-dy/2, max(y)+dy/2),
			aspect = 'auto',
			norm = mpl.colors.LogNorm(),
			interpolation = 'none',
			)
		ax.set_xlim(min(x), max(x))
		ax.set_ylim(min(y), max(y))
		
		c = plt.colorbar(
			im,
			ax = ax,
			format = mpl.ticker.LogFormatterSciNotation(),
			ticks = mpl.ticker.LogLocator(),
			)
		
		return contourplot_container(fig, ax, im, c, savedir=self.fig_savedir)

class dr_wrap_base():
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
	
class dr_stat(dr_wrap_base):
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

class dr_sm(dr_wrap_base):
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

if __name__ == "__main__":
	class disp_rel_from_yaver(m_scl_SBC15, dr_yaver_base):
		pass
	
	dr = disp_rel_from_yaver()
	dr.plot_komega(1)
	
	plt.show()
