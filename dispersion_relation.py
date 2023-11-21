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
		k_tilde_max = 20, #plot limit
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
		self.k_tilde_max = k_tilde_max
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
		ikx_min = np.argmin(np.abs(0 - self.kx*L_0))
		iomega_max = np.argmin(np.abs(self.omega_tilde_max - self.omega/omega_0))
		iomega_min = np.argmin(np.abs(0 - self.omega/omega_0))
		
		kx = self.kx[ikx_min:ikx_max]
		omega = self.omega[iomega_min:iomega_max]
		uz_fft = self.uz_fft[iomega_min:iomega_max,ikx_min:ikx_max]
		
		iz_surf = np.argmin(np.abs(z - self.grid.z))
		
		fig,ax = plt.subplots()
		data = np.abs(omega[:,None]*uz_fft[:,:,iz_surf]/(omega_0*D**2)) #NOTE: multiplying by omega to take 'running difference'
		data = np.where(data == 0, np.nan, data) #replace 0 with nan so that log scaling works.
		
		return kx*L_0, omega/omega_0, data
	
	def plot_komega(self, z):
		k_tilde, omega_tilde, data = self.prep_data_for_plot(z)
		
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

class disp_rel_nonorm_from_yaver(disp_rel_from_yaver):
	def get_scales(self):
		cs_d = np.sqrt(self.param.cs2cool)
		g = np.abs(self.param.gravz)
		self.L_0 = cs_d**2/g
		self.omega_0 = g/cs_d
		
		self.D = 1

if __name__ == "__main__":
	dr = disp_rel_from_yaver()
	dr.plot_komega(1)
	
	plt.show()
