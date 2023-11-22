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

def fit_mode(dr, k_tilde, z, target_om, tol=3):
	"""
	Given a disp_rel_from_yaver instance, find the amplitude of a particular mode as a function of depth
	
	Arguments:
		dr: disp_rel_from_yaver instance
		k_tilde: float, wavenumber at which to find the amplitude
		z: float, z-coordinate at which to read the data
		target_om: consider the mode whose omega_tilde at the given k_tilde is nearest to target_om.
		tol: float. If the standard deviation about the continuum is stdev, only consider peaks that are above tol*stdev + continuum
		TODO: iz: int
	
	TODO: SBC15 just say "To determine the shape of the f and p mode eigenfunctions, we derived them from the z-dependent spectrum of uz by selecting kx = 2 and ω = 1.31, 2.09, 2.77, 3.67 corresponding to the f mode and the first three p modes, p0, 1, 2 , respectively". Is that enough?
	Algorithm:
		Given a target omega, find the mode near it by the following procedure.
			1. Find the 'continuum' by averaging over a sufficiently wide band of omega
			2. Detect 'provisional peaks' by taking all peaks which are sufficiently larger than the continuum
				Perhaps 'sufficiently larger' can be related to the variance of P near the peak (but need to be a bit careful about this; the band needs to be wide enough that the variance is not affected by the presence of a genuine peak)
			3. Among the provisional peaks, return the peak which is closest to the target omega.
	TODO: (20-Nov 2023, meeting) Nishant mentioned that the spectra are always smoothed before fitting the peak. But I find that the peak width drastically changes even for smoothing widths of 3 or 5 points. That in fact suggests that rather than reporting the peak amplitude, we should be reporting the area under the curve above the continuum (or perhaps even some sort of 'equivalent width'). Need to think a bit. Perhaps it is fine, since the mode eigenfunction is undetermined up to a multiplicative factor.
		Singh et al. (2015, eq. 19) indeed define something called a mode mass.
	TODO: ()meeting, 21-Nov-2023) Nishant mentioned that to remove the continuum, one can fit a polynomial. Let the user input the band of frequencies. Fit more than one Lorentzian (needed for the fanning part)
		Singh 2016 (the f-mode paper) mention that the continuum is fit by a parabola.
	"""
	iz = np.argmin(np.abs(z - dr.grid.z))
	ik = np.argmin(np.abs(k_tilde - dr.kx*dr.L0))
	data = np.abs(dr.omega*dr.uz_fft[:,ik,iz]/(dr.omega_0*dr.D**2)) #NOTE: multiplying by omega to take 'running difference'
	
	om_tilde = dr.omega/dr.omega_0
	itarget = np.argmin(np.abs(om_tilde - target_om))
	
	data = smooth(data, 3) #smooth the data so that we get neat profiles.
	#TODO: smoothing as above currently leads to artefacts near omega = +- omega_max
	
	bw = 0.5 #half-width of the band in omega_tilde in which the 'continuum' is calculated TODO: make this an arg. TODO: also make this the window about target_om in which a peak will be found? Would I need to try to handle the case where there are multiple peaks within this band?
	continuum, stdev = get_continuum(data, bw, dr)
	
	d_om_tilde = om_tilde[1] - om_tilde[0]
	bw_i = int(np.round(bw/d_om_tilde))
	i_min = np.max(itarget - bw_i, 0)
	i_max = np.min(itarget + bw_i, len(data))
	data_near_target = (data - continuum)[i_min: i_max]
	omt_near_target = omega_tilde[i_min: i_max]
	
	model = lambda om, om_0, gam, A: (A*gam/np.pi)/((om -om_0)**2 + gam**2) #Lorentzian
	
	popt, pcov = scipy.optimize.curve_fit(
		model,
		omt_near_target,
		data_near_target,
		p0 = np.array([target_om, 0, np.max(data_near_target)]),
		#TODO: sigma = ?? stdev may not be correct, but not sure what else to put.
		bounds = (
			np.array([target_om - bw,-np.inf,-np.inf]),
			np.array([target_om + bw,np.inf, np.inf]),
			)
		)
	
	return {
		'A': popt[2],
		'omega_tilde': popt[0],
		'gam': popt[1],
		}

# def bw_to_i(bw, om_tilde):
# 	"""
# 	Given a bandwidth in om_tilde, convert it to the equivalent number of grid points.
# 	"""
# 	
# 	d_om_tilde = om_tilde[1] - om_tilde[0]
# 	return int(np.round(bw/d_om_tilde))

def get_continuum(data, bw, dr):
	"""
	bw: half-width of the band in omega_tilde in which the 'continuum' is calculated
	"""
	om_tilde = dr.omega/dr.omega_0
	d_om_tilde = (dr.omega[1] - dr.omega[0])/dr.omega_0
	bw_i = int(np.round(bw/d_om_tilde))
	continuum = smooth(data, bw_i)
	stdev = np.sqrt(smooth(data**2, bw_i) - continuum**2)
	return continuum, stdev

def smooth(data, n):
	"""
	data: numpy array
	n: int, half-width of the smoothing filter (top hat)
	"""
	weight = np.ones(2*n+1)
	weight = weight/np.sum(weight)
	return scipy.signal.convolve(data, weight, mode='same')

if __name__ == "__main__":
	dr = disp_rel_from_yaver()
	dr.plot_komega(1)
	
	plt.show()
