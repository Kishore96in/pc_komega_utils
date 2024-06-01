import scipy.special
import os
import numpy as np

def step(z, z0, width):
	return (scipy.special.erf((z-z0)/width) + 1)/2

def calc_K_prof(z, sim):
	"""
	Calculate the profile of thermal conductivity when iheatcond='K-profile'
	"""
	hcond0 = sim.param['hcond0']
	hcond1 = sim.param['hcond1']
	hcond2 = sim.param['hcond2']
	widthss = sim.param['widthss']
	z1 = sim.param['z1']
	z2 = sim.param['z2']
	
	prof = 1 + (hcond1-1)*step(z, z1, -widthss) + (hcond2-1)*step(z, z2, widthss)
	return hcond0*prof

class fig_saver():
	"""
	savefig: bool
	savedir: string, path to save the figure
	"""
	def __init__(self, savefig, savedir):
		self.savefig = savefig
		self.savedir = savedir
	
	def __call__(self, fig, name, **kwargs):
		if not self.savefig:
			return
		
		if not os.path.exists(self.savedir):
			#Create directory if it does not exist
			os.makedirs(self.savedir)
		elif not os.path.isdir(self.savedir):
			raise FileExistsError(f"Save location {self.savedir} exists but is not a directory.")
		
		loc = os.path.join(self.savedir, name)
		loc_dir = os.path.dirname(loc)
		if not os.path.exists(loc_dir):
			os.makedirs(loc_dir)
		fig.savefig(loc, **kwargs)

def get_av(av, name, it1=-500):
	return np.average(getattr(av.xy, name)[it1:], axis=0)

def smooth_tophat(a, n, axis=0):
	"""
	Smooth array a along axis axis with a top hat of width (2*n+1). The first n and last n points of the smoothed array contain the same value as the next point towards the center.
	
	a: numpy array
	n: int, such that width of the smoothing filter (top hat) is 2*n+1
	axis: int. Which axis of arr to smooth.
	"""
	a = np.moveaxis(a, axis, 0)
	sm = np.zeros_like(a)
	for i in range(-n, n+1):
		sm += np.roll(a, i, axis=0)
	sm = sm/(2*n+1)
	
	#Edge correction
	if n!= 0:
		sm[:n] = sm[n]
		sm[-n:] = sm[-n-1]
	
	sm = np.moveaxis(sm, 0, axis)
	return sm

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

def smooth_gauss(data, n):
	"""
	data: numpy array
	n: int, such that FWHM of the smoothing filter (Gaussian) is 2*n+1.
	"""
	sig = (2*n+1)/2.355
	
	weight = scipy.signal.windows.gaussian(6*sig, std=sig)
	weight = weight/np.sum(weight)
	return scipy.signal.convolve(data, weight, mode='same')
