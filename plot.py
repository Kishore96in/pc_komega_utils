"""
Theoretical dispersion relations which I may want to mark on k-omega diagrams.
"""

import numpy as np

def fmode(dr, plot=None, ax=None):
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
