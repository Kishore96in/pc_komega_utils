"""
Defined only for compatibility with older scripts. Do not use in new code.
"""
from . import read, plot, fit

class disp_rel_from_yaver(read.m_scl_SBC15, read.dr_yaver_base):
	pass

class disp_rel_nonorm_from_yaver(read.m_scl_SBC15, read.dr_yaver_base):
	@property
	def cbar_label_default(self):
		return  r"$\tilde{{\omega}} \hat{{u}}$"
	
	def scale_data(self, data):
		data = np.moveaxis(data, self.data_axes['omega_tilde'], -1) # for broadcasting
		#NOTE: multiplying by omega to take 'running difference'
		data = np.abs(self.omega_tilde * data)
		data = np.moveaxis(data, -1, self.data_axes['omega_tilde'])
		return data

class disp_rel_from_yaver_L0_HP(read.m_scl_HP, read.m_dscl_rdbyD2, read.dr_yaver_base):
	pass

class disp_rel_from_dvar(read.m_dscl_rdbyD2, read.m_scl_HP, read.dr_dvar_base):
	pass

oplot_dr_f = plot.fmode
smooth = fit.smooth
