from .dispersion_relation import *
from .fit import *

#The following are defined for compatibility with old scripts
class disp_rel_from_yaver(m_scl_SBC15, dr_yaver_base):
	pass

class disp_rel_nonorm_from_yaver(m_scl_SBC15, dr_yaver_base):
	@property
	def cbar_label_default(self):
		return  r"$\tilde{{\omega}} \hat{{u}}$"
	
	def scale_data(self, data):
		data = np.moveaxis(data, self.data_axes['omega_tilde'], -1) # for broadcasting
		#NOTE: multiplying by omega to take 'running difference'
		data = np.abs(self.omega_tilde * data)
		data = np.moveaxis(data, -1, self.data_axes['omega_tilde'])
		return data

class disp_rel_from_yaver_L0_HP(m_scl_HP, m_dscl_rdbyD2, dr_yaver_base):
	pass

class disp_rel_from_dvar(m_dscl_rdbyD2, m_scl_HP, dr_dvar_base):
	pass
