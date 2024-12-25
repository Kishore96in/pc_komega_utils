"""
Only meant for quick interactive use; may change without notice.
"""

from . import read
from . import power

class dr_yaver(
	read.m_dscl_d2,
	read.m_scl_HP,
	read.dr_yaver_base,
	): pass

class dr_dvar(
	read.m_dscl_d2,
	read.m_scl_HP,
	read.dr_dvar_base,
	): pass

class dr_pxy(
	read.m_dscl_d2,
	read.m_scl_HP,
	read.dr_pxy_base,
	): pass

class dr_pxy_h5(
	read.m_dscl_d2,
	read.m_scl_HP,
	power.dat2h5.m_pxy_h5,
	read.dr_pxy_base,
	): pass
