"""
Only meant for quick interactive use; may change without notice.
"""

from . import read

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
