import numpy as np
import pickle

om = np.linspace(0.3,0.7,100)

A = 1e-5
om_0 = 0.5
gam = 0.01
sigma = 1e-5

rand = np.random.default_rng().normal(loc=1e-4, scale=sigma, size=len(om))

data = rand + (A*gam/np.pi)/((om - om_0)**2 + gam**2)

inputs = {
	'data_near_target': data,
	'omt_near_target': om,
	'sigma': sigma,
	}

with open("inputs.pickle", 'wb') as f:
	pickle.dump(inputs, f)
