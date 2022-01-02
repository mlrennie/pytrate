# mean and standard deviation for uncertainty in the concentration
mu_ext_coeff = -0.00
sd_ext_coeff =  0.05

# conversion factors for concentration (conversion to Molar)
AVAIL_CONC_UNITS = {'M':1e0,
                    'mM':1e-3,
                    'uM':1e-6,
                    'nM':1e-9}
    
# units for rate measurements
AVAIL_OBS_UNITS_RATE = {"uM/sec":1e-6,
                        "M/min":1./60.,
                        "M/sec":1}

# units for other measurements (TO BE IMPLEMENTED)