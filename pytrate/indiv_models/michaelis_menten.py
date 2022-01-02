__description__ = \
"""
michaelis_menten.py

Model describing initial rates for a simple enzyme.
"""
__author__ = "Martin L. Rennie"
__date__ = "10th Oct 2018"

import numpy as np

from .base import Model

class MichaelisMenten(Model):
    """
    Initial rates of a simple enzyme.
    
    v = kcat*Et / (Km + S0)
    
    Model Parameters:
    -----------------
    kcat ~ Catalytic rate (M-1 s-1)
    Km ~ MichaelisMenten constant (M-1)
    
    Independent Variables:
    --------------------
    Et ~ Enzyme concentration (M)
    S0 ~ Initial substrate concentration (M)
    
    Dependent Variables:
    --------------------
    v ~ observed initial (steady state) rate (M-1 s-1)
    """

    def __init__(self,
                 concs,
                 obs):
        """
        Run checks to make sure the inputs are valid for this model type.
        """
        
        # run the initialization based on the super class Model
        Model.__init__(self,concs,obs)
        
        if(self.n_species != 2):
            sys.exit("ERROR: Must be two concentration species entered. \
                      {} detected".format(self.n_species))
        
        if(self.n_obs !=1):
            sys.exit("ERROR: Must be one observable entered. \
                      {} detected".format(self.n_obs))
        
        print("First column of data file assumed to be enzyme concentration")
        print("Second column of data file assumed to be substrate concentration\n")
    
    def param_definition(log10_Km=-6,log10_kcat=6,
                         conc_corr_enz=1.0,conc_corr_sub=1.0):
        pass

    @property
    def obs_calc(self):
        """
        Calculate the steady state rates that would be observed for a given value
        of Michealis-Menten constant (Km) and catalytic rate (kcat).
        """
        
        # correct for concentration measurement uncertainties
        Et = self._concs[0][0] * self.param_values["conc_corr_enz"]
        S0 = self._concs[0][1] * self.param_values["conc_corr_sub"] 
        
        to_return = 10**self.param_values["log10_kcat"] * Et * S0 / (10**self.param_values["log10_Km"] + S0)

        return np.array([to_return])
    
    @property
    def x_var(self):
        """
        Return the variable to be plotted on the x-axis. Depends on model.
        """
        return self._concs[0][1]
    
    @property
    def x_label(self):
        """
        Return the label for the x-axis.
        """
        return "[Initial Substrate]"
    
    @property
    def y_var(self):
        """
        3D plot. Return the variable to be plotted on the y-axis. Depends on model.
        """
        return self._concs[0][1]
    
    @property
    def y_label(self):
        """
        3D plt. Return the label for the y-axis.
        """
        return "[Enzyme]"
    
