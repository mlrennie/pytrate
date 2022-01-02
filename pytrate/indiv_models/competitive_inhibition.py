__description__ = \
"""
michaelis_menten.py

Model describing initial rates for a simple enzyme.
"""
__author__ = "Martin L. Rennie"
__date__ = "10th Oct 2018"

import numpy as np

from .base import Model

class CompetitiveInhibition(Model):
    """
    Initial rates of a simple enzyme in the presence of a 
    competitive inhibitor (binding the enzyme).
    """

    def __init__(self,
                 concs,
                 obs):
        """
        Run checks to make sure the inputs are valid for this model type.
        """
        
        # run the initialization based on the super class Model
        Model.__init__(self,concs,obs)
        
        if(self.n_species != 3):
            sys.exit("ERROR: Must be three concentration species entered. \
                      {} detected".format(self.n_species))
        
        if(self.n_obs !=1):
            sys.exit("ERROR: Must be one observable entered. \
                      {} detected".format(self.n_obs))
        
        print("First column of data file assumed to be enzyme concentration")
        print("Second column of data file assumed to be substrate concentration\n")
        print("Second column of data file assumed to be inhibitor concentration\n")
    
    def param_definition(log10_Km=-5,log10_Ki=-6,log10_kcat=5,
                          conc_corr_enz=1.0,conc_corr_sub=1.0,conc_corr_inh=1.0):
        pass

    @property
    def obs_calc(self):
        """
        Calculate the steady state rates that would be observed across shots for a given value
        of Michealis-Menten constant and catalytic rate.
        """
        
        # correct for concentration measurement uncertainties
        Et = self._concs[0][0] * self.param_values["conc_corr_enz"]
        S0 = self._concs[0][1] * self.param_values["conc_corr_sub"] 
        It = self._concs[0][2] * self.param_values["conc_corr_inh"] 
        
        to_return = 10**self.param_values["log10_kcat"] * Et * S0 / ((self.param_values["Km"]*(1+It/self.param_values["Ki"]) + S0))

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
        return self._concs[0][2]
    
    @property
    def y_label(self):
        """
        3D plot. Return the label for the y-axis.
        """
        return "[Inhibitor]"
    
    @property
    def z_var(self):
        """
        4D plot. Return the variable to be plotted on the z-axis. Depends on model.
        """
        return self._concs[0][0]
    
    @property
    def z_label(self):
        """
        4D plot. Return the label for the z-axis.
        """
        return "[Enzyme]"
    
