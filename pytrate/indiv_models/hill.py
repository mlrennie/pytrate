__description__ = \
"""
michaelis_menten.py

Model describing initial rates for a simple enzyme.
"""
__author__ = "Martin L. Rennie"
__date__ = "10th Oct 2018"

import numpy as np

from .base import Model

class SSIS(Model):
    """
    Ligand binding model with empirical Hill cooperativity. Basal signal level subtracted.
    
    Fobs = Fmin + Fmax * L^nH / (Kd^nH + L^nH)
    
    Model Parameters:
    -----------------
    Kd ~ Dissociation constant for the ligand (M-1)
    nH ~ Hill coefficient (unitless)
    Fmin ~ Observable for unliganded protein (unit of observable)
    Fmax ~ Observable for liganded protein (unit of observable)
    
    Independent Variables:
    --------------------
    L ~ Free ligand concentration (M)
    
    Dependent Variables:
    --------------------
    Fobs ~ observed signal (unit of observable)
    """

    def __init__(self,
                 concs,
                 obs):
        """
        Run checks to make sure the inputs are valid for this model type.
        """
        
        # run the initialization based on the super class Model
        Model.__init__(self,concs,obs)
        
        if(self.n_species != 1):
            sys.exit("ERROR: Must be one concentration species entered. \
                      {} detected".format(self.n_species))
        
        if(self.n_obs !=1):
            sys.exit("ERROR: Must be one observable entered. \
                      {} detected".format(self.n_obs))
        
        print("First column of data file assumed to be ligand concentration")
    
    def param_definition(log10_Kd=-5,nH=1.0,
                         Fmin=0.,Fmax=1.,
                         conc_corr_lig=1.0):
        pass

    @property
    def obs_calc(self):
        """
        Calculate the observable as a function of total protein and ligand concenration.
        """
        
        # correct for concentration measurement uncertainties
        L = self._concs[0][0] * self.param_values["conc_corr_lig"] 
        
        to_return = self.param_values["Fmin"] + self.param_values["Fmax"] * L**self.param_values["nH"] / (
                        10**self.param_values["log10_Kd"]**self.param_values["nH"] 
                        + L**self.param_values["nH"])

        return np.array([to_return])
    
    @property
    def x_var(self):
        """
        Return the variable to be plotted on the x-axis. Depends on model.
        """
        return self._concs[0][0]
    
    @property
    def x_label(self):
        """
        Return the label for the x-axis.
        """
        return "[Free Ligand]"
    
