__description__ = \
"""
michaelis_menten.py

Model describing initial rates for a simple enzyme.
"""
__author__ = "Martin L. Rennie"
__date__ = "20th Jul 2021"

import numpy as np
import sys

from .base import Model

class SSIS(Model):
    """
    Ligand binding model for a single set of identical and independent sites
    using a generic observable e.g. fluorescence.
    
    Fobs = F0*[P] + F1*[PL].
    
    Model Parameters:
    -----------------
    Kd ~ Dissociation constant for the ligand (M)
    n_sites ~ Apparent number of sites
    F0 ~ Observable for unliganded protein (unit of observable)
    F1 ~ Observable for liganded protein (unit of observable)
    
    Independent Variables:
    --------------------
    Pt ~ Total protein concentration (M)
    Lt ~ Total ligand concentration (M)
    
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
        
        if(self.n_species != 2):
            sys.exit("ERROR: Must be two concentration species entered. \
                      {} detected".format(self.n_species))
        
        if(self.n_obs !=1):
            sys.exit("ERROR: Must be one observable entered. \
                      {} detected".format(self.n_obs))
        
        print("First column of data file assumed to be protein concentration")
        print("Second column of data file assumed to be ligand concentration\n")
    
    def param_definition(log10_Kd=-5,n_sites=1.0,
                         F0=0.0,F1=1.,
                         conc_corr_prot=1.0,conc_corr_lig=1.0):
        pass

    @property
    def obs_calc(self):
        """
        Calculate the observable as a function of total protein and ligand concenration.
        """
        
        # correct for concentration measurement uncertainties
        Pt = self._concs[0][0] * self.param_values["conc_corr_prot"]
        Lt = self._concs[0][1] * self.param_values["conc_corr_lig"] 
        
        a = Lt + self.param_values["n_sites"]*Pt + 10**self.param_values["log10_Kd"]
        
        to_return = self.param_values["F0"] + (self.param_values["F1"]-self.param_values["F0"]) * (a 
                                    - np.sqrt(a**2 - 4*Lt*self.param_values["n_sites"]*Pt)
                                    ) / (2*self.param_values["n_sites"]*Pt)

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
        return "[Total Ligand]"
        
    @property
    def y_var(self):
        """
        3D plot. Return the variable to be plotted on the y-axis. Depends on model.
        """
        return self._concs[0][0]
    
    @property
    def y_label(self):
        """
        3D plot. Return the label for the y-axis.
        """
        return "[Total Protein]"
        
    #@property
    def obs_calc_user(self,Lt,Pt):
        """
        Calculate the observable as a function of total protein and ligand concenration.
        """ 
        
        a = Lt + self.param_values["n_sites"]*Pt + 10**self.param_values["log10_Kd"]
        
        to_return = self.param_values["F0"] + (self.param_values["F1"]-self.param_values["F0"]) * (a 
                                    - np.sqrt(a**2 - 4*Lt*self.param_values["n_sites"]*Pt)
                                    ) / (2*self.param_values["n_sites"]*Pt)

        return np.array([to_return])
    
    
