__description__ = \
"""
simple_ternary_association.py

Model for a ternary association, where a pair of the components 
are assumed to be at identical concentrations.
"""
__author__ = "Martin L. Rennie"
__date__ = "20th Aug 2019"

import numpy as np

from .base import Model

class STA(Model):
    """
    Three-component ternary interaction (A,B,C) assuming no intermediates.
    A property of A is observed and total concentrations of [B] and [C] are kept equimolar.
    
    Fobs = F0 + (F1-F0)*[ABC]/[A]t.
    
    Model Parameters:
    -----------------
    Kd ~ Dissociation constant for the assembly (M^2)
    F0 ~ Observable for component A (unit of observable)
    F1 ~ Observable for component A in complex with B and C (unit of observable)
    
    Independent Variables:
    --------------------
    At ~ Total concentration of component A (M)
    BCt ~ Total concentration of component B = component C (M)
    
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
    
    def param_definition(log10_Kd=-5,
                         F0=0.0,F1=1.,
                         conc_corr_A=1.0,conc_corr_BC=1.0):
        pass

    @property
    def obs_calc(self):
        """
        Calculate the observable as a function of total concenrations.
        Cubic equation: Ka*[BC]^3 + (Ka*[A]t-[BC]t)*[BC]^2 + [BC] - [BC]t = 0
        Cubic solution as per Cardano (1545).
        """
        
        # correct for concentration measurement uncertainties
        At = self._concs[0][0] * self.param_values["conc_corr_A"]
        BCt = self._concs[0][1] * self.param_values["conc_corr_BC"] 
        
        b = At-BCt*10**self.param_values["log10_Kd"]
        c = 10**self.param_values["log10_Kd"]
        d = -10**self.param_values["log10_Kd"]*BCt
        
        q = (-b**3/27.+b*c/6.-d/2.)
        
        conc_BC = np.cbrt(q+np.sqrt(q**2+(c/3.-b**2/9.)**3)) + \
                  np.cbrt(q-np.sqrt(q**2+(c/3.-b**2/9.)**3)) - b/3.
                                  
        
        to_return = self.param_values["F0"] + (self.param_values["F1"]-self.param_values["F0"]) * (
                    conc_BC**2/(10**self.param_values["log10_Kd"]+conc_BC**2))

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
    
