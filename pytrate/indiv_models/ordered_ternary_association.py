__description__ = \
"""
ordered_ternary_association.py

Model for a ternary association, where a pair of the components 
are assumed to be at identical total concentrations.
"""
__author__ = "Martin L. Rennie"
__date__ = "20th Aug 2019"

import numpy as np
from scipy.optimize import root as solve_mass_balance
from scipy.optimize import OptimizeResult

from .base import Model

class OTA(Model):
    """
    Three-component ternary interaction (A,B,C) assuming a BC intermediate.
    Property of A is observed and total concentrations of [B] and [C] are kept equimolar.
    
    Fobs = F0 + (F1-F0)*[ABC]/[A]t.
    
    Model Parameters:
    -----------------
    Kd1 ~ Dissociation constant for the BC complex (M)
    Kd2 ~ Dissociation constant for the ABC complex into A and BC (M)
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
    
    def param_definition(log10_Kd1=-5,log10_Kd2=-5,
                         F0=0.0,F1=1.,
                         conc_corr_A=1.0,conc_corr_BC=1.0):
        pass

    @property
    def obs_calc(self):
        """
        Calculate the observable as a function of total concenrations.
        Mass balance equations numerically solved.
        """
        
        # correct for concentration measurement uncertainties
        At = self._concs[0][0] * self.param_values["conc_corr_A"]
        BCt = self._concs[0][1] * self.param_values["conc_corr_BC"] 
        
        num_points = len(At)
        
        A = np.zeros((num_points),dtype=float)
        BC = np.zeros((num_points),dtype=float)
        
        # call function to compute the free species by numerical solution of the mass balance equations        
        (A, BC) = solve_mb(num_points, 
            10**-self.param_values["log10_Kd1"], 10**-self.param_values["log10_Kd2"], 
            At,BCt)
        
        to_return = self.param_values["F0"] + (self.param_values["F1"]-self.param_values["F0"]) * (
                    10**-self.param_values["log10_Kd1"]*10**-self.param_values["log10_Kd2"]*BC**2*A / At)

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
 
def solve_mb(N_points, K1, K2, At, BCt):
    """
    Solve mass balance equations for the model.
    
    [A]t  = [A] + K1*K2*[BC]^2*[A]
    [BC]t = [BC] + K1*[BC]^2 + K1*K2*[BC]^2*[A]
    """
                
    A = np.zeros(N_points)
    BC = np.zeros(N_points)
    
    for i in range(N_points):
        
        # mass balance equations
        def equations(x):
            A,BC = x
            return [A + K1*K2*BC**2*A - At[i], \
                    BC + K1*BC**2 + K1*K2*BC**2*A - BCt[i]]
    
        sol = OptimizeResult(success=False)
        Atmp = -1
        BCtmp = -1
        j = 1.
        # try to solve using total concentrations as initial guesses. Maximum number of iterations is large as gradient may be very shallow
        sol = solve_mass_balance(equations,(At[i],BCt[i]),method='lm',options={'maxiter':2000})
        Atmp,BCtmp = sol.x
        # if no solution try to solve with different initial conditions and solver options
        if(not sol.success and i!=0):
            sol = solve_mass_balance(equations,(A[i-1],BC[i-1]),method='lm',options={'factor':1,'maxiter':8000})
            Atmp,BCtmp = sol.x
            # if still no solution...bugger
            if(not sol.success):
                print("ERROR: Could not find solution...")
    
        A[i] = Atmp
        BC[i] = BCtmp
        
    return (A,BC)
 
