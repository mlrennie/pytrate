__description__ = \
"""
simple_ternary_association.py

Model for a ternary association, where a pair of the components 
are assumed to be at identical concentrations.
"""
__author__ = "Martin Rennie"
__date__ = "10th Nov 2021"

import numpy as np
from scipy.optimize import root as solve_mass_balance
from scipy.optimize import OptimizeResult
import sys

from .base import Model

class TPOL_nMS(Model):
    """
    Three-component ternary interaction (A,B,L) assuming the fractions of A,B and complexes are monitored
    
    Fobs1 = [A]/([A]+[B]+[AB]+[AL]+[BL]+[ABL])
    Fobs2 = [AL]/([A]+[B]+[AB]+[AL]+[BL]+[ABL])
    Fobs3 = [B]/([A]+[B]+[AB]+[AL]+[BL]+[ABL])
    Fobs4 = [BL]/([A]+[B]+[AB]+[AL]+[BL]+[ABL])
    Fobs5 = [ABL]/([A]+[B]+[AB]+[AL]+[BL]+[ABL])
    
    Model Parameters:
    -----------------
    K_AL ~ Association constant for ligand binding to component A
    K_BL ~ Association constant for ligand binding to component B    
    K_AB ~ Association constant for complexation of A and B 
    a ~ Cooperativity of the interaction
    
    Independent Variables:
    --------------------
    At ~ Total concentration of component A (M)
    Bt ~ Total concentration of component B (M)
    Lt ~ Total concentration of component L (M)
    
    Dependent Variables:
    --------------------
    Fobs1 ~ observed fraction of state A
    Fobs2 ~ observed fraction of state AL
    Fobs3 ~ observed fraction of state B
    Fobs4 ~ observed fraction of state BL
    Fobs5 ~ observed fraction of state ABL
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
        
        if(self.n_obs !=5):
            sys.exit("ERROR: Must be five observables entered. \
                      {} detected".format(self.n_obs))
        
        
        print("First column of data file assumed to be ligand concentration")
        print("Second column of data file assumed to be component A concentration")
        print("Third column of data file assumed to be component B concentration\n")
    
    def param_definition(log10_K_AB=-5,log10_K_AL=-5,log10_K_BL=-5,log10_a=0,
                         conc_corr_A=1.0,conc_corr_B=1.0,conc_corr_L=1.0):
        pass
    
    @property
    def obs_calc(self):
        """
        Calculate the observable as a function of total concenrations.
        Mass balance equations numerically solved.
        """
        
        # correct for concentration measurement uncertainties
        Lt = self._concs[0][0] * self.param_values["conc_corr_L"]
        At = self._concs[0][1] * self.param_values["conc_corr_A"]
        Bt = self._concs[0][2] * self.param_values["conc_corr_B"]  
        
        num_points = len(At)
        
        A = np.zeros((num_points),dtype=float)
        B = np.zeros((num_points),dtype=float)
        L = np.zeros((num_points),dtype=float)
        
        # call function to compute the free species by numerical solution of the mass balance equations        
        (A, B, L) = solve_mb(num_points, 
            10**self.param_values["log10_K_AB"], 10**self.param_values["log10_K_AL"],
            10**self.param_values["log10_K_BL"], 10**self.param_values["log10_a"],
            At,Bt,Lt)
        
        # use free species concentrations to compute concentrations of complexes
        AB  = 10**self.param_values["log10_K_AB"]*A*B
        AL  = 10**self.param_values["log10_K_AL"]*A*L
        BL  = 10**self.param_values["log10_K_BL"]*B*L
        ABL = 10**self.param_values["log10_a"]*10**self.param_values["log10_K_AB"]*10**self.param_values["log10_K_AL"]*10**self.param_values["log10_K_BL"]*A*B*L
        
        tot = A + B + AB + AL + BL + ABL
        
        # output experimental measurement
        to_return = np.array([A/tot,AL/tot,B/tot,BL/tot,ABL/tot])

        return to_return
    
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
        return "[Total L]"
        
    @property
    def y_var(self):
        """
        3D plot. TO BE IMPLEMENTED
        """
        return self._concs[0][1]
    
    @property
    def y_label(self):
        """
        3D plot.  TO BE IMPLEMENTED
        """
        return "[Total A]"
        
    @property
    def y2_var(self):
        """
        3D plot.  TO BE IMPLEMENTED
        """
        return self._concs[0][2]
    
    @property
    def y2_label(self):
        """
        3D plot.  TO BE IMPLEMENTED
        """
        return "[Total B]"
    
def solve_mb(N_points, K_AB, K_AL, K_BL, a, At, Bt, Lt):
    """
    Solve mass balance equations for the model.
    
    [A]t = [A] + K_AB*[A]*[B] + K_AL*[A]*[L] + a*K_AB*K_AL*K_BL*[A]*[B]*[L]
    [B]t = [B] + K_AB*[A]*[B] + K_BL*[B]*[L] + a*K_AB*K_AL*K_BL*[A]*[B]*[L]
    [L]t = [L] + K_AL*[A]*[L] + K_BL*[B]*[L] + a*K_AB*K_AL*K_BL*[A]*[B]*[L]
    """
                
    A = np.zeros(N_points)
    B = np.zeros(N_points)
    L = np.zeros(N_points)
    
    for i in range(N_points):
        
        # mass balance equations
        def equations(x):
            A,B,L = x
            return [A + K_AB*A*B + K_AL*A*L + a*K_AB*K_AL*K_BL*A*B*L - At[i], \
                    B + K_AB*A*B + K_BL*B*L + a*K_AB*K_AL*K_BL*A*B*L - Bt[i], \
                    L + K_AL*A*L + K_BL*B*L + a*K_AB*K_AL*K_BL*A*B*L - Lt[i]]
    
        sol = OptimizeResult(success=False)
        Atmp = -1
        Btmp = -1
        Ltmp = -1
        j = 1.
        # try to solve using total concentrations as initial guesses. Maximum number of iterations is large as gradient may be very shallow
        sol = solve_mass_balance(equations,(At[i],Bt[i],Lt[i]),method='lm',options={'maxiter':2000})
        Atmp,Btmp,Ltmp = sol.x
        # if no solution try to solve with different initial conditions and solver options
        if(not sol.success and i!=0):
            sol = solve_mass_balance(equations,(A[i-1],B[i-1],L[i-1]),method='lm',options={'factor':1,'maxiter':8000})
            Atmp,Btmp,Ltmp = sol.x
            # if still no solution...bugger
            if(not sol.success):
                print("ERROR: Could not find solution...")
    
        A[i] = Atmp
        B[i] = Btmp
        L[i] = Ltmp
        
    return (A,B,L)
