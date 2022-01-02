__description__ = \
"""
base.py

Base class for model description.
"""
__author__ = "Martin L. Rennie"
__date__ = "10th Oct 2018"

import sys
import inspect
import numpy as np
from .. import fit_param

from ..fixed_values import AVAIL_CONC_UNITS

class Model:
    """
    Base class from which all models should be sub-classed.
    """

    def __init__(self,
                 concs,
                 obs):

        """

        Parameters
        ----------

        data_file : string
               directory location of file containing the data
        """
        
        self._concs = concs
        self._obs = obs
        
        self.n_species = len(concs[0])
        self.n_obs = len(obs[0])
        
        self._initialize_param()

    def param_definition(self):
        pass

    @property
    def obs_calc(self):
        return np.array(())
    
#    @property
#    def obs_err(self):
#        return self.param_values["obs_err"]

    @property
    def mole_ratio(self):
        """
        Molar ratio of titrant to stationary species.  If not yet initialized,
        send return empty array.
        """

        try:
            return self._Ltot[1:]/self._Ptot[1:]
        except AttributeError:
            return np.array([],dtype=float)

    def _initialize_param(self,param_names=None,param_guesses=None):
        """
        Initialize the parameters.  
        """

        self._params = {}

        if param_names == None:
            param_names = []
        if param_guesses == None:
            param_guesses = []

        # Grab parameter names and guesses from the self.param_definition function
        a = inspect.getargspec(self.param_definition)

        if type(a.args) != None:

            args = list(a.args)
            try:
                args.remove("self")
            except ValueError:
                pass

        if len(args) != 0:
                
            if len(args) != len(a.defaults):
                err = "all parameters in self.param_definition must have a default value.\n"
                raise ValueError(err)

            param_names.extend(args)
            param_guesses.extend(list(a.defaults))
        
        # Add modeled error in the observable TO BE IMPLEMENTED
        #param_names.extend(["obs_err"])
        #param_guesses.extend([1e-3])
        
        for i, p in enumerate(param_names):
            self._params[p] = fit_param.FitParameter(p,guess=param_guesses[i])

        self._param_names = param_names[:]
        self._param_names.sort()


    # -------------------------------------------------------------------------
    # parameter names

    @property
    def param_names(self):
        """
        The parameters for the model.
        """

        return self._param_names

    # -------------------------------------------------------------------------
    # parameter objects

    @property
    def parameters(self):
        """
        Return FitParam objects associated with the model.
        """

        return self._params

    # -------------------------------------------------------------------------
    # parameter values

    @property
    def param_values(self):
        """
        Values for each parameter in the model.
        """

        return dict([(p,self._params[p].value) for p in self._param_names])  
 

    def update_values(self,param_values):
        """
        Update parameter values for fit. param_values is a dictionary with
        with some number of parameter names.
        """

        for p in param_values.keys():
            self._params[p].value = param_values[p]

    # -------------------------------------------------------------------------
    # parameter stdev

    @property
    def param_stdevs(self):
        """
        Standard deviation for each parameter in the model.
        """

        return dict([(p,self._params[p].stdev) for p in self._param_names])  
 

    def update_param_stdevs(self,param_stdevs):
        """
        Update parameter stdev for fit. param_stdevs is a dictionary with
        with some number of parameter names.
        """

        for p in param_stdevs.keys():
            self._params[p].stdev = param_stdevs[p]

    # -------------------------------------------------------------------------
    # parameter ninetyfive

    @property
    def param_ninetyfives(self):
        """
        95% confidence intervals for each parameter in the model.
        """

        return dict([(p,self._params[p].ninetyfive) for p in self._param_names])  
 

    def update_ninetyfives(self,param_ninetyfives):
        """
        Update parameter 95% for fit. param_ninetyfives is a dictionary with
        with some number of parameter names.
        """

        for p in param_ninetyfives.keys():
            self._params[p].ninetyfive = param_ninetyfives[p]

    # -------------------------------------------------------------------------
    # parameter guesses

    @property
    def param_guesses(self):
        """
        Guesses for each parameter in the model.
        """

        return dict([(p,self._params[p].guess) for p in self._param_names])  

    def update_guesses(self,param_guesses):
        """
        Update parameter guesses for fit. param_guesses is a dictionary with
        with some number of parameter names.
        """

        for p in param_guesses.keys():
            self._params[p].guess = param_guesses[p]

    # -------------------------------------------------------------------------
    # parameter ranges

    @property
    def param_guess_ranges(self):
        """
        Return parameter ranges.
        """

        return dict([(p,self._params[p].guess_range) for p in self._param_names])  

    def update_guess_ranges(self,param_ranges):
        """
        Update parameter ranges.  param_ranges is a dictionary of paramters
        keyed to two-entry lists/tuples or ranges.
        """

        for p in param_ranges.keys():
            self._params[p].guess_range = param_ranges[p]


    # -------------------------------------------------------------------------
    # fixed parameters

    @property
    def fixed_param(self):
        """
        Return the fixed parameters.
        """

        return dict([(p,self._params[p].fixed) for p in self._param_names])  

    def update_fixed(self,fixed_param):
        """
        Fix parameters.  fixed_param is a dictionary of parameters keyed to their
        fixed values.  If the value is None, the parameter is removed from the
        fixed parameters dictionary and will float.
        """

        for p in fixed_param.keys():
        
            if fixed_param[p] == None:
                self._params[p].fixed = False
            else:
                self._params[p].fixed = True
                self._params[p].value = fixed_param[p]


    # -------------------------------------------------------------------------
    # types of prior distribution for each parameter

    @property
    def dist_type(self):
        """
        Return parameter prior distribution types.
                None - no uncertainty in the concentrations
                1 - normally distributed uncertainty
                2 - uniform uncertainty between two bounds
                3 - normally distributed uncertainty but with different errors
                    for each concentration data point (TO BE IMPLEMENTED)
        """
        return dict([(p,self._params[p].dist_type) for p in self._param_names])
        
    def update_dist_type(self,dist_type):
        """
        Update parameter prior distribution types.  dist_type is a dictionary of paramters
        keyed to one-entry lists/tuples or ranges.
        """

        for p in dist_type.keys():
            self._params[p].dist_type = dist_type[p]
        
    # -------------------------------------------------------------------------
    # parameter distribution variables

    @property
    def dist_vars(self):
        """
        Return parameter prior distribution variables.
        """

        return dict([(p,self._params[p].dist_vars) for p in self._param_names])  

    def update_dist_vars(self,dist_vars):
        """
        Update parameter prior distribution variables.  dist_vars is a dictionary of paramters
        keyed to two-entry lists/tuples or ranges.
        """

        for p in dist_vars.keys():
            self._params[p].dist_vars = dist_vars[p]

    # -------------------------------------------------------------------------
    # parameter aliases

    @property
    def param_aliases(self):
        """
        Return parameter aliases.
        """

        return dict([(p,self._params[p].alias) for p in self._param_names
                     if self._params[p].alias != None])  

    def update_aliases(self,param_alias):
        """
        Update parameter aliases.  param_alias is a dictionary of parameters keyed
        to their aliases (used by the global fit).  If the value is None, the parameter
        alias is removed.
        """

        for p in param_alias.keys():
            self._params[p].alias = param_alias[p]
