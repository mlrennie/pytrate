__description__ = \
"""
Main class for holding fit parameters, including guesses, values, ranges, etc.
"""
__date__ = ""
__author__ = ""

import copy
import numpy as np

class FitParameter:
    """
    Class for storing and manipulating generic fit parameters.
    """

    def __init__(self,name,guess=None,guess_range=None,fixed=False,dist_type=None,dist_vars=None,
                 alias=None):
        """
        Initialize class.  Parameters:

        name: name of parameter (string)
        guess: parameter guess (float). If None, guess will be set to 1.0.
        guess_range: range of reasonable guesses (list-like object of 2 floats).
                     If None, there will be no bounds.  
        fixed: whether or not the parameter is fixed (bool)
        dist_type: type of prior distribution
                     0 - no uncertainty in the concentrations
                     1 - uniform uncertainty between two bounds
                     2 - normally distributed uncertainty 
                     3 - normally distributed uncertainty but with different errors
                            for each concentration data point (TO BE IMPLEMENTED)
                     If None, will default to 1.
        dist_vars: variables for the prior distribution on fit for parameter 
                (list-like object of 2 floats). If None, bounds will be set 
                to (None,None).  If (None,5), no lower bound, upper bound of 5.
                If normally distributed, {2,3} then object should be (mu,sd)
        alias: alias for parameter name, for linking to global paramter names. (str)
               If None, no alias is made.
        """

        self.name = name
        self.guess = guess
        self.guess_range = guess_range
        self.fixed = fixed
        self.dist_type = dist_type 
        self.dist_vars = dist_vars        
        self.alias = alias
       
        self._initialize_fit_results() 

    def _initialize_fit_results(self):
        """
        Set fit results to start (stdev, ninetyfive, value to guess).
        """
    
        self.value = self.guess
        self._stdev = np.inf
        self._ninetyfive = [-np.inf,np.inf]

    #--------------------------------------------------------------------------
    # parameter name

    @property
    def name(self):
        """
        Name of the parameter.
        """

        return self._name

    @name.setter
    def name(self,n):
        
        self._name = str(n)

    #--------------------------------------------------------------------------
    # parameter value

    @property
    def value(self):
        """
        Value of the parameter.
        """

        return self._value

    @value.setter
    def value(self,v):
        """
        If value is set to None, set value to self.guess value.
        """

        if v != None:
            self._value = v
        else:
            self._value = self.guess

    #--------------------------------------------------------------------------
    # parameter stdev

    @property
    def stdev(self):
        """
        Standard deviation on the parameter.
        """

        return self._stdev

    @stdev.setter
    def stdev(self,s):
        """
        Set the standard deviation of the parameter.
        """

        self._stdev = s

    #--------------------------------------------------------------------------
    # parameter 95% confidence

    @property
    def ninetyfive(self):
        """
        95% confidence interval on the parameter.
        """

        return self._ninetyfive

    @ninetyfive.setter
    def ninetyfive(self,value):
        """
        Set the 95% confidence interval on the parameter.
        """

        if len(value) != 2:
            err = "ninetyfive requires a list-like with length 2.\n"
            raise ValueError(err)

        self._ninetyfive[0] = value[0]
        self._ninetyfive[1] = value[1]

    #--------------------------------------------------------------------------
    # parameter guess

    @property
    def guess(self):
        """
        Guess for the parameter.
        """

        return self._guess

    @guess.setter
    def guess(self,g):
        """
        Set the guess.  If None, set to 1.0. 
        """

        if g != None:
            self._guess = g
        else:
            self._guess = 1.0

        self._value = self._guess
        self._initialize_fit_results()

    #--------------------------------------------------------------------------
    # parameter guess_range

    @property
    def guess_range(self):
        """
        Range of reasonable guesses for the parameter. 
        """

        return self._guess_range

    @guess_range.setter
    def guess_range(self,g):
        """
        Set range of reasonable guesses.  If None, choose reasonable guess range
        based on parameter name.
        """

        if g != None:
            try:
                if len(g) != 2:
                    raise TypeError
            except TypeError:
                err = "Guess range must be list-like object of length 2.\n"
                raise ValueError(err)

            self._guess_range = copy.deepcopy(g)
        else:
            self._guess_range = [-np.inf,np.inf]

        self._initialize_fit_results()

    #--------------------------------------------------------------------------
    # parameter fixed-ness. 

    @property
    def fixed(self):
        """
        Whether or not the parameter if fixed.
        """

        return self._fixed

    @fixed.setter
    def fixed(self,bool_value):
        """
        Fix or unfix the parameter.
        """
        
        self._fixed = bool(bool_value)
        self._initialize_fit_results()

    #--------------------------------------------------------------------------
    # prior distribution type for fit.

    @property
    def dist_type(self):
        """
        Fit prior distribution type.  Either a integer between 0 and 3 or None.
        """

        return self._dist_type

    @dist_type.setter
    def dist_type(self,t):
        """
        Set fit distribution type. 
        """

        if t != None:
            try:
                if t != 0 and t != 1 and t != 2 and t != 3:
                    raise TypeError
            except TypeError:
                err = "Distribution type must be an integer between 0 and 3\n"
                raise ValueError(err)
        
            self._dist_type = t
            
        else:
            self._dist_type = 1

        self._initialize_fit_results()

    #--------------------------------------------------------------------------
    # prior distribution variables for fit.

    @property
    def dist_vars(self):
        """
        Fit prior distribution variables.  Either list of distribution variables or None.
        """

        return self._dist_vars

    @dist_vars.setter
    def dist_vars(self,b):
        """
        Set fit distribution variables. 
        """

        if b != None:
            try:
                if len(b) != 2:
                    raise TypeError
            except TypeError:
                err = "Distribution variables must be list-like object of length 2\n"
                raise ValueError(err)
        
            self._dist_vars = tuple(copy.deepcopy(b))
            
        else:
            self._dist_vars = (-np.inf,np.inf)

        self._initialize_fit_results()

    #--------------------------------------------------------------------------
    # parameter alias
   
    @property
    def alias(self):
        """
        Parameter alias.  Either string or None.
        """

        return self._alias
    
    @alias.setter
    def alias(self,a):
        """
        Set alias.
        """

        try:
            if self._alias != None and self._alias != a and a != None:
                err = "Could not set alias to {:} because it is already set to {:}".format(a,self._alias)
                raise ValueError(err)
        except AttributeError:
            pass

        self._alias = a

        self._initialize_fit_results()
