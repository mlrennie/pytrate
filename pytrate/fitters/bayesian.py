__description__ = \
"""
Fitter subclass for performing bayesian (MCMC) fits.
"""
__author__ = ""
__date__ = ""

from .base import Fitter

import emcee, corner

import numpy as np
import scipy.optimize as optimize

import multiprocessing

class BayesianFitter(Fitter):
    """
    """
    def __init__(self,num_walkers=100,initial_walker_spread=1e-4,ml_guess=False,
                 num_steps=100,burn_in=0.1,num_threads=1):
        """
        Initialize the bayesian fitter

        Parameters
        ----------       
 
        num_walkers : int > 0
            how many markov chains to have in the analysis
        initial_walker_spread : float
            each walker is initialized with parameters sampled from normal 
            distributions with mean equal to the initial guess and a standard
            deviation of guess*initial_walker_spread 
        ml_guess : bool
            if true, do an ML optimization to get the initial guess
        num_steps:
            number of steps to run the markov chains
        burn_in : float between 0 and 1
            fraction of samples to discard from the start of the run
        num_threads : int or `"max"`
            number of threads to use.  if `"max"`, use the total number of 
            cpus. [NOT YET IMPLEMENTED] 
        """
        
        if(ml_guess):
            raise ValueError("Maximum likelihood not functional at present, please switch off")
        
        Fitter.__init__(self)

        self._num_walkers = num_walkers
        self._initial_walker_spread = initial_walker_spread
        self._ml_guess = ml_guess
        self._num_steps = num_steps
        self._burn_in = burn_in

        self._num_threads = num_threads
        if self._num_threads == "max":
            self._num_threads = multiprocessing.cpu_count()

        if not type(self._num_threads) == int and self._num_threads > 0:
            err = "num_threads must be 'max' or a positive integer\n"
            raise ValueError(err)

        if self._num_threads != 1:
            err = "multithreading has not yet been (fully) implemented.\n"
            raise NotImplementedError(err)

        self._success = None

        self.fit_type = "bayesian"    
    
    def ln_prior(self,param):
        """
        Log priors of the fit parameters. Priors are either 
                 (0) uninformative (no uncertainty)
                 (1) uniform distributed between two bounds
                 (2) normally distributed
                 (3) normally distributed uncertainty but with different errors
                                   for each concentration data point (TO BE IMPLEMENTED)
        
        Parameters
        ----------

        param : array of floats
            parameters to fit
        types : array of types
        
        Returns
        -------

        float value for log of prior. 
        """
        
        # if there is no uncertainty the ln_prior is zero
        n_params = len(param)
        ln_prior_count = np.zeros(n_params)
                
        # treat each parameter separately, allowing for different types
        for i in range(n_params):
            if(self._dist_types[i]==1):
                # If the parameter falls outside of the bounds, make the prior -infinity
                if (param[i] < self._dist_vars[0,i]) or (param[i] > self._dist_vars[1,i]):
                    return -np.inf
                # Otherwise does not contribute to the probability
            
            elif(self._dist_types[i]==2):
                mu = self._dist_vars[0,i]
                sigma2 = self._dist_vars[1,i]**2
                ln_prior_count[i] = -0.5*(np.sum((param[i] - mu)**2/sigma2 + np.log(sigma2)))
            
            elif(self._dist_types[i]==3):
                err = "Separate errors for each concentration point has not been implemented yet"
                raise NotImplementedError(err)
        
        return np.sum(ln_prior_count)   
        
    def ln_prob(self,param):
        """
        Posterior probability of model parameters.

        Parameters
        ----------

        param : array of floats
            parameters to fit

        Returns
        -------

        float value for log posterior proability
        """

        # Calculate prior.  If not finite, this solution has an -infinity log 
        # likelihood
        ln_prior = self.ln_prior(param)
        if not np.isfinite(ln_prior):
            return -np.inf
        
        # Calcualte likelihood.  If not finite, this solution has an -infinity
        # log likelihood
        ln_like = self.ln_like(param)
        if not np.isfinite(ln_like):
            return -np.inf
        
        #print(ln_like,ln_prior)
        # log posterior is log priors plus log likelihood 
        return ln_like + ln_prior

    def fit(self,model,parameters,dist_types,dist_vars,y_obs,y_err=None,param_names=None):
        """
        Fit the parameters.       
 
        Parameters
        ----------

        model : callable
            model to fit.  model should take "parameters" as its only argument.
            this should (usually) be GlobalFit._y_calc
        parameters : array of floats
            parameters to be optimized.  usually constructed by GlobalFit._prep_fit
        dist_types : list
            list containing prior distribution types
        dist_vars : list
            list of two lists containing prior distribution variables
        y_obs : array of floats
            observations in an concatenated array
        y_err : array of floats or None
            standard deviation of each observation.  If None, a normal error,
            constant across observations, is assumed            
        param_names : array of str
            names of parameters.  If None, parameters assigned names p0,p1,..pN
        """

        self._model = model
        self._y_obs = y_obs

        # Convert the distribution types into a 1d numpy array
        self._dist_types = np.array(dist_types)
        
        # Convert the distribution variables into a 2d numpy array
        self._dist_vars = np.array(dist_vars)

        if param_names is None:
            self._param_names = ["p{}".format(i) for i in range(len(parameters))]
        else:
            self._param_names = param_names[:] 

        # TO IMPLEMENT: If no error is specified, assign as a fit variable 
        self._y_err = y_err
        #####
        
        # Make initial guess (ML or just whatever the paramters sent in were)
        # TO IMPLEMENT: Bound fitting
        if self._ml_guess:
            fn = lambda *args: -self.weighted_residuals(*args)
            ml_fit = optimize.least_squares(fn,x0=parameters)
            self._initial_guess = np.copy(ml_fit.x)
        else:
            self._initial_guess = np.copy(parameters)
        
        # Create walker positions 

        # Size of perturbation in parameter depends on the scale of the parameter 
        perturb_size = self._initial_guess*self._initial_walker_spread
 
        ndim = len(parameters)
        pos = [self._initial_guess + np.random.randn(ndim)*perturb_size
               for i in range(self._num_walkers)]

        # Sample using walkers
        self._fit_result = emcee.EnsembleSampler(self._num_walkers, ndim, self.ln_prob,
                                                 threads=self._num_threads)
        self._fit_result.run_mcmc(pos, self._num_steps)

        # Create list of samples
        to_discard = int(round(self._burn_in*self._num_steps,0))
        self._samples = self._fit_result.chain[:,to_discard:,:].reshape((-1,ndim))
        self._lnprob = self._fit_result.lnprobability[:,:].reshape(-1)

        # Get mean and standard deviation 
        self._estimate = np.mean(self._samples,axis=0)
        self._stdev = np.std(self._samples,axis=0)

        # Calculate 95% confidence intervals
        self._ninetyfive = []
        lower = int(round(0.025*self._samples.shape[0],0))
        upper = int(round(0.975*self._samples.shape[0],0))
        for i in range(self._samples.shape[1]):
            nf = np.sort(self._samples[:,i])
            self._ninetyfive.append([nf[lower],nf[upper]])

        self._ninetyfive = np.array(self._ninetyfive)

        self._success = True
        
        print("Mean acceptance fraction: {0:.3f} \n"
                .format(np.mean(self._fit_result.acceptance_fraction)))

    @property
    def fit_info(self):
        """
        Information about the Bayesian run.
        """

        output = {}
        output["Num walkers"] = self._num_walkers
        output["Initial walker spread"] = self._initial_walker_spread
        output["Use ML guess"] = self._ml_guess
        output["Num steps"] = self._num_steps
        output["Burn in"] = self._burn_in
        output["Final sample number"] = len(self._samples[:,0])
        output["Num threads"] = self._num_threads
        
        return output

    @property
    def samples(self):
        """
        Bayesian samples.
        """
        
        return self._samples

