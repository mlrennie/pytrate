__description__ = \
"""
global_fit.py

Classes for doing nonlinear regression of global models against multiple ITC
experiments.
"""
__author__ = ""
__date__ = ""

from . import fitters
from . global_connectors import GlobalConnector

import numpy as np
import scipy
import scipy.optimize as optimize
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d.axes3d import Axes3D

import copy, inspect, warnings, sys, datetime

class FitNotRunError(Exception):
    """
    Throw when the fit has not been run but the output only makes sense after
    the fit has been done.
    """

    pass


class GlobalFit:
    """
    Class for regressing models against an arbitrary number of ITC experiments.
    """

    def __init__(self):
        """
        Set up the main binding model to fit.
        """

        # Objects for holding global parameters
        self._global_param_keys = []
        self._global_params = {}
        self._global_param_mapping = {}

        # List of experiments 
        self._expt_dict = {}
        self._expt_list_stable_order = []

    def add_experiment(self,experiment):
        """
        Add an experiment to the fit

        Parameters
        ----------

        experiment: an ITCExperiment instance
        """

        name = experiment.experiment_id

        # Record the experiment
        self._expt_dict[name] = experiment
        self._expt_list_stable_order.append(name)

        # Delete the fitter if we remove an experiment.  It is no longer valid
        self.delete_current_fit()


    def remove_experiment(self,experiment):
        """
        Remove an experiment from the analysis.

        Parameters
        ----------
        
        experiment: an ITCExperiment instance
        """

        expt_name = experiment.experiment_id

        # Go through all global parameters
        global_param_map = list(self._global_param_mapping.keys())
        for k in global_param_map:

            # If the experiment links to that
            for expt in self._global_param_mapping[k]:
                if expt_name == expt[0]:
                    self._global_param_mapping[k].remove(expt)

                    if len(self._global_param_mapping[k]) == 0:
                        self.remove_global(k)

        self._expt_dict.pop(expt_name)
        self._expt_list_stable_order.remove(expt_name)

        # Delete the fitter if we remove an experiment.  It is no longer valid
        self.delete_current_fit()

    def link_to_global(self,expt,expt_param,global_param_name):
        """
        Link a local experimental fitting parameter to a global fitting
        parameter.

        Parameters
        ----------

        expt : an ITCExperiment instance
        expt_param : string
            key pointing to experimental parameter
        global_param_name : string OR global_connector method
            the global parameter this individual parameter should point to
        """

        expt_name = expt.experiment_id

        # If the experiment hasn't already been added to the global fitter, add it
        try:
            self._expt_dict[expt_name]
        except KeyError:
            self.add_experiment(expt)

        # Make sure the experimental paramter is actually in the experiment
        if expt_param not in self._expt_dict[expt_name].model.param_names:
            err = "Parameter {} not in experiment {}\n".format(expt_param,expt_name)
            raise ValueError(err)

        # Update the alias from the experiment side
        self._expt_dict[expt_name].model.update_aliases({expt_param:
                                                         global_param_name})

        # Update the alias from the global side
        e = self._expt_dict[expt_name]
        if global_param_name not in self._global_param_keys:

            # Make new global parameter and create link
            self._global_param_keys.append(global_param_name)
            self._global_param_mapping[global_param_name] = [(expt_name,expt_param)]

            # If this is a "dumb" global parameter, store a FitParameter
            # instance with the data in it.
            if type(global_param_name) == str:
                self._global_params[global_param_name] = copy.copy(e.model.parameters[expt_param])

            # If this is a GlobalConnector method, store the GlobalConnector
            # instance as the paramter
            elif issubclass(global_param_name.__self__.__class__,GlobalConnector):
                self._global_params[global_param_name] = global_param_name.__self__

            else:
                err = "global variable class not recongized.\n"
                raise ValueError(err)

        else:
            # Only add link, but do not make a new global parameter
            if expt_name not in [m[0] for m in self._global_param_mapping[global_param_name]]:
                self._global_param_mapping[global_param_name].append((expt_name,expt_param))

        self.delete_current_fit()

    def unlink_from_global(self,expt,expt_param):
        """
        Remove the link between a local fitting parameter and a global
        fitting parameter.

        Parameters
        ----------

        expt : ITCExperiment instance
        expt_param : string
            experimental parameter to unlink from global
        """

        expt_name = expt.experiment_id

        # Make sure the experimental parameter is actually in the experiment
        if expt_param not in self._expt_dict[expt_name].model.param_names:
            err = "Parameter {} not in experiment {}\n".format(expt_param,expt_name)
            raise ValueError(err)

        global_name = self._expt_dict[expt_name].model.parameters[expt_param].alias

        # remove global --> expt link
        self._global_param_mapping[global_name].remove((expt_name,expt_param))
        if len(self._global_param_mapping[global_name]) == 0:
            self.remove_global(global_name)

        # remove expt --> global link
        self._expt_dict[expt_name].model.update_aliases({expt_param:None})

        self.delete_current_fit()

    def remove_global(self,global_param_name):
        """
        Remove a global parameter, unlinking all local parameters.

        global_param_name: string
            global parameter name
        """

        if global_param_name not in self._global_param_keys:
            err = "Global parameter {} not defined.\n".format(global_param_name)
            raise ValueError(err)

        # Remove expt->global mapping from each experiment
        for k in self._global_param_mapping.keys():

            for expt in self._global_param_mapping[k]:

                expt_name = expt[0]
                expt_params = self._expt_dict[expt_name].model.param_aliases.keys()
                for p in expt_params:
                    if self._expt_dict[expt_name].model.param_aliases[p] == global_param_name:
                        self._expt_dict[expt_name].model.update_aliases({p:None})
                        break

        # Remove global data
        self._global_param_keys.remove(global_param_name)
        self._global_param_mapping.pop(global_param_name)
        self._global_params.pop(global_param_name)

        self.delete_current_fit()

    def fit(self,fitter=fitters.MLFitter):
        """
        Public function that performs the fit. 
        
        Parameters
        ----------

        fitter : subclass of fitters.Fitter
            Fitter specifies how the fit should be done.  It defaults to a 
            maximum-likelihood method.  If the subclass is passed, it is
            initialized with default parameters.  If an instance of the 
            subclass is passed, it will be used as-is. 
        """

        # Prep the fit (creating arrays that properly map between the the
        # Mapper instance and numpy arrays for regression).
        self._prep_fit()
       
        # If the fitter is not intialized, initialize it 
        if inspect.isclass(fitter):
            self._fitter = fitter()
        else:
            self._fitter = fitter
        
        # Perform the fit.
        self._fitter.fit(self._y_calc,
                         self._flat_param,
                         self._flat_param_dist_type,
                         self._flat_dist_vars,
                         self._y_obs,
                         self._y_err,
                         self._flat_param_name)

        # Take the output of the fit (numpy arrays) and map it back to specific
        # parameters using Mapper.
        self._parse_fit()

    def _prep_fit(self):
        """
        Prep the fit, creating all appropriate parameter mappings etc.
        """

        self._flat_param = []
        self._flat_param_dist_type = []
        self._flat_dist_vars = [[],[]]
        self._flat_param_mapping = []
        self._flat_param_type = []
        self._flat_param_name = []

        self._flat_global_connectors_seen = []

        flat_param_counter = 0

        # Go through global variables
        for k in self._global_param_mapping.keys():
    
            # Otherwise, there is just one parameter to enumerate over.
            if type(k) == str:
                enumerate_over = {k:self._global_params[k]}
                param_type = 1
 
            # If this is a global connector, enumerate over all parameters in
            # that connector 
            elif issubclass(k.__self__.__class__,GlobalConnector):

                if k.__self__ in self._flat_global_connectors_seen:
                    continue

                enumerate_over = self._global_params[k].params
                self._flat_global_connectors_seen.append(k.__self__)
                param_type = 2

            else:
                err = "global variable class not recongized.\n"
                raise ValueError(err)
            
            # Now update parameter values, bounds, and mapping 
            for e in enumerate_over.keys(): 

                # write fixed parameter values to the appropriate experiment,
                # then skip
                if enumerate_over[e].fixed:
                    fixed_value = enumerate_over[e].value
                    for expt, expt_param in self._global_param_mapping[k]:
                        self._expt_dict[expt].model.update_fixed({expt_param:fixed_value}) 
                    continue

                self._flat_param.append(enumerate_over[e].guess)
                self._flat_param_dist_type.append(enumerate_over[e].dist_type)
                self._flat_dist_vars[0].append(enumerate_over[e].dist_vars[0])
                self._flat_dist_vars[1].append(enumerate_over[e].dist_vars[1])
                self._flat_param_mapping.append((k,e))
                self._flat_param_type.append(param_type)
                self._flat_param_name.append(e)

                flat_param_counter += 1

        # Go through every experiment
        y_obs = []
        for k in self._expt_dict.keys():                                       

            e = self._expt_dict[k]                                             

            for p in e.model.param_names:

                # If the parameter is fixed, ignore it.
                if e.model.fixed_param[p]:
                    continue

                # If the parameter is global, ignore it.
                try:
                    e.model.param_aliases[p]
                    continue
                except KeyError:
                    pass

                # If not fixed or global, append the parameter to the list of
                # floating parameters
                self._flat_param.append(e.model.param_guesses[p])
                self._flat_param_dist_type.append(e.model.dist_type[p])
                self._flat_dist_vars[0].append(e.model.dist_vars[p][0])
                self._flat_dist_vars[1].append(e.model.dist_vars[p][1])
                self._flat_param_mapping.append((k,p))
                self._flat_param_type.append(0)
                self._flat_param_name.append(p)

                flat_param_counter += 1

        # Create observed y and y err arrays for the likelihood function
        y_obs = []
        y_err = []
        for k in self._expt_dict.keys():                                       
            y_obs.extend(self._expt_dict[k].obs_meas)
            y_err.extend(self._expt_dict[k].obs_stdev)

        self._y_obs = np.array(y_obs)
        self._y_err = np.array(y_err)

    def _y_calc(self,param=None):
        """
        Calculate observations using the given model parameters.
        """
        
        # Update parameters
        for i in range(len(param)):

            # local variable
            if self._flat_param_type[i] == 0:
                experiment = self._flat_param_mapping[i][0]
                parameter_name = self._flat_param_mapping[i][1]
                self._expt_dict[experiment].model.update_values({parameter_name:param[i]})    

            # Vanilla global variable
            elif self._flat_param_type[i] == 1:
                param_key = self._flat_param_mapping[i][0]
                for experiment, parameter_name in self._global_param_mapping[param_key]:
                    self._expt_dict[experiment].model.update_values({parameter_name:param[i]}) 

            # Global connector global variable
            elif self._flat_param_type[i] == 2:
                connector = self._flat_param_mapping[i][0].__self__
                param_name = self._flat_param_mapping[i][1]
                connector.update_values({param_name:param[i]})

            else:
                err = "Paramter type {} not recongized.\n".format(self._flat_param_type[i])
                raise ValueError(err) 

        # Look for connector functions
        for connector_function in self._global_param_keys:            

            if type(connector_function) == str:
                continue

            # If this is a method of GlobalConnector...
            if issubclass(connector_function.__self__.__class__,GlobalConnector):

                # Update experiments with the value spit out by the connector function
                for expt, param in self._global_param_mapping[connector_function]: 
                    e = self._expt_dict[expt]                                                 
                    value = connector_function(e)
                    self._expt_dict[expt].model.update_values({param:value})                  
        
        # Calculate using the model         
        y_calc = []
        for k in self._expt_dict.keys(): 
            y_calc.extend(self._expt_dict[k].obs_calc)
        
        return np.array(y_calc)

    def _parse_fit(self):
        """
        Parse the fit results.
        """

        # Store the result
        for i in range(len(self._fitter.estimate)):
                
            # local variable
            if self._flat_param_type[i] == 0:

                experiment = self._flat_param_mapping[i][0]
                parameter_name = self._flat_param_mapping[i][1]

                self._expt_dict[experiment].model.update_values({parameter_name:self._fitter.estimate[i]})
                self._expt_dict[experiment].model.update_param_stdevs({parameter_name:self._fitter.stdev[i]})
                self._expt_dict[experiment].model.update_ninetyfives({parameter_name:self._fitter.ninetyfive[i]})

            # Vanilla global variable
            elif self._flat_param_type[i] == 1:

                param_key = self._flat_param_mapping[i][0]
                for k, p in self._global_param_mapping[param_key]:
                    self._expt_dict[k].model.update_values({p:self._fitter.estimate[i]})
                    self._global_params[param_key].value = self._fitter.estimate[i]
                    self._global_params[param_key].stdev = self._fitter.stdev[i]
                    self._global_params[param_key].ninetyfive = self._fitter.ninetyfive[i]

            # Global connector global variable
            elif self._flat_param_type[i] == 2:
                connector = self._flat_param_mapping[i][0].__self__
                param_name = self._flat_param_mapping[i][1]

                # HACK: if you use the params[param_name].value setter function,
                # it will break the connector.  This is because I expose the 
                # thermodynamic-y stuff of interest via .__dict__ rather than 
                # directly via params.  So, this has to use the .update_values
                # method.
                connector.update_values({param_name:self._fitter.estimate[i]})
                connector.params[param_name].stdev = self._fitter.stdev[i]
                connector.params[param_name].ninetyfive = self._fitter.ninetyfive[i]

            else:
                err = "Paramter type {} not recognized.\n".format(self._flat_param_type[i])
                raise ValueError(err) 

    def delete_current_fit(self):
        """
        Delete the current experiment (if it exists).
        """

        try:
            del self._fitter
        except AttributeError:
            pass

        self._prep_fit()

    def plot(self,correct_concs=False,color_list=None,
             data_symbol="o",linewidth=1.5,num_samples=100,
             logplot=True,ploterrors=True,figsize=(5.5,6),
             smooth=False):
        """
        Plot the experimental data and fit results.

        Parameters
        ----------
        correct_concs : bool
            correct the total concentrations using fx_competent
        color_list : list of things matplotlib can interpret as colors
            color of each series
        data_symol : character
            symbol to use to plot data
        linewidth : float
            width of line for fits
        num_samples : int 
            number of samples to draw when drawing fits like Bayesian fits with
            multiple fits.
        logplot : bool
            plot data and fit on a log scale for the independent variable      
        ploterrors : bool
            include error bars on the plot
        smooth : bool
            make smooth curves (in testing)

        Returns matplotlib Figure and AxesSubplot instances that can be further
        manipulated by the user of the API.
        """

        # Make graph of appropraite size
        fig = plt.figure(figsize=figsize)

        # Create two panel graph
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) 
        ax = []
        ax.append(fig.add_subplot(gs[0]))
        ax.append(fig.add_subplot(gs[1],sharex=ax[0]))

        # Clean up graphs
        for i in range(2):
            #ax[i].spines['top'].set_visible(False)
            #ax[i].spines['right'].set_visible(False)

            ax[i].yaxis.set_ticks_position('left')
            ax[i].xaxis.set_ticks_position('bottom')

        # Nothing to plot
        if len(self._expt_list_stable_order) < 1:
            return fig, ax

        # Add labels to top plot and remove x-axis
        u = "" # units
        ax[0].set_ylabel("Observable")
        plt.setp(ax[0].get_xticklabels(), visible=False)

        # Add labels to the residuals plot
        x_meas = self._expt_dict[self._expt_list_stable_order[0]].model.x_var
        if(logplot):
            ax[1].semilogx([np.min(x_meas),np.max(x_meas)],[0,0],"--",linewidth=1.0,color="gray")
        else:
            ax[1].plot([np.min(x_meas),np.max(x_meas)],[0,0],"--",linewidth=1.0,color="gray")
        ax[1].set_xlabel(self._expt_dict[self._expt_list_stable_order[0]].model.x_label)
        ax[1].set_ylabel("residuals")

        # Make list of colors
        if color_list == None:
            N = len(self._expt_list_stable_order)
            color_list = [plt.cm.brg(i/N) for i in range(N)]

        # Sanity check on colors
        if len(color_list) < len(self._expt_list_stable_order):
            err = "Number of specified colors is less than number of experiments.\n"
            raise ValueError(err)

        try:
            # If there are samples:
            if len(self._fitter.samples) > 0:
                s = self._fitter.samples
                these_samples = s[np.random.randint(len(s),size=num_samples)]
            else:
                these_samples = [self._fitter.estimate]
        except AttributeError:

            # If fit has not been done, create dummy version
            self._prep_fit()
            these_samples = [np.array(self._flat_param)]

        # If there are multiple samples, assign them partial transparency
        if len(these_samples) == 1:
            alpha = 1.0
        else:
            alpha = 0.1

        for i, s in enumerate(these_samples):

            # Update calculation for this sample
            self._y_calc(s)
            for j, expt_name in enumerate(self._expt_list_stable_order):
                
                # Extract fit info for this experiment
                e = self._expt_dict[expt_name]
                for k in range(e.model.n_obs):
                    
                    x_meas = e.model.x_var
                    y_meas = e.obs_meas[k]
                    calc = self._expt_dict[expt_name].obs_calc[k]
                    
                    # In testing, plot smooth lines
                    if(smooth and logplot):
                        x_smooth = np.logspace(np.log10(np.min(x_meas)),
                                            np.log10(np.max(x_meas)),
                                            500)
                        x2_smooth = np.logspace(np.log10(np.min(e.model.y_var)),
                                            np.log10(np.max(e.model.y_var)),
                                            500)
                        calc = e.model.obs_calc_user(Lt=x_smooth,Pt=x2_smooth)
    
                    #if len(calc) > 0:
                    #
                    #     Try to correct total concentrations for competent fraction
                    #     TO BE IMPLEMENTED
                    #    if correct_concs:
                    #        try:
                    #            ltot = ltot/e.param_values["fx_lig_competent"]
                    #            ptot = ptot/e.param_values["fx_prot_competent"]
                    #            
                    #        except KeyError:
                    #            pass
                    
                    # Draw fit lines and residuals
                    
                    if len(e.obs_calc[k]) > 0:
                        if(logplot and not smooth):
                            ax[0].semilogx(x_meas,calc,color=color_list[k],linewidth=linewidth,alpha=alpha)
                            #ax[1].semilogx(x_meas,(calc-y_meas),data_symbol,color=color_list[j],alpha=alpha,markersize=8) 
                        # In testing, plot smooth lines
                        elif(smooth and logplot):
                            ax[0].semilogx(x_smooth,calc,color=color_list[k],linewidth=linewidth,alpha=alpha)
                        else:
                            ax[0].plot(x_meas,calc,color=color_list[k],linewidth=linewidth,alpha=alpha)
                            #ax[1].plot(x_meas,(calc-y_meas),data_symbol,color=color_list[j],alpha=alpha,markersize=8)     
                    
                    # If this is the last sample, plot the experimental data
                    if(i == len(these_samples) - 1):
                        if(ploterrors):
                            ax[0].errorbar(x_meas,y_meas,e.obs_stdev[k],fmt=data_symbol,color=color_list[k],markersize=8)
                        else:
                            ax[0].plot(x_meas,y_meas,data_symbol,color=color_list[k],markersize=8)
        
        fig.set_tight_layout(True)

        return fig, ax
        
    def corner_plot(self,filter_params=()):
        """
        Create a "corner plot" that shows distributions of values for each
        parameter, as well as cross-correlations between parameters.

        Parameters
        ----------
        param_names : list
            list of parameter names to include.  if None all parameter names
        """
  
        try: 
            return self._fitter.corner_plot(filter_params)
        except AttributeError:
            # If the fit has not been done, return an empty plot
            dummy_fig = plt.figure(figsize=(5.5,6))
            return dummy_fig
 
    # -------------------------------------------------------------------------
    # Properties describing fit results

    @property
    def fit_as_csv(self):
        """
        Return a csv-style string of the fit.
        """

        if len(self._expt_list_stable_order) < 1:
            return "# No experiments loaded."

        out = ["# Fit successful? {}\n".format(self.fit_success)]
        out.append("# {}\n".format(datetime.datetime.now()))

        u = "M/sec"
        out.append("# Units: {}\n".format(u))
        
        fit_stats_keys = list(self.fit_stats.keys())
        fit_stats_keys.sort()
        fit_stats_keys.remove("Fit type")  
 
        out.append("# {}: {}\n".format("Fit type",self.fit_stats["Fit type"])) 
        for k in fit_stats_keys:
            out.append("# {}: {}\n".format(k,self.fit_stats[k]))

        out.append("type,name,exp_file,value,stdev,bot95,top95,fixed,guess,dist_type,dist_var_1,dist_var_2\n") 
        for k in self.fit_param[0].keys():

            param_type = "global"
            data_file = "NA"

            fixed = self.global_param[k].fixed

            param_name = k
            value = self.fit_param[0][k]
            stdev = self.fit_stdev[0][k]
            ninetyfive = self.fit_ninetyfive[0][k]
            guess = self.global_param[k].guess
            dist_type = self.global_param[k].dist_type
            dist_var_1 = self.global_param[k].dist_vars[0]
            dist_var_2 = self.global_param[k].dist_vars[1]

            out.append("{:},{:},{:},{:.5e},{:.5e},{:.5e},{:.5e},{:},{:.5e},{:},{:.5e},{:.5e}\n".format(param_type,
                                                                                                   param_name,
                                                                                                   data_file,
                                                                                                   value,
                                                                                                   stdev,
                                                                                                   ninetyfive[0],
                                                                                                   ninetyfive[1],
                                                                                                   fixed,
                                                                                                   guess,
                                                                                                   dist_type,
                                                                                                   dist_var_1,
                                                                                                   dist_var_2))

        for i in range(len(self.fit_param[1])):

            expt_name = self._expt_list_stable_order[i]

            param_type = "local"
            data_file = self._expt_dict[expt_name].data_file

            for k in self.fit_param[1][i].keys():

                try:
                    alias = self._expt_dict[expt_name].model.parameters[k].alias
                    if alias != None:
                        continue
                except AttributeError:
                    pass

                fixed = self._expt_dict[expt_name].model.parameters[k].fixed

                param_name = k 
                value = self.fit_param[1][i][k]
                stdev = self.fit_stdev[1][i][k]
                ninetyfive = self.fit_ninetyfive[1][i][k]
                guess = self._expt_dict[expt_name].model.parameters[k].guess
                dist_type = self._expt_dict[expt_name].model.parameters[k].dist_type
                dist_var_1 = self._expt_dict[expt_name].model.parameters[k].dist_vars[0]
                dist_var_2 = self._expt_dict[expt_name].model.parameters[k].dist_vars[1]

                out.append("{:},{:},{:},{:.5e},{:.5e},{:.5e},{:.5e},{:},{:.5e},{:},{:.5e},{:.5e}\n".format(param_type,
                                                                                                       param_name,
                                                                                                       data_file,
                                                                                                       value,
                                                                                                       stdev,
                                                                                                       ninetyfive[0],
                                                                                                       ninetyfive[1],
                                                                                                       fixed,
                                                                                                       guess,
                                                                                                       dist_type,
                                                                                                       dist_var_1,
                                                                                                       dist_var_2))


        return "".join(out)
 

    @property
    def global_param(self):
        """
        Return all of the unique global parameters as FitParameter instances.
        """

        connectors_seen = []

        # Global parameters
        global_param = {}
        for g in self._global_param_keys:
            if type(g) == str:
                global_param[g] = self._global_params[g]
            else:
                if g.__self__ not in connectors_seen:
                    connectors_seen.append(g.__self__)
                    for p in g.__self__.params.keys():
                        global_param[p] = g.__self__.params[p]

        return global_param
                

    @property
    def fit_param(self):
        """
        Return the fit results as a dictionary that keys parameter name to fit
        value.  This is a tuple with global parameters first, then a list of
        dictionaries for each local fit.
        """

        # Global parameters
        global_out_param = {}
        for g in self.global_param.keys():
            global_out_param[g] = self.global_param[g].value

        # Local parameters
        local_out_param = []
        for expt_name in self._expt_list_stable_order:
            local_out_param.append(self._expt_dict[expt_name].model.param_values)

        return global_out_param, local_out_param

    @property
    def fit_stdev(self):
        """
        Return the param stdev as a dictionary that keys parameter name to fit
        stdev. This is a tuple with global parameters first, then a list of
        dictionaries for each local fit.
        """

        # Global parameters
        global_out_stdev = {}
        for g in self.global_param.keys():
            global_out_stdev[g] = self.global_param[g].stdev

        # Local parameters
        local_out_stdev = []
        for expt_name in self._expt_list_stable_order:
            local_out_stdev.append(self._expt_dict[expt_name].model.param_stdevs)

        return global_out_stdev, local_out_stdev

    @property
    def fit_ninetyfive(self):
        """
        Return the param 95% confidence as a dictionary that keys parameter name
        confidence. This is a tuple with global parameters first, then a list of
        dictionaries for each local fit.
        """

        # Global parameters
        global_out_ninetyfive = {}
        for g in self.global_param.keys():
            global_out_ninetyfive[g] = self.global_param[g].ninetyfive

        # Local parameters
        local_out_ninetyfive = []
        for expt_name in self._expt_list_stable_order:
            local_out_ninetyfive.append(self._expt_dict[expt_name].model.param_ninetyfives)

        return global_out_ninetyfive, local_out_ninetyfive

    @property
    def fit_success(self):
        """
        Return fit success.
        """
        try:
            return self._fitter.success
        except AttributeError:
            return None

    @property
    def fit_num_obs(self):
        """
        Return the number of observations used for the fit.
        """

        return len(self._y_obs)


    @property
    def fit_num_param(self):
        """
        Return the number of parameters fit.
        """

        return len(self._flat_param)


    @property
    def fit_stats(self):
        """
        Stats about the fit as a dictionary.
        """

        # Only return something if the fit has already been done
        try:
            self._fitter
        except AttributeError:
            return {}

        output = {}
     
        output["num_obs"] = self.fit_num_obs
        output["num_param"] = self.fit_num_param
        output["df"] = self.fit_num_obs - self.fit_num_param
 
        # Create a vector of calcluated and observed values.  
        y_obs = [] 
        y_estimate = []
        for k in self._expt_dict:
            y_obs.extend(self._expt_dict[k].obs_meas)
            y_estimate.extend(self._expt_dict[k].obs_calc)
        y_estimate= np.array(y_estimate)
        y_obs = np.array(y_obs)

        P = self.fit_num_param
        N = self.fit_num_obs
 
        sse = np.sum((y_obs -          y_estimate)**2)
        sst = np.sum((y_obs -      np.mean(y_obs))**2)
        ssm = np.sum((y_estimate - np.mean(y_obs))**2)

        output["Fit type"] = self._fitter.fit_type
        
        fit_info = self._fitter.fit_info
        for x in fit_info.keys():
            output["  {}: {}".format(self._fitter.fit_type,x)] = fit_info[x]

        # Calcluate R**2 and adjusted R**2
        if sst == 0.0:
            output["Rsq"] = np.inf
            output["Rsq_adjusted"] = np.inf
        else:
            Rsq = 1 - (sse/sst)
            Rsq_adjusted = Rsq - (1 - Rsq)*P/(N - P - 1)

            output["Rsq"] = Rsq
            output["Rsq_adjusted"] = Rsq_adjusted
        
        # calculate F-statistic
        msm = (1/P)*ssm
        mse = 1/(N - P - 1)*sse
        if mse == 0.0:
            output["F"] = np.inf
        else:
            output["F"] = msm/mse

        # Calcluate log-likelihood
        lnL = self._fitter.ln_like(self._fitter.estimate) 
        output["ln(L)"] = lnL
        
        # AIC and BIC
        P_all = P + 1 # add parameter to account for implicit residual
        output["AIC"] = 2*P_all  - 2*lnL
        output["BIC"] = P_all*np.log(N) - 2*lnL
        output["AICc"] = output["AIC"] + 2*(P_all + 1)*(P_all + 2)/(N - P_all - 2)

        return output

    # -------------------------------------------------------------------------
    # Properties describing currently loaded parameters and experiments

    @property
    def experiments(self):
        """
        Return a list of associated experiments.
        """

        out = []
        for expt_name in self._expt_list_stable_order:
            out.append(self._expt_dict[expt_name])

        return out

    #--------------------------------------------------------------------------
    # parameter names

    @property
    def param_names(self):
        """
        Return parameter names. This is a tuple of global names and then a list
        of parameter names for each experiment.
        """

        global_param_names = list(self.global_param.keys())

        final_param_names = []
        for expt_name in self._expt_list_stable_order:
            e = self._expt_dict[expt_name]

            param_names = copy.deepcopy(e.model.param_names)

            # Part of the global command names.
            for k in e.model.param_aliases.keys():
                param_names.remove(k)

            final_param_names.append(param_names)

        return global_param_names, final_param_names

    #--------------------------------------------------------------------------
    # parameter aliases

    @property
    def param_aliases(self):
        """
        Return the parameter aliases.  This is a tuple.  The first entry is a
        dictionary of gloal parameters mapping to experiment number; the second
        is a map between experiment number and global parameter names.
        """

        expt_to_global = []
        for expt_name in self._expt_list_stable_order:
            e = self._expt_dict[expt_name]
            expt_to_global.append(copy.deepcopy(e.model.param_aliases))

        return self._global_param_mapping, expt_to_global

    #--------------------------------------------------------------------------
    # parameter guesses

    @property
    def param_guesses(self):
        """
        Return parameter guesses. This is a tuple of global names and then a list
        of parameter guesses for each experiment.
        """

        global_param_guesses = {}
        for p in self.global_param.keys():
            global_param_guesses[p] = self.global_param[p].guess

        final_param_guesses = []
        for expt_name in self._expt_list_stable_order:
            e = self._expt_dict[expt_name]
            param_guesses = copy.deepcopy(e.model.param_guesses)

            for k in e.model.param_aliases.keys():
                param_guesses.pop(k)

            final_param_guesses.append(param_guesses)

        return global_param_guesses, final_param_guesses

    def update_guess(self,param_name,param_guess,expt=None):
        """
        Update the one of the guesses for this fit.  If the experiment is None,
        set a global parameter.  Otherwise, set the specified experiment.

        Parameters
        ----------

        param_name: string 
            name of parameter to set
        param_guess: float
            value to set parameter to
        expt_name: ITCExperiment instance OR None
            experiment to update guess of
        """

        if expt == None:
            try:
                self.global_param[param_name].guess = param_guess
            except KeyError:
                err = "param \"{}\" is not global.  You must specify an experiment.\n".format(param_name)
                raise KeyError(err)
        else:
            self._expt_dict[expt.experiment_id].model.update_guesses({param_name:param_guess})

        self.delete_current_fit()

    #--------------------------------------------------------------------------
    # parameter ranges

    @property
    def param_ranges(self):
        """
        Return the parameter ranges for each fit parameter. This is a tuple.
        Global parameters are first, a list of local parameter ranges are next.
        """

        global_param_ranges = {}
        for p in self.global_param.keys():
            global_param_ranges[p] = self.global_param[p].guess_range

        final_param_ranges = []
        for expt_name in self._expt_list_stable_order:
            e = self._expt_dict[expt_name]
            param_ranges = copy.deepcopy(e.model.param_guess_ranges)

            for k in e.model.param_aliases.keys():
                param_ranges.pop(k)

            final_param_ranges.append(param_ranges)


        return global_param_ranges, final_param_ranges

    def update_range(self,param_name,param_range,expt=None):
        """
        Update the range of a parameter for this fit.  If the experiment is None,
        set a global parameter.  Otherwise, set the specified experiment.

        Parameters
        ----------

        param_name: string 
            name of parameter to set
        param_guess: float
            value to set parameter to
        expt_name: ITCExperiment instance OR None
            experiment to update guess of
        """

        try:
            if len(param_range) != 2:
                raise TypeError
        except TypeError:
            err = "Parameter range must be a list or tuple of length 2"
            raise TypeError(err)

        if expt == None:
            try:
                self.global_param[param_name].guess_range = param_range
            except KeyError:
                err = "param \"{}\" is not global.  You must specify an experiment.\n".format(param_name)
                raise KeyError(err)
        else:
            self._expt_dict[expt.experiment_id].model.update_guess_ranges({param_name:param_range})

        self.delete_current_fit()

    #--------------------------------------------------------------------------
    # fixed parameters

    @property
    def fixed_param(self):
        """
        Return the fixed parameters of the fit.  This is a tuple,  Global fixed
        parameters are first, a list of local fixed parameters is next.
        """

        global_fixed_param = {}
        for p in self.global_param.keys():
            global_fixed_param[p] = self._global_params[p].fixed

        final_fixed_param = []
        for expt_name in self._expt_list_stable_order:

            e = self._expt_dict[expt_name]

            fixed_param = copy.deepcopy(e.model.fixed_param)

            for k in e.model.param_aliases.keys():
                try:
                    fixed_param.pop(k)
                except KeyError:
                    pass

            final_fixed_param.append(fixed_param)

        return global_fixed_param, final_fixed_param

    def update_fixed(self,param_name,param_value,expt=None):
        """
        Fix fit parameters.  If expt is None, set a global parameter. Otherwise,
        fix individual experiment parameters.  if param_value is set to None,
        fixed value is removed.

        Parameters
        ----------

        param_name: string 
            name of parameter to set
        param_guess: float
            value to set parameter to
        expt_name: ITCExperiment instance OR None
            experiment to update guess of
        """

        if expt == None:
            try:
                if param_value == None:
                    self.global_param[param_name].fixed = False
                else:
                    self.global_param[param_name].fixed = True
                    self.global_param[param_name].value = param_value
            except KeyError:
                err = "param \"{}\" is not global.  You must specify an experiment.\n".format(param_name)
                raise KeyError(err)

        else:
            self._expt_dict[expt.experiment_id].model.update_fixed({param_name:param_value})

        self.delete_current_fit()

    # -------------------------------------------------------------------------
    # types of prior distribution for each parameter

    @property
    def param_dist_type(self):
        """
        Return the parameter distribution types for each fit parameter. This is a tuple.
        Global parameters are first, a list of local parameter ranges are next.
        """

        global_param_dist_type = {}
        for p in self.global_param.keys():
            global_param_dist_type[p] = copy.deepcopy(self.global_param[p].dist_type)

        final_param_dist_type = []
        for expt_name in self._expt_list_stable_order:
            e = self._expt_dict[expt_name]
            param_dist_type = copy.deepcopy(e.model.dist_type)

            for k in e.model.param_aliases.keys():
                param_dist_type.pop(k)

            final_param_dist_type.append(param_dist_type)


        return global_param_dist_type, final_param_dist_type

    def update_dist_type(self,param_name,param_dist_type,expt=None):
        """
        Update the distribution types of a parameter for this fit.  If the experiment is None,
        set a global parameter.  Otherwise, set the specified experiment.

        Parameters
        ----------

        param_name: string 
            name of parameter to set
        param_dist_type: float
            value to set parameter prior distribution to
        expt_name: Experiment instance OR None
            experiment to update guess of
        """
        try:
            if param_dist_type != 0 and param_dist_type != 1 and param_dist_type != 2 and param_dist_type != 3:
                raise TypeError
        except TypeError:
            err = "Parameter distribution type must be a an integer between 0 and 3"
            raise TypeError(err)

        if expt == None:
            try:
                self.global_param[param_name].dist_type = param_dist_type
            except KeyError:
                err = "param \"{}\" is not global.  You must specify an experiment.\n".format(param_name)
                raise KeyError(err)
        else:
            self._expt_dict[expt.experiment_id].model.update_dist_type({param_name:param_dist_type})

        self.delete_current_fit()
        
    # -------------------------------------------------------------------------
    # parameter distribution variables

    @property
    def dist_vars(self):
        """
        Return the parameter distributions for each fit parameter. This is a tuple.
        Global parameters are first, a list of local parameter ranges are next.
        """

        global_dist_vars = {}
        for p in self.global_param.keys():
            global_dist_vars[p] = copy.deepcopy(self.global_param[p].dist_vars)

        final_dist_vars = []
        for expt_name in self._expt_list_stable_order:
            e = self._expt_dict[expt_name]
            dist_vars = copy.deepcopy(e.model.dist_vars)

            for k in e.model.param_aliases.keys():
                dist_vars.pop(k)

            final_dist_vars.append(dist_vars)

        return global_dist_vars, final_dist_vars

    def update_dist_vars(self,param_name,dist_vars,expt=None):
        """
        Update the distribution variables of a parameter for this fit.  If the experiment is None,
        set a global parameter.  Otherwise, set the specified experiment.

        Parameters
        ----------

        param_name: string 
            name of parameter to set
        param_guess: float
            value to set parameter to
        expt_name: Experiment instance OR None
            experiment to update guess of
        """

        try:
            if len(dist_vars) != 2:
                raise TypeError
        except TypeError:
            err = "Parameter dist_vars must be a list or tuple of length 2"
            raise TypeError(err)

        if expt == None:
            try:
                self.global_param[param_name].dist_vars = dist_vars
            except KeyError:
                err = "param \"{}\" is not global.  You must specify an experiment.\n".format(param_name)
                raise KeyError(err)
        else:
            self._expt_dict[expt.experiment_id].model.update_dist_vars({param_name:dist_vars})

        self.delete_current_fit()

    #--------------------------------------------------------------------------
    # Functions for updating values directly (used in gui)

    def guess_to_value(self):
        """
        Set all parameter values back to their guesses.
        """       
 
        for p in self.global_param.keys():
            self.global_param[p].value = self.global_param[p].guess

        for expt_name in self._expt_list_stable_order:
            for n, p in self._expt_dict[expt_name].model.parameters.items():
                p.value = p.guess

    def update_value(self,param_name,param_value,expt=None):
        """
        Update the one of the values for this fit.  If the experiment is None,
        set a global parameter.  Otherwise, set the specified experiment.

        Parameters
        ----------

        param_name: string 
            name of parameter to set
        param_guess: float
            value to set parameter to
        expt_name: ITCExperiment instance OR None
            experiment to update guess of
        """

        if expt == None:
            try:
                self.global_param[param_name].value = param_value
            except KeyError:
                err = "param \"{}\" is not global.  You must specify an experiment.\n".format(param_name)
                raise KeyError(err)
        else:
            self._expt_dict[expt.experiment_id].model.update_values({param_name:param_value})

