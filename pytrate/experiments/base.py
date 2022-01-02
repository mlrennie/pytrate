__description__ = \
"""
experiments.py

Classes for loading experimental steady state data and associating those data with a
model.

Units: 
    
"""
__author__ = ""
__date__ = ""

import random, string, os
import csv
import numpy as np
import sys

from ..fixed_values import AVAIL_CONC_UNITS

class BaseExperiment:
    """
    Class that holds an experiment and a model that describes it.
    """
    
    def __init__(self,data_file,model,uncertainty=1e-12,
                 **model_kwargs):
        """

        Parameters
        ----------

        data_file: string
            ...
        model: Model subclass instance
            Model subclass to use for modeling
        uncertainty: float > 0.0
            uncertainty in observations, assumed to be constant.
            TO IMPLEMENT-read in measured uncertainties
 
        **model_kwargs: any keyword arguments to pass to the model.  Any
                        keywords passed here will override whatever is
                        stored in the data_file.
        """

        self.data_file = data_file

        # load data observations
        extension = self.data_file.split(".")[-1]
        (self._concs,self._obs) = self._read_data(self.data_file)
        
        self._obs_meas = self._obs[0]
        # set the observed errors read from the file as standard deviations
        # TO IMPLEMENT standard errors from the data
        self._obs_stdev = self._obs[1]
        self._obs_stdev[self._obs_stdev==0] = uncertainty
        
        # Initialize model using information read from data file
        self._model = model(self._concs,
                            self._obs,
                            **model_kwargs)

        r = "".join([random.choice(string.ascii_letters) for i in range(20)])
        self._experiment_id = "{}_{}".format(self.data_file,r)


    @property
    def obs_calc(self):
        """
        Return observables calculated by the model with parameters defined in params
        dictionary.
        """

        if len(self._model.obs_calc) == 0:
            return np.array(())

        return np.array(self._model.obs_calc)

    @property
    def param_values(self):
        """
        Values of fit parameters.
        """

        return self._model.param_values

    @property
    def param_stdevs(self):
        """
        Standard deviations on fit parameters.
        """

        return self._model.param_stdevs

    @property
    def param_ninetyfives(self):
        """
        95% confidence intervals on fit parmeters.
        """

        return self._model.param_ninetyfives

    @property
    def model(self):
        """
        Fitting model.
        """

        return self._model

    @property
    def obs_meas(self):
        """
        Return experimental observables.
        """
        return self._obs_meas

    @obs_meas.setter
    def obs_meas(self,obs):
        """
        Set the observables.
        """
        
        self._obs = obs_meas

    @property
    def obs_stdev(self):
        """
        Standard deviation on the uncertainty of the observables.
        """

        return self._obs_stdev

    @obs_stdev.setter
    def obs_stdev(self,obs_stdev):
        """
        Set the standard deviation on the uncertainty of the observables.
        """

        self._obs_stdev = obs_stdev

    @property
    def mole_ratio(self):
        """
        Return the mole ratio of titrant to stationary.
        """
        return self._model.mole_ratio

    @property
    def experiment_id(self):
        """
        Return a unique experimental id.
        """

        return self._experiment_id
    
    # -------------------------------------------------------------------------        
    # function to read in the data in a specified format
    def _read_data(self,file_name):
        """
        Function to read experimental details for a biophysical/biochemical 
        titration from a standard template.
        
        Returns an ordered list containing numpy arrays for the 
        concentrations, concentration uncertainties, 
        observations, and observation uncertainties.
        """
        
        # initialize some things
        err_obs = 'No'
        err_concs = 'No'
        data = []
        header = []
        
        # read in file and determine header region
        # update to read UTF-8 from Excel
        with open(file_name,newline='') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.readline())
            csvfile.seek(0)
            reader = csv.reader(csvfile,dialect)
            
            for row in reader:
                try:
                    float(row[0])
                    data.append(row)
                except ValueError:
                    header.append(row)
        
        # process header information        
        for head in header:
            # essential header information
            if(head[0] == 'Conc units:'):
                try:
                    conc_conv = AVAIL_CONC_UNITS[head[1]]
                except KeyError:
                    sys.exit("ERROR: Concentration units must be one of",
                                     "'M', 'mM', 'uM', or 'nM'.\n ",
                                     "Check for erroneous spaces.")
            # TO IMPLEMENT: CONVERT OBSERVABLE UNITS
            elif(head[0] == 'Obs units:'):
                obs_units = head[1]
            
            # optional header information
            elif(head[0] == 'Type:'):
                exp_type = head[1]
            
            else:
                data_head = head
        
        data = list(map(list, zip(*data)))
        if(len(data_head) != len(data)):
            sys.exit("ERROR: Number of data columns and headers do not match.",
                     "\n       All columns must have headers.")
        
        concs = []
        obs = []
        concs_err = []
        obs_err = []
        
        n_concs = 0
        n_obs   = 0
        
        for i in range(len(data_head)):
            # err columns are processed with the column to which it represents
            # the uncertainty
            if(data_head[i][-3:].lower()=='err'):
                continue
            
            # process conc columns
            if(data_head[i][:4].lower()=='conc'):
                if(data_head[i+1][-3:].lower()=='err'):
                    if(data_head[i][4:].lower()==data_head[i+1][4:-4].lower()):
                        concs.append(data[i])
                        concs_err.append(data[i+1])
                        n_concs = n_concs+1
                    else:
                        sys.exit("ERROR: Columns must have format of ",
                                 "'conc xxx',with 'conc xxx err' adjacent")
                else:
                    concs.append(data[i])
                    concs_err.append([None]*len(data[i]))
                    n_concs = n_concs+1
            
            # process obs columns TO IMPLEMENT ERROR COLUMNS
            #if(data_head[i][:3].lower()=='obs' and len(data_head)>i+1):
            #    if(data_head[i+1][-3:].lower()=='err'):
            #        if(data_head[i][3:].lower()==data_head[i+1][3:-4].lower()):
            #            obs.append(data[i])
            #            obs_err.append(data[i+1])
            #            print("\nIMPORTANT: Reading in an 'err' column -",
            #                  "assumed to be STANDARD DEVIATIONs.\n")
            #        else:
            #            sys.exit("ERROR: Columns must have format of ",
            #                     "'obs xxx' with 'obs xxx err' adjacent")
            if(data_head[i][:3].lower()=='obs'):
                # if no error given estimate from data
                obs.append(data[i])
                obs_err.append([0.]*len(data[i]))
                n_obs = n_obs+1
        
        # convert to numpy arrays - remove overhead
        try:
            concs_and_err = np.array([concs,concs_err],dtype='float64')
            del concs, concs_err
        except ValueError:
            sys.exit("ERROR: Missing data point")
        try:
            obs_and_err = np.array([obs,obs_err],dtype='float64')
            del obs, obs_err
        except ValueError:
            sys.exit("ERROR: Missing data point")
                
		# convert concentrations to molar (M)
        return (conc_conv*concs_and_err,obs_and_err)
            