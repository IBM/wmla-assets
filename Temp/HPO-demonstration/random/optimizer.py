#!/usr/bin/env python
###############################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2020 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
###############################################################################
import numpy as np
import os,sys

from plugins.core.logger import logger

from plugins.core.base_plugin_opt import BasePluginOptimizer
"""
Define user customized plugin optimizer for hyper parameter tuning
The class name "PluginOptimizer" should not be changed
user need to implement the search method at minimal
"""
class PluginOptimizer(BasePluginOptimizer):
    
    """
    create an Optimizer with parameters
    param: 
        - name, string, plugin optimizer name
        - hyper_parameters, list, hyper parameters that need to be tuned
        - kwargs, dict, algorithm parameters passed by hpo task submission
                  rest body, the parameter value type is string
    """
    
    def __init__(self, name, hyper_parameters, **kwargs):
        super(PluginOptimizer, self).__init__(name, hyper_parameters, **kwargs)
        
        # get all hyper parameters that need to be tuned
        logger.info("all tuning hyper parameters: \n{}".format(hyper_parameters))
        
        self._hyper_parameters = hyper_parameters
        self._exp_history = []
        
        # get all optimizer search parameters that user passed
        logger.info("all optimizer search parameters: \n{}".format(kwargs))
        
        # get optimizer parameters, the parameters value is string
        if kwargs.get('random_seed'):
            self._random_seed = int(kwargs.get('random_seed'))
            np.random.seed(self._random_seed)
            #self.rnd = np.random.RandomState(1234)

    
    """
    search new set of candidate hyper-parameters
    param: 
        - number_samples, int, number of hyper parameter candidates requested
        - last_exp_results, list, the execution results of last suggested hyper-
                    parameter sets
    return: hyper_params, list, suggested hyper-parameter sets to run
    """
    def search(self, number_samples, last_exp_results):

        logger.info("last exps results:\n{}".format(last_exp_results))
        if not last_exp_results is None and len(last_exp_results) > 0:
            self._exp_history.extend(last_exp_results)
        
        # start random search of the hyper-parameters
        exp_list = []
        for i in range(number_samples):
            hypers = {}
            for hp in self._hyper_parameters:
                type = hp.get('type')
                if type == "Range":
                    val = self._getRandomValueFromRange(hp)
                elif type == "Discrete":
                    val = self._getRandomValueFromDiscrete(hp)
                else:
                    raise Exception("un-supported type {} for random search.".format(type))
                hypers[hp.get('name')] = val
            exp_list.append(hypers)
            
        logger.info("suggest next exps list:\n{}".format(exp_list))
        return exp_list
    
    def get_state(self):
        return {'rng_state': np.random.get_state()}
    
    def set_state(self, state_dict):
        np.random.set_state(state_dict.get('rng_state'))
        
    def _getRandomValueFromRange(self, hp):
                
        data_type = hp.get('dataType')
        if data_type == "DOUBLE":
            val = hp.get('minDbVal') + np.random.rand() * (hp.get('maxDbVal') - hp.get('minDbVal'))
        elif data_type == "INT":
            val = np.random.randint(hp.get('minIntVal'), hp.get('maxIntVal'))
        else:
            raise Exception("un-supported data type {} for random range search.".format(data_type))
        
        logger.debug("next {} val: {}".format(hp.get('name'), val))
        return val


    def _getRandomValueFromDiscrete(self, hp):
                
        data_type = hp.get('dataType')
        if data_type == "DOUBLE":
            vals = hp.get('discreteDbVal')
        elif data_type == "INT":
            vals = hp.get('discreteIntVal')
        else:
            vals = hp.get('discreateStrVal')
            
        val = vals[np.random.randint(len(vals))]
        
        logger.debug("next {} val: {}".format(hp.get('name'), val))
        return val
