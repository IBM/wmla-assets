
import numpy as np
from plugins.core.logger import logger

from plugins.core.base_plugin_opt import BasePluginOptimizer

class PluginOptimizer(BasePluginOptimizer):

    def __init__(self, name, hyper_parameters, **kwargs):
        super(PluginOptimizer, self).__init__(name, hyper_parameters, **kwargs)
        logger.info("all tuning hyper parameters: \n{}".format(hyper_parameters))
        self._hyper_parameters = hyper_parameters
        self.cubeIndex = 0
        self.gridCube = self._buildGridCube()
        logger.info("grid cube size: \n{}".format(len(self.gridCube)))

    def _buildGridCube(self):
        hpdict = {}
        for hp in self._hyper_parameters:
            name = hp.get('name')
            type = hp.get('type')
            dataType = hp.get('dataType')
            if type == 'Discrete':
                if dataType == 'DOUBLE':
                    hpdict[name] = hp.get('discreteDbVal')
                if dataType == 'INT':
                    hpdict[name] = hp.get('discreteIntVal')
                if dataType == 'STR':
                    hpdict[name] = hp.get('discreateStrVal')

            if type == 'Range':
                if dataType == 'DOUBLE':
                    hpdict[name] = np.arange(hp.get('minDbVal'),hp.get('maxDbVal'), 0.01)
                if dataType == 'INT':
                    hpdict[name] = list(range(hp.get('minIntVal'), hp.get('maxIntVal'), 1))
        logger.info(hpdict)
        items = sorted(hpdict.items())
        keys, values = zip(*items)
        logger.info(keys)
        logger.info(values)

        grids = []
        from itertools import product
        for v in product(*values):
            params = dict(zip(keys, v))
            grids.append(params)
        return grids

    def search(self, number_samples, last_exp_results=None):

        if self.cubeIndex + number_samples <= len(self.gridCube) - 1:
            suggestions = self.gridCube[self.cubeIndex: self.cubeIndex + number_samples]
            self.cubeIndex += number_samples
        elif self.cubeIndex < len(self.gridCube) - 1:
            suggestions = self.gridCube[self.cubeIndex:]
            self.cubeIndex = len(self.gridCube) - 1
        else:
            suggestions = []

        return suggestions
