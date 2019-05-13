#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019, RISE ETC AB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL RISE 
# ETC AB BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Except as contained in this notice, the name of RISE ETC AB shall not be
# used in advertising or otherwise to promote the sale, use or other dealings 
# in this Software without prior written authorization from RISE ETC AB.

"""
.. module:: uncertainty
   :platform: Unix, Windows
   :synopsis: Contains high-level abstractions for managing uncertainty in
       ensemble predictions.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

This module contains high-level abstractions for managing uncertainty in
ensemble predictions. The module provides the following functionality:
    
==================  =============  =============
name                type           summary
==================  =============  =============
EnsemblePrediction  class          Abstract base class of ensemble predictions.
NormalStatistics    class          A class implementing confidence bounds based
                                   on normal statistics.
==================  =============  =============

"""

import abc
import numpy as np
from scipy import stats
from typing import Iterable, Tuple


class EnsemblePrediction(abc.ABC):
    """ Abstract base class for ensemble predictions.
    
    """
    
    def __init__(self, data: Iterable) -> None:
        """ Constructor. The object will hold a reference to the ensemble data.
        
        :param data: The ensemble data. Must be iterable and each element is
            considered an individual prediction. This is normally a List or
            numpy array.
        :type data: Iterable
        
        """
        self.data = data
        
    @abc.abstractmethod
    def estimate_ci(self, alpha: float) -> Tuple[float, float]:
        """ An abstract method for calculating the confidence interval.
        
        :param alpha: The alpha parameter for computing the confidence bound.
        :type alpha: float
        :returns: A 2-Tuple of floats r, where r[0] is the lower bound and r[1]
            is the upper bound.
        
        """
        pass
    
    @abc.abstractmethod
    def resample(self, num_samples: int) -> Iterable:
        """ An abstract method for resampling from a distribution based on the
        ensemble statistics. The resampled population can be used for e.g.,
        Monte-Carlo estimation.
        
        :param num_samples: The number of samples to draw from the 
            distribution.
        :type num_samples: int
        :returns: An Iterable, the resampled population.
        
        """
        pass
    

class NormalStatistics(EnsemblePrediction):
    """ Ensemble statistics assuming a normal distribution of independent
    predictions.
    
    **Usage**
    >>> norm_stats = NormalStatistics([pred_1, pred_2,...])
    >>> mean, conf_int = normal_stats.mean, norm_stats.estimate_ci(alpha=0.95)
    
    
    """
    
    def __init__(self, data: np.array) -> None:
        """ Constructor.
        
        :param data: The ensemble data. Must be iterable and each element is
            considered an individual prediction. This is normally a List or
            numpy array.
        :type data: numpy array
        
        """
        super(NormalStatistics, self).__init__(data)
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.num_data = len(self.data)
    
    def estimate_ci(self, alpha: float = 0.95) -> Tuple[float, float]:
        """ Returns confidence interval based on the mean and standard error.
        
        :param alpha: The alpha parameter of the confidence interval.
        :type alpha: float
        :returns: A 2-Tuple r, where r[0] is the lower bound and r[1] is the 
            higher bound.
        
        """
        return stats.t.interval(alpha, self.num_data-1, loc=self.mean, 
            scale=stats.sem(self.data))
    
    def resample(self, num_samples: int = 100) -> np.array:
        """ Pulls a sample from the normal distribution fit over self.data.
        
        :param num_samples: The number of samples to draw from the 
            distribution.
        :type num_samples: int
        :returns: A numpy array, the resampled population.
        
        """
        return np.random.randn(num_samples) * self.std + self.mean
