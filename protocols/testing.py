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
.. module:: testing
   :platform: Unix, Windows
   :synopsis: Contains protocols for training neural networks.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

Contains high-level abstraction for managing models and their pre- and 
postprocessing. The module provides the following functionality:

=================================  =============  =============================
name                               type           summary
=================================  =============  =============================
TestingProtocol                    class          Abstract base class for
                                                  testing protocols.
ProcessTestingProtocol             class          Parent class for testing on
                                                  process data.
ProcessSynchedTimeWithSensitivity  class          Protocol for testing process 
                                                  models using time-synched
                                                  process data.
gradient_sensitivity_analysis      method         Calculates gradients of the
                                                  output with respect to 
                                                  inputs.
=================================  =============  =============================

"""

from __future__ import absolute_import

import abc
import numpy as np
import functools
from abstractions import model_manager as mm
from abstractions import generators
from tensorflow import keras
from typing import Any, Callable, Generator, Optional, Tuple, Union
from misc import utils


def gradient_sensitivity_analysis(
        model: keras.models.Model, 
        input_space: Union[np.array, Generator],
        batch_axis: int = 0,
        num_steps: Optional[int] = 1000,
        directional: bool = False,
        interim_layer: Optional[str] = None
    ) -> np.array:
    """ Calculates the mean gradient of the output of a model with respect to 
    inputs over some input data.
    
    .. note::
        In most cases, this should be used on a scalar regression network.
    
    :param model: A Keras model to evaluate.
    :type model: keras.models.Model
    :param input_space: An Iterable that serves data to the model. Either a
        numpy array or a generator object.
    :type input_space: Union[np.array, Generator]
    :param batch_axis: This is the axis along which the gradients are averaged. 
    :type batch_axis: int
    :param num_steps: The number of steps, i.e., the number of batches to feed
        in total. If None, then the generator or array will be completely 
        exhausted. If an infinite generator is provided, this results in an
        infinite loop.
    :type num_steps: Optional[int]
    :param directional: If True, signed gradients will be computed. If False,
        the absolute value of gradients will be returned. Directional gradients
        not only give an estimation of parameter importance, but also of the
        direction of change: if a directional gradient is negative, it means
        that when the input decreases, the output increases.
    :type directional: bool
    :param interim_layer: Optional, the name of an interim layer. If provided,
        gradients with respect to this layer will be computed instead of
        gradients with respect to the input. Useful when one wants to see the
        importance of latent variables.
    :type interim_layers: Optional[str]
    :returns: A numpy array, the evaluated gradients.
    :rtype: numpy.array
    
    """    
    input_tensor = model.input
    if interim_layer is not None:
        interim_tensor = model.get_layer(interim_layer).output
    output_tensor = model.output
    
    if interim_layer is not None:
        J, = keras.backend.gradients(output_tensor, interim_tensor)
    else:
        J, = keras.backend.gradients(output_tensor, input_tensor)
    
    with keras.backend.get_session() as sess:
        for i, batch in enumerate(input_space):
            dydx = J.eval(feed_dict={input_tensor: batch[0]}, session=sess)
            if i == 0:
                if not directional:
                    grads = np.sum(dydx**2, axis=batch_axis)
                else:
                    grads = np.sum(dydx, axis=batch_axis)
            else:
                if not directional:
                    grads += np.sum(dydx**2, axis=batch_axis)
                else:
                    grads += np.sum(dydx, axis=batch_axis)
            if num_steps is not None:
                if i > num_steps:
                    break
            utils.log1l("Done {} of {}".format(i, num_steps))
        grads /= i * dydx.shape[batch_axis]
        if not directional:
            grads = np.sqrt(grads)
    
    return grads


class TestingProtocol(abc.ABC):
    """ Base class for testing protocols. All testing protocols must implement
    the 'evaluate' method.
    
    """
    
    def __init__(
            self, 
            metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]] = None
        ) -> None:
        """ Constructor. Stored optional metrics.
        
        :param metrics: Optional, a Tuple of metric names or metric functions.
        :type metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]]
        
        """
        self.metrics = metrics
    
    @abc.abstractmethod
    def evaluate(*args, **kwargs) -> Any:
        """ Abstract method. Implementations should evaluate a model in some 
        way.
        
        """
        pass
    

class ProcessTestingProtocol(TestingProtocol):
    """ An abstract protocol for process data. Process data is a table of 
    scalars where columns are variables or features and rows are observations
    in time. Each column is a time history.
    
    """

    def __init__(
            self,
            metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]] = None,
            is_random: bool = False
        ) -> None:
        """ The same as the parent class' constructor, except it stores an
        additional switch, is_random.
        
        :param metrics: Optional, a Tuple of metric names or metric functions.
        :type metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]]
        :param is_random: This is a switch that controls if data is fed
            sequentially or randomly to the model.
        :type is_random: bool
        
        """
        super(ProcessTestingProtocol, self).__init__(metrics)
        self.is_random = is_random
    
    @abc.abstractmethod
    def evaluate(*args, **kwargs) -> Any:
        pass


class ProcessSynchedTimeWithSensitivity(ProcessTestingProtocol):
    """ A protocol for testing on time-synched process data. Time-synched
    process data is a table of scalars, where (by the default) the first
    column stores the time coordinate. All other columns are interpolated to
    this common time coordinate. The time coordinate in exogenous and 
    endogenous data can be different, but the endogenous time coordinate values
    must form a subset of the exogenous time coordinate values. Exogenous and
    endogenous data will be matched based on this shared time coordinate.
    
    """    
    
    def __init__(
            self,
            metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]] = None,
            is_random: bool = False,
            sampling_axis: int = 0
        ) -> None:
        """ The same as the parent class' constructor, except it stores an
        additional parameter, sampling_axis.
        
        :param metrics: Optional, a Tuple of metric names or metric functions.
        :type metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]]
        :param is_random: This is a switch that controls if data is fed
            sequentially or randomly to the model.
        :type is_random: bool
        :param sampling_axis: The column ID where the time coordinate is stored
            in the exogenous and endogenous tables. It must be the same column
            in both.
        :type sampling_axis: int
        
        """
        super(ProcessSynchedTimeWithSensitivity, self).__init__(
                metrics,
                is_random
            )
        self.sampling_axis = sampling_axis
        
    def _get_pdg(
            self,
            num_batch,
            num_history,
            seed
        ) -> Callable:
        return functools.partial(
                generators.TableOfScalars,
                n_batch=num_batch,
                n_history=num_history,
                random_mode=self.is_random,
                sampling_axis=self.sampling_axis,
                random_seed=seed
            )
    
    def evaluate(
            self,
            model_manager: mm.ModelManager,
            num_batch: int = 10,
            num_steps: int = 100,
            exogenous_train: Optional[np.array] = None,
            endogenous_train: Optional[np.array] = None,
            exogenous_test: Optional[np.array] = None,
            endogenous_test: Optional[np.array] = None,
            random_seed: Optional[int] = None
        ) -> Tuple:
        """ Evaluates the passed model. Depending on the data provided, reports
        and returns the following:
        
            * Accuracy evaluation on the training dataset
            * Accuracy evaluation on the testing dataset
            * Sensitivities on the training dataset
            * Sensitivities on the testing dataset
        
        .. note::
            Here is the pattern for providing exogenous and endogenous data:
                * Exogenous and endogenous data must be provided in pairs
                * If both exogenous and endogenous data are provided for a
                  training/test set, both accuracy evaluation and sensitivity
                  analysis will be carried out on that set.
            
        :param model_manager: A ModelManager object that stores a trained
            model.
        :type model_manager: model_manager.ModelManager
        :param num_batch: The batch size of served batches.
        :type num_batch: int
        :param num_steps: The number of batches to serve.
        :type num_steps: int
        :param exogenous_train: The exogenous part of the training dataset.
        :type exogenous_train: numpy.array
        :param endogenous_train: The endogenous part of the training dataset.
        :type endogenous_train: numpy.array
        :param random_seed: A random seed for the data generator. Use this if
            you want to reproduce previous results.
        :type random_seed: Optional[int]
        :returns: A Tuple that stores results from training and test accuracy
            evaluation and training and test sensitivity analysis, 
            respectively.
        :rtype: Tuple
        
        """
        # preprocess training data
        exogenous_train_ = model_manager.preprocessor(exogenous_train)
        
        # preprocess test data
        exogenous_test_ = model_manager.preprocessor(exogenous_test)
        
        sens_test, acc_test, sens_train, acc_train = False, False, False, False
        
        if exogenous_test is not None:
            if endogenous_test is None:
                raise ValueError("If test exogenous data are provided, test "
                                 "endogenous data must be provided as well.")
            print("Test exogenous data provided, carrying out sensitivity "
                  "analysis...")
            gen_train = self._get_pdg(
                num_batch,
                int(model_manager.model.input.shape[1]),
                random_seed
            )(exogenous_train_, endogenous_train)
            sens_test = True
            print("Test endogenous and test exogenous data are provided, "
                  "evaluating accuracy...")
            acc_test = True
        
        if exogenous_train is not None:
            if endogenous_train is None:
                raise ValueError("If train exogenous data are provided, train "
                                 "endogenous data must be provided as well.")
            print("Train exogenous data provided, carrying out sensitivity "
                  "analysis...")
            gen_test = self._get_pdg(
                num_batch,
                int(model_manager.model.input.shape[1]),
                random_seed
            )(exogenous_test_, endogenous_test)
            sens_train = True
            print("Train Y and train X are provided, evaluating accuracy...")
            acc_train = True
            
        if acc_train:
            print("Evaluating on training data...")
            results_train = model_manager.model.evaluate_generator(
                    gen_train,
                    steps=num_steps
                )
        
        if acc_test:
            print("Evaluating on test data...")
            results_test = model_manager.model.evaluate_generator(
                    gen_test,
                    steps=num_steps
                )
            
        if sens_train:
            print("Calculating sensitivity on training data...")
            grad_train = gradient_sensitivity_analysis(
                    model_manager.model,
                    gen_train,
                    num_steps=num_steps
                )
            
        if sens_test:
            print("Calculating sensitivity on training data...")
            grad_test = gradient_sensitivity_analysis(
                    model_manager.model,
                    gen_test,
                    num_steps=num_steps
                )
            
        return results_train, results_test, grad_train, grad_test
