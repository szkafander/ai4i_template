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
.. module:: model_manager
   :platform: Unix, Windows
   :synopsis: Contains high-level abstraction for managing models and their
       pre- and postprocessing.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

Contains high-level abstraction for managing models and their pre- and 
postprocessing. The module provides the following functionality:

================================  =============  ==============================
name                              type           summary
================================  =============  ==============================
ModelManager                      class          Handler class and wrapper for 
                                                 models.
================================  =============  ==============================

"""

from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow import keras
from abstractions import pre_post
from misc import utils
from typing import Callable, Optional, Tuple, Union


class ModelManager:
    """ Handler and wrapper class for managing Keras models. Maintains
    topologies, pre- and postprocessors, miscellaneous tensors, training/test 
    protocols and saving and loading.
    
    **Usage**
    
    >>> # construct from input and output tensors
    >>> model_mngr = ModelManager(input, topology, preprocessor, postprocessor)
    >>> # construct from a saved Keras model
    >>> model_mngr = ModelManager(path_to_model, preprocessor, postprocessor)
    >>> # construct from a recipe
    >>> model_mngr = ModelManager.from_recipe(path_to_recipes)
    
    """
    
    def __init__(
            self, 
            input_: Optional[tf.Tensor] = None,
            topology: Optional[tf.Tensor] = None, 
            misc_tensors: Optional[Tuple] = None,
            model_path: Optional[str] = None,
            model: keras.models.Model = None,
            preprocessor: Optional[pre_post.Processor] = None, 
            postprocessor: Optional[pre_post.Processor] = None
        ) -> None:
        """ Constructor.
        
        :param input_: A Tensorflow tensor, the input placeholder.
        :type input_: tensorflow.Tensor
        :param topology: A Tensorflow tensor, the output tensor. The graph of
            this tensor includes input_ as a node.
        :type topology: tensorflow.Tensor
        :param misc_tensors: An optional Tuple of miscellaneous tensorflow
            Tensors. These might be used in some specific training protocols.
        :type misc_tensors: Tuple[tensorflow.tensor,...]
        :param model_path: An optional path to a Keras model stored on disk.
        :type model_path: str
        :param model: A keras.models.Model object.
        :type model: keras.models.Model
        :param preprocessor: A Processor that will be applied to data passed to
            self.predict prior to prediction.
        :type preprocessor: Processor
        :param postprocessor: A Processor that will be applied to predictions
            prior to returning them.
        :type postprocessor: Processor
        :raises: ValueError
        
        """
        self.input = input_
        self.topology = topology
        self.misc_tensors = misc_tensors
        self.model_path = model_path
        if model_path is not None:
            model = keras.models.load_model(utils.relative_path(model_path))
        if model is not None:
            self.model = model
        elif self.input is not None and self.topology is not None:
            self.model = keras.models.Model(
                    inputs=[self.input], 
                    outputs=[self.topology]
                )
        else:
            raise ValueError("You must provide either an input and topology "
                             "OR a model description.")
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
    
    @staticmethod
    def from_recipe(recipe: Union[str, dict]) -> "ModelManager":
        """ Instantiates a ModelManager object from a recipe. A recipe is a 
        text file.
        
        Recipes follow the JSON format. Here is an example recipe:
        
        .. code-block:: text
        
            {
              "model":
                {
                  "path": "segmenter/dcnn_filt24_kernel9x9_pexp135"
                },
              "preprocess":
                {
                  "take_channel": "blue",
                  "subtract_mean": 119.01272772,
                  "standardize": 21.73939139
                },
              "postprocess":
                {
                  "extract_slag_signal": ["masks/segmentation_mask.png", 0.1]
                }
            }
        
        
        Top-level keywords in recipes specify the type of object to return from
        the description under them. For example, 'model' tells that a model or
        ModelManager instance should be returned. 'preprocess' and 
        'postprocess' tell ModelManager.from_recipe what type of pre- and
        postprocessors to create when instantiating. See the module pre_post
        about more details on Processors.
        
        .. note::
            Any paths given in a recipe should be treated as relative paths.
            See misc.utils.relative_path.
            
        :param recipe: A string specifying a path to a recipe text file or a
            dictionary, i.e., from json.load(recipe).
        :type recipe: Union[str, dict]
        :returns: A ModelManager object.
        :rtype: ModelManager
        :raises: ValueError
        
        """
        if isinstance(recipe, str):
            recipe = utils.read_recipe(recipe)
        try:
            recipe["model"]["path"]
        except KeyError:
            raise ValueError("Recipe must contain a model path under "
                             "['model']['path'].")
        model_path = recipe["model"]["path"]
        preprocessor = pre_post.Processor.from_recipe(recipe, "pre")
        postprocessor = pre_post.Processor.from_recipe(recipe, "post")
        return ModelManager(
                input_=None, 
                topology=None, 
                model_path=model_path, 
                model=None,
                preprocessor=preprocessor, 
                postprocessor=postprocessor
            )
        
    def save_recipe(self, recipe_path: str) -> None:
        """ Not implemented.
        
        """
        raise NotImplementedError("This function is not implemented. "
                                  "Check back later.")
        
    def compile_model(
            self, 
            loss: str, 
            optimizer: Union[str, keras.optimizers.Optimizer], 
            metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]] = None
        ) -> None:
        """ Compiles self.model. Invokes self.model.compile.
        
        :param loss: The name of the Keras loss.
        :type loss: str
        :param optimizer: The name of the Keras optimizer or an instance of
            keras.optimizers.Optimizer.
        :type optimizer: Union[str, keras.optimizers.Optimizer]
        :param metrics: Optional Tuple of metric names or keras Metric objects.
        :type metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]]
        
        """
        self.model.compile(optimizer, loss, metrics)
    
    def predict(self, data: np.array) -> np.array:
        if self.preprocessor is not None:
            data = self.preprocessor(data)
        data = self.model.predict(data)
        if self.postprocessor is not None:
            data = self.postprocessor(data)
        return data
    
    # TODO: implicit training protocol handling
    # def train(
    #         training_protocol: training.TrainingProtocol
    #     ) -> Tuple[keras.models.Model, np.array]:
    #     raise NotImplementedError("This function is not implemented. "
    #                               "Check back later.")
    
    def summary(self) -> None:
        """ Prints out a summary of the ModelManager. This is a summary of
        self.model and a summary of pre- and postprocessors.
        
        """
        print("Model manager summary:")
        print("Preprocessor:")
        print(self.preprocessor)
        print("Model summary:")
        self.model.summary()
        print("Postprocessor:")
        print(self.postprocessor)
