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
.. module:: training
   :platform: Unix, Windows
   :synopsis: Contains protocols for training neural networks.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

Contains high-level abstraction for managing models and their pre- and 
postprocessing. The module provides the following functionality:

================================  =============  ==============================
name                              type           summary
================================  =============  ==============================
TrainingProtocol                  class          Abstract base class for
                                                 training protocols.
ImageAugmentingProtocol           class          Parent class for training on
                                                 image data with augmentation.
ImageClassificationXYAugmenting   class          Classification on image data 
                                                 with augmentation. Image data 
                                                 is in memory.
ImageSegmentationXYAugmenting     class          Segmentation of image data
                                                 with augmentation. Image data
                                                 is in memory.
ProcessInterpolatedInMemory       class          Training on process data. Data
                                                 are in memory. Data are time-
                                                 regular (interpolated).
ProcessSynchedTimeSampler         class          Training on process data. Data
                                                 are in memory and regular.
                                                 Observations in x and y are
                                                 matched based on time.
================================  =============  ==============================

"""

from __future__ import absolute_import

import abc
import numpy as np
import functools
from abstractions import model_manager as mm
from abstractions import generators
from tensorflow import keras
from tensorflow.keras import callbacks
from typing import Callable, Generator, Optional, Tuple, Union
from misc import utils


# Default callbacks
_history = callbacks.History()
_early_stopping = callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", mode="max", patience=5, verbose=1)
_early_stopping_regression = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, verbose=1)
_early_stopping_regression_noval = callbacks.EarlyStopping(
        monitor="loss", mode="min", patience=5, verbose=1)
_lr_reducer = callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
_lr_reducer_noval = callbacks.ReduceLROnPlateau(monitor="loss",factor=0.5, 
                                                patience=3, verbose=1)
_callbacks = [_history, _early_stopping, _lr_reducer]
_callbacks_regression = [_history, _early_stopping_regression, _lr_reducer]


class TrainingProtocol(abc.ABC):
    """ Abstract base class for training protocols.
    
    All protocols must implement the compile_and_train method.
    
    """
    
    def __init__(
            self, 
            loss: Union[str, Callable],
            optimizer: Union[str, keras.optimizers.Optimizer],
            metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]],
            callbacks: Optional[Tuple[callbacks.Callback,...]]
        ) -> None:
        """ Training protocols are intialized based on at least the type of
        loss, the optimizer, optional metrics and optional callbacks.
        
        :param loss: The type of loss. Can be the name of loss or a Keras loss.
        :type loss: Union[str, Callable]
        :param optimizer: The name of the optimizer or the Keras optimizer.
        :type optimizer: Union[str, keras.optimizers.Optimizer]
        :param metrics: Optional metrics. Can be the a Tuple of metric names or
            a Tuple of metric objects.
        :type metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]]
        :param callbacks: Optional callbacks. A Tuple of Keras callback 
            objects.
        :type callbacks: Optional[Tuple[callbacks.Callback,...]]
        
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.callbacks = callbacks
    
    @abc.abstractmethod
    def compile_and_train(
            *args, 
            **kwargs
        ) -> Tuple[keras.models.Model, np.array]:
        """ The main training method.
        
        """
        pass


class ImageAugmentingProtocol(TrainingProtocol):
    """ Parent class for protocols that train based on image data with 
    augmentation. Stores defaults for image augmentation.
    
    """
    def __init__(
            self, 
            loss: Union[str, Callable],
            optimizer: Union[str, keras.optimizers.Optimizer],
            metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]],
            callbacks: Optional[Tuple[callbacks.Callback,...]],
            aug_args: Optional[dict] = None
        ) -> None:
        """ Constructor.
        
        :param loss: The type of loss. Can be the name of loss or a Keras loss.
        :type loss: Union[str, Callable]
        :param optimizer: The name of the optimizer or the Keras optimizer.
        :type optimizer: Union[str, keras.optimizers.Optimizer]
        :param metrics: Optional metrics. Can be the a Tuple of metric names or
            a Tuple of metric objects.
        :type metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]]
        :param callbacks: Optional callbacks. A Tuple of Keras callback 
            objects.
        :type callbacks: Optional[Tuple[callbacks.Callback,...]]
        :param aug_args: Image augmentation parameters. See 
            keras.utils.preprocessing.ImageDataGenerator for more information.
        :type aug_args: Optional[dict]
        
        """
        super(ImageAugmentingProtocol, self).__init__(
                loss, 
                optimizer,
                metrics,
                callbacks
            )
        self.aug_args = aug_args

    # Default spatial augmentation params.
    # If an argument is not specified in the constructor, these are used 
    # instead.
    AUG_PARAMS_DEFAULT = {
            "featurewise_center": False,
            "featurewise_std_normalization": False,
            "rotation_range": 15,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "horizontal_flip": True,
            "vertical_flip": True,
            "validation_split": 0.3,
            "zoom_range": 0.1,
            "fill_mode": 'nearest'
        }
    
    @abc.abstractmethod
    def compile_and_train(
            *args,
            **kwargs
        ) -> Tuple[keras.models.Model, np.array]:
        """ Still abstract.
        
        """
        pass
    
    def get_idg(self) -> keras.preprocessing.image.ImageDataGenerator:
        """ Returns the ImageDataGenerator.
        
        :returns: An ImageDataGenerator parametrized by the protocol.
        :rtype: keras.utils.preprocessing.ImageDataGenerator
        
        """
        args = utils._arg_or_default(self.AUG_PARAMS_DEFAULT, self.aug_args)
        datagen = keras.preprocessing.image.ImageDataGenerator(**args)
        return datagen


class ImageClassificationXYAugmenting(ImageAugmentingProtocol):
    """ Classification protocol with in-memory data.
    
    """
    
    def __init__(
            self, 
            optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
            loss: Union[str, Callable] = "categorical_crossentropy",
            metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]] = None,
            callbacks: Optional[Tuple[callbacks.Callback,...]] = None,
            aug_args: Optional[dict] = None
        ) -> None:
        """ Constructor. Has relevant defaults for loss, callbacks and metrics.
        
        :param loss: The type of loss. Can be the name of loss or a Keras loss.
        :type loss: Union[str, Callable]
        :param optimizer: The name of the optimizer or the Keras optimizer.
        :type optimizer: Union[str, keras.optimizers.Optimizer]
        :param metrics: Optional metrics. Can be the a Tuple of metric names or
            a Tuple of metric objects.
        :type metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]]
        :param callbacks: Optional callbacks. A Tuple of Keras callback 
            objects.
        :type callbacks: Optional[Tuple[callbacks.Callback,...]]
        :param aug_args: Image augmentation parameters. See 
            keras.utils.preprocessing.ImageDataGenerator for more information.
        :type aug_args: Optional[dict]
        
        """
        if metrics is None:
            metrics = ["categorical_accuracy",]
        if callbacks is None:
            callbacks = _callbacks
        super(ImageClassificationXYAugmenting, self).__init__(
                loss, optimizer, metrics, callbacks, aug_args)
        
    def compile_and_train(
            self,
            model_manager: mm.ModelManager,
            x: np.array,
            y: np.array,
            model_path: str,
            validation_split: Optional[float] = 0.2,
            num_workers: Optional[int] = 10,
            num_batch: int = 10,
            num_epochs: int = 100,
            num_shows: int = 10,
            class_weights: Optional[Tuple[float,...]] = None,
            seed: Optional[int] = None
        ) -> Tuple[keras.models.Model, np.array]:
        """ Compiles and trains model.
        
        :param model_manager: A ModelManager object that contains the model to
            train.
        :type model_manager: model_manager.ModelManager
        :param x: The image data. Size num_images X width X height X channels.
        :type x: numpy.array
        :param y: The classification labels. Should be one-hot encoded. Size
            num_images X num_classes.
        :type y: numpy.array
        :param model_path: The path of the saved model. Model is saved at 
            checkpoints.
        :type model_path: str
        :param validation_split: Optional. If provided, X and Y are split into
            disjoint training and validation sets.
        :type validation_split: Optional[float]
        :param num_workers: Optional. If provided, multiple threads will be
            used to generate data for training. The argument is the number of
            workers to use.
        :type num_workers: Optional[int]
        :param num_batch: The number of batches.
        :type num_batch: int
        :param num_epochs: The number of epochs.
        :type num_epochs: int
        :param num_shows: This is approximately the number of times a single
            observation will be 'seen' by the network per epoch. If increased,
            more steps will be taken in each epoch.
        :type num_shows: int
        :param class_weights: Optional. If provided, classes will be weighted
            based on on this. This is a Tuple of floats such that 
            len(class_weights) == num_classes.
        :type class_weights: Optional[Tuple[float,...]]
        :param seed: Optional. If provided, the random state of the protocol
            will be fixed with this seed. Use this if you want to reproduce
            results later.
        :type seed: Optional[int]
        :returns: A tuple of a keras model and a training log in form of a
            numpy array.
        :rtype: Tuple[keras.models.Model, numpy.array]
        
        """
        
        # compile model
        model_manager.compile_model(
                loss=self.loss,
                optimizer=self.optimizer,
                metrics=self.metrics
            )
        
        # csv logger callback
        csv_logger = callbacks.CSVLogger(model_path+"_log.csv")
    
        # model checkpoint callback
        checkpoint = keras.callbacks.ModelCheckpoint(
                model_path, 
                monitor="val_categorical_accuracy",
                mode="max",
                verbose=1, 
                save_best_only=True
            )
        
        # preprocess data - exclude spatial transformations here
        x = model_manager.preprocessor(x)
        
        # set up generators
        datagen_x = self.get_idg()
        datagen_x.fit(x, seed=seed, augment=True)
        
        # shuffle data
        random_ind = np.random.choice(range(len(x)), len(x), replace=False)
        x = x[random_ind]
        y = y[random_ind]
        
        # training generator
        gen_train = datagen_x.flow(
                x, 
                y=y,
                seed=seed,
                shuffle=True,
                batch_size=num_batch, 
                subset='training'
            )
        
        # validation generator
        if validation_split is not None:
            gen_val = datagen_x.flow(
                    x,
                    y=y,
                    seed=seed,
                    shuffle=True,
                    batch_size=num_batch,
                    subset='validation'
                )

        # handle multithreading generators
        if num_workers is not None:
            multiprocessing = True
        else:
            multiprocessing = False
        
        # default uniform class weights
        if class_weights is None:
            # infer number of classes from topology
            num_classes = model_manager.model.output.shape.as_list()[-1]
            class_weights = np.ones(num_classes)
        
        # train
        spe_train = np.ceil(num_shows * len(x) / gen_train.batch_size 
                            * (1-validation_split))
        spe_val = np.ceil(num_shows * len(x) / gen_val.batch_size
                          * validation_split)
        if validation_split is not None:
            history = model_manager.model.fit_generator(
                gen_train, 
                steps_per_epoch=spe_train, 
                epochs=num_epochs, 
                verbose=1, 
                use_multiprocessing=multiprocessing,
                workers=num_workers,
                class_weight=class_weights,
                validation_data=gen_val,
                validation_steps=spe_val,
                callbacks=[*self.callbacks, checkpoint, csv_logger]
            )
        else:
            history = model_manager.model.fit_generator(
                gen_train, 
                steps_per_epoch=spe_train, 
                epochs=num_epochs, 
                verbose=1, 
                use_multiprocessing=multiprocessing,
                workers=num_workers,
                class_weight=class_weights,
                callbacks=[*self.callbacks, checkpoint, csv_logger]
            )
            
        # save history
        utils.save_history(history, model_path)
        
        # reload best model
        model = keras.models.load_model(model_path)
        
        self.model = model
        
        return model, history


class ImageSegmentationXYAugmenting(ImageClassificationXYAugmenting):
    """ Segmentation protocol on image data. Image data are in memory.
    
    """
        
    def compile_and_train(
            self,
            model_manager: mm.ModelManager,
            x: np.array,
            y: np.array,
            model_path: str,
            validation_split: Optional[float] = 0.2,
            num_workers: Optional[int] = 10,
            num_batch: int = 10,
            num_epochs: int = 100,
            num_shows: int = 10,
            seed: Optional[int] = None
        ) -> Tuple[keras.models.Model, np.array]:
        """ Compiles and trains model.
        
        :param model_manager: A ModelManager object that contains the model to
            train.
        :type model_manager: model_manager.ModelManager
        :param x: The image data. Size num_images X width X height X channels.
        :type x: numpy.array
        :param y: The classification labels. Should be one-hot encoded. Size
            num_images X num_classes.
        :type y: numpy.array
        :param model_path: The path of the saved model. Model is saved at 
            checkpoints.
        :type model_path: str
        :param validation_split: Optional. If provided, X and Y are split into
            disjoint training and validation sets.
        :type validation_split: Optional[float]
        :param num_workers: Optional. If provided, multiple threads will be
            used to generate data for training. The argument is the number of
            workers to use.
        :type num_workers: Optional[int]
        :param num_batch: The number of batches.
        :type num_batch: int
        :param num_epochs: The number of epochs.
        :type num_epochs: int
        :param num_shows: This is approximately the number of times a single
            observation will be 'seen' by the network per epoch. If increased,
            more steps will be taken in each epoch.
        :type num_shows: int
        :param seed: Optional. If provided, the random state of the protocol
            will be fixed with this seed. Use this if you want to reproduce
            results later.
        :type seed: Optional[int]
        :returns: A tuple of a keras model and a training log in form of a
            numpy array.
        :rtype: Tuple[keras.models.Model, numpy.array]
        
        """
        
        return super(ImageSegmentationXYAugmenting, self).compile_and_train(
                model_manager,
                x,
                y,
                model_path,
                validation_split,
                num_workers,
                num_batch,
                num_epochs,
                num_shows,
                None,
                seed
            )

        
class ProcessInterpolatedInMemory(TrainingProtocol):
    """ Base class for trainin on process data (table of scalars).
    
    """
    
    def __init__(
            self, 
            loss: Union[str, Callable],
            optimizer: Union[str, keras.optimizers.Optimizer],
            history_length: int,
            metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]] = None,
            callbacks: Optional[Tuple[callbacks.Callback,...]] = None
        ) -> None:
        """ Constructor. Has relevant defaults callbacks.
        
        :param loss: The type of loss. Can be the name of loss or a Keras loss.
        :type loss: Union[str, Callable]
        :param optimizer: The name of the optimizer or the Keras optimizer.
        :type optimizer: Union[str, keras.optimizers.Optimizer]
        :param metrics: Optional metrics. Can be the a Tuple of metric names or
            a Tuple of metric objects.
        :type metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]]
        :param callbacks: Optional callbacks. A Tuple of Keras callback 
            objects.
        :type callbacks: Optional[Tuple[callbacks.Callback,...]]
        
        """
        if callbacks is None:
            callbacks = _callbacks_regression
        super(ProcessInterpolatedInMemory, self).__init__(
                loss, 
                optimizer,
                metrics,
                callbacks
            )
        self.history_length = history_length
    
    @abc.abstractmethod
    def compile_and_train(
            *args, 
            **kwargs
        ) -> Tuple[keras.models.Model, np.array]:
        pass


class ProcessSynchedTimeSampler(ProcessInterpolatedInMemory):
    """ Protocol for training on time-synched process data. Process data
    are in memory. Process data are regular, i.e., all features are 
    interpolated to a common time coordinate.
    
    This protocol offers two-step training with a PCA block. In this case, the
    PCA block will be trained first. Then, PCA encoder weights will be frozen
    and the remainder of the network will be trained.
    
    """
    
    def __init__(
            self, 
            loss: Union[str, Callable] = "mse",
            optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
            history_length: int = 1000,
            metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]] = None,
            callbacks: Optional[Tuple[callbacks.Callback,...]] = None,
            sampling_axis: int = 0,
            is_random: bool = True,
            pre_pca: bool = False,
            pca_components_name: Optional[str] = "pca_components",
            pca_reconstruction_name: Optional[str] = "pca_reconstruction"
        ) -> None:
        """ Constructor. Has relevant defaults for the loss, optimizer,
        callbacks and metrics.
        
        :param loss: The type of loss. Can be the name of loss or a Keras loss.
        :type loss: Union[str, Callable]
        :param optimizer: The name of the optimizer or the Keras optimizer.
        :type optimizer: Union[str, keras.optimizers.Optimizer]
        :param metrics: Optional metrics. Can be the a Tuple of metric names or
            a Tuple of metric objects.
        :type metrics: Optional[Union[Tuple[str,...], Tuple[Callable,...]]]
        :param callbacks: Optional callbacks. A Tuple of Keras callback 
            objects.
        :type callbacks: Optional[Tuple[callbacks.Callback,...]]
        :param sampling_axis: The column in the process data of the time 
            coordinate.
        :type sampling_axis: int
        :param is_random: If True, the time coordinate will be sampled in 
            random order.
        :type is_random: bool
        :param pre_pca: If True, a PCA block will be searched for in 
            model_manager.misc_tensors. If found, it will be trained prior to
            training the task network.
        :type pre_pca: bool
        :param pca_components_name: The name of the layer that contains the PCA
            components. Only relevant if pre_pca is True.
        :type pca_components_name: str
        :param pca_reconstruction_name: The name of the layer that contains the
            PCA reconstruction. Only relevant if pre_pca is True.
        :type pca_reconstruction_name: str
        
        """
        super(ProcessSynchedTimeSampler, self).__init__(
                loss, 
                optimizer,
                history_length,
                metrics,
                callbacks
            )
        self.sampling_axis = sampling_axis
        self.is_random = is_random
        self.pre_pca = pre_pca
        self.pca_components_name = pca_components_name
        self.pca_reconstruction_name = pca_reconstruction_name
        
    def _get_pdg(
            self, 
            num_batch, 
            num_history, 
            seed
        ) -> generators.TableOfScalars:
        return functools.partial(
                generators.TableOfScalars,
                n_batch=num_batch,
                n_history=num_history,
                random_mode=self.is_random,
                sampling_axis=self.sampling_axis,
                random_seed=seed
            )
    
    def _get_ae_pdg(
            self,
            num_batch,
            num_history,
            seed
        ) -> Generator:
        def _get_gen(
                exogenous, 
                endogenous, 
                num_batch_=None, 
                num_history_=None, 
                seed_=None
            ):
            pdg = self._get_pdg(
                    num_batch_, 
                    num_history_, 
                    seed_
                )(exogenous, endogenous)
            for batch in pdg:
                yield batch[0], batch[0]
        return functools.partial(
                _get_gen,
                num_batch_=num_batch,
                num_history_=num_history,
                seed_=seed
            )
    
    def compile_and_train(
            self,
            model_manager: mm.ModelManager,
            exogenous: np.array,
            endogenous: np.array,
            model_path: str,
            validation_split: Optional[float] = 0.2,
            num_workers: Optional[int] = 10,
            num_batch: int = 10,
            num_shows: float = 0.01,
            num_epochs: int = 100,
            seed: Optional[int] = None
        ) -> Tuple[keras.models.Model, np.array]:
        """ Compiles and trains model.
        
        :param model_manager: ModelManager object.
        :type model_manager: model_manager.ModelManager
        
        """
        if num_workers is not None:
            multiprocessing = True
        else:
            num_workers = 1
            multiprocessing = False
            
        # preprocess data
        exogenous_ = model_manager.preprocessor(exogenous)
        
        if self.pre_pca:
            print("Training PCA layer...")
            gen_ae = self._get_ae_pdg(
                    num_batch, 
                    self.history_length, 
                    seed
                )(exogenous_, endogenous)
            spe = np.ceil(num_shows * len(endogenous) / num_batch)
            input_layer = model_manager.model.input
            misc_names = [tensor for tensor in model_manager.misc_tensors 
                          if self.pca_reconstruction_name in tensor.name]
            if len(misc_names) > 1:
                raise ValueError("There are multiple PCA reconstruction "
                                 "tensors in model_manager.misc_tensors. Not "
                                 "clear which one to use.")
            else:
                output_layer = misc_names[0]
            model = keras.models.Model(
                    inputs=[input_layer],
                    outputs=[output_layer])
            model.compile("adam", loss="mse")
            model.fit_generator(
                    gen_ae,
                    steps_per_epoch=spe,
                    epochs=num_epochs,
                    verbose=1,
                    use_multiprocessing=multiprocessing,
                    workers=num_workers,
                    callbacks=[keras.callbacks.EarlyStopping(
                            monitor="loss", 
                            patience=5, 
                            verbose=2, 
                            mode="min"
                        )]
                )
        
        if self.pre_pca:
            print("Freezing PCA layer...")
            model_manager.model.get_layer(
                    self.pca_components_name).trainable = False
        
        # compile model
        model_manager.compile_model(
                loss=self.loss,
                optimizer=self.optimizer,
                metrics=self.metrics
            )
        
        # split train-val
        if validation_split is not None:
            ind_split = int(len(endogenous) * validation_split)
            end_train = endogenous[:ind_split, :]
            end_val = endogenous[ind_split:, :]
        else:
            end_train = endogenous
        
        # get generators
        gen_train = self._get_pdg(
                num_batch, 
                self.history_length,
                seed
            )(exogenous_, end_train)
        
        if validation_split is not None:
            gen_val = self._get_pdg(
                    num_batch, 
                    self.history_length,
                    seed
                )(exogenous_, end_val)
            
        # csv logger callback
        csv_logger = callbacks.CSVLogger(model_path+"_log.csv")
    
        # model checkpoint callback
        if validation_split is not None:
            checkpoint = keras.callbacks.ModelCheckpoint(
                    model_path, 
                    monitor="val_loss",
                    mode="min",
                    verbose=1, 
                    save_best_only=True
                )
        else:
            checkpoint = keras.callbacks.ModelCheckpoint(
                    model_path, 
                    monitor="loss",
                    mode="min",
                    verbose=1, 
                    save_best_only=True
                )
                
        # train
        if validation_split is not None:
            spe_train = np.ceil(num_shows * len(endogenous) / num_batch
                                * (1-validation_split))
            spe_val = np.ceil(num_shows * len(endogenous) / num_batch
                              * validation_split)
        else:
            spe_train = np.ceil(num_shows * len(endogenous) / num_batch)
        
        if validation_split is not None:
            history = model_manager.model.fit_generator(
                gen_train, 
                steps_per_epoch=spe_train, 
                epochs=num_epochs, 
                verbose=1, 
                use_multiprocessing=multiprocessing,
                workers=num_workers,
                validation_data=gen_val,
                validation_steps=spe_val,
                callbacks=[*self.callbacks, checkpoint, csv_logger]
            )
        else:
            history = model_manager.model.fit_generator(
                gen_train, 
                steps_per_epoch=spe_train, 
                epochs=num_epochs, 
                verbose=1, 
                use_multiprocessing=multiprocessing,
                workers=num_workers,
                callbacks=[*self.callbacks, checkpoint, csv_logger]
            )
            
        # save history
        utils.save_history(history, model_path)
        
        # reload best model
        model = keras.models.load_model(model_path)
        
        self.model = model
        
        return model, history
