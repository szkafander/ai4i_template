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
.. module:: pre_post
   :platform: Unix, Windows
   :synopsis: Contains high-level abstraction of pre- and postprocessing 
       functionality.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

This module contains high-level abstraction of pre- and postprocessing 
functionality, in form of a chainable processor pipeline object. The module 
provides the following functionality:
    
================================  =============  ==============================
name                              type           summary
================================  =============  ==============================
Processor                         class          A chainable pipeline of 
                                                 processing functions.
estimate_image_standardization    method         Returns a Processor that does
                                                 standardization given an image
                                                 dataset.
estimate_process_standardization  method         Returns a Processor that does
                                                 standardization given a scalar
                                                 dataset.
function_from_recipe_line         method         A helper function that returns
                                                 a function given a recipe
                                                 line.
================================  =============  ==============================

"""

from __future__ import absolute_import

import functools
import inspect
from processing import pre_post_fns
from misc import utils
import numpy as np
import pprint
from typing import Any, Callable, Optional, Tuple, Union

    
# this is a lookup that maps function names in recipes to function names in 
# image_processing
recipe_names = {
        "rescale": pre_post_fns.rescale,
        "take_channel": pre_post_fns.take_channel,
        "subtract_mean": pre_post_fns.subtract_mean,
        "standardize": pre_post_fns.standardize,
        "column_normalize": pre_post_fns.normalize_columns,
        "column_standardize": pre_post_fns.standardize_columns,
        "mask_image": pre_post_fns.MaskImage,
        "sum_values": pre_post_fns.sum_values,
        "spatial_transform": pre_post_fns.SpatialTransformAugmentation,
        "extract_slag_signal": pre_post_fns.SlagSignalExtractor
    }


def function_from_recipe_line(recipe_line: str) -> Callable:
    """ Returns a Callable from a recipe line. The recipe is a text file. The 
    argument recipe_line is a line of this recipe. Function_from_recipe_line
    attempts to returns a Callable, either a function or an instance of
    PersistentPrePostFunction, parametrized by whatever is found in 
    recipe_line.
    
    :param recipe_line: A line of text.
    :type recipe_line: str
    :returns: A Callable, normally a function or an instance of 
        pre_post_fns.PersistentPrePostFunction, parameterized by what is found 
        in recipe_line.
    
    """
    function_name = recipe_line[0]
    if function_name in recipe_names:
        function = recipe_names[function_name]
        kwargs = pre_post_fns._fns_kwargs[function]
        kwarg_values = utils._make_iterable(recipe_line[1])
        kwargs = {kwarg: kwarg_value for kwarg, kwarg_value 
                  in zip(kwargs, kwarg_values)}
        if inspect.isclass(function):
            return function(**kwargs)
        else:
            return functools.partial(function, **kwargs)
    else:
        return None


class Processor:
    """ A processor class, instances of which are chainable pipelines of 
    Callables.
    
    :ivar functions: An iterable of functions, the function chain or pipeline.
        Elements of self.functions act upon a single input argument - this
        argument is referred to as the 'argument' in the documentation of the
        Processor class.
    
    **Usage**
    
    >>> # instantiate from a list of functions
    >>> processor = Processor([function_1, function_2,...])
    >>> processed = processor(data)
    
    
    >>> # instantiate from a recipe
    >>> processor = Processor.from_recipe(recipe_path)
    >>> processed = processor(data)
    
    
    >>> # instantiate as a chain of member methods
    >>> processor = Processor.take_axis(channel="blue").range_normalize((0,1))
    >>> processed = processor(data)
    

    """
    
    # this maps abbreviations used in recipes to names used in this class
    _abbr = {
            "pre": "preprocess",
            "post": "postprocess"
        }
    
    def __init__(self, *functions) -> None:
        """ Constructor.
        
        :param functions: Callables. These are executed at __call__ in the 
            order they were provided to __init__.
        :type functions: Callable, any number
        
        """
        self.functions = tuple([function for function in functions 
                          if function is not None])
    
    def __call__(self, data: Any) -> Any:
        """ __call__ override, applies Callables one-by-one in self.functions
        to data.
        
        :param data: The data to apply functions to.
        :type data: Any
        :returns: Any type, data acted upon by self.functions. For example,
            self.functions[1](self.functions[0](data)).
            
        .. warning::
            The argument data, or transformed outputs based on data must be 
            understandable by subsequent functions in self.functions. 
            
        
        """
        for function in self.functions:
            data = function(data)
        return data
    
    def __str__(self) -> str:
        """ String representation.
        
        """
        return ("Processor object with the following function "
                "chain: \n {}".format(pprint.pformat(self.functions)))
    
    @staticmethod
    def from_recipe(
            recipe: Union[str, dict], 
            step: str = "pre"
        ) -> "Processor":
        """ Returns a Processor object from a recipe. A recipe is a text file.
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
        'postprocess' tell Processor.from_recipe what functions to use to
        instantiate Processors for pre- or postprocessing. Under a top-level
        keyword, a JSON object lists the function names as keywords and
        function parameters as values. To understand parameters, see the module
        processing.pre_post_fns. To understand function names (keywords in the
        recipe), please see pre_post.recipe_names. To understand top-level
        keywords, please see pre_post.Processor._abbr.
        
        .. note::
            Any paths given in a recipe should be treated as relative paths.
            See misc.utils.relative_path.
        
        :param recipe: A string specifying a path to a recipe text file or a
            dictionary, i.e., from json.load(recipe).
        :type recipe: Union[str, dict]
        :param step: A string specifying what to return: a preprocessor or a 
            postprocessor. Valid values are 'pre' and 'post', otherwise given
            in pre_post.Processor._abbr.
        :returns: A Processor from the recipe.
        :rtype: Processor
        
        """
        if step in Processor._abbr:
            step = Processor._abbr[step]
        if isinstance(recipe, str):
            recipe = utils.read_recipe(recipe)
        try:
            recipe = recipe[step]
        except:
            return None
        return Processor(*[function_from_recipe_line(line) for line 
                         in recipe.items()])
            
    def add_function(
            self, 
            function: Union[Tuple[Callable], Callable]
        ) -> "Processor":
        """ Adds a function to self.functions and returns a new Processor
        object.
        
        .. note::
            The added function is a parametrized Callable that takes a single
            argument. This single argument will be acted upon by the function 
            chain.
        
        :param function: The function to add. This will be appended to 
            self.functions.
        :type function: Callable
        :returns: A new Processor with the updated functions.
        :rtype: Processor
        
        """
        functions = self.functions + utils._make_iterable(function)
        return Processor(*functions)
    
    def range_normalize(
            self, 
            output_range: Tuple[float, float] = (0,1)
        ) -> "Processor":
        """ Adds range normalization to the function chain. Returns a new 
        Processor with updated functions.
        
        :param output_range: The argument will be scaled between this range.
        :type output_range: Tuple[float, float]
        :returns: A Processor object with updated functions.
        :rtype: Processor
        
        """
        return self.add_function(functools.partial(
                pre_post_fns.rescale, output_range=output_range))
    
    def mean_normalize(
            self, 
            mean: Optional[float] = None
        ) -> "Processor":
        """ Adds mean normalization to the function chain. Returns a new 
        Processor with updated functions.
        
        :param mean: This will be subtracted from the argument. If not 
            provided, the mean will be calculated on-the-fly.
        :type mean: Optional[float]
        :returns: A Processor object with updated functions.
        :rtype: Processor
        
        """
        return self.add_function(functools.partial(
                pre_post_fns.subtract_mean, mean=mean))
    
    def standardize(
            self, 
            std: Optional[float] = None
        ) -> "Processor":
        """ Adds standardization to the function chain. Returns a new 
        Processor with updated functions.
        
        :param std: The argument will be divided by this value. If not 
            provided, the std will be calculated on-the-fly.
        :type std: Optional[float]
        :returns: A Processor object with updated functions.
        :rtype: Processor
        
        """
        return self.add_function(functools.partial(
                pre_post_fns.standardize, std=std))
    
    def column_normalize(
            self, 
            mean: Optional[Union[np.array, float]] = None
        ) -> "Processor":
        """ Adds column normalization to the function chain. Returns a new 
        Processor with updated functions.
        
        :param mean: This will be subtracted from the argument. If not 
            provided, the mean will be calculated on-the-fly. This can be a
            single float or a numpy array. If a numpy array, it will be 
            subtracted columnwise. In this case, mean has to be a 1D array
            and len(mean) == data.shape[1].
        :type mean: Optional[Union[numpy.array, float]]
        :returns: A Processor object with updated functions.
        :rtype: Processor
        
        """
        return self.add_function(functools.partial(
                pre_post_fns.normalize_columns, mean=mean))
        
    def column_standardize(
            self, 
            std: Optional[Union[np.array, float]] = None
        ) -> "Processor":
        """ Adds column standardization to the function chain. Returns a new 
        Processor with updated functions.
        
        :param std: The argument will be divided by this. If not 
            provided, the std will be calculated on-the-fly. This can be a
            single float or a numpy array. If a numpy array, it will be 
            applied columnwise. In this case, std has to be a 1D array
            and len(std) == data.shape[1].
        :type std: Optional[Union[numpy.array, float]]
        :returns: A Processor object with updated functions.
        :rtype: Processor
        
        """
        return self.add_function(functools.partial(
                pre_post_fns.standardize_columns, std=std))
    
    def mask(
            self, 
            mask: Optional[np.array] = None
        ) -> "Processor":
        """ Adds low-level masking to the function chain. Returns a new 
        Processor with updated functions.
        
        :param mask: A binary mask. The size of the mask and the argument must
            be the same. If mask is None, the argument will be returned 
            unchanged.
        :type mask: Optional[numpy.array]
        :returns: A Processor object with updated functions.
        :rtype: Processor
        
        """
        return self.add_function(functools.partial(
                pre_post_fns.mask, mask=mask))
    
    def take_axis(
            self, 
            channel: Union[str, int] = "blue", 
            squeeze: bool = False
        ) -> "Processor":
        """ Takes a specific array axis before proceeding on the function 
        chain. Returns a new Processor with updated functions.
        
        :param channel: A string or integer that specifies the axis to take. If
            a string, values can be 'red', 'green' and 'blue' (this assumes a 
            color image). The channel is the last dimension of the argument.
        :type channel: Union[str, int]
        :returns: A Processor object with updated functions.
        :rtype: Processor
        
        """
        return self.add_function(functools.partial(
                pre_post_fns.take_channel, channel=channel, 
                squeeze=squeeze))
    
    # Persistent functions - these all store some internal state. Unlike
    # regular functions in the chain, these are class instances.
    def spatial_transform(
            self, 
            rotation_range: float = 15.0,
            width_shift_range: float = 0.0,
            height_shift_range: float = 0.0,
            zoom_range: float = 0.0
        ) -> Callable:
        """ Adds spatial transformation augmentation to the function chain.
        Spatial transformation augmentation applies a random rigid 
        transformation to an image argument.
        
        :param rotation_range: The +/- range of random rotation in angles.
        :type rotation_range: float
        :param width_shift_range: The +/- range of random horizon��z��[�Lʡ�xN���.,6���M�K)6�rNUW�*�Y��q����OI��w�����1��u6��S{�i֭Ye<�f�\[��8l�+{vy�5/�|��p�D�YQ_��1�X���V�2�I��k����y�g��ٌ$���3�xqh�9�d*���ڴ�s%.22H/�������7��*1�t� ��67��Bm�$NŁ�����l$3�
!ᕙr��B3�x�Q�9J�X/���,uT���S9Kt�=����ѷ���://�>p^7�#����m���N7v�v�~̌�)шZ+-4e)|Y�,`��
d�����[#B�i�{��J�8B.��o_*/��-ػ�e��R;��2?�=if�������.�lrq��$P� 3U�#�����Ƨs����Da氚@cY��	H�\��%Lwn��(c%�*�W;�/�t�^<7�e�&@��C�3�2bQڵ�@�*M���̹�&#��z���^Ta+�G%����I{�+,_L�.�.�����r���y��]֓ݑ�=4�1�y��8��&������n��M�=��q�����H�}����C�0���� ��	�a�%:��~1'̧���΂�D6��4�RXבg?ӭ0ȯ�|�	
A�Ұ�ė�d�]%�%�`�ncx-���.Z��f�T�����N��6= f0/���j�L-�t -|�K�d�66��'�
���ab��C��n�Ŋ(h�=�>85��<��9�ɹደ����������ګ���i�� O]f�Z4oΗn>���^����������� 8�t4N�y��4��%�"ͥwQ~ �&)#�P�c���w�@Ƈވ�ܙ�K���9��P*�'?;�'H�����7�m
�Yv�_��Y����u�N����?Ø�߀Ъ!��7!�m����{ӧ�Hݻ��4��1n�A�����+9�d̓��#�s��|1[�Nj�����{����H���f���<�n�DtC�E��`"�J��pJ��/x�Y������"��dh&��JC&Ė���=]��cl���;T�;?s�m�P�a��wp���Rݤ�2�䔛؄�'��T���5��3�?F��!'�|���ΖE�K	�� 6��@B����3`�,�9�MM�ۈ�q
�Qｾ�
{�HX���T����B���~C@b2���P̭F'�@!��홿b�Fz��(��Dϴ~�k [�g���ȏT	�@i�k�K��a\,U���F��!,�Dk0���j�8+�f�u<~!9(>���m��+��_��-*L���M��א`v�^"�+���u�SpK��n�`�mo��[<^�/.�E�/��/�6�{�
-�T����X��c;ʫU�
hZ��\�lL�N=�v����)��=��dES����"5���AJ��l��X��_�T�XG����bO�4&�qWm�3j`��ֳp�tW�!z�X���m���j<	�H�ut�}�0�a�H�\�vL��c1-+'��[U�+d4�D�"4�ޣ�R�b�%�j@x������,"}�8��Rvװ�i�~�t�^�F�����0�@���C���%�*$nh����w�n4QJc���|X�Wp�D��
Q��X�_
���v��Ѣ�-?�� ߋ\Q6kbz�;Ԃ��=�����5މ���iS�\�E�b�i1�q Dѝ�Q�1�Ԝ��[���8�<��|.>�-��\��^)�o<`<��oA��A�����;��(\�l��=h�s/ɲmp�L�gz��G��e�lS����_�7����Է�	>q5L�o����(w���F���}����f�\���o��)u=D�0l�R��_WLp=}���*O�\Nt��=ґbʀto���bdҺCD����`�����H�qbe��&���zu�o��bo^AT��G�ki�J?.-]Ⱥ\���:���)A^`��3<"�������UI���,��sy���d����8�6(��*;,���w��:F�uj=b�-m&�P�<�Æ�^���n�����m9��+OuΈ,�D�8c����$��;ou`��o�:L����4�֬2Y3~���u
���=�<�J>EV8I����/���p�����`��$�յ
��s�<�3��lF�OәS�84��Y2��imZ�����Z]�]�t]��c��e	�i��J��cs�6	C��@�I�me6��	������L�ot��w�˨���A���lx�:�\�ѩ�%:���ŀ���Z�V���E8��]�UӆF�s�6�zA�;{�a?fF��hD�������N��a2WY�G�������˽Mf%��Q!�z�/�����]��S�S��NA��Ϟ43k�hp��W�[6� ]l(v���Ǒ��Zv�ӹ��J}�0sXM�����$O�����;���?���q���P�B/��
�2} G�!�b�(��X�W����_�\D��=�_c/��ԣ����=��/&KT���xZI\9��L�<t��.��������NWxI�ej�z7Hr�&���yl���|$���o~�!�[���GlP҄ǰ qْ`_?�����SZg�e"��k�j)��ȅ�����V��|�����	iXR�K^�Ȯ���@�h�1��f�rQ�lm3@*Y�ZHg'\q�3����s����yM:�>�%g2���v���01�P�![	V��bE�T�� P��s�ن�����E���l�����r�d�U��봍ZB��.3tI-�7�K7[�d����{Cq��y���� �l:'��<�fe�k��һ(?����?g�ȱ���;� �Co�p�L�%F���`(ʊ����_�}���v�۶��,��/R����A�:S'b�o��a��o@h��ʿ����F޽��k���Id���7� Lz��ڕ}2�I����M��-b'5�U
P��=BGXs$EIz�PI�g��K"�!e��RR0y%�h8�|݅<�,�e]�|R�w24�~�!b��z��1�q�}��*����q]�6A(��0Xy��;8\tw�n�hcr�Ml�
��v��_ǚJ�ٙ����Z󐓀_>��vg˂�饄�W��}Z ��r���C����&�mD�8�Ѩ�^_}�=v$,���J*�W��|����ΉG�! 1HSD(�V�c��j���_�u#=��~�]L�gZ?�5��ɳ��
N�G��v�4ߵإ��0.��|fr�����F�� tFk5S��A��:�?���u��Ķzj��^�/��&T@�P�kH0;�/ʕ�e�:�)�%�`7W0xԶ�r[�-/�����_�?�=�_�W�In}Q�Zر�ժH4-B�@�A6�z��H;������ОB���"�)j�����R� %�I���I,�L��/M�V,���srU��y縫6�50ֆx�Y8�U�+���g,Њ���6T��{5��H$�::��u��0P�k�	];&u�㱘������2�A"�Bo�Q`)~���O5� ��Ohn�`�>o�I���k��4�{?y:f/m#^��GY_ �b��z�U74zk��;p7�(���XrO>,���m��k�(�^��/��pU�@�hQti��E�(�51��jAj�Ȏ?r��q��D�S��)��΢f1D����8 �����֘Qj��í��r�p	�@>�ޖ�].zt��70��巋 Yߠ}�S���K�d.e6U���ֹ�d�6�S&	�
z�3=Iˣd�L��[�g�/כ��a}�[���&�Yz~H��u�g�DJ�>�����Z�^t\���ʔ��"\6�[)�¯+&��>��y��'�H.'�H��H1e@��7C�y12i�!�r�Ec0FPBR�	�@��8���`��~V�:�7V[�7� �C��#��J���.d].��weth� /0�ĂX���ނ\몤��F��t���EWS2Ǐ��Y�G�BX�I����`T#A�:�1���6�r�l��aCn����w�GS��R����SՕ���:gDb�F�1���Sj���:0mq�7w�����q�ukV��?�V�:[�ʞ]��@�%�"+�$Q�{Vԗq|L8V��끅U�����Z��9t��rz6#	���)^�`�,��j�6��\��̇