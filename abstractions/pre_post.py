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
        :param width_shift_range: The +/- range of random horizonêÔzÄÔ[ÚLÊ¡²xN‡¹½.,6³ßİMıK)6ÚrNUW*êœYˆ‰qÆĞ÷ïOI¨éwŞêÀ´Å1ßÜu6˜şS{ÇiÖ­Ye<²fü\[Ñë8lé+{vyâ5/”|Š¬p’DîYQ_Æñ1áXÓ¬VÁ2üIş«kşéçĞyôgÊéÙŒ$œŸ¦3§xqh‚9³d*ª‘ÓÚ´Ús%.22H/µº„»¸éºÆ7¥Ç*1ËtÓ ü•67ÇæBm†$NÅ¤“¾ÛÊl$3¦
!á•™rßèB3îx—QÑ9JƒX/Ùğ,uT¹–¢S9Ktø=®™‹ŸÑ·µ¤­://‹>p^7»î«¦#Œ”çĞmŞõ‚N7vövÃ~ÌŒ€)ÑˆZ+-4e)|Y,`ñÃ
d®²ªú[#BİiÉï£—{›ÌJ£8B.õào_*/â“ã-Ø»úe§è§R;‚2?Ÿ=ifÖ®ÑàœÃ¯.·lrqºØ$Pì 3U#ÿœµìÆ§sı“•úDaæ°š@cYëÃ	H\õƒ%Lwn‘(c%ã*ÂW;ô/¡t…^<7¼eú&@ìCÔ3Ä2bQÚµ±@¯*MÃû¿Ì¹ˆ&#ÄÿzŒ¿Æ^Ta+¨G%¶ë×ÇI{Ø+,_L–.¨.ˆƒıñ´’¸rà™Âyèäµ]Ö“İ‘Ù=4¨1Êy¶8®ğ&’ÔËÔàõnäMâ=±ÛqóØøìùHü}ßüØCê·0×íØ  ¤	aâ²%:À¾~1'Ì§¡§´Î‚ËD6Ï×4ÕRX×‘g?Ó­0È¯ù|Ÿ	
AÛÒ°¤Ä—¼d‘]%ô%€`Ñncx-Í€å¢.ZÙÚf€T²ôµÎN¸â6= f0/ıÑçjñL-óšt -|´KÎdò66¬'ì
ÔÕıab¨¡C¶¬n»ÅŠ(h©= >85ªç<³‡9ÚÉ¹á‹°åÙîôÃ§÷å¨ÉÚ«˜‰×iµ„ O]fè’Z4oÎ—n>¶ÊÉ^³£Õ÷†âÚãó‹» 8Ùt4N™yöÍ4Ê¢%Ö"Í¥wQ~ ÿ&)#ÎP‘c¡ûwÔ@Æ‡ŞˆáÜ™èKŒèï9¦ÁP*”'?;ˆ'H¿šû”í7í¬·m
ßYv¡_¤ĞY½ƒŠu¦NÄêßö?Ã˜õß€Ğª!¨•7!ßmëı¼{Ó§×Hİ»“È4£½1nÌA˜ô´ûµ+9údÌ“ÂÓ#Ïs›|1[ÄNj¸ª š{„°æHŠ’ôf¡’¼<În—DtCÊE¥¤`"òJøÑpJùº/x¢YŞËºÒù¤"®ïdh&ŒıJC&Ä–×ôÜ=]¢clãÌû;T†;?sãºm‚P÷a°ò†‹wp¸èîRİ¤Ñ2Æä”›Ø„ê'æíT§¿5•³3Í?Fëµæ!'¿|ÂÇíÎ–EÓK	›¯ 6û´@BŸåÔÁ3`‡,³9äMMâÛˆˆq
£Qï½¾ú
{ìHXÈÓñ•T¯º›ùBıï~C@b2¦ˆPÌ­F'Æ@!Õöí™¿bëFz§ı(»˜DÏ´~Æk [“g„›œÈT	í@i¾k±K§Éa\,UùÌäFÁÿ!,Dk0èŒÖj¦8+ƒf«u<~!9(>ëö‰mõÔ+½â_³-*L¨€ÚM¡Ú×`v¾^"”+£ËŞuÒSpK’Án®`ğ¨moå¶à[<^¸/.àEß/éî/¾6¬{¿
-®T“Üú¢Xµ°c;Ê«U‘
hZ„\ƒlLõN=‘v‘›±)†¡=…¸dESÔşÜï"5µ¥ÆAJ®“lí“Xâ™•_šT­XG×çäªbOó4&ÎqWm¶3j`¬ñÖ³pä«tW !zÏX »Ïm¨ë÷j<	‘H€utã}ë0•a H×\ºvLêÇc1-+'Óå[Uó+d4ƒD…"4ŞŞ£ÀRübß%Ÿj@xƒŸĞÜÎÁ,"}Ş8š“Rv×°Íi÷~òtÌ^ÚF¼œ²0¾@ÅÂCõõÊ%ª*$nhôÖÚİwàn4QJc‹±ä|XöWpÛDû×
Q¦½XÃ_
‚—áªvòÑ¢Ş-?éÒ ß‹\Q6kbzû;Ô‚ÔÌ=äƒãØ5Ş‰Š§üiSŞ\EÍbˆi1ÿq DÑÿQ­1£Ôœı‡[û¡å8à<¸|.>¼-‘»\ôè^)’o<`<ÛËoA²¾Aû¾§Âí§;–É(\ÊlªÌ=h­s/É²mp§Lôgz’–GÉÎe™lS·´ÏÖ_®7ÙÏÃúÔ·Ü	>q5LŞo³ôü(wëâÏF‰”}ş•ïáfµ\½è¸ÁoÅ•)u=D¸0l·Rö…_WLp=}¨×ó*Oæ‘\Nt‘=Ò‘bÊ€to†üóbdÒºCDåş‹Æ`Œ „¤úHqbe¹Á&Åı¬zuÄo¬¶bo^AT‡ûGÜki•J?.-]Èº\ÌïÊ:èĞÖ)A^`’‰3<"°Æåÿà½¸ÖUI·«,­ésy‹®¦dßı³86(…°*;,’‘ówÁ¨:F‚uj=bê-m&åPÙ<§Ã†Ü^›Ùïn¦ş¿¥m9§ª+OuÎˆ,ÄD8cèû÷§$Ôô;ou`Úâ˜oî:Lÿ©½ã4ëÖ¬2Y3~®­èu
¶ô•=»<ñšJ>EV8I¢÷¬¨/ãø˜p¬ÀéÖ«`™ş$ÿÕµ
ÿôsè<ú3åôlFÎOÓ™S¼84ÁœY2ÕÈimZí¹™¤—Z]Â]Üt]ã›Òc•˜e	ºiş€J››cs¡6	C§â@ÒIßme6’Ó	…†ğÊL¹ot¡‡w¼Ë¨è¥A¬—Àlx–:ª\ËÑ©œ%:ü×ÌÅ€ÏèÛZÒV——E8¯›]÷UÓ†FÊsè6ïzA§;{»a?fFÀ”hD­•š²”¾¬N°øa2WYÕGı­¡î´ä÷ÑË½Mf%Š‡Q!—zğ·/•ñÉñŠì]ı²SôS©NA™ŸÏ43k×hpÎáW—[6¹ ]l(v€™ªÇ‘ÆÎZvãÓ¹şÉJ}¢0sXM ±¬õá$O®úÁ¦‰;·ÈÇ?”±’qá«ú—PºB/›
Ş2} Gö!êb±(íÚX W•¦áı_æ\D“â=Æ_c/ª°Ô£Ûõëã¤=ì–/&KTÄÁşxZI\9ğƒLá€<tòÚ.ëÉîÈìÔå¼ÛNWxIêejğz7Hr‚&ñØí¸ylüö|$ş¾Îo~ì!õ[˜ëöGlPÒ„Ç° qÙ’`_?Š˜æÓĞSZgÁe"›çkšj)¬ëÈ…³‰Ÿ†éVä×|¾Ï‰ í	iXRâK^²È®’úˆ@°h·1¼–fÀrQ­lm3@*YúZHg'\q›3˜—şèŒsµø¦–yM:€>Ú%g2ùÖÎvêêş01ÔP![	V·İbE´TŒ PœÕsÙ†ÃíäÜğEØÇòŒl÷úá¿ÓûrÔdíUÌÄë´ZB€§.3tI-š7çK7[åˆd¯ÙÑê{CqíñyŠÅİ œl:'„Ì<ûfeÑk‘æÒ»(?€“”‘?g¨È±ĞıÏ;ê ãCoÄpîLô%Fô÷Ó`(ÊŠ“ŸÄ¤_Í}Êö›vÖÛ¶…ï,»Ğ/Rè¬ŞÀŠAÅ:S'bõoûŸaÌúo@hÕÔÊ¿›ïˆ¶õşFŞ½éÓk¤îİIdšÑŞ7æ LzÚıÚ•}2æIáé‘ç¹ÀM¾˜-b'5ÜU
PÍÆ=BGXs$EIz³PIŞg·ŠK"º!e¢RR0y%üh8¥|İ…<Ñ,ïe]é|R×w24Æ~¥!bËëzî®ÑÀ1¶qæ}‚*Ã†Ÿ¹q]Š6A(û0XyÃÅ;8\tw©nÒhcrÊMlÂ
õóvªÓ_ÇšJ„Ù™æ£õZó“€_>áãvgË‚¢é¥„ÍW€›}Z ¡Ïrêà°C–¿Ùò¦&ñmDÄ8…Ñ¨÷^_}…=v$,äéøJ*‹WİÍ|¡ş÷Î‰G¿! 1HSD(æV£c jûöÌ_±u#=„Ó~”]L¢gZ?ã5€­É³ÂÍ
NäGª„v 4ßµØ¥Óä0.–ª|fr£àÿ‰–F¢µ tFk5Sœ•A³Õ:?¿ŸuûŠÄ¶zj†•^ñ/ÈÙ&T@í¦PíkH0;ß/Ê•Ñeïƒ:é…)¸%É`7W0xÔ¶·r[ğ-/Üğ¢Šï—ô÷_›?Ö=_…WªIn}Q¬ZØ±åÕªH4-BŠ@®A6¦z§H;‚ÀÈÍØÃĞBÜ²Š")jî÷‘šÚRã %×I¶ÀöI,ñL‚Ê/MªV,‡£ësrU±§yç¸«6Û50Ö†xëY8òUº+‚Ğ½g,ĞŠŒİç6Tõ{5„H$À::Šñ¾u˜Ê0P¤k®	];&u†ã±˜–•“éò­ªù2šA"BoïQ`)~±ï‡O5  ¼ÁOhnç`‘>oÍI©‚»kØæ4‚{?y:f/m#^ÎÀGY_ Èbá¡úzåU74zkíî;p7š(¥±ÅXrO>,û«¸m¢ık…(Ó^¬á/ÁËpU»@ùhQï–Ÿti€ïE®(›51½ıjAjæÈ?rˆÁqìšïDÅSş´)ï®Î¢f1D†´˜ÿ8 ¢èÎÿ¨Ö˜QjÎşÃ­ıĞrœp	Ü@>—Ş–È].zt¯É70íå·‹ Yß }ßSáöÓK‡d.e6Uæ´†Ö¹—dÙ6¸S&	
z‡3=IË£dç²L¶©[Úgë/×›ìça}ê[îŸ¸&ï·Yz~H”»uñg£DJÇ>ÿÊ÷ğ³Z®^t\ˆà·âÊ”ºŠ"\6[)ûÂ¯+&¸>Ôëy•‡'óH.'ºHÈéH1e@º7Cşy12iİ!¢rÿEc0FPBRı	¤@‡8±²Ü`“â~V½:â7V[±7¯ ªC‰ı#îµŒ´J¥—–.d].ƒæwethë” /0ÉÄ‚XãòğŞ‚\ëª¤ÛÕF–Öt¹¼EWS2ÇïşYœG”BX•IƒÈù»`T#AŒ:µ1õ–€6“r¨lÓaCn¯‹Íìw·GSÿßRŠ¶œSÕ•§Š†:gDb¢Fœ1ôıûSjú·:0mqÌ7w¦ÿÔŞqšukV¬?×Vô:[úÊ]ø@Í%Ÿ"+œ$Q‡{VÔ—q|L8Vàôë…U°Ì’ÿêZ…ú9tı™rz6#	ç§éÌ)^š`Î,™Šjä´6­ö\‰‹Ì‡