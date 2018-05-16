#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:46:36 2018

@author: henriquebueno
"""

#O trabalho utilizou a API Tensorflow Object Detection.
#https://github.com/tensorflow/models/tree/master/research/object_detection

#Para instalar a API, siga os passos abaixo:
#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
#Para validar a instalação, execute o comando abaixo:
#python object_detection/builders/model_builder_test.py


#IMPORTS =============================================================================
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

#Env setup =============================================================================
# This is needed to display the images.
#%matplotlib inline

#Object detection imports =============================================================================
#Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util
















print("acabou")