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
import sys
import tensorflow as tf
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

#==============================================================================
#==============================================================================
#Model preparation =======================================
#==============================================================================
#==============================================================================

#Variables ==============================================================================
#Any model exported using the export_inference_graph.py tool can be loaded here simply by changing PATH_TO_CKPT to point to a new .pb file.
#By default we use an "SSD with Mobilenet" model here. See the detection model zoo for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = 'celular_inference_graph'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

#NUM_CLASSES = 90
NUM_CLASSES = 1

#Load a (frozen) Tensorflow model into memory.==========================================
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
#Loading label map ============================================================
#Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#print(categories)
category_index = label_map_util.create_category_index(categories)

#Helper code ============================================================
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
#==============================================================================
#==============================================================================
#Detection
#==============================================================================
#==============================================================================

PATH_TO_TEST_IMAGES_DIR = str(sys.argv[1])#tem que ser o caminho completo

directory = os.fsencode(PATH_TO_TEST_IMAGES_DIR)
listaDeArquivos = list()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        listaDeArquivos.append(filename)
        continue
    else:
        continue

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}'.format(i)) for i in listaDeArquivos ]

#metodo principal que faz a inferencia de uma imagem com base em um grafo pre treinado
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

# =========================================================================
# =========================================================================
#Metodos auxiliares =======================================================
# =========================================================================
# =========================================================================

def temTelefone(detection_boxes1, detection_classes1, detection_scores1, fileName):

    xEstimado = -1
    yEstimado = -1
    
    #so estou considerando uma classe entao pegarei o box com maior score
    indice = np.argmax(detection_scores1)
    box = detection_boxes1[indice]

    ymin = box[0]
    xmin = box[1]
    ymax = box[2]
    xmax = box[3]
    
    xEstimado=(xmax+xmin)/2
    yEstimado=(ymax+ymin)/2
    
    return xEstimado, yEstimado

######################################################################################
#predicao ############################################################################
######################################################################################

for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  #print("abriu " + str(image.filename))
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  detection_boxes1 = output_dict['detection_boxes']
  detection_classes1 = output_dict['detection_classes']
  detection_scores1 = output_dict['detection_scores']
     
  xEstimado, yEstimado = temTelefone(detection_boxes1, detection_classes1, detection_scores1, image.filename)

  if((xEstimado==-1)and(yEstimado==-1)):
    print("NAO ACHOU "+  str(image_path))
  else: 
    print("ACHOU " + "xEst " + str(xEstimado) + " yEst " + str(yEstimado) + " "+  str(image_path))
    
print("acabou")