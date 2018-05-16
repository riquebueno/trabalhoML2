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
import math
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
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
PATH_TO_TEST_IMAGES_DIR = '/Users/henriquebueno/DOUTORADO/2018.01/ml/trabalho2/find_phone_data'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,59,60,61,62,63,64,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134] ]
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in [0,1] ]

# Size, in inches, of the output images.
#IMAGE_SIZE = (12, 8)
#IMAGE_SIZE = (490, 326)
IMAGE_SIZE = (5, 5)

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

    #im_width=490
    #im_height=326
    
    #print("(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)")
    #print("box: " + str(box))
    
    ymin = box[0]
    xmin = box[1]
    ymax = box[2]
    xmax = box[3]
    #print("(xmin, xmax, ymin, ymax)")
    #print(xmin, xmax, ymin, ymax)
    
    xEstimado=(xmax+xmin)/2
    yEstimado=(ymax+ymin)/2
    #print("(xEstimado, yEstimado)")
    #print(xEstimado,yEstimado)
    
    return xEstimado, yEstimado

#carrega o arquivo com o nome dos arquivos e as coordenadas dos celulares em cada
#arquivo em uma matriz onde cada linha é um registro do arquivo labels.txt e cada
#linha tem 3 valores: NOME_ARQUIVO X Y
#ATENCAO: os 3 valores de cada linha sao armazenados como string
def carregaArquivo(arquivo):

    #abre o arquivo
    file = open(arquivo, "r")

    #le todas as linhas
    linhas=file.readlines()
    
    #fecha arquivo
    file.close()

    return linhas

#Funcao que recebe como entrada um array com os registros de entrada e um
#string com o NOME_ARQ. O retorno é a posicao do arquivo NOME_ARQ em registros.
#Caso ele não encontre o nome, ele retorna -1
def localiza_X_e_Y(linhas, nomeArquivo):
    indiceLinha = -1
    xRetorno = -1
    yRetorno = -1
    
    pos=0
    for linha in linhas:
        temp = linha.split()
        nome=temp[0]
        nome = PATH_TO_TEST_IMAGES_DIR+"/" + nome
        #print(nome)
        if(nome==nomeArquivo):
            indiceLinha = pos
            break
        pos+=1
        
    if(indiceLinha != -1):
        linha = linhas[pos]
        linhaTokenizada = linha.split()
        xRetorno=float(linhaTokenizada[1])
        yRetorno=float(linhaTokenizada[2])
        
    return xRetorno, yRetorno

#calculo de distancia
def distancia(x1,y1,x2,y2):
     dist = ((x1 - x2)**2 + (y1-y2)**2)**.5 
     return dist

######################################################################################
#predicao ############################################################################
######################################################################################

arquivo = "/Users/henriquebueno/DOUTORADO/2018.01/ml/trabalho2/labels.txt"
linhas = carregaArquivo(arquivo)
#pos = localizaArquivoEmLinhas(linhas, "129.jpg")
#print(pos)
#linha = linhas[pos]
#linhaTokenizada = linha.split()
#print(linhaTokenizada)

sim = 0
nao = 0
simComDistancia=0

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
    #print("Não: " + str(image.filename))
    print("NAO")
    nao+=1
  else:
    #vai consultar o arquivo do professor com as coordenadas para encontrar as coord
    #corretas de image.filename. Atencao: o nome do arquivo tem que estar completo com a pasta
    xCorreto, yCorreto = localiza_X_e_Y(linhas, image.filename)
    
    dist = distancia(xEstimado,yEstimado,xCorreto,yCorreto)
    if(dist<0.05):
        simComDistancia+=1
    
    #print("Sim: " + str(image.filename) + " x, y= " + str(xEstimado) + "-" + str(yEstimado))
    print("SIM " + str(dist) + " xEst " + str(xEstimado) + " yEst " + str(yEstimado) + " xCorr " + str(xCorreto) + " yCorr " + str(yCorreto) + image_path)
    sim+=1
    
print("% sim: " + str(sim/(sim+nao)) + " " + str(sim))
print("% nao: " + str(nao/(sim+nao)) + " " + str(nao))
print("% sim com distancia < 0.05: " + str(simComDistancia/(sim+nao)) + " " + str(simComDistancia))

print("acabou")