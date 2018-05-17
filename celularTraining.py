#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:28:46 2018

@author: henriquebueno
"""

import os
import time

# =============================================================================
# =============================================================================
#vou fazer algumas configuracoes no sistema operacional =======================
# =============================================================================
# =============================================================================

#dentro da pasta research rodar...para resolver o problema "ImportError: No module named nets" faca:
#- protoc object_detection/protos/*.proto --python_out=.
#- export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim

#estou na pasta "research/object_detection"
print(str(os.getcwd())+" --------------------------------")
#vou para a pasta "research"
os.chdir("..")

#vou rodar o comando "protoc object_detection/protos/*.proto --python_out=."
os.system("protoc object_detection/protos/*.proto --python_out=.")

#vou rodar o comando "export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim"
pythonPathOriginal = os.environ.get('PYTHONPATH')
pwd = os.getcwd()
pwdComSlim = pwd + "/slim"
total = pwd + ":" + pwdComSlim
os.environ['PYTHONPATH'] = pythonPathOriginal + ":" + total
#vou voltar para a pasta "research/object_detection"
os.chdir("object_detection")


#=============================================================================
#=============================================================================
#Vou verificar se ja existe um modelo pronto e vou move-lo para outra pasta 
#=============================================================================
#=============================================================================

#mv celular_inference_graph/ celular_inference_graph5/
saidaDeLs = os.system("ls celular_inference_graph")
if(saidaDeLs==0):#ja existe um diretorio entao preciso move-lo
    tempo = time.time()
    novoNomeDir = "celular_inference_graph" + str(tempo)
    os.system("mv celular_inference_graph/ " + novoNomeDir)

#=============================================================================
#=============================================================================
#Vou rodar o script PARA EXPORTAR O TREINAMENTO JA FEITO =====================
#=============================================================================
#=============================================================================
comando = "python3 export_inference_graph.py     --input_type image_tensor     --pipeline_config_path training/ssd_mobilenet_v1_pets.config     --trained_checkpoint_prefix training/model.ckpt-16767     --output_directory celular_inference_graph"
os.system(comando)

print("acabou")