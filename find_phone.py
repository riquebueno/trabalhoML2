#vou fazer um classificador. Para isso vou quebrar a imagem em vários pedaços
#a saida de cada imagem sera um vetor onde apenas uma posicao contera o valor 1
#e todas as demais posicoes conterao o valor 0

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

print("Tensor Flow version: " + str(tf.__version__))
print("Numpy version: " + str(np.version.version))

#estou adicionando o grafo gerado no treinamento ao grafo default
saver = tf.train.import_meta_graph("/tmp/my_model_final_henrique_bueno.ckpt.meta")

#Inicialização da sessão e execução do treinamento
with tf.Session() as sess:
    
    #restaurando o modelo criado pelo script train_phone_finder.py
    saver.restore(sess, "/tmp/my_model_final_henrique_bueno.ckpt")
    
    x1 = tf.get_default_graph().get_tensor_by_name('x:0')
    sess.run(x1)
    print(x1)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


