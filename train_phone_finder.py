#vou fazer um classificador. Para isso vou quebrar a imagem em vários pedaços
#a saida de cada imagem sera um vetor onde apenas uma posicao contera o valor 1
#e todas as demais posicoes conterao o valor 0

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

print("Tensor Flow version: " + str(tf.__version__))
print("Numpy version: " + str(np.version.version))

#Definição das variáveis fora do Tensor Flow
n_epochs = 300

#Definição de todos os nós do grafo Tensor Flow
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

#Inicialização de todos os nós do grafo Tensor Flow
init = tf.global_variables_initializer()

#Criação do saver para persistir o modelo criado em disco
saver = tf.train.Saver()

#Inicialização da sessão e execução do treinamento
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:#checkpoint every 100 epochs
            #persistindo o modelo criado até o momento
            save_path = saver.save(sess, "/tmp/my_model_henrique_bueno.ckpt")
            #save_path = saver.save(sess, "./my_model_henrique_bueno.ckpt")
        
        #sess.run(training_op)
        print("epoch: " + str(epoch))
        
    valor_f = f.eval()
    print("f= " + str(valor_f))
    
    #persistindo a versão final do modelo criado
    save_path = saver.save(sess, "/tmp/my_model_final_henrique_bueno.ckpt")
    #save_path = saver.save(sess, "./my_model_final_henrique_bueno.ckpt")
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


