TO DO
- IDENTIFICAR QUANTAS ENTRADAS EU CONSIGO IDENTIFICAR COM CELULAR E QUANTAS EU NAO IDENTIFICO (52% do total).
- DEPOIS DISSO, PARA AQUELAS QUE EU CONSIGO, VERIFICAR AS DISTANCIAS (2% do total).
% sim: 0.5193798449612403
% nao: 0.4806201550387597
% sim com distancia < 0.005: 0.023255813953488372
- Precisarei partir para treinar a rede.
- Marcar as imagens do dataset.
- Ver video 3.

- baixar https://github.com/tzutalin/labelImg
- extrair o zip, entrar na pasta e executar 
- make qt5py3
- python labelImg.py
- baixar https://github.com/datitran/raccoon_dataset
- Marcar todas as imagens com o programa https://github.com/tzutalin/labelImg. Ele gera arquivos xml para todas as imagens com as coordenadas do box desenhado.
- Agora vou converter o xml para csv. Pegar arquivo https://raw.githubusercontent.com/datitran/raccoon_dataset/master/xml_to_csv.py
- arquivos xml convertidos para a pasta data/train e data/test. Coloquei train e test iguais.
- peguei o arquivo https://raw.githubusercontent.com/datitran/raccoon_dataset/master/generate_tfrecord.py e alterei o label racoon para celular
- dentro de research rodar "sudo python3 setup.py install"
- para gerar .record a partir do .csv faca: 
--python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
--python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
- Criar o arquivo "ssd_mobilenet_v1_pets.config" e colocar dentro
- curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
- Put the config in the training directory, and extract the ssd_mobilenet_v1 in the models/object_detection directory
- copiar as pastas abaixo para models/object_detection: data, images, ssd..., training
- rodei deu erro: pandas.core.computation' has no attribute 'expressions
- executei: conda update dask
- outro erro: ModuleNotFoundError: No module named 'deployment'
- dentro de research rodar: protoc object_detection/protos/*.proto --python_out=.
And...
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
- copiar training/object-detection.pbtxt para dentro da pasta object-detection/data
- comando para treinar: python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
- detalhes para treinar em https://pythonprogramming.net/training-custom-objects-tensorflow-object-detection-api-tutorial/?completed=/creating-tfrecord-files-tensorflow-object-detection-api-tutorial/
- ESTA TREINANDO
- QUERO ACOMPANHAR COM TENSOR BOARD
- conda install -c anaconda tensorflow-tensorboard
- python -m tensorboard.main --logdir='training'
- acessar http://mbp-de-henrique.lan:6006/
---------------AGORA PRECISO EXPORTAR O MODELO E TESTA-LO---------------------
- dentro de research rodar: export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
- dentro do object-detection: 
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-1241 \
    --output_directory celular_inference_graph
- Booting up jupyter notebook and opening the object_detection_tutorial.ipynb, let's make a few changes. First, head to the Variables section, and let's change the model name, and the paths to the checkpoint and the labels:
- Next, we can just delete the entire Download Model section, since we don't need to download anymore.
- RESULTADOS
- DESCARTAR checkpoint 1241 % sim: 0.0 % nao: 1.0 % sim com distancia < 0.005: 0.0
- checkpoint 2355 % sim: 1.0 % nao: 0.0 % sim com distancia < 0.05: 0.0
- checkpoint 8087 % sim: 1.0 % nao: 0.0 % sim com distancia < 0.05: 0.015503875968992248
- checkpoint 14171 % sim: 1.0 % nao: 0.0 % sim com distancia < 0.05: 0.007751937984496124
- checkpoint 16767 % sim: 1.0 129 % nao: 0.0 0 % sim com distancia < 0.05: 0.9922480620155039 128
- ok CONTINUAR TREINANDO A REDE E PERIODICAMENTE TESTAR DESEMPENHO
- SEPARAR CÓDIGO EM 2 SCRIPTS PYTHON
- script celularDetection.py
    -- ok criar versão zero
    -- ok alterar forma de receber parâmetros
- script celularTraining.py
    -- ok criar versão zero
    -- alterar forma de receber parametros
- ARRUMAR CÓDIGOS, COLOCAR COMENTARIOS E HEADER COM EXPLICACAO DO TRABALHO
- EMPACOTAR OS 2 SCRIPTS E OS OUTROS ARQUIVOS
- RENOMEAR OS 2 SCRIPTS
- ESCREVER RELATÓRIO










PAREI AQUI https://pythonprogramming.net/creating-tfrecord-files-tensorflow-object-detection-api-tutorial/ -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


- Estudar dataset.
- Configurações
      -- Instalar as bibliotecas: pillow, lxml, jupyter e matplotlib
      -- Baixar e extrair zip: https://github.com/tensorflow/models
      -- Entrar na pasta "/models-master/research" e executar o comando "protoc object_detection/protos/*.proto --python_out=."
      -- Na pasta "/models-master/research/object_detection" abri o Jupyter "object_detection_tutorial.ipynb"
      -- Executar e a saída será um cachorro ;)
- Abrir Jupyter com "jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000"
- {'id': 77, 'name': 'cell phone'}
- [ymin, xmin, ymax, xmax] [0.08139385 0.7841443  0.20364265 0.87176406] 0.jpg 0.8306 0.1350

# observações
- todas as imagens possuem um celular
- labels.txt armazena a base de treinamentos com 129 (nomeDaImagem.jpg x y)
- 8 imagens de testes
- Entregar 2 scripts: 
      > python train_phone_finder.py ~/find_phone
      > python find_phone.py ~/find_phone_test_images/51.jpg
        0.2551 0.3129
 - A phone is considered to be detected correctly on a test image if your output is within a radius of 0.05 (normalized distance) centered on the phone.
- your algorithm is expected to detect a phone correctly on 4 out of the 8 test images and to detect at least 70% correctly on the provided labeled dataset. 
- If you do not have enough time, please focus on a submission with clean, well structured code, rather than on the perfect performance.
- Additionally, you are welcome to attach your notes regarding possible next steps for your detector and ways to improve the data collection of the customer.

# enunciado
You’ve been tasked to implement a prototype of a visual object detection system for a customer. The task is to find a location of a phone dropped on the floor from a single RGB camera image. Read the attached PDF for further instructions.

# trabalhoML2
Código do trabalho 2 da disciplina Machine Learning do doutorado no Instituto de Computação da UFF no primeiro semestre de 2018.
