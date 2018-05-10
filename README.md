TO DO
- Ler enunciado do trabalho no classroom.
- Estudar dataset.
- Configurações
      -- Instalar as bibliotecas: pillow, lxml, jupyter e matplotlib
      -- Baixar e extrair zip: https://github.com/tensorflow/models
      -- Entrar na pasta "/models-master/research" e executar o comando "protoc object_detection/protos/*.proto --python_out=."
      -- Na pasta "/models-master/research/object_detection" abri o Jupyter "object_detection_tutorial.ipynb"
      -- Executar e a saída será um cachorro ;)
- Abrir Jupyter com "jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000"
- {'id': 77, 'name': 'cell phone'}

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
