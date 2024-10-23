import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    # Carregar a imagem com a dimensão esperada
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)  # Converter para array NumPy
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão extra para batch (1, 224, 224, 3)
    img_array = preprocess_input(img_array)  # Aplicar a normalização esperada para VGG16/VGG19
    return img_array

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'modelo', 'combined_model')
model_combined_path = os.path.join(MODEL_PATH, 'model_combined_image_input_vgg16_vgg19.h5')

# Carregar o modelo combinado
model_combined = load_model(model_combined_path)

# Caminho para a imagem de teste
img_path = 'C:\\PythonProjects\\obstacle-detection\\images-treino\\nonclear.120.jpg'  # Substitua pelo caminho da sua imagem

# Pré-processar a imagem
preprocessed_img = load_and_preprocess_image(img_path)

# Fazer a previsão
prediction = model_combined.predict(preprocessed_img)

# Interpretar o resultado da previsão
if prediction >= 0.5:
    print("Nenhum obstaculo detectado")
else:
    print("Obstaculo detectado.")
