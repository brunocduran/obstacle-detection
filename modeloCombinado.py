from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate, Input
import tensorflow as tf
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Função para renomear camadas de maneira profunda
def rename_layers(model, prefix):
    for layer in model.layers:
        layer._name = prefix + layer.name
    return model

# Criar o caminho completo para o arquivo do modelo denso
dense_model_path = os.path.join(BASE_PATH, 'modelo', 'model', 'model.h5')

# Carregar a parte densa já treinada
model_dense = load_model(dense_model_path)

# Renomear as camadas do modelo denso
model_dense = rename_layers(model_dense, 'dense_')

MODEL_PATH = os.path.join(BASE_PATH, 'modelo', 'combined_model')
TFLITE_MODEL_COMBINED_PATH = os.path.join(BASE_PATH, 'modelo', 'combined_model.tflite')

# CNN Parameters
IMAGE_CHANNELS = 3
POOLING = None  # None, 'avg', 'max'
image_size = (224, 224)

# Criar uma única entrada
input_image = Input(shape=(224, 224, 3))

# Carregar os modelos VGG16 e VGG19 pré-treinados e aplicá-los na mesma entrada
vgg16_base = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
vgg16_base = rename_layers(vgg16_base, 'vgg16_')
vgg16_features = vgg16_base(input_image)

vgg19_base = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
vgg19_base = rename_layers(vgg19_base, 'vgg19_')
vgg19_features = vgg19_base(input_image)

# Congelar as camadas do VGG16 e VGG19
for layer in vgg16_base.layers:
    layer.trainable = False
for layer in vgg19_base.layers:
    layer.trainable = False

# Achatar as features de VGG16 e VGG19
flatten_vgg16 = Flatten()(vgg16_features)
flatten_vgg19 = Flatten()(vgg19_features)

# Concatenar as features extraídas de VGG16 e VGG19
concatenated_features = Concatenate()([flatten_vgg16, flatten_vgg19])

# Passar as features concatenadas para a parte densa já treinada
dense_output = model_dense(concatenated_features)

# Criar o novo modelo que aceita imagem bruta como entrada e utiliza VGG16 + VGG19 para extração de features
model_combined = Model(inputs=input_image, outputs=dense_output)

# Compilar o modelo combinado
model_combined.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Salvar o modelo Keras
model_combined_path = os.path.join(MODEL_PATH, 'model_combined_image_input_vgg16_vgg19.h5')
model_combined.save(model_combined_path)

# Converter para TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model_combined)
tflite_model = converter.convert()

# Salvar o modelo TensorFlow Lite
with open(TFLITE_MODEL_COMBINED_PATH, 'wb') as f:
    f.write(tflite_model)

print("Modelo combinado gerado e salvo com sucesso.")
