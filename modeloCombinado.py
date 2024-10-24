from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
import tensorflow as tf
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Função para renomear camadas de maneira profunda
def rename_layers(model, prefix):
    for layer in model.layers:
        layer._name = prefix + layer.name
    return model

def gerarModeloCombinado():
    # Criar o caminho completo para o arquivo do modelo denso
    dense_model_path = os.path.join(BASE_PATH, 'modelo', 'model', 'model.h5')

    # Carregar a parte densa já treinada
    model_dense = load_model(dense_model_path)

    # Renomear as camadas do modelo denso
    model_dense = rename_layers(model_dense, 'dense_')

    MODEL_PATH = os.path.join(BASE_PATH, 'modelo', 'combined_model')
    TFLITE_MODEL_COMBINED_PATH = os.path.join(BASE_PATH, 'modelo', 'combined_model.tflite')

    # Parâmetros do modelo
    IMAGE_CHANNELS = 3
    image_size = (224, 224)

    # Criar uma única entrada para as imagens
    input_image = Input(shape=(224, 224, 3))

    # Carregar o MobileNetV2 pré-treinado
    mobilenetv2_base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=image_size + (IMAGE_CHANNELS,))
    mobilenetv2_base = rename_layers(mobilenetv2_base, 'mobilenetv2_')

    # Aplicar MobileNetV2 na entrada da imagem
    mobilenetv2_features = mobilenetv2_base(input_image)

    # Congelar as camadas do MobileNetV2
    for layer in mobilenetv2_base.layers:
        layer.trainable = False

    # Achatar as features de MobileNetV2
    flatten_mobilenetv2 = Flatten()(mobilenetv2_features)

    # Passar as features achatadas para a parte densa já treinada
    dense_output = model_dense(flatten_mobilenetv2)

    # Criar o novo modelo que aceita imagem bruta como entrada e utiliza MobileNetV2 para extração de features
    model_combined = Model(inputs=input_image, outputs=dense_output)

    # Compilar o modelo combinado
    model_combined.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Salvar o modelo Keras
    model_combined_path = os.path.join(MODEL_PATH, 'model_combined_image_input_mobilenetv2.h5')
    model_combined.save(model_combined_path)

    # Converter para TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model_combined)
    tflite_model = converter.convert()

    # Salvar o modelo TensorFlow Lite
    with open(TFLITE_MODEL_COMBINED_PATH, 'wb') as f:
        f.write(tflite_model)

    print("Modelo combinado com MobileNetV2 gerado e salvo com sucesso.")

if __name__ == "__main__":
    gerarModeloCombinado()
