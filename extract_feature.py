# Esse arquivo é responsável por gerar as features das imagens da base de teste

# base libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import time
import os

# transformation
from keras_preprocessing.image import ImageDataGenerator

# Garantir reprodutibilidade dos resultados
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Definindo paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, 'images-treino')
RESULT_PATH = os.path.join(BASE_PATH, 'features', 'features.csv')

def load_data():
    # Definir extensões de arquivos válidos (imagens)
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Listar apenas arquivos que possuem extensões de imagem válidas
    filenames = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid_extensions)]

    categories = []

    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'clear':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df

def feature_model_extract(df):
    start = time.time()

    # Extrai features usando MobileNetV2
    model_type = 'MobileNetV2'
    modelMobileNetV2, preprocessing_functionMobileNetV2, image_sizeMobileNetV2 = create_model(model_type)
    features_MobileNetV2 = extract_features(df, modelMobileNetV2, preprocessing_functionMobileNetV2, image_sizeMobileNetV2)

    end = time.time()

    time_feature_extraction = end - start

    return features_MobileNetV2, time_feature_extraction

def create_model(model_type):
    IMAGE_CHANNELS = 3
    POOLING = 'avg'  # 'avg' pooling para MobileNetV2

    # Carrega o modelo e a função de pré-processamento
    if model_type == 'MobileNetV2':
        image_size = (224, 224)
        from keras.api.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING,
                            input_shape=image_size + (IMAGE_CHANNELS,))

    else:
        raise ValueError("Error: Model not implemented.")

    preprocessing_function = preprocess_input

    return model, preprocessing_function, image_size

def extract_features(df, model, preprocessing_function, image_size):
    # Atualiza os nomes de categoria
    df["category"] = df["category"].replace({1: 'clear', 0: 'non-clear'})

    datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=20,  # Rotação aleatória até 20 graus
        width_shift_range=0.2,  # Deslocamento horizontal
        height_shift_range=0.2,  # Deslocamento vertical
        shear_range=0.2,  # Cisalhamento aleatório
        zoom_range=0.2,  # Zoom aleatório
        horizontal_flip=True,  # Inversão horizontal
        fill_mode='nearest'  # Como preencher os novos pixels gerados
    )

    total = df.shape[0]
    batch_size = 4

    # Calcula o número correto de steps
    steps = int(np.ceil(total / batch_size))

    generator = datagen.flow_from_dataframe(
        df,
        DATASET_PATH,
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Realiza a predição com base no número de steps calculado
    features = model.predict(generator, steps=steps)

    return features

# ----------------------- MAIN ------------------------------------------------
def main_extract_feature():
    # Carregando as imagens em um dataframe
    df = load_data()

    # Extraindo as características das imagens
    features, time_feature_extraction = feature_model_extract(df)

    # Convertendo as características em um dataframe
    df_csv = pd.DataFrame(features)

    # Salvando o dataframe em um arquivo CSV
    df_csv.to_csv(RESULT_PATH)

    print(f"Extração de features concluída em {time_feature_extraction:.2f} segundos.")

# Este bloco garante que o código seja executado apenas quando o arquivo for executado diretamente
if __name__ == "__main__":
    main_extract_feature()