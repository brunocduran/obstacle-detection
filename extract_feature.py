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
    # Filtra apenas arquivos de imagem válidos
    filenames = [f for f in os.listdir(DATASET_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
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

    # Extrai features usando VGG16
    model_type = 'VGG16'
    modelVGG16, preprocessing_functionVGG16, image_sizeVGG16 = create_model(model_type)
    features_VGG16 = extract_features(df, modelVGG16, preprocessing_functionVGG16, image_sizeVGG16)

    # Extrai features usando VGG19
    model_type = 'VGG19'
    modelVGG19, preprocessing_functionVGG19, image_sizeVGG19 = create_model(model_type)
    features_VGG19 = extract_features(df, modelVGG19, preprocessing_functionVGG19, image_sizeVGG19)

    # Concatenar as features extraídas de VGG16 e VGG19
    features = np.hstack((features_VGG16, features_VGG19))

    end = time.time()

    time_feature_extraction = end - start

    return features, time_feature_extraction


def create_model(model_type):
    IMAGE_CHANNELS = 3
    POOLING = None  # Nenhum, 'avg', 'max'

    # Carrega o modelo e a função de pré-processamento
    if model_type == 'VGG16':
        image_size = (224, 224)
        from keras.applications.vgg16 import VGG16, preprocess_input
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING,
                      input_shape=image_size + (IMAGE_CHANNELS,))

    elif model_type == 'VGG19':
        image_size = (224, 224)
        from keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING,
                      input_shape=image_size + (IMAGE_CHANNELS,))

    else:
        raise ValueError("Error: Model not implemented.")

    preprocessing_function = preprocess_input

    from keras.layers import Flatten
    from keras.models import Model

    output = Flatten()(model.layers[-1].output)
    model = Model(inputs=model.inputs, outputs=output)

    return model, preprocessing_function, image_size


def extract_features(df, model, preprocessing_function, image_size):
    # Atualiza os nomes de categoria
    df["category"] = df["category"].replace({1: 'clear', 0: 'non-clear'})

    datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
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
# Carregando as imagens em um dataframe
df = load_data()

# Extraindo as características das imagens
features, time_feature_extraction = feature_model_extract(df)

# Convertendo as características em um dataframe
df_csv = pd.DataFrame(features)

# Salvando o dataframe em um arquivo CSV
df_csv.to_csv(RESULT_PATH)

print(f"Extração de features concluída em {time_feature_extraction:.2f} segundos.")
